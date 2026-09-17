//! Logger setup for the `pgraph` binary.
//!
//! Binary-local on purpose. It cannot live in `cli.rs`, whose own module doc
//! says "No I/O and no business logic live here", and it must not live in
//! `lib.rs`, which would push `env_logger` into the library's dependency
//! surface — the library emits through the `log` facade and leaves the choice
//! of sink to whoever links it.
//!
//! # Where the log goes
//!
//! Never stdout: that is the Java-compatible statistics channel, and a log
//! line there would break a naive `diff` against the reference pipeline.
//! Always stderr, plus a copy in `<--log-dir>/<stage>.log` — the artefact
//! `build_pg.sh` used to leave in `logs/`, and the reason `--log-dir` exists.
//!
//! # Line format
//!
//! A copy of webgraph-rs `cli/src/lib.rs:998-1037`, so a `pgraph` run and a
//! `webgraph` run in the same pipeline produce the same shape:
//!
//! ```text
//! 2026-09-17 10:58:01.123 1m5s200ms INFO [ThreadId(1)] pgraph - threads=112 …
//! ```
//!
//! That is: the UTC wall clock (`jiff::Timestamp::strftime` formats in UTC, so
//! this is the same instant `env_logger` used to print with a trailing `Z`,
//! only without the `T` and the `Z`), a compact span of time elapsed since the
//! logger was installed, the level, the thread id, the `log` target and the
//! message. (`progress_logger!` sets the target to the `module_path!()` of its
//! *construction* site; every pipeline-loop logger is built in `main.rs`, so
//! in practice they all log under `pgraph` and `RUST_LOG` cannot single one of
//! them out.) It replaces `env_logger`'s own
//! `[<UTC timestamp> <LEVEL> <target>]` prefix.
//!
//! # Colour
//!
//! Disabled, explicitly. The tee means the sink is a `Target::Pipe`, which
//! `anstream` never detects as a terminal, so colour was already off in
//! practice; `WriteStyle::Never` states it instead of leaving it to depend on
//! a detection that happens to fail, and guarantees the log FILE can never
//! acquire ANSI escapes.

use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use std::time::{Instant, SystemTime};

use log::{debug, warn};

/// A `Write` that forwards to stderr and, when one could be opened, to
/// `<log-dir>/<stage>.log` as well.
///
/// `--log-dir` used to be created and then never written to, which both
/// implied a contract the binary did not honour and lost the artefacts
/// `build_pg.sh` left in `logs/`. Failing to open or write the file is never
/// fatal: the log is a convenience, the run is not.
struct TeeWriter {
    file: Option<File>,
}

impl Write for TeeWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if let Some(f) = self.file.as_mut() {
            let _ = f.write_all(buf);
        }
        std::io::stderr().write(buf)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        if let Some(f) = self.file.as_mut() {
            let _ = f.flush();
        }
        std::io::stderr().flush()
    }
}

/// Installs `env_logger` on stderr, teed into `<log_dir>/<stage>.log`.
///
/// `verbose` is the `-v` count: 0 = info, 1 = debug, 2 or more = trace.
///
/// `RUST_LOG` *layers on top of* that level rather than replacing it, which is
/// why the builder is `Builder::new().filter_level(level).parse_default_env()`
/// and NOT `Builder::from_env(Env::default().default_filter_or(..))`.
/// `filter_level` pushes a catch-all `Directive { name: None, .. }`
/// (`env_filter/src/filter.rs`), and directive matching falls through to it
/// for any target no explicit directive names — so
/// `RUST_LOG=utxo2webgraph::arcs=debug` raises that one module and leaves
/// everything else at the `-v` level. With `default_filter_or` the default
/// string is consulted only when `RUST_LOG` is *unset*, so the same command
/// would silence `pgraph`, `split`, `compress` and webgraph itself, and `-v`
/// would become a no-op.
///
/// Installing the logger twice is not an error here: `try_init`'s failure is
/// deliberately ignored, so a test harness that already installed one keeps
/// it.
pub fn init(verbose: u8, log_dir: &Path, stage: &str) {
    use jiff::fmt::friendly::{Designator, Spacing, SpanPrinter};
    use jiff::SpanRound;

    let level = match verbose {
        0 => log::LevelFilter::Info,
        1 => log::LevelFilter::Debug,
        _ => log::LevelFilter::Trace,
    };
    // The working directories are created in `run()`, after this; do it here
    // too so the very first log line already lands in the file.
    let _ = fs::create_dir_all(log_dir);
    let path = log_dir.join(format!("{stage}.log"));
    let file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .ok();
    let have_file = file.is_some();

    let start = Instant::now();
    let printer = SpanPrinter::new()
        .spacing(Spacing::None)
        .designator(Designator::Compact);
    let span_round = SpanRound::new()
        .largest(jiff::Unit::Day)
        .smallest(jiff::Unit::Millisecond)
        .days_are_24_hours();

    let mut builder = env_logger::Builder::new();
    builder
        .filter_level(level)
        .parse_default_env()
        .write_style(env_logger::WriteStyle::Never)
        .target(env_logger::Target::Pipe(Box::new(TeeWriter { file })))
        .format(move |buf, record| {
            let Ok(ts) = jiff::Timestamp::try_from(SystemTime::now()) else {
                return Err(std::io::Error::other("Failed to get timestamp"));
            };
            let style = buf.default_level_style(record.level());
            let elapsed = start.elapsed();
            let span = jiff::Span::new()
                .seconds(elapsed.as_secs() as i64)
                .milliseconds(elapsed.subsec_millis() as i64);
            let span = span.round(span_round).expect("Failed to round span");
            writeln!(
                buf,
                "{} {} {style}{}{style:#} [{:?}] {} - {}",
                ts.strftime("%F %T%.3f"),
                printer.span_to_string(&span),
                record.level(),
                std::thread::current().id(),
                record.target(),
                record.args()
            )
        });
    let _ = builder.try_init();

    if have_file {
        debug!("logging to stderr and {}", path.display());
    } else {
        warn!("could not open {} for logging; stderr only", path.display());
    }
}
