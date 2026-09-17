//! `pgraph` binary. A Rust port of the Bitcoin Payment Graph pipeline by
//! **Matteo Loporchio**.
//!
//! This file is dispatch and orchestration only: thread-pool setup, the
//! disk-space and node-count preflight, and the fused pipelines that
//! `build_pg.sh` and `builder.sh` used to drive with `java` and GNU `sort`.
//! Logger setup lives next door in [`logging`]; all of the algorithm lives in
//! the library.
//!
//! It is the only file in the crate that may use `anyhow`: the library returns
//! typed `PgError`s so that `--lenient` and `--on-missing-source` can *match*
//! on them, and the binary adds human-readable context on top. Exit codes are
//! exactly 0 and 1, the only two the Java program ever produced.

use std::fs::{self, File};
use std::io::{BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::Parser;
use dsi_progress_logger::prelude::*;
use log::{debug, info, warn};

use utxo2webgraph::arcs::{
    ArcFileFormat, ArcSorter, BinaryArcSink, SortOpts, SortStats, SortedArcs, TsvArcSink,
};
use utxo2webgraph::cli::{
    ArcFormatArg, BuildArgs, BuildPgArgs, Cli, Command, CommonOpts, CompressArgs, EdgeListArgs,
    InputFormatArg, InputSelector, NumNodesSpec, SortEdgesArgs, SplitArgs, StageArg,
};
use utxo2webgraph::compress::{self, CompressOpts};
use utxo2webgraph::edge_list::{self, EdgeListOpts};
use utxo2webgraph::nodemap::DenseNodeMap;
use utxo2webgraph::split::{self, ChunkChain, SplitOpts};
use utxo2webgraph::{
    free_space_bytes, take_iter_error, total_memory_bytes, ArcCodec, ArcSink, NodeId,
};

mod logging;

/// The `BufReader` size used for every transaction-list read. Shared with the
/// library so the `split` stage and the edge-list stage agree.
const READ_BUFFER: usize = utxo2webgraph::split::SPLIT_READ_BUF;

/// Upper bound on the default memory budget, however much RAM the box has.
const MEMORY_CAP: u64 = 192 * (1u64 << 30);

/// Measured output slots per input byte (0.01603 in chunk_15, 0.01647 in
/// chunk_28); used only for the space preflight.
const OUTPUTS_PER_BYTE: f64 = 0.0165;

/// Measured raw arcs per input byte (0.05505 .. 0.08395 across samples).
const ARCS_PER_BYTE: f64 = 0.070;

/// Bytes of text edge list produced per input byte.
const EL_BYTES_PER_BYTE: f64 = 1.55;

/// Bytes of text node map produced per input byte.
const NM_BYTES_PER_BYTE: f64 = 0.41;

/// The three files `BVGraph.store` writes, plus the optional Elias-Fano index.
///
/// Spelled out here rather than imported from `webgraph`, because only
/// `compress.rs` may depend on that crate.
const BVGRAPH_EXTENSIONS: [&str; 3] = ["graph", "offsets", "properties"];
/// Extension of the optional random-access index (`--build-ef`).
const EF_EXTENSION: &str = "ef";

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            // `{:#}` prints the whole anyhow context chain, so a dangling
            // reference surfaces as
            //   Error: edge-list stage failed: line 412, tx 18601: input #0
            //   references unknown output (17000, 1); ...
            // which is the diagnostic the Java NullPointerException never gave.
            eprintln!("Error: {e:#}");
            ExitCode::from(1)
        }
    }
}

fn run() -> Result<()> {
    // 1. Parse.
    let cli = Cli::parse();
    let common = cli.common.clone();

    // 2. Logger. Never stdout: that is the Java-compatible statistics channel,
    //    and a log line there would break a naive diff. stderr always, plus a
    //    copy in `<--log-dir>/<stage>.log`, which is what `build_pg.sh`
    //    produced (`logs/pg_el_builder.log`, `logs/webgraph_builder.err`) and
    //    what the directory is created for.
    let stage = stage_name(&cli.command);
    logging::init(common.verbose, &common.log_dir, stage);

    // 3. TMPDIR, before any thread exists. NON-NEGOTIABLE: webgraph's external
    //    sort calls bare `tempfile::tempdir()` and ignores
    //    `BvCompConf::tmp_dir`, so without this a parallel N=28 run spills tens
    //    of GB into /tmp on the system disk.
    std::env::set_var("TMPDIR", &common.tmp_dir);

    // 4. One global Rayon pool, built once.
    let threads = resolve_threads(&common);
    if let Err(e) = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
    {
        // Only possible if something already initialised the pool; not fatal.
        warn!("could not set the global Rayon pool to {threads} threads: {e}");
    }

    // 5. Working directories.
    create_dir(&common.tmp_dir)?;
    create_dir(&common.log_dir)?;
    // A run killed by a signal never runs `TempDir`'s destructor; reap what
    // dead pids left behind before the space preflight measures the volume.
    let reaped = utxo2webgraph::arcs::reap_stale_run_dirs(&common.tmp_dir);
    if reaped > 0 {
        info!(
            "reaped {reaped} stale spill director{}",
            if reaped == 1 { "y" } else { "ies" }
        );
    }

    let memory = resolve_memory(&common);
    info!(
        "threads={threads}  memory={}  tmp-dir={}  log-dir={}  mode={:?}  stats={:?}",
        human_bytes(memory),
        common.tmp_dir.display(),
        common.log_dir.display(),
        common.mode(),
        common.stats_style()
    );

    // 6/7. Preflight and dispatch live in the per-stage functions, which know
    //      their own inputs and outputs.
    match cli.command {
        Command::Split(ref a) => cmd_split(&common, a).context("split stage failed"),
        Command::EdgeList(ref a) => cmd_edge_list(&common, a).context("edge-list stage failed"),
        Command::SortEdges(ref a) => cmd_sort_edges(&common, a).context("sort-edges stage failed"),
        Command::Compress(ref a) => cmd_compress(&common, a).context("compress stage failed"),
        Command::BuildPg(ref a) => cmd_build_pg(&common, a).context("build-pg stage failed"),
        Command::Build(ref a) => cmd_build(&common, a).context("build stage failed"),
    }
}

// ---------------------------------------------------------------- startup

/// The subcommand's name, used for the log file name.
fn stage_name(c: &Command) -> &'static str {
    match c {
        Command::Split(_) => "split",
        Command::EdgeList(_) => "edge-list",
        Command::SortEdges(_) => "sort-edges",
        Command::Compress(_) => "compress",
        Command::BuildPg(_) => "build-pg",
        Command::Build(_) => "build",
    }
}

/// Copies the Java-compatible statistics lines (which go to stdout) into the
/// log, so `<log-dir>/<stage>.log` is a complete record of the run.
fn log_java_stats(stats: &edge_list::EdgeListStats) {
    info!(
        "statistics: Processed: {} transactions (after {} seconds). / Nodes: {}\tEdges: {}",
        stats.tx_count, stats.elapsed_secs, stats.node_slots, stats.edge_count
    );
}

/// Resolved worker-thread count.
fn resolve_threads(c: &CommonOpts) -> usize {
    c.threads
        .filter(|&n| n > 0)
        .unwrap_or_else(|| {
            // NOT `rayon::current_num_threads()`: merely calling that
            // initialises the global pool with Rayon's own defaults, so the
            // `build_global()` below would then fail with "already
            // initialized" and `--threads` would be silently ignored.
            // `available_parallelism` is what Rayon's default would have been
            // anyway, and it touches no pool.
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        })
        .max(1)
}

/// Resolved memory budget: `--memory`, else `min(45% of RAM, 192 GiB)`.
fn resolve_memory(c: &CommonOpts) -> u64 {
    c.memory.unwrap_or_else(|| {
        let fraction = (total_memory_bytes() as f64 * 0.45) as u64;
        fraction.clamp(1u64 << 30, MEMORY_CAP)
    })
}

/// `arcs_per_run = memory / (2 * arc_size)`.
///
/// The factor 2 is the radix sort's scratch buffer, which is the same size as
/// the data. Dropping it would make a nominal 128 GiB budget cost 256 GiB of
/// RSS and OOM on the first full-scale run.
fn arcs_per_run(memory: u64, codec: ArcCodec) -> u64 {
    (memory / (2 * codec.record_size() as u64)).max(1)
}

fn create_dir(p: &Path) -> Result<()> {
    fs::create_dir_all(p).with_context(|| format!("could not create directory {}", p.display()))
}

/// Creates the parent directory of an output file, if it has one.
fn create_parent(p: &Path) -> Result<()> {
    match p.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => create_dir(parent),
        _ => Ok(()),
    }
}

fn human_bytes(n: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = n as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{n} B")
    } else {
        format!("{value:.2} {}", UNITS[unit])
    }
}

// ------------------------------------------------------------- guardrails

/// Resolves the `InputSelector` into an ordered list of files.
///
/// HARD RULE: all paths are validated here, BEFORE any work starts.
/// `builder.sh` checked each chunk's existence inside its copy loop, so a
/// missing `chunk_14` surfaced only after 13 chunks had already been copied
/// into a temporary file.
fn resolve_inputs(sel: &InputSelector, chunks: Option<usize>) -> Result<Vec<PathBuf>> {
    let paths = match (&sel.chunk_dir, chunks) {
        (Some(dir), Some(n)) => {
            if n == 0 {
                bail!("--chunks must be at least 1");
            }
            (1..=n)
                .map(|i| dir.join(format!("chunk_{i:02}.txt")))
                .collect::<Vec<_>>()
        }
        (Some(_), None) => bail!("--chunk-dir requires --chunks"),
        (None, _) => sel.input.clone(),
    };
    if paths.is_empty() {
        bail!("no input files selected; pass -i FILE... or --chunk-dir DIR --chunks N");
    }
    for p in &paths {
        if !p.is_file() {
            bail!("input file not found: {}", p.display());
        }
    }
    Ok(paths)
}

/// Stat every resolved input and return their total size in bytes.
///
/// A pure probe: it writes nothing, logs nothing and refuses nothing. The
/// total is load-bearing downstream — it feeds `estimate`, the `require_space`
/// disk preflight, `guard_projected_node_count` and every `--dry-run` plan.
fn measure_inputs(paths: &[PathBuf]) -> Result<u64> {
    let mut total = 0u64;
    for p in paths {
        let len = fs::metadata(p)
            .with_context(|| format!("could not stat {}", p.display()))?
            .len();
        total = total.saturating_add(len);
    }
    Ok(total)
}

/// HARD RULE: stay inside the project. Outputs default to relative paths under
/// the working directory; anything absolute or escaping upwards was typed by
/// the user, so it is allowed but logged, never silently obeyed.
fn note_output_location(what: &str, path: &Path) {
    let escapes = path.is_absolute()
        || path
            .components()
            .any(|c| matches!(c, std::path::Component::ParentDir));
    if escapes {
        warn!(
            "{what} will be written outside the working directory, at {} (explicit path)",
            path.display()
        );
    } else {
        debug!("{what} -> {}", path.display());
    }
}

/// Space estimates derived from the input size, used by the preflight.
#[derive(Copy, Clone, Debug)]
struct Estimate {
    outputs: f64,
    arcs: f64,
    edge_list_bytes: u64,
    node_map_bytes: u64,
    arc_bytes: u64,
}

fn estimate(input_bytes: u64, codec: ArcCodec) -> Estimate {
    let bytes = input_bytes as f64;
    let arcs = bytes * ARCS_PER_BYTE;
    Estimate {
        outputs: bytes * OUTPUTS_PER_BYTE,
        arcs,
        edge_list_bytes: (bytes * EL_BYTES_PER_BYTE) as u64,
        node_map_bytes: (bytes * NM_BYTES_PER_BYTE) as u64,
        arc_bytes: (arcs * codec.record_size() as f64) as u64,
    }
}

/// Refuses to start rather than filling a volume.
fn require_space(path: &Path, needed: u64, what: &str) -> Result<()> {
    match free_space_bytes(path) {
        Some(free) => {
            if free < needed {
                bail!(
                    "not enough free space for {what}: {} needs about {}, but only {} is free \
                     (use --tmp-dir to point at another filesystem, or --no-text-edge-list)",
                    path.display(),
                    human_bytes(needed),
                    human_bytes(free)
                );
            }
            debug!(
                "space check: {} needs ~{}, {} free on {}",
                what,
                human_bytes(needed),
                human_bytes(free),
                path.display()
            );
            Ok(())
        }
        None => {
            warn!(
                "could not determine free space on {}; skipping the {what} space check",
                path.display()
            );
            Ok(())
        }
    }
}

/// Refuses a run whose projected node count the 32-bit WebGraph format cannot
/// hold, BEFORE the hours of work that would be thrown away.
///
/// `compress::prepare` rejects `num_nodes > i32::MAX` — `it.unimi.dsi.webgraph`
/// (as opposed to `it.unimi.dsi.big.webgraph`) is 32-bit throughout — but that
/// is the very last stage. With `--num-nodes node-map` the count is the node
/// map's `next_id`, which is proportional to the input and therefore
/// predictable here: at N = 28 the estimate is ~2.23e9, above
/// `i32::MAX = 2.147e9`, so a `pgraph build 28 --num-nodes node-map` used to
/// die after the read pass, the sort, a 51 GiB node map and a 195 GiB edge
/// list.
///
/// `from-arcs` (the default) cannot be predicted this way — it is the largest
/// endpoint actually seen — so it only gets a warning.
fn guard_projected_node_count(spec: NumNodesSpec, projected_outputs: f64) -> Result<()> {
    const LIMIT: f64 = i32::MAX as f64;
    let over = projected_outputs > LIMIT;
    match spec {
        NumNodesSpec::NodeMap if over => bail!(
            "--num-nodes node-map would need about {projected_outputs:.3e} nodes, above the \
             i32::MAX = {} the 32-bit WebGraph format can hold; the compressor would reject \
             the graph after hours of work. Use --num-nodes from-arcs (what build_pg.sh \
             produced), or split the input into fewer chunks.",
            i32::MAX
        ),
        _ if over => {
            warn!(
                "the input is projected to create about {projected_outputs:.3e} output slots, \
                 above i32::MAX; the graph can only be written if the arcs reach fewer than \
                 {} distinct nodes",
                i32::MAX
            );
            Ok(())
        }
        _ => Ok(()),
    }
}

/// Warns when `/proc/sys/vm/max_map_count` is too low for the parallel path.
///
/// webgraph's `ParSortPairs` memory-maps every batch, and its own docs warn
/// about `ENOMEM` from this limit. Better to see it now than at the merge, six
/// hours in.
fn check_max_map_count(threads: usize, parallel: bool) {
    if !parallel {
        return;
    }
    let expected = threads.saturating_mul(threads).max(1024);
    match fs::read_to_string("/proc/sys/vm/max_map_count") {
        Ok(text) => match text.trim().parse::<usize>() {
            Ok(limit) if limit < expected => warn!(
                "/proc/sys/vm/max_map_count is {limit}, below the ~{expected} mappings the \
                 parallel path may need; raise it with `sysctl -w vm.max_map_count={}`",
                expected * 2
            ),
            Ok(limit) => debug!("/proc/sys/vm/max_map_count = {limit} (need ~{expected})"),
            Err(_) => debug!("could not parse /proc/sys/vm/max_map_count"),
        },
        Err(_) => debug!("/proc/sys/vm/max_map_count is unreadable on this system"),
    }
}

// ------------------------------------------------------ atomic file output

/// An output file written to a sibling temporary path and renamed on success.
///
/// `build_pg.sh` had no `set -e`, and `(sort | uniq) > $EL_FILE` truncates the
/// target *before* `sort` runs. That is exactly how `graph/pg_el_1.tsv`,
/// `pg_el_10.tsv` and `pg_el_15.tsv` all came to be 0 bytes in this checkout
/// while the script cheerfully continued to the next stage. Nothing here ever
/// destroys a good output before its replacement is complete.
struct AtomicOut {
    final_path: PathBuf,
    work_path: PathBuf,
    rename: bool,
}

impl AtomicOut {
    /// Prepares an output. Non-regular destinations such as `/dev/null` are
    /// written in place, since they cannot be renamed onto.
    fn new(final_path: &Path) -> Result<Self> {
        create_parent(final_path)?;
        let is_special = fs::metadata(final_path)
            .map(|m| !m.is_file())
            .unwrap_or(false);
        if is_special {
            return Ok(Self {
                final_path: final_path.to_path_buf(),
                work_path: final_path.to_path_buf(),
                rename: false,
            });
        }
        let name = final_path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "out".to_string());
        let work = final_path.with_file_name(format!(".{name}.pgtmp{}", std::process::id()));
        Ok(Self {
            final_path: final_path.to_path_buf(),
            work_path: work,
            rename: true,
        })
    }

    /// The path the producer must write to.
    fn path(&self) -> &Path {
        &self.work_path
    }

    /// Publishes the finished file under its final name.
    fn commit(self) -> Result<()> {
        if self.rename {
            fs::rename(&self.work_path, &self.final_path).with_context(|| {
                format!(
                    "could not rename {} to {}",
                    self.work_path.display(),
                    self.final_path.display()
                )
            })?;
        }
        Ok(())
    }
}

impl Drop for AtomicOut {
    fn drop(&mut self) {
        // A dropped-without-commit output is a failed run: remove the partial
        // file instead of leaving a plausible-looking truncated one behind.
        if self.rename && self.work_path.exists() {
            let _ = fs::remove_file(&self.work_path);
        }
    }
}

/// A dot-free sibling basename for the three BVGraph files.
///
/// Dot-free because `sanitize_basename` rejects a `.` in a basename: webgraph's
/// `with_extension` would strip everything after it.
fn bvgraph_temp_basename(final_basename: &Path) -> PathBuf {
    let name = final_basename
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "pg".to_string());
    final_basename.with_file_name(format!("{name}-pgtmp{}", std::process::id()))
}

/// Renames `<tmp>.graph`, `.offsets`, `.properties` (and `.ef`) into place.
fn commit_bvgraph(tmp: &Path, final_basename: &Path, with_ef: bool) -> Result<()> {
    let mut exts: Vec<&str> = BVGRAPH_EXTENSIONS.to_vec();
    if with_ef {
        exts.push(EF_EXTENSION);
    }
    for ext in exts {
        let from = tmp.with_extension(ext);
        if !from.exists() {
            if ext == EF_EXTENSION {
                continue;
            }
            bail!(
                "the compressor did not produce {}; treating the run as incomplete \
                 (webgraph writes .properties last, so a missing one means a crash)",
                from.display()
            );
        }
        let to = final_basename.with_extension(ext);
        fs::rename(&from, &to)
            .with_context(|| format!("could not rename {} to {}", from.display(), to.display()))?;
    }
    Ok(())
}

/// Removes any leftover temporary BVGraph files after a failure.
fn clean_bvgraph_temp(tmp: &Path) {
    for ext in BVGRAPH_EXTENSIONS
        .iter()
        .chain(std::iter::once(&EF_EXTENSION))
    {
        let _ = fs::remove_file(tmp.with_extension(ext));
    }
}

// ------------------------------------------------------------------ split

fn cmd_split(common: &CommonOpts, a: &SplitArgs) -> Result<()> {
    let opts = SplitOpts {
        start: a.start.clone(),
        end: a.end.clone(),
        months: a.months,
        tz: a.tz.clone(),
        append: a.append,
        create_empty: !a.no_create_empty,
        skip_out_of_range: a.skip_out_of_range,
    };

    let boundaries = split::compute_boundaries(&opts)?;
    let table = split::format_boundary_table(&boundaries);

    // `--show-boundaries` is a pure query: print and stop.
    if a.show_boundaries {
        print!("{table}");
        std::io::stdout().flush().ok();
        return Ok(());
    }
    info!(
        "chunk boundaries ({} chunks, tz={}):\n{table}",
        boundaries.upper.len(),
        boundaries.tz
    );

    let inputs = vec![a.input.clone()];
    let input_bytes = measure_inputs(&inputs)?;
    note_output_location("chunks", &a.output_dir);

    if common.dry_run {
        println!("DRY RUN: pgraph split");
        println!(
            "  input       : {} ({})",
            a.input.display(),
            human_bytes(input_bytes)
        );
        println!("  output dir  : {}", a.output_dir.display());
        println!("  timezone    : {}", opts.tz);
        println!("  chunks      : {}", boundaries.upper.len());
        println!(
            "  mode        : {}",
            if opts.append { "append" } else { "truncate" }
        );
        print!("{table}");
        return Ok(());
    }

    // The split is a pure sequential copy: input bytes in, the same bytes out.
    create_dir(&a.output_dir)?;
    require_space(&a.output_dir, input_bytes, "the chunk files")?;

    let file =
        File::open(&a.input).with_context(|| format!("could not open {}", a.input.display()))?;
    let reader = BufReader::with_capacity(READ_BUFFER, file);

    let started = Instant::now();
    let mut pl = progress_logger![display_memory = true, log_interval = common.log_interval];
    let stats = split::split(reader, &a.output_dir, &opts, &mut pl)?;

    info!(
        "split done in {:?}: {} records written, {} dropped before --start",
        started.elapsed(),
        stats.records_written,
        stats.records_before_start
    );
    if let Some((record, ts)) = stats.stopped_early_at {
        // `splitter.sh` used `exit`, not `next`. Preserved, but never silent.
        warn!(
            "stopped early at record {record} (timestamp {ts} is at or after --end); \
             everything after it was NOT read. Pass --skip-out-of-range to keep scanning."
        );
    }
    // The per-chunk table makes the run self-verifying against the boundaries.
    for c in &stats.chunks {
        info!(
            "chunk {:02}  {:>12} records  {:>12}  first_ts={}  last_ts={}  {}",
            c.index,
            c.records,
            human_bytes(c.bytes),
            c.first_ts
                .map(|t| t.to_string())
                .unwrap_or_else(|| "-".into()),
            c.last_ts
                .map(|t| t.to_string())
                .unwrap_or_else(|| "-".into()),
            c.path.display()
        );
    }
    Ok(())
}

// -------------------------------------------------------------- edge-list

fn cmd_edge_list(common: &CommonOpts, a: &EdgeListArgs) -> Result<()> {
    let paths = resolve_inputs(&a.inputs, a.chunks)?;
    let input_bytes = measure_inputs(&paths)?;
    let codec: ArcCodec = a.arc_codec.into();
    let est = estimate(input_bytes, codec);

    note_output_location("raw arcs", &a.arcs);
    if let Some(nm) = &a.node_map {
        note_output_location("node map", nm);
    }

    if common.dry_run {
        println!("DRY RUN: pgraph edge-list");
        print_input_plan(&paths, input_bytes)?;
        println!("  arcs        : {} ({:?})", a.arcs.display(), a.arc_format);
        match &a.node_map {
            Some(p) => println!("  node map    : {}", p.display()),
            None => println!("  node map    : (not written)"),
        }
        print_estimate_plan(&est, a.arc_format == ArcFormatArg::Tsv);
        return Ok(());
    }

    // Space preflight: refuse rather than fill the volume.
    let arcs_bytes = match a.arc_format {
        ArcFormatArg::Tsv => est.edge_list_bytes,
        ArcFormatArg::Binary => est.arc_bytes,
    };
    require_space(&a.arcs, arcs_bytes, "the raw arc file")?;
    if let Some(nm) = &a.node_map {
        require_space(nm, est.node_map_bytes, "the node map")?;
    }

    let mut node_map = DenseNodeMap::sized(common.mode(), a.max_tx_id);

    // The sinks are themselves atomic (`arcs::AtomicOut`: sibling temp file,
    // fsync where it means anything, rename on success, unlink on failure), so
    // they are handed the FINAL path. Wrapping them in a second atomic layer
    // renamed the file twice through two different temp-name conventions and
    // left `..<name>.pgtmp<pid>.pgraph-tmp-<pid>` orphans that nothing cleaned.
    create_parent(&a.arcs)?;
    let mut sink: Box<dyn ArcSink> = match a.arc_format {
        ArcFormatArg::Tsv => Box::new(TsvArcSink::create(&a.arcs)?),
        ArcFormatArg::Binary => Box::new(BinaryArcSink::create(&a.arcs, codec)?),
    };

    let opts = EdgeListOpts {
        mode: common.mode(),
        on_missing_source: a.on_missing_source.into(),
        progress_every: a.progress_every,
        stats: common.stats_style(),
        quiet_stats: false,
    };

    // Java measured from just before the read loop (line 47) and INCLUDED the
    // node-map write in the final elapsed time (line 95). Same here.
    let started = Instant::now();
    let reader = BufReader::with_capacity(READ_BUFFER, ChunkChain::new(paths)?);
    let mut pl = progress_logger![display_memory = true, log_interval = common.log_interval];
    let mut stats = edge_list::build_edge_list(reader, &mut node_map, &mut *sink, opts, &mut pl)?;
    drop(sink);

    if let Some(nm_path) = &a.node_map {
        // `write_node_map` opens the path itself, so this one keeps main's
        // atomic wrapper.
        let phase = Instant::now();
        info!("writing the node map to {}", nm_path.display());
        let nm_out = AtomicOut::new(nm_path)?;
        let mut pl = progress_logger![display_memory = true, log_interval = common.log_interval];
        let written = edge_list::write_node_map(&node_map, nm_out.path(), &mut pl)?;
        nm_out.commit()?;
        info!(
            "node map written in {:?}: {} rows, {}",
            phase.elapsed(),
            node_map.next_id(),
            human_bytes(written)
        );
    }

    stats.elapsed_secs = started.elapsed().as_secs();
    edge_list::print_stats(&stats, common.stats_style());
    log_java_stats(&stats);
    report_edge_list_stats(&stats);
    Ok(())
}

/// Logs the counters that `--stats java` cannot show without breaking the
/// byte-compatible stdout contract.
fn report_edge_list_stats(s: &edge_list::EdgeListStats) {
    info!(
        "transactions={} output_slots={} distinct_nodes={} raw_arcs={}",
        s.tx_count, s.node_slots, s.distinct_nodes, s.edge_count
    );
    if s.skipped_lines > 0 {
        warn!("{} malformed line(s) skipped", s.skipped_lines);
    }
    if s.blank_lines > 0 {
        warn!("{} blank line(s) skipped", s.blank_lines);
    }
    if s.non_utf8_lines > 0 {
        warn!(
            "{} line(s) were not valid UTF-8 and were decoded with U+FFFD substitution, \
             as Java's InputStreamReader did",
            s.non_utf8_lines
        );
    }
    if s.dangling_refs > 0 {
        warn!(
            "{} dangling input reference(s); the Java program would have died on the first one",
            s.dangling_refs
        );
    }
    if s.zero_output_txs > 0 {
        warn!(
            "{} transaction(s) with no outputs emitted no edges (Java emitted a phantom edge \
             into node 0 for each of their inputs)",
            s.zero_output_txs
        );
    }
    if s.distinct_nodes < s.node_slots {
        warn!(
            "`Nodes:` reports {} output slots, but only {} distinct nodes exist; \
             Java's counter (line 64) counts slots, so a repeated txId overcounts",
            s.node_slots, s.distinct_nodes
        );
    } else if s.distinct_nodes > s.node_slots {
        warn!(
            "`Nodes:` reports {} output slots, but {} distinct nodes exist; the \
             extra {} were minted by --on-missing-source create and are not outputs \
             of any transaction in this input",
            s.node_slots,
            s.distinct_nodes,
            s.distinct_nodes - s.node_slots
        );
    }
}

// ------------------------------------------------------------- sort-edges

fn cmd_sort_edges(common: &CommonOpts, a: &SortEdgesArgs) -> Result<()> {
    if a.edge_list.is_none() && a.sorted_arcs.is_none() {
        bail!("nothing to do: pass --edge-list FILE and/or --sorted-arcs FILE");
    }
    let inputs = vec![a.input.clone()];
    let input_bytes = measure_inputs(&inputs)?;
    let threads = resolve_threads(common);
    let memory = resolve_memory(common);
    let codec: ArcCodec = a.arc_codec.into();

    if let Some(p) = &a.edge_list {
        note_output_location("sorted edge list", p);
    }
    if let Some(p) = &a.sorted_arcs {
        note_output_location("sorted binary arcs", p);
    }

    if common.dry_run {
        println!("DRY RUN: pgraph sort-edges");
        println!(
            "  arcs        : {} ({})",
            a.input.display(),
            human_bytes(input_bytes)
        );
        println!("  sort algo   : {:?}   codec: {codec:?}", a.sort_algo);
        println!(
            "  memory      : {}  ({} arcs/run)",
            human_bytes(memory),
            arcs_per_run(memory, codec)
        );
        println!("  dedup       : {}", !a.no_dedup);
        return Ok(());
    }

    let opts = SortOpts {
        algo: a.sort_algo.into(),
        codec,
        dedup: !a.no_dedup,
        memory_bytes: memory,
        tmp_dir: common.tmp_dir.clone(),
        threads,
        keep_intermediate: common.keep_intermediate,
    };
    info!(
        "sorting with algo={:?} codec={codec:?} memory={} ({} arcs per run) threads={threads}",
        opts.algo,
        human_bytes(memory),
        arcs_per_run(memory, codec)
    );

    let mut pl = progress_logger![display_memory = true, log_interval = common.log_interval];
    let (sorted, stats) = sort_arc_file(&a.input, a.input_format, codec, opts, &mut pl)?;
    report_sort_stats(&stats);

    // `write_tsv`/`write_binary` go through `arcs::AtomicOut`, which already
    // writes to a sibling temp file and renames on success.
    if let Some(p) = &a.edge_list {
        create_parent(p)?;
        let n = sorted.write_tsv(p, &mut pl)?;
        info!("wrote {n} arcs to {}", p.display());
    }
    if let Some(p) = &a.sorted_arcs {
        create_parent(p)?;
        let n = sorted.write_binary(p, &mut pl)?;
        info!("wrote {n} arcs to {}", p.display());
    }
    Ok(())
}

/// Reads an arc file in the requested (or sniffed) format and sorts it.
fn sort_arc_file(
    path: &Path,
    format: InputFormatArg,
    codec: ArcCodec,
    opts: SortOpts,
    pl: &mut impl ProgressLog,
) -> Result<(SortedArcs, SortStats)> {
    let resolved = resolve_format(path, format, codec)?;
    let mut sorter = ArcSorter::new(opts)?;
    pl.item_name("arc");
    pl.start(format!("Reading the arcs of {}...", path.display()));
    match resolved {
        ArcFileFormat::Tsv => {
            let (iter, slot) = utxo2webgraph::arcs::read_tsv_arcs(path)?;
            for (src, dst) in iter {
                sorter.push(src as NodeId, dst as NodeId)?;
                pl.light_update();
            }
            take_iter_error(&slot)?;
        }
        ArcFileFormat::Binary(c) => {
            let (iter, slot) = utxo2webgraph::arcs::read_binary_arcs(path, c)?;
            for (src, dst) in iter {
                sorter.push(src as NodeId, dst as NodeId)?;
                pl.light_update();
            }
            take_iter_error(&slot)?;
        }
    }
    pl.done();
    sorter.finish()?;
    Ok(sorter.into_sorted(pl)?)
}

/// Resolves `--input-format`, sniffing the file when it says `auto`.
fn resolve_format(path: &Path, format: InputFormatArg, codec: ArcCodec) -> Result<ArcFileFormat> {
    Ok(match format {
        InputFormatArg::Auto => {
            // The codec matters even here: `auto` only sniffs text versus
            // binary; a fixed-width file's record width is not inferable from
            // its length, so `--arc-codec` decides it in both arms.
            let f = utxo2webgraph::arcs::detect_format(path, codec)?;
            debug!("sniffed {} as {f:?}", path.display());
            f
        }
        InputFormatArg::Tsv => ArcFileFormat::Tsv,
        InputFormatArg::Binary => ArcFileFormat::Binary(codec),
    })
}

fn report_sort_stats(s: &SortStats) {
    info!(
        "sorted {} raw arcs into {} distinct ({} runs, {} spilled, max node id {})",
        s.raw_arcs,
        s.distinct_arcs,
        s.runs,
        human_bytes(s.spilled_bytes),
        s.max_node_id
    );
    if s.duplicates_removed > 0 {
        // Measured zero across chunk_01, chunk_05 and a 200 MB sample of
        // chunk_20, so a non-zero count is worth looking at.
        warn!(
            "{} duplicate arc(s) removed; the corpus is expected to contain none",
            s.duplicates_removed
        );
    }
}

// --------------------------------------------------------------- compress

fn cmd_compress(common: &CommonOpts, a: &CompressArgs) -> Result<()> {
    compress::sanitize_basename(&a.output_prefix)?;
    let inputs = vec![a.input.clone()];
    let input_bytes = measure_inputs(&inputs)?;
    let threads = resolve_threads(common);
    let memory = resolve_memory(common);
    let codec: ArcCodec = a.arc_codec.into();
    note_output_location("BVGraph", &a.output_prefix);
    check_max_map_count(threads, a.parallel);

    if common.dry_run {
        println!("DRY RUN: pgraph compress");
        println!(
            "  edge list   : {} ({})",
            a.input.display(),
            human_bytes(input_bytes)
        );
        println!(
            "  output      : {}.{{graph,offsets,properties}}",
            a.output_prefix.display()
        );
        println!("  num nodes   : {:?}", a.num_nodes);
        println!("  parallel    : {}", a.parallel);
        return Ok(());
    }

    let format = resolve_format(&a.input, a.input_format, codec)?;

    // Pass 1: count the arcs and find the largest endpoint. `compress_*_iter`
    // needs the arc count, and `--num-nodes from-arcs` needs `max + 1`.
    let phase = Instant::now();
    let (num_arcs, max_node_id) = scan_arcs(&a.input, format, common.log_interval)?;
    info!(
        "edge list: {num_arcs} arcs, max node id {max_node_id} (scanned in {:?})",
        phase.elapsed()
    );
    if num_arcs == 0 && !a.allow_empty {
        // Java's ArcListASCIIGraph died here with
        // `Expected integer, found Token[EOF], line 1`, two stages downstream
        // of the real failure. Fail here, clearly, instead.
        return Err(utxo2webgraph::PgError::EmptyEdgeList.into());
    }

    let from_arcs = if num_arcs == 0 {
        0
    } else {
        max_node_id as usize + 1
    };
    let num_nodes = resolve_num_nodes(a.num_nodes, a.node_map.as_deref(), from_arcs)?;

    let opts = CompressOpts {
        num_nodes,
        tmp_dir: common.tmp_dir.clone(),
        threads,
        memory_bytes: memory,
        parallel: a.parallel,
        build_ef: a.build_ef,
        allow_empty: a.allow_empty,
        verify: a.verify.into(),
        log_interval: common.log_interval,
    };

    // Pass 2: stream the sorted arcs straight into the compressor. No
    // `ArrayListMutableGraph`, hence no `-Xmx200g`.
    let tmp = bvgraph_temp_basename(&a.output_prefix);
    create_parent(&a.output_prefix)?;
    let result = (|| -> Result<compress::CompressStats> {
        let stats = match format {
            ArcFileFormat::Tsv => {
                let (iter, slot) = utxo2webgraph::arcs::read_tsv_arcs(&a.input)?;
                let s = compress::compress_sorted_iter(iter, num_arcs, &tmp, &opts)?;
                take_iter_error(&slot)?;
                s
            }
            ArcFileFormat::Binary(c) => {
                let (iter, slot) = utxo2webgraph::arcs::read_binary_arcs(&a.input, c)?;
                let s = compress::compress_sorted_iter(iter, num_arcs, &tmp, &opts)?;
                take_iter_error(&slot)?;
                s
            }
        };
        compress::verify_graph(&tmp, num_nodes, num_arcs, opts.verify)?;
        Ok(stats)
    })();
    let stats = match result {
        Ok(s) => s,
        Err(e) => {
            clean_bvgraph_temp(&tmp);
            return Err(e);
        }
    };
    commit_bvgraph(&tmp, &a.output_prefix, a.build_ef)?;
    info!(
        "wrote {}.{{graph,offsets,properties}}: {} nodes, {} arcs, {} bits",
        a.output_prefix.display(),
        stats.nodes,
        stats.arcs,
        stats.bits
    );
    Ok(())
}

/// Counts the arcs and finds the largest endpoint in one streaming pass.
///
/// 195 GiB of text at `N = 28`: a silent nine-minute phase without a progress
/// logger. A fixed-width binary file has an exact arc count in its length, so
/// that case gets a percentage and an ETA; a TSV's line count does not exist
/// until it has been read, so that case reports a rate and a running total.
fn scan_arcs(
    path: &Path,
    format: ArcFileFormat,
    log_interval: std::time::Duration,
) -> Result<(u64, NodeId)> {
    let mut count = 0u64;
    let mut max = 0u64;
    let mut pl = progress_logger![
        item_name = "arc",
        display_memory = true,
        log_interval = log_interval
    ];
    if let ArcFileFormat::Binary(c) = format {
        let len = fs::metadata(path)
            .with_context(|| format!("could not stat {}", path.display()))?
            .len();
        pl.expected_updates(Some((len / c.record_size() as u64) as usize));
    }
    pl.start(format!("Scanning {} to count the arcs...", path.display()));
    match format {
        ArcFileFormat::Tsv => {
            let (iter, slot) = utxo2webgraph::arcs::read_tsv_arcs(path)?;
            for (src, dst) in iter {
                count += 1;
                max = max.max(src as u64).max(dst as u64);
                pl.light_update();
            }
            take_iter_error(&slot)?;
        }
        ArcFileFormat::Binary(c) => {
            let (iter, slot) = utxo2webgraph::arcs::read_binary_arcs(path, c)?;
            for (src, dst) in iter {
                count += 1;
                max = max.max(src as u64).max(dst as u64);
                pl.light_update();
            }
            take_iter_error(&slot)?;
        }
    }
    pl.done_with_count(count as usize);
    Ok((count, max))
}

/// Resolves `--num-nodes`, logging both candidate values when they differ.
///
/// `node-map` makes every UTXO a node; `from-arcs` reproduces
/// `ArcListASCIIGraph`'s `max(id) + 1`, which is smaller because recent
/// unspent outputs never appear in an arc. The choice changes the
/// `.properties` `nodes=` value and every node-indexed downstream array.
fn resolve_num_nodes(
    spec: NumNodesSpec,
    node_map: Option<&Path>,
    from_arcs: usize,
) -> Result<usize> {
    match spec {
        NumNodesSpec::FromArcs => {
            info!(
                "num_nodes = {from_arcs} (from-arcs: max endpoint + 1, as ArcListASCIIGraph did)"
            );
            Ok(from_arcs)
        }
        NumNodesSpec::Explicit(n) => {
            if n < from_arcs {
                bail!(
                    "--num-nodes {n} is smaller than the {from_arcs} nodes the arcs require; \
                     webgraph would silently drop the out-of-range arcs"
                );
            }
            info!("num_nodes = {n} (explicit; from-arcs would give {from_arcs})");
            Ok(n)
        }
        NumNodesSpec::NodeMap => {
            let path = node_map
                .ok_or_else(|| anyhow::anyhow!("--num-nodes node-map requires --node-map FILE"))?;
            let next_id = node_map_next_id(path)?;
            announce_num_nodes(next_id, from_arcs);
            if next_id < from_arcs {
                bail!(
                    "the node map at {} declares {next_id} nodes, fewer than the {from_arcs} the \
                     arcs require; the two files do not belong to the same run",
                    path.display()
                );
            }
            Ok(next_id)
        }
    }
}

fn announce_num_nodes(node_map: usize, from_arcs: usize) {
    if node_map == from_arcs {
        info!("num_nodes = {node_map} (node-map and from-arcs agree)");
    } else {
        info!(
            "num_nodes = {node_map} (node map); from-arcs would give {from_arcs}, \
             {} fewer - the highest-numbered outputs are unspent and appear in no arc",
            node_map - from_arcs
        );
    }
}

/// `max(id) + 1` over a `txId \t offset \t id` node map.
///
/// Reading the maximum id is stricter than counting lines: it is correct even
/// if the file was concatenated, deduplicated or reordered.
fn node_map_next_id(path: &Path) -> Result<usize> {
    use std::io::BufRead;
    let file = File::open(path)
        .with_context(|| format!("could not open the node map {}", path.display()))?;
    let reader = BufReader::with_capacity(READ_BUFFER, file);
    let mut max: u64 = 0;
    let mut rows: u64 = 0;
    for (i, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("while reading {}", path.display()))?;
        let text = line.trim_end_matches(['\r', '\n']);
        if text.is_empty() {
            continue;
        }
        let id = text.rsplit('\t').next().unwrap_or("");
        let id: u64 = id.parse().with_context(|| {
            format!(
                "node map {} line {}: third column {:?} is not an integer",
                path.display(),
                i + 1,
                id
            )
        })?;
        max = max.max(id);
        rows += 1;
    }
    if rows == 0 {
        bail!("the node map {} is empty", path.display());
    }
    debug!("node map {}: {rows} rows, max id {max}", path.display());
    Ok(max as usize + 1)
}

// --------------------------------------------------------------- build-pg

fn cmd_build_pg(common: &CommonOpts, a: &BuildPgArgs) -> Result<()> {
    compress::sanitize_basename(&a.output_prefix)?;
    let paths = resolve_inputs(&a.inputs, a.chunks)?;
    let input_bytes = measure_inputs(&paths)?;
    let threads = resolve_threads(common);
    let memory = resolve_memory(common);
    let codec: ArcCodec = a.arc_codec.into();
    let est = estimate(input_bytes, codec);
    check_max_map_count(threads, a.parallel);
    // BEFORE any work: `compress::prepare` rejects num_nodes > i32::MAX, and
    // that check used to fire only at the very last stage — after the read
    // pass, the sort, the 51 GiB node map and the 195 GiB edge list. At N=28
    // the node-map estimate is ~2.23e9, above i32::MAX.
    guard_projected_node_count(a.num_nodes, est.outputs)?;

    note_output_location("node map", &a.node_map);
    note_output_location("edge list", &a.edge_list);
    note_output_location("BVGraph", &a.output_prefix);

    if common.dry_run {
        println!("DRY RUN: pgraph build-pg");
        print_input_plan(&paths, input_bytes)?;
        println!(
            "  node map    : {}",
            if a.no_node_map {
                "(not written)".to_string()
            } else {
                a.node_map.display().to_string()
            }
        );
        println!(
            "  edge list   : {}",
            if a.no_text_edge_list {
                "(not written)".to_string()
            } else {
                a.edge_list.display().to_string()
            }
        );
        println!(
            "  BVGraph     : {}.{{graph,offsets,properties}}",
            a.output_prefix.display()
        );
        println!(
            "  sort algo   : {:?}   codec: {codec:?}   parallel: {}",
            a.sort_algo, a.parallel
        );
        println!(
            "  memory      : {}  ({} arcs/run)",
            human_bytes(memory),
            arcs_per_run(memory, codec)
        );
        print_estimate_plan(&est, !a.no_text_edge_list);
        return Ok(());
    }

    // Space preflight, before a single byte is written.
    let mut tmp_needed = est.arc_bytes;
    if arcs_per_run(memory, codec) >= est.arcs as u64 {
        // The whole arc set fits in one in-RAM run: nothing spills.
        tmp_needed = 0;
    }
    if tmp_needed > 0 {
        require_space(&common.tmp_dir, tmp_needed, "the sort spill runs")?;
    }
    if !a.no_text_edge_list {
        require_space(&a.edge_list, est.edge_list_bytes, "the text edge list")?;
    }
    if !a.no_node_map {
        require_space(&a.node_map, est.node_map_bytes, "the node map")?;
    }

    let sort_opts = SortOpts {
        algo: a.sort_algo.into(),
        codec,
        dedup: true,
        memory_bytes: memory,
        tmp_dir: common.tmp_dir.clone(),
        threads,
        keep_intermediate: common.keep_intermediate,
    };
    info!(
        "build-pg: {} input file(s), {} of transactions; ~{:.3e} outputs, ~{:.3e} raw arcs expected",
        paths.len(),
        human_bytes(input_bytes),
        est.outputs,
        est.arcs
    );

    let mut node_map = DenseNodeMap::sized(common.mode(), a.max_tx_id);
    let mut sorter = ArcSorter::new(sort_opts)?;

    let el_opts = EdgeListOpts {
        mode: common.mode(),
        on_missing_source: a.on_missing_source.into(),
        progress_every: a.progress_every,
        stats: common.stats_style(),
        quiet_stats: false,
    };

    let started = Instant::now();
    let reader = BufReader::with_capacity(READ_BUFFER, ChunkChain::new(paths)?);
    let mut pl = progress_logger![display_memory = true, log_interval = common.log_interval];
    let mut stats =
        edge_list::build_edge_list(reader, &mut node_map, &mut sorter, el_opts, &mut pl)?;

    if !a.no_node_map {
        let phase = Instant::now();
        info!(
            "writing the node map ({} rows) to {}",
            node_map.next_id(),
            a.node_map.display()
        );
        let out = AtomicOut::new(&a.node_map)?;
        let bytes = edge_list::write_node_map(&node_map, out.path(), &mut pl)?;
        out.commit()?;
        info!(
            "node map written in {:?}: {}",
            phase.elapsed(),
            human_bytes(bytes)
        );
    }

    // Java's elapsed time (line 95) is measured from before the read loop and
    // includes writing the node map. Reproduced exactly.
    stats.elapsed_secs = started.elapsed().as_secs();
    edge_list::print_stats(&stats, common.stats_style());
    log_java_stats(&stats);
    report_edge_list_stats(&stats);

    let node_map_next_id = node_map.next_id();
    drop(node_map);

    let phase = Instant::now();
    info!("sorting {} raw arcs", stats.edge_count);
    let (sorted, sort_stats) = sorter.into_sorted(&mut pl)?;
    info!("sort finished in {:?}", phase.elapsed());
    report_sort_stats(&sort_stats);

    if !a.no_text_edge_list {
        // `write_tsv` is atomic on its own; see `cmd_edge_list`.
        let phase = Instant::now();
        info!("writing the text edge list to {}", a.edge_list.display());
        create_parent(&a.edge_list)?;
        let n = sorted.write_tsv(&a.edge_list, &mut pl)?;
        info!(
            "wrote {n} sorted arcs to {} in {:?}",
            a.edge_list.display(),
            phase.elapsed()
        );
    }
    if common.keep_intermediate {
        let path = kept_arcs_path(&common.tmp_dir, &a.output_prefix);
        let n = sorted.write_binary(&path, &mut pl)?;
        info!("kept {n} sorted binary arcs at {}", path.display());
    }

    let from_arcs = if sorted.num_arcs() == 0 {
        0
    } else {
        sorted.max_node_id() as usize + 1
    };
    let num_nodes = match a.num_nodes {
        // In the fused pipeline the node map is still in memory, so
        // `node-map` never has to re-read a 55 GB file.
        NumNodesSpec::NodeMap => {
            announce_num_nodes(node_map_next_id as usize, from_arcs);
            node_map_next_id as usize
        }
        other => resolve_num_nodes(other, Some(&a.node_map), from_arcs)?,
    };

    if sorted.num_arcs() == 0 && !a.allow_empty {
        return Err(utxo2webgraph::PgError::EmptyEdgeList.into());
    }

    let comp_opts = CompressOpts {
        num_nodes,
        tmp_dir: common.tmp_dir.clone(),
        threads,
        memory_bytes: memory,
        parallel: a.parallel,
        build_ef: false,
        allow_empty: a.allow_empty,
        verify: a.verify.into(),
        log_interval: common.log_interval,
    };
    compress_and_commit(&sorted, &a.output_prefix, &comp_opts)?;
    info!("build-pg completed in {:?}", started.elapsed());
    Ok(())
}

/// Compresses into a dot-free temporary basename, verifies, then renames.
fn compress_and_commit(sorted: &SortedArcs, basename: &Path, opts: &CompressOpts) -> Result<()> {
    create_parent(basename)?;
    let tmp = bvgraph_temp_basename(basename);
    let result = (|| -> Result<compress::CompressStats> {
        let stats = compress::compress_sorted(sorted, &tmp, opts)?;
        compress::verify_graph(&tmp, opts.num_nodes, sorted.num_arcs(), opts.verify)?;
        Ok(stats)
    })();
    let stats = match result {
        Ok(s) => s,
        Err(e) => {
            clean_bvgraph_temp(&tmp);
            return Err(e);
        }
    };
    commit_bvgraph(&tmp, basename, opts.build_ef)?;
    info!(
        "wrote {}.{{graph,offsets,properties}}: {} nodes, {} arcs, {} bits",
        basename.display(),
        stats.nodes,
        stats.arcs,
        stats.bits
    );
    Ok(())
}

/// Where `--keep-intermediate` leaves the sorted binary arcs, and where
/// `--from-stage sort-edges` looks for them.
fn kept_arcs_path(tmp_dir: &Path, output_prefix: &Path) -> PathBuf {
    let stem = output_prefix
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "pg".to_string());
    tmp_dir.join(format!("{stem}_arcs.bin"))
}

// ------------------------------------------------------------------ build

fn cmd_build(common: &CommonOpts, a: &BuildArgs) -> Result<()> {
    if a.chunks == 0 {
        bail!("N must be a positive integer");
    }
    // Validate every chunk BEFORE any work: `builder.sh` checked inside its
    // copy loop, so a missing chunk_14 surfaced after 13 chunks had been
    // copied into a 135 GB temporary file.
    let paths: Vec<PathBuf> = (1..=a.chunks)
        .map(|i| a.chunk_dir.join(format!("chunk_{i:02}.txt")))
        .collect();
    for p in &paths {
        if !p.is_file() {
            bail!("chunk file not found: {}", p.display());
        }
    }

    // Exactly the names `build_pg.sh` derived, so `graph/` and any downstream
    // tooling stay compatible.
    let n = a.chunks;
    let node_map = a.graph_dir.join(format!("pg_nm_{n}.tsv"));
    let edge_list = a.graph_dir.join(format!("pg_el_{n}.tsv"));
    let output_prefix = a.graph_dir.join(format!("pg_{n}"));
    create_dir(&a.graph_dir)?;

    match a.from_stage {
        None | Some(StageArg::EdgeList) => {
            let args = BuildPgArgs {
                inputs: InputSelector {
                    input: paths,
                    chunk_dir: None,
                },
                chunks: None,
                node_map,
                edge_list,
                output_prefix,
                no_text_edge_list: a.no_text_edge_list,
                no_node_map: a.no_node_map,
                sort_algo: a.sort_algo,
                arc_codec: a.arc_codec,
                max_tx_id: a.max_tx_id,
                on_missing_source: a.on_missing_source,
                num_nodes: a.num_nodes,
                progress_every: a.progress_every,
                parallel: a.parallel,
                allow_empty: a.allow_empty,
                verify: a.verify,
            };
            cmd_build_pg(common, &args)
        }
        Some(StageArg::SortEdges) => {
            // Resume from the binary arcs left by --keep-intermediate.
            let arcs = kept_arcs_path(&common.tmp_dir, &output_prefix);
            if !arcs.is_file() {
                bail!(
                    "--from-stage sort-edges needs {}, which a previous run with \
                     --keep-intermediate would have left there",
                    arcs.display()
                );
            }
            info!("resuming at sort-edges from {}", arcs.display());
            let sort_args = SortEdgesArgs {
                input: arcs,
                // Binary, written by this very binary with `--arc-codec`.
                input_format: InputFormatArg::Binary,
                edge_list: if a.no_text_edge_list {
                    None
                } else {
                    Some(edge_list.clone())
                },
                sorted_arcs: None,
                sort_algo: a.sort_algo,
                arc_codec: a.arc_codec,
                no_dedup: false,
            };
            if sort_args.edge_list.is_none() {
                bail!("--from-stage sort-edges with --no-text-edge-list has nothing to produce");
            }
            cmd_sort_edges(common, &sort_args)?;
            let compress_args = CompressArgs {
                input: edge_list,
                output_prefix,
                input_format: InputFormatArg::Tsv,
                arc_codec: a.arc_codec,
                num_nodes: a.num_nodes,
                node_map: Some(node_map),
                parallel: a.parallel,
                build_ef: false,
                allow_empty: a.allow_empty,
                verify: a.verify,
            };
            cmd_compress(common, &compress_args)
        }
        Some(StageArg::Compress) => {
            if !edge_list.is_file() {
                bail!(
                    "--from-stage compress needs {}, which a previous run would have written",
                    edge_list.display()
                );
            }
            info!("resuming at compress from {}", edge_list.display());
            let compress_args = CompressArgs {
                input: edge_list,
                output_prefix,
                input_format: InputFormatArg::Tsv,
                arc_codec: a.arc_codec,
                num_nodes: a.num_nodes,
                node_map: Some(node_map),
                parallel: a.parallel,
                build_ef: false,
                allow_empty: a.allow_empty,
                verify: a.verify,
            };
            cmd_compress(common, &compress_args)
        }
    }
}

// -------------------------------------------------------------- dry-run IO

fn print_input_plan(paths: &[PathBuf], total: u64) -> Result<()> {
    println!(
        "  inputs      : {} file(s), {}",
        paths.len(),
        human_bytes(total)
    );
    for p in paths {
        let len = fs::metadata(p).map(|m| m.len()).unwrap_or(0);
        println!("      {:>12}  {}", human_bytes(len), p.display());
    }
    Ok(())
}

fn print_estimate_plan(est: &Estimate, text_edge_list: bool) {
    println!("  estimates   :");
    println!("      output slots (nodes) ~ {:.3e}", est.outputs);
    println!("      raw arcs             ~ {:.3e}", est.arcs);
    println!(
        "      binary arcs          ~ {}",
        human_bytes(est.arc_bytes)
    );
    if text_edge_list {
        println!(
            "      text edge list       ~ {}",
            human_bytes(est.edge_list_bytes)
        );
    }
    println!(
        "      text node map        ~ {}",
        human_bytes(est.node_map_bytes)
    );
}
