//! Chunk splitting and chunked input. Ported from `splitter.sh` and
//! `builder.sh` by **Matteo Loporchio**.
//!
//! Two things live here:
//!
//! * [`ChunkChain`], a concatenating [`Read`] over `chunk_01.txt .. chunk_NN.txt`
//!   that replaces `builder.sh`'s 135 GB temporary file;
//! * [`split`], the six-month chunker that replaces `splitter.sh`'s awk
//!   program, including its **Europe/Rome local-midnight** boundaries.
//!
//! # Timezone
//!
//! `splitter.sh` computes every boundary with a bare
//! `date -d "$Y-$M-01 00:00:00" +%s` — **no `-u`** — on a machine whose
//! `/etc/localtime` points at `Europe/Rome`, so all 28 boundaries are *local*
//! midnights. January boundaries are `UTC - 3600` (CET) and July boundaries
//! `UTC - 7200` (CEST).
//!
//! The 132 GB of chunks already on disk prove it: `head -1 chunks/chunk_02.txt`
//! has `t = 1246400389`, which is **above** the local 2009-07-01 boundary
//! (1246399200) but **below** the UTC one (1246406400). Under UTC boundaries
//! that record would belong to chunk 01.
//!
//! The default is therefore the literal string `"Europe/Rome"` and **not**
//! `$TZ`: pinning the zone means a cron job running with `TZ=UTC` cannot
//! silently produce different chunks from an interactive run.

use std::collections::VecDeque;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use chrono::{Datelike, TimeZone, Utc};
use chrono_tz::Tz;
use dsi_progress_logger::prelude::*;

use crate::{PgError, PgResult};

/// Read buffer used for each file of a [`ChunkChain`].
const CHAIN_BUF: usize = 16 << 20;
/// Read buffer callers should wrap the master transaction list in before
/// handing it to [`split`]. Exported so the binary and the library agree on a
/// single value.
pub const SPLIT_READ_BUF: usize = 32 << 20;
/// Write buffer used by [`split`] for the chunk currently being filled.
const SPLIT_WRITE_BUF: usize = 16 << 20;

// ---------------------------------------------------------------------------
// ChunkChain
// ---------------------------------------------------------------------------

/// A [`Read`] that concatenates several files, in order, with no intermediate
/// copy.
///
/// This is `builder.sh`'s
/// `for i in 1..N: cat chunks/chunk_%02d.txt >> $(mktemp --tmpdir=".")`
/// without the temporary file. That file was written **into the project
/// directory** and its measured cumulative sizes are 4.4 GB at N=10, 23.9 GB at
/// N=15, 62.5 GB at N=20 and 135.3 GB at N=28 — a full write plus a full
/// re-read that produce nothing, and an orphan `combined_chunks_XXXXXX.txt`
/// next to the source data whenever a crash beat the `EXIT` trap.
///
/// Every path is validated **before** any work starts, unlike the shell, which
/// checked existence inside the copy loop and so only discovered a missing
/// `chunk_14.txt` after copying thirteen chunks.
///
/// Callers are expected to wrap this in a [`BufReader`] of their own if they
/// want line-oriented access; each underlying file already gets a 16 MiB
/// buffer.
#[derive(Debug)]
pub struct ChunkChain {
    paths: Vec<PathBuf>,
    queue: VecDeque<PathBuf>,
    cur: Option<BufReader<File>>,
    cur_path: PathBuf,
    total_len: u64,
}

impl ChunkChain {
    /// Builds a chain over `paths`, in the given order.
    ///
    /// Every path must exist and be readable; the error names the offending
    /// file.
    pub fn new(paths: Vec<PathBuf>) -> PgResult<Self> {
        let mut total_len = 0u64;
        for p in &paths {
            let md = fs::metadata(p).map_err(|e| PgError::io(p.clone(), e))?;
            if !md.is_file() {
                return Err(PgError::io(
                    p.clone(),
                    io::Error::new(io::ErrorKind::InvalidInput, "not a regular file"),
                ));
            }
            total_len += md.len();
        }
        Ok(ChunkChain {
            queue: paths.iter().cloned().collect(),
            paths,
            cur: None,
            cur_path: PathBuf::new(),
            total_len,
        })
    }

    /// Builds a chain over `chunk_01.txt .. chunk_NN.txt` inside `dir`.
    ///
    /// This is exactly `./builder.sh N`'s input selection. All `n` files are
    /// validated up front.
    pub fn from_chunk_dir(dir: &Path, n: usize) -> PgResult<Self> {
        if n == 0 {
            return Err(PgError::other(
                "the number of chunks must be at least 1".to_string(),
            ));
        }
        let mut paths = Vec::with_capacity(n);
        for i in 1..=n {
            let p = dir.join(format!("chunk_{i:02}.txt"));
            if !p.is_file() {
                return Err(PgError::io(
                    p,
                    io::Error::new(io::ErrorKind::NotFound, "chunk file not found"),
                ));
            }
            paths.push(p);
        }
        Self::new(paths)
    }

    /// Total number of bytes the chain will yield, known up front.
    pub fn total_len(&self) -> u64 {
        self.total_len
    }

    /// The files this chain reads, in order.
    pub fn paths(&self) -> &[PathBuf] {
        &self.paths
    }
}

impl Read for ChunkChain {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        loop {
            if self.cur.is_none() {
                match self.queue.pop_front() {
                    None => return Ok(0),
                    Some(p) => {
                        let f = File::open(&p).map_err(|e| {
                            io::Error::new(e.kind(), format!("{}: {e}", p.display()))
                        })?;
                        self.cur = Some(BufReader::with_capacity(CHAIN_BUF, f));
                        self.cur_path = p;
                    }
                }
            }
            match self.cur.as_mut().expect("just opened").read(buf) {
                Ok(0) => {
                    self.cur = None;
                    continue;
                }
                Ok(n) => return Ok(n),
                Err(e) => {
                    return Err(io::Error::new(
                        e.kind(),
                        format!("{}: {e}", self.cur_path.display()),
                    ))
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Boundaries
// ---------------------------------------------------------------------------

/// Options for [`split`] and [`compute_boundaries`].
#[derive(Clone, Debug)]
pub struct SplitOpts {
    /// First date, inclusive, as `YYYY-MM-DD`.
    pub start: String,
    /// Last date, exclusive, as `YYYY-MM-DD`. Must land exactly on a boundary.
    pub end: String,
    /// Months per chunk.
    pub months: u32,
    /// IANA timezone the boundaries are computed in; `utc` is accepted.
    pub tz: String,
    /// Append to the chunk files instead of truncating them. This is what
    /// `splitter.sh` did, and it doubles every chunk on a re-run.
    pub append: bool,
    /// Create every chunk file in range, even the ones that receive no record.
    pub create_empty: bool,
    /// Keep scanning past the first out-of-range record instead of stopping.
    pub skip_out_of_range: bool,
}

impl Default for SplitOpts {
    fn default() -> Self {
        SplitOpts {
            start: "2009-01-01".to_string(),
            end: "2023-01-01".to_string(),
            months: 6,
            tz: "Europe/Rome".to_string(),
            append: false,
            create_empty: true,
            skip_out_of_range: false,
        }
    }
}

/// The resolved chunk boundaries.
///
/// `upper[k]` is the **exclusive** upper bound of chunk `k + 1`; chunk 1's
/// inclusive lower bound is `start_ts` and chunk `k + 1`'s is `upper[k - 1]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Boundaries {
    /// Inclusive lower bound of the first chunk, as a Unix timestamp.
    pub start_ts: i64,
    /// Exclusive upper bounds, one per chunk, ascending.
    pub upper: Vec<i64>,
    /// Canonical name of the timezone the bounds were computed in.
    pub tz: String,
}

/// Resolves a timezone name, accepting `utc` in any case.
fn resolve_tz(name: &str) -> PgResult<Tz> {
    if name.eq_ignore_ascii_case("utc") {
        return Ok(Tz::UTC);
    }
    name.parse::<Tz>()
        .map_err(|_| PgError::BadTimezone(name.to_string()))
}

/// Parses `YYYY-MM-DD` into its three components.
fn parse_date(s: &str) -> PgResult<(i32, u32, u32)> {
    let mut it = s.split('-');
    let bad = || PgError::BadDate(s.to_string());
    let y = it.next().ok_or_else(bad)?;
    let m = it.next().ok_or_else(bad)?;
    let d = it.next().ok_or_else(bad)?;
    if it.next().is_some() {
        return Err(bad());
    }
    let y: i32 = y.parse().map_err(|_| bad())?;
    let m: u32 = m.parse().map_err(|_| bad())?;
    let d: u32 = d.parse().map_err(|_| bad())?;
    Ok((y, m, d))
}

/// Local midnight of `(y, m, d)` in `tz`, as a Unix timestamp.
fn midnight(tz: Tz, y: i32, m: u32, d: u32) -> PgResult<i64> {
    tz.with_ymd_and_hms(y, m, d, 0, 0, 0)
        .single()
        .map(|dt| dt.timestamp())
        .ok_or_else(|| {
            PgError::other(format!(
                "{y:04}-{m:02}-{d:02} 00:00:00 does not exist exactly once in {tz}; \
                 pick another boundary or timezone"
            ))
        })
}

/// Computes the chunk boundaries, reproducing `splitter.sh` lines 36-59.
///
/// The shell starts at the start year/month and repeatedly adds `months`,
/// emitting the **resulting** date as the next exclusive upper bound, while the
/// current year is still before the end year. With the defaults that yields the
/// 28 six-month boundaries from 2009-07-01 to 2023-01-01.
///
/// The last boundary is then checked against the end date. `splitter.sh`'s
/// `while (current < n && timestamp >= boundary[current])` clamp is a no-op
/// **only** because `END_TS == boundary[28]`; if `--end` were ever pushed past
/// the last boundary, the final chunk would silently become a catch-all for
/// everything after it. That is why a mismatch is
/// [`PgError::BoundaryMismatch`] rather than a warning.
pub fn compute_boundaries(opts: &SplitOpts) -> PgResult<Boundaries> {
    if opts.months == 0 {
        return Err(PgError::other("--months must be at least 1"));
    }
    let tz = resolve_tz(&opts.tz)?;
    let (sy, sm, sd) = parse_date(&opts.start)?;
    let (ey, em, ed) = parse_date(&opts.end)?;
    let start_ts = midnight(tz, sy, sm, sd)?;
    let end_ts = midnight(tz, ey, em, ed)?;

    let mut upper = Vec::new();
    let (mut y, mut m) = (sy, sm);
    while (y, m) < (ey, em) {
        let total = m - 1 + opts.months;
        y += (total / 12) as i32;
        m = total % 12 + 1;
        upper.push(midnight(tz, y, m, 1)?);
        // A malformed range (e.g. a start after the end) must not spin.
        if upper.len() > 100_000 {
            return Err(PgError::other(
                "refusing to generate more than 100000 chunk boundaries",
            ));
        }
    }

    match upper.last() {
        Some(last) if *last == end_ts => {}
        Some(last) => {
            return Err(PgError::BoundaryMismatch {
                last: *last,
                end: end_ts,
            })
        }
        None => {
            return Err(PgError::BoundaryMismatch {
                last: start_ts,
                end: end_ts,
            })
        }
    }

    Ok(Boundaries {
        start_ts,
        upper,
        tz: tz.name().to_string(),
    })
}

/// Renders the boundary table as plain text.
///
/// Used by `--show-boundaries` and logged at `Info` on every real run, so the
/// timezone choice is always in the record. The `utc` column is the timestamp
/// the same civil midnight would have had under UTC, and `delta` is
/// `local - utc` (−3600 in January, −7200 in July).
pub fn format_boundary_table(b: &Boundaries) -> String {
    let tz = resolve_tz(&b.tz).unwrap_or(Tz::UTC);
    let mut s = String::with_capacity(64 * (b.upper.len() + 3));
    s.push_str(&format!(
        "Chunk boundaries ({}), start {} ({})\n",
        b.tz,
        b.start_ts,
        civil_date(tz, b.start_ts)
    ));
    s.push_str(
        "chunk  interval                    upper bound   local ts       utc ts         delta\n",
    );
    let mut lower = b.start_ts;
    for (i, up) in b.upper.iter().enumerate() {
        let up_date = civil_date(tz, *up);
        let utc_ts = utc_midnight_of(tz, *up);
        s.push_str(&format!(
            "{:>5}  {} -> {}   {}    {:<14} {:<14} {}\n",
            i + 1,
            civil_date(tz, lower),
            up_date,
            up_date,
            up,
            utc_ts,
            up - utc_ts
        ));
        lower = *up;
    }
    s
}

/// `YYYY-MM-DD` of `ts` in `tz`.
fn civil_date(tz: Tz, ts: i64) -> String {
    match tz.timestamp_opt(ts, 0).single() {
        Some(dt) => format!("{:04}-{:02}-{:02}", dt.year(), dt.month(), dt.day()),
        None => "????-??-??".to_string(),
    }
}

/// Timestamp of the UTC midnight of the civil date `ts` names in `tz`.
fn utc_midnight_of(tz: Tz, ts: i64) -> i64 {
    match tz.timestamp_opt(ts, 0).single() {
        Some(dt) => Utc
            .with_ymd_and_hms(dt.year(), dt.month(), dt.day(), 0, 0, 0)
            .single()
            .map(|u| u.timestamp())
            .unwrap_or(ts),
        None => ts,
    }
}

// ---------------------------------------------------------------------------
// split
// ---------------------------------------------------------------------------

/// Per-chunk result of a [`split`] run.
#[derive(Clone, Debug, Default)]
pub struct ChunkStat {
    /// 1-based chunk number, matching the `chunk_NN.txt` file name.
    pub index: usize,
    /// Where the chunk was written.
    pub path: PathBuf,
    /// Records written to this chunk.
    pub records: u64,
    /// Bytes written to this chunk.
    pub bytes: u64,
    /// Timestamp of the first record, if any.
    pub first_ts: Option<i64>,
    /// Timestamp of the last record, if any.
    pub last_ts: Option<i64>,
}

/// Result of a [`split`] run.
#[derive(Clone, Debug, Default)]
pub struct SplitStats {
    /// One entry per chunk, in order.
    pub chunks: Vec<ChunkStat>,
    /// Records dropped because they predate `start`.
    pub records_before_start: u64,
    /// Records written across all chunks.
    pub records_written: u64,
    /// `(record number, timestamp)` of the first record at or after `end`.
    pub stopped_early_at: Option<(u64, i64)>,
}

/// Opens one chunk file, truncating or appending as requested.
fn open_chunk(path: &Path, append: bool) -> PgResult<BufWriter<File>> {
    let mut o = OpenOptions::new();
    o.write(true).create(true);
    if append {
        o.append(true);
    } else {
        o.truncate(true);
    }
    let f = o.open(path).map_err(|e| PgError::io(path, e))?;
    Ok(BufWriter::with_capacity(SPLIT_WRITE_BUF, f))
}

/// Flushes, fsyncs and closes a chunk writer — the awk's `close()`, but with
/// the errors actually checked.
fn finish_chunk(mut w: BufWriter<File>, path: &Path) -> PgResult<()> {
    w.flush().map_err(|e| PgError::io(path, e))?;
    // `fsync` is meaningless (and returns EINVAL) on a non-regular
    // destination; see `crate::sync_if_durable`.
    crate::sync_if_durable(w.get_ref(), path)?;
    Ok(())
}

/// Parses the leading Unix timestamp of a record: everything up to the first
/// `,`. The rest of the line is never touched — and never even decoded, so a
/// record with a stray non-UTF-8 byte past the timestamp is copied through
/// verbatim instead of killing a 135 GB pass.
fn leading_timestamp(line: &[u8]) -> Option<i64> {
    let end = line.iter().position(|b| *b == b',').unwrap_or(line.len());
    std::str::from_utf8(&line[..end]).ok()?.trim().parse().ok()
}

/// Splits a chronologically sorted transaction list into chunk files.
///
/// A faithful port of the awk program in `splitter.sh` (lines 102-121):
/// a single forward pass, exactly one open writer at a time, records copied
/// **verbatim** including their terminator, and `current` never decreasing.
/// A missing final newline is preserved.
///
/// Two deliberate divergences, both documented in the README:
///
/// * **chunk files are truncated by default.** The awk used `>>`, so re-running
///   `splitter.sh` without clearing `chunks/` silently *doubled* every file.
///   [`SplitOpts::append`] restores the old behaviour.
/// * **every chunk file in range is created**, even for an interval with no
///   records. The awk created none, after which `builder.sh` would fail with
///   "chunk file not found". [`SplitOpts::create_empty`] turns this off.
///
/// The awk's `exit` on the first record at or after `end` is preserved for
/// fidelity, but it is now logged loudly with the record number, the timestamp
/// and the bytes consumed so far; [`SplitOpts::skip_out_of_range`] turns it
/// into a `continue`.
///
/// `pl` is ticked once per input record. At full scale this is a 135 GB
/// sequential copy that used to print the boundary table and then nothing at
/// all until it finished. No `expected_updates`: the record count of the
/// master list is not known without reading it, which is what this pass is.
pub fn split<R: BufRead>(
    mut input: R,
    out_dir: &Path,
    opts: &SplitOpts,
    pl: &mut impl ProgressLog,
) -> PgResult<SplitStats> {
    let b = compute_boundaries(opts)?;
    let n = b.upper.len();
    let end_ts = *b
        .upper
        .last()
        .expect("compute_boundaries rejects an empty table");

    fs::create_dir_all(out_dir).map_err(|e| PgError::io(out_dir, e))?;
    let paths: Vec<PathBuf> = (1..=n)
        .map(|i| out_dir.join(format!("chunk_{i:02}.txt")))
        .collect();

    let mut stats = SplitStats {
        chunks: paths
            .iter()
            .enumerate()
            .map(|(i, p)| ChunkStat {
                index: i + 1,
                path: p.clone(),
                ..Default::default()
            })
            .collect(),
        ..Default::default()
    };

    if opts.create_empty {
        for p in &paths {
            let w = open_chunk(p, opts.append)?;
            finish_chunk(w, p)?;
        }
    }

    log::info!("{}", format_boundary_table(&b));
    // "record", matching the `records_written` / `records_before_start`
    // counters `main.rs` reports when the split finishes.
    pl.item_name("record");
    pl.start(format!("Splitting into {n} chunks..."));

    let mut current = 0usize;
    let mut writer: Option<BufWriter<File>> = None;
    // Bytes, not a `String`: the splitter is a byte-exact copy, and
    // `read_line`'s UTF-8 validation would both cost a full validating pass
    // over 135 GB and abort the run on a single stray byte.
    let mut line: Vec<u8> = Vec::with_capacity(256);
    let mut line_no: u64 = 0;
    let mut bytes_read: u64 = 0;

    loop {
        line.clear();
        let read = input.read_until(b'\n', &mut line)?;
        if read == 0 {
            break;
        }
        bytes_read += read as u64;
        line_no += 1;
        pl.light_update();

        if line.iter().all(|b| b.is_ascii_whitespace()) {
            // awk compares an empty $1 as a string and takes the `next`
            // branch, so a blank line is dropped there too.
            continue;
        }
        let ts = match leading_timestamp(&line) {
            Some(ts) => ts,
            None => {
                let end = line.iter().position(|b| *b == b',').unwrap_or(line.len());
                return Err(PgError::BadInteger {
                    line: line_no,
                    field: "timestamp",
                    value: String::from_utf8_lossy(&line[..end]).into_owned(),
                });
            }
        };

        // awk: `if (timestamp < start_ts) next`
        if ts < b.start_ts {
            stats.records_before_start += 1;
            continue;
        }
        // awk: `if (timestamp >= end_ts) exit`
        if ts >= end_ts {
            if stats.stopped_early_at.is_none() {
                stats.stopped_early_at = Some((line_no, ts));
                log::warn!(
                    "record {line_no} has timestamp {ts} >= the end boundary {end_ts}: \
                     {} after {bytes_read} bytes consumed (splitter.sh's awk exited here, \
                     silently dropping the rest of the input)",
                    if opts.skip_out_of_range {
                        "skipping it and continuing"
                    } else {
                        "stopping"
                    }
                );
            }
            if opts.skip_out_of_range {
                continue;
            }
            break;
        }

        // awk: `while (current < n && timestamp >= boundary[current]) { close(...); current++ }`
        // (1-based there, 0-based here, hence the `n - 1` clamp).
        let mut target = current;
        while target < n - 1 && ts >= b.upper[target] {
            target += 1;
        }
        if writer.is_none() || target != current {
            if let Some(w) = writer.take() {
                finish_chunk(w, &paths[current])?;
            }
            writer = Some(open_chunk(&paths[target], opts.append)?);
            current = target;
        }

        let w = writer.as_mut().expect("a writer is open");
        w.write_all(&line)
            .map_err(|e| PgError::io(&paths[current], e))?;

        let c = &mut stats.chunks[current];
        c.records += 1;
        c.bytes += line.len() as u64;
        if c.first_ts.is_none() {
            c.first_ts = Some(ts);
        }
        c.last_ts = Some(ts);
        stats.records_written += 1;
    }

    if let Some(w) = writer.take() {
        finish_chunk(w, &paths[current])?;
    }
    pl.done_with_count(line_no as usize);
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// The 28 exclusive upper bounds produced by `splitter.sh` on this machine,
    /// as measured with `date -d ... +%s` under `Europe/Rome`.
    const ROME_UPPER: [i64; 28] = [
        1246399200, 1262300400, 1277935200, 1293836400, 1309471200, 1325372400, 1341093600,
        1356994800, 1372629600, 1388530800, 1404165600, 1420066800, 1435701600, 1451602800,
        1467324000, 1483225200, 1498860000, 1514761200, 1530396000, 1546297200, 1561932000,
        1577833200, 1593554400, 1609455600, 1625090400, 1640991600, 1656626400, 1672527600,
    ];

    #[test]
    fn default_boundaries_match_the_reference_table() {
        let b = compute_boundaries(&SplitOpts::default()).unwrap();
        assert_eq!(b.tz, "Europe/Rome");
        assert_eq!(b.start_ts, 1230764400);
        assert_eq!(b.upper.len(), 28);
        assert_eq!(b.upper.as_slice(), &ROME_UPPER[..]);
    }

    #[test]
    fn utc_boundaries_differ_from_local_ones() {
        let opts = SplitOpts {
            tz: "utc".to_string(),
            ..Default::default()
        };
        let b = compute_boundaries(&opts).unwrap();
        assert_eq!(b.tz, "UTC");
        assert_eq!(b.start_ts, 1230768000);
        assert_eq!(b.upper[0], 1246406400);
        assert_eq!(b.upper[1], 1262304000);
        assert_eq!(*b.upper.last().unwrap(), 1672531200);
        assert_ne!(b.upper.as_slice(), &ROME_UPPER[..]);
        // The two tables never agree on any boundary: the offset is always
        // 3600 or 7200 seconds.
        for (local, utc) in ROME_UPPER.iter().zip(b.upper.iter()) {
            let d = utc - local;
            assert!(d == 3600 || d == 7200, "unexpected delta {d}");
        }
    }

    #[test]
    fn an_end_off_the_boundary_grid_is_rejected() {
        let opts = SplitOpts {
            end: "2023-02-01".to_string(),
            ..Default::default()
        };
        match compute_boundaries(&opts) {
            Err(PgError::BoundaryMismatch { last, end }) => {
                assert!(last > end, "last {last} should overshoot end {end}");
            }
            other => panic!("expected BoundaryMismatch, got {other:?}"),
        }
    }

    #[test]
    fn unknown_timezone_and_bad_date_are_typed_errors() {
        let opts = SplitOpts {
            tz: "Mars/Olympus".to_string(),
            ..Default::default()
        };
        assert!(matches!(
            compute_boundaries(&opts),
            Err(PgError::BadTimezone(_))
        ));
        let opts = SplitOpts {
            start: "2009/01/01".to_string(),
            ..Default::default()
        };
        assert!(matches!(
            compute_boundaries(&opts),
            Err(PgError::BadDate(_))
        ));
    }

    #[test]
    fn boundary_table_renders_every_chunk() {
        let b = compute_boundaries(&SplitOpts::default()).unwrap();
        let t = format_boundary_table(&b);
        assert!(t.contains("Europe/Rome"));
        assert!(t.contains("2009-07-01"));
        assert!(t.contains("1672527600"));
        // header + title + 28 rows
        assert_eq!(t.lines().count(), 30);
    }

    /// Builds a two-chunk configuration around the 2009-07-01 boundary.
    fn two_chunk_opts() -> SplitOpts {
        SplitOpts {
            start: "2009-01-01".to_string(),
            end: "2010-01-01".to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn split_places_records_on_the_right_side_of_a_boundary() {
        let dir = tempfile::tempdir().unwrap();
        let opts = two_chunk_opts();
        let b = compute_boundaries(&opts).unwrap();
        assert_eq!(b.upper.len(), 2);
        let cut = b.upper[0]; // 1246399200

        let mut input = String::new();
        for i in 0..5 {
            input.push_str(&format!("{},0,{},1,0,0,0::a,1,1\n", b.start_ts + i, i));
        }
        for i in 0..5 {
            input.push_str(&format!("{},0,{},1,0,0,0::a,1,1\n", cut + i, 5 + i));
        }

        let stats = split(Cursor::new(input.clone()), dir.path(), &opts, no_logging!()).unwrap();
        assert_eq!(stats.records_written, 10);
        assert_eq!(stats.records_before_start, 0);
        assert!(stats.stopped_early_at.is_none());
        assert_eq!(stats.chunks[0].records, 5);
        assert_eq!(stats.chunks[1].records, 5);
        assert_eq!(stats.chunks[0].first_ts, Some(b.start_ts));
        assert_eq!(stats.chunks[1].first_ts, Some(cut));
        assert_eq!(stats.chunks[1].last_ts, Some(cut + 4));

        let c1 = fs::read_to_string(dir.path().join("chunk_01.txt")).unwrap();
        let c2 = fs::read_to_string(dir.path().join("chunk_02.txt")).unwrap();
        assert_eq!(c1.lines().count(), 5);
        assert_eq!(c2.lines().count(), 5);
        // Bytes are preserved exactly, and nothing is lost or duplicated.
        assert_eq!(format!("{c1}{c2}"), input);
        assert_eq!(c1.len() as u64, stats.chunks[0].bytes);
        assert_eq!(c2.len() as u64, stats.chunks[1].bytes);
    }

    #[test]
    fn split_creates_every_chunk_file_even_when_empty() {
        let dir = tempfile::tempdir().unwrap();
        let opts = two_chunk_opts();
        let b = compute_boundaries(&opts).unwrap();
        let input = format!("{},0,0,1,0,0,0::a,1,1\n", b.start_ts);
        let stats = split(Cursor::new(input), dir.path(), &opts, no_logging!()).unwrap();
        assert_eq!(stats.chunks[1].records, 0);
        let empty = dir.path().join("chunk_02.txt");
        assert!(
            empty.is_file(),
            "chunk_02.txt must exist even with no records"
        );
        assert_eq!(fs::metadata(&empty).unwrap().len(), 0);
    }

    #[test]
    fn split_stops_at_the_first_out_of_range_record() {
        let dir = tempfile::tempdir().unwrap();
        let opts = two_chunk_opts();
        let b = compute_boundaries(&opts).unwrap();
        let end = *b.upper.last().unwrap();
        let input = format!(
            "{},0,0,1,0,0,0::a,1,1\n{},0,1,1,0,0,0::a,1,1\n{},0,2,1,0,0,0::a,1,1\n",
            b.start_ts,
            end,
            b.start_ts + 1
        );
        let stats = split(Cursor::new(input), dir.path(), &opts, no_logging!()).unwrap();
        assert_eq!(stats.stopped_early_at, Some((2, end)));
        assert_eq!(
            stats.records_written, 1,
            "the third record is never reached"
        );

        // ... unless the operator asks for it.
        let dir2 = tempfile::tempdir().unwrap();
        let opts2 = SplitOpts {
            skip_out_of_range: true,
            ..two_chunk_opts()
        };
        let input = format!(
            "{},0,0,1,0,0,0::a,1,1\n{},0,1,1,0,0,0::a,1,1\n{},0,2,1,0,0,0::a,1,1\n",
            b.start_ts,
            end,
            b.start_ts + 1
        );
        let stats2 = split(Cursor::new(input), dir2.path(), &opts2, no_logging!()).unwrap();
        assert_eq!(stats2.stopped_early_at, Some((2, end)));
        assert_eq!(stats2.records_written, 2);
    }

    #[test]
    fn split_drops_records_before_the_start_and_keeps_a_missing_final_newline() {
        let dir = tempfile::tempdir().unwrap();
        let opts = two_chunk_opts();
        let b = compute_boundaries(&opts).unwrap();
        let input = format!(
            "1,0,0,1,0,0,0::a,1,1\n{},0,1,1,0,0,0::a,1,1",
            b.start_ts + 3
        );
        let stats = split(Cursor::new(input), dir.path(), &opts, no_logging!()).unwrap();
        assert_eq!(stats.records_before_start, 1);
        assert_eq!(stats.records_written, 1);
        let c1 = fs::read_to_string(dir.path().join("chunk_01.txt")).unwrap();
        assert!(!c1.ends_with('\n'), "a missing final newline is preserved");
    }

    #[test]
    fn chunk_chain_concatenates_its_files() {
        let dir = tempfile::tempdir().unwrap();
        let mut expected = String::new();
        let mut paths = Vec::new();
        for i in 1..=3usize {
            let p = dir.path().join(format!("chunk_{i:02}.txt"));
            let body = format!("file {i} line a\nfile {i} line b\n");
            fs::write(&p, &body).unwrap();
            expected.push_str(&body);
            paths.push(p);
        }

        let chain = ChunkChain::from_chunk_dir(dir.path(), 3).unwrap();
        assert_eq!(chain.total_len(), expected.len() as u64);
        assert_eq!(chain.paths(), paths.as_slice());
        let mut got = String::new();
        BufReader::new(chain).read_to_string(&mut got).unwrap();
        assert_eq!(got, expected);

        // And the same through `new`.
        let mut got2 = String::new();
        ChunkChain::new(paths)
            .unwrap()
            .read_to_string(&mut got2)
            .unwrap();
        assert_eq!(got2, expected);
    }

    #[test]
    fn chunk_chain_names_the_missing_file_up_front() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("chunk_01.txt"), "x\n").unwrap();
        match ChunkChain::from_chunk_dir(dir.path(), 2) {
            Err(PgError::Io { path, .. }) => {
                assert!(path.ends_with("chunk_02.txt"), "got {}", path.display());
            }
            other => panic!("expected a named I/O error, got {other:?}"),
        }
    }
}
