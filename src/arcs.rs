//! Arc storage, sorting and deduplication. Replaces the
//! `sort --temporary-directory=./tmp -t$'\t' -k1,1n -k2,2n | uniq` stage of
//! `build_pg.sh` by **Matteo Loporchio**.
//!
//! # Why not shell out to GNU `sort`
//!
//! At `N = 28` chunks the raw edge list is roughly `9.5e9` arcs, about 210 GB
//! of text. Shelling out to GNU `sort` would:
//!
//! 1. move ~840 GB through the filesystem (read the 210 GB input, write ~210 GB
//!    of temporary runs, read them back, write the 210 GB result) in order to
//!    re-parse decimal integers that this process already holds in registers;
//! 2. cap its own parallelism: GNU `sort`'s `--parallel` default is
//!    `min(nproc, 8)` and its merge phase is essentially serial, on a box with
//!    112 cores;
//! 3. hide failures. `build_pg.sh` has neither `set -e` nor `set -o pipefail`,
//!    and `(sort | uniq) > $EL_FILE` truncates the target *before* `sort` runs.
//!    That is exactly how `graph/pg_el_1.tsv`, `graph/pg_el_10.tsv` and
//!    `graph/pg_el_15.tsv` all ended up 0 bytes in this checkout.
//!
//! Instead, arcs are packed into a single integer, sorted in RAM with an LSD
//! radix sort, spilled to fixed-width binary runs only when the memory budget
//! is exhausted, and merged back with a streaming k-way merge that performs the
//! `uniq` step on parsed integers rather than on text.
//!
//! # Sorting an arc is sorting one integer
//!
//! [`crate::pack_arc`] maps `(src, dst)` to `(src << 32) | dst`. Sorting those
//! `u64`s ascending is *exactly* `sort -t$'\t' -k1,1n -k2,2n`, and
//! [`Vec::dedup`] on the sorted vector is *exactly* `uniq` — with the bonus
//! that numerically equal but textually different records (`007\t5` versus
//! `7\t5`, which survive `sort | uniq` and are then fatal to the Java BVGraph
//! loader) cannot exist at all.
//!
//! # Run sizing, and the trap to avoid
//!
//! ```text
//! arcs_per_run = memory_bytes / (2 * codec.record_size())
//! ```
//!
//! The factor two is the radix sort's scratch buffer: `rdst`'s LSD radix
//! allocates a second buffer the same size as the data. Computing
//! `memory_bytes / record_size` instead would make a nominal 128 GiB budget
//! cost 256 GiB of resident memory and OOM on the first full-scale run.
//!
//! # Expected dedup yield: zero
//!
//! A duplicate `(src, dst)` arc requires the same `(prevTxId, prevOffset)` pair
//! to appear twice as an input of the **same** transaction. That was measured
//! zero times over all of `chunk_01`, all of `chunk_05` and a 200 MB sample of
//! `chunk_20`. Buffers are therefore sized for `distinct ~= raw`, and a
//! non-zero `duplicates_removed` is logged at `WARN` because it is genuinely
//! interesting.
//!
//! This module deliberately depends on neither `webgraph`, `clap` nor `anyhow`.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use log::{debug, info, warn};

use crate::{
    new_error_slot, pack_arc, sync_if_durable, take_iter_error, ArcCodec, ArcSink, ErrorSlot,
    NodeId, PgError, PgResult, SortAlgo,
};

/// Size of every `BufReader`/`BufWriter` in this module.
///
/// 16 MiB is large enough that the 210 GB text edge list at `N = 28` costs
/// about 13 000 `write(2)` calls per GB rather than millions.
const IO_BUF: usize = 16 << 20;

/// Smallest run the sorter will accept. A run below this size means the memory
/// budget was mis-specified (a typo such as `--memory 8K`), and would produce
/// millions of run files whose merge would dominate the runtime.
const MIN_ARCS_PER_RUN: usize = 1024;

/// Arcs the run buffer reserves up front, however large the budget is.
///
/// The buffer used to be created with `Vec::with_capacity(arcs_per_run)`, i.e.
/// ~103 GiB of address space for the default 192 GiB budget — even for a
/// 976 KB input. Linux overcommit made that harmless in practice (RSS stayed
/// at 14 MB), but it broke every `ulimit -v`, made every VSZ reading
/// meaningless, and put the tool out of reach of a container with a strict
/// address-space limit. The `Vec` still grows to `arcs_per_run` on demand, so
/// the spill threshold and the memory budget are unchanged.
const INITIAL_ARCS_CAPACITY: usize = 1 << 20;

// ---------------------------------------------------------------------------
// Options and statistics
// ---------------------------------------------------------------------------

/// Configuration for [`ArcSorter`] and [`sort_file`].
#[derive(Clone, Debug)]
pub struct SortOpts {
    /// Run-sorting backend. [`SortAlgo::Radix`] is silently downgraded to
    /// [`SortAlgo::Pdq`] when `codec` is [`ArcCodec::Wide`]; see
    /// [`ArcSorter::effective_algo`].
    pub algo: SortAlgo,
    /// In-memory and on-disk arc representation.
    pub codec: ArcCodec,
    /// Remove duplicate arcs, reproducing the `uniq` of `build_pg.sh`.
    ///
    /// A duplicate arc is *fatal* to the Java `it.unimi.dsi.webgraph` loader
    /// (`BVGraph.updateBins` computes `mostSignificantBit(0) == -1` and dies
    /// with `ArrayIndexOutOfBoundsException: Index -1`), so this should stay
    /// `true` for anything that will be compressed.
    pub dedup: bool,
    /// Total memory budget for one sort run, **including** the radix sort's
    /// scratch buffer. See the module documentation for the arithmetic.
    pub memory_bytes: u64,
    /// Directory that spill runs are created under.
    pub tmp_dir: PathBuf,
    /// Worker threads for [`SortAlgo::Pdq`]. `0` means "use the ambient
    /// `rayon` pool".
    pub threads: usize,
    /// Keep spill runs after a successful run, in `tmp_dir`, under stable
    /// names. When `false` the runs live in a private temporary directory that
    /// is removed when the [`SortedArcs`] is dropped.
    pub keep_intermediate: bool,
}

impl Default for SortOpts {
    /// A conservative 256 MiB budget. `main.rs` overrides `memory_bytes` from
    /// `--memory` (whose own default is `min(45% of RAM, 192 GiB)`); this
    /// library-level default only has to be safe, not fast, because the buffer
    /// is reserved eagerly to honour the budget.
    fn default() -> Self {
        SortOpts {
            algo: SortAlgo::default(),
            codec: ArcCodec::default(),
            dedup: true,
            memory_bytes: 256 << 20,
            tmp_dir: PathBuf::from("./tmp"),
            threads: 0,
            keep_intermediate: false,
        }
    }
}

/// What the sort did.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct SortStats {
    /// Arcs pushed into the sorter, duplicates included.
    pub raw_arcs: u64,
    /// Arcs surviving deduplication. Equals `raw_arcs` when `dedup` is off.
    pub distinct_arcs: u64,
    /// `raw_arcs - distinct_arcs`. Expected to be zero on this corpus.
    pub duplicates_removed: u64,
    /// Number of spill runs written. Zero means the whole sort stayed in RAM.
    pub runs: usize,
    /// Bytes written to spill runs.
    pub spilled_bytes: u64,
    /// Largest node id observed on either endpoint.
    pub max_node_id: NodeId,
}

// ---------------------------------------------------------------------------
// Atomic output files
// ---------------------------------------------------------------------------

/// A buffered output file that only becomes visible under its final name once
/// it has been fully written, flushed and `fsync`ed.
///
/// The Java program opened both of its outputs with `new PrintWriter(String)`,
/// which truncates at open and before any parsing: a crash on line 1 destroyed
/// the previous good output (observed — `nm_02.tsv` and `el_02.tsv` are both
/// 0 bytes in the reference directory). Writing to a sibling temporary path and
/// renaming on success makes that impossible.
///
/// Non-regular destinations (`/dev/null`, a fifo, a process substitution) are
/// written through directly, because renaming over them would be wrong.
struct AtomicOut {
    final_path: PathBuf,
    temp_path: Option<PathBuf>,
    writer: Option<BufWriter<File>>,
}

impl AtomicOut {
    fn create(path: &Path) -> PgResult<Self> {
        let direct = match std::fs::metadata(path) {
            Ok(md) => !md.is_file(),
            Err(_) => false,
        };
        if direct {
            let file = File::create(path).map_err(|e| PgError::io(path, e))?;
            return Ok(AtomicOut {
                final_path: path.to_path_buf(),
                temp_path: None,
                writer: Some(BufWriter::with_capacity(IO_BUF, file)),
            });
        }
        let name = path.file_name().ok_or_else(|| {
            PgError::other(format!("{} has no file name component", path.display()))
        })?;
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| PgError::io(parent, e))?;
        }
        let temp = parent.join(format!(
            ".{}.pgraph-tmp-{}",
            name.to_string_lossy(),
            std::process::id()
        ));
        let file = File::create(&temp).map_err(|e| PgError::io(&temp, e))?;
        Ok(AtomicOut {
            final_path: path.to_path_buf(),
            temp_path: Some(temp),
            writer: Some(BufWriter::with_capacity(IO_BUF, file)),
        })
    }

    /// Writes a whole buffer, naming the destination only if it fails.
    ///
    /// The path is *not* cloned on the happy path. It used to be, once per
    /// arc, purely so the error branch could name the file: at `9.5e9` arcs
    /// that is `9.5e9` `malloc`/`memcpy`/`free` triples, measured at 6.67x the
    /// cost of the write itself (10.00 s versus 1.50 s over 50 000 000 arcs).
    #[inline]
    fn write_all(&mut self, bytes: &[u8]) -> PgResult<()> {
        let w = self
            .writer
            .as_mut()
            .expect("AtomicOut used after commit; this is a bug in utxo2webgraph");
        match w.write_all(bytes) {
            Ok(()) => Ok(()),
            Err(e) => Err(PgError::io(&self.final_path, e)),
        }
    }

    /// Flushes, `fsync`s and publishes the file under its final name.
    fn commit(&mut self) -> PgResult<()> {
        let Some(mut writer) = self.writer.take() else {
            return Ok(());
        };
        let written_path = self
            .temp_path
            .clone()
            .unwrap_or_else(|| self.final_path.clone());
        writer.flush().map_err(|e| PgError::io(&written_path, e))?;
        // Not `sync_all` directly: `fsync` on `/dev/null`, a fifo or a process
        // substitution returns EINVAL, which used to fail an otherwise
        // completed run.
        sync_if_durable(writer.get_ref(), &written_path)?;
        drop(writer);
        if let Some(temp) = self.temp_path.take() {
            std::fs::rename(&temp, &self.final_path).map_err(|e| PgError::io(&temp, e))?;
            // Best effort: make the rename itself durable. A failure here is
            // not worth aborting a completed multi-hour run over.
            if let Some(parent) = self.final_path.parent() {
                let dir = if parent.as_os_str().is_empty() {
                    Path::new(".")
                } else {
                    parent
                };
                if let Ok(d) = File::open(dir) {
                    let _ = d.sync_all();
                }
            }
        }
        Ok(())
    }
}

impl Drop for AtomicOut {
    /// Removes the temporary file when the output was never committed.
    ///
    /// Without this, every failed run left its partial output behind under
    /// `.<name>.pgraph-tmp-<pid>` — up to 195 GiB at `N = 28`, and, because the
    /// name carries the pid, a *new* orphan on every retry.
    fn drop(&mut self) {
        if self.writer.is_none() {
            return; // committed (or already cleaned up)
        }
        // Close the file before unlinking, so the space is reclaimed at once.
        self.writer = None;
        if let Some(temp) = self.temp_path.take() {
            match std::fs::remove_file(&temp) {
                Ok(()) => debug!("removed the uncommitted {}", temp.display()),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => warn!("could not remove the uncommitted {}: {e}", temp.display()),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Sinks
// ---------------------------------------------------------------------------

/// An [`ArcSink`] that counts arcs and writes nothing.
///
/// Used by `--dry-run` and by benchmarks that want the parsing cost without
/// the I/O cost.
#[derive(Debug, Default)]
pub struct NullArcSink {
    /// Arcs seen so far.
    pub count: u64,
}

impl NullArcSink {
    /// Creates an empty counter.
    pub fn new() -> Self {
        NullArcSink { count: 0 }
    }
}

impl ArcSink for NullArcSink {
    #[inline]
    fn push(&mut self, _src: NodeId, _dst: NodeId) -> PgResult<()> {
        self.count += 1;
        Ok(())
    }

    fn finish(&mut self) -> PgResult<()> {
        Ok(())
    }
}

/// Writes arcs as `{src}\t{dst}\n`.
///
/// The formatting is **canonical and load-bearing**: decimal, no padding, no
/// sign, no leading zeros, a single TAB, LF only. `007\t5` and `7\t5` compare
/// equal under `sort -k1,1n -k2,2n` yet survive `uniq`, and a duplicate arc is
/// fatal to the Java BVGraph loader. This reproduces Java's
/// `printf("%d\t%d\n", ...)` byte for byte.
///
/// Integers are rendered with [`itoa`], not `write!`: at `9.5e9` lines
/// `std::fmt`'s per-call overhead is the difference between minutes and an
/// hour.
pub struct TsvArcSink {
    out: AtomicOut,
    itoa: itoa::Buffer,
    count: u64,
}

impl TsvArcSink {
    /// Creates the sink. The destination only appears under `path` once
    /// [`ArcSink::finish`] has succeeded.
    pub fn create(path: &Path) -> PgResult<Self> {
        Ok(TsvArcSink {
            out: AtomicOut::create(path)?,
            itoa: itoa::Buffer::new(),
            count: 0,
        })
    }

    /// Number of arcs written so far.
    pub fn count(&self) -> u64 {
        self.count
    }
}

impl ArcSink for TsvArcSink {
    #[inline]
    fn push(&mut self, src: NodeId, dst: NodeId) -> PgResult<()> {
        // Two itoa renderings plus two literal bytes; no formatting machinery,
        // and no allocation whatsoever on this path.
        let mut scratch = [0u8; 48];
        let mut n = 0;
        {
            let s = self.itoa.format(src);
            scratch[n..n + s.len()].copy_from_slice(s.as_bytes());
            n += s.len();
        }
        scratch[n] = b'\t';
        n += 1;
        {
            let s = self.itoa.format(dst);
            scratch[n..n + s.len()].copy_from_slice(s.as_bytes());
            n += s.len();
        }
        scratch[n] = b'\n';
        n += 1;
        self.out.write_all(&scratch[..n])?;
        self.count += 1;
        Ok(())
    }

    fn finish(&mut self) -> PgResult<()> {
        self.out.commit()
    }
}

/// Writes arcs as raw little-endian fixed-width records with no header, so that
/// `file length / record size` is the arc count and a torn write is detectable
/// as a non-multiple length (see [`PgError::TruncatedRun`]).
pub struct BinaryArcSink {
    out: AtomicOut,
    codec: ArcCodec,
    count: u64,
}

impl BinaryArcSink {
    /// Creates the sink. The destination only appears under `path` once
    /// [`ArcSink::finish`] has succeeded.
    pub fn create(path: &Path, codec: ArcCodec) -> PgResult<Self> {
        Ok(BinaryArcSink {
            out: AtomicOut::create(path)?,
            codec,
            count: 0,
        })
    }

    /// Number of arcs written so far.
    pub fn count(&self) -> u64 {
        self.count
    }
}

impl ArcSink for BinaryArcSink {
    #[inline]
    fn push(&mut self, src: NodeId, dst: NodeId) -> PgResult<()> {
        match self.codec {
            ArcCodec::Packed32 => {
                let key = pack_arc(src, dst).ok_or(PgError::ArcCodecOverflow(src.max(dst)))?;
                self.out.write_all(&key.to_le_bytes())?;
            }
            ArcCodec::Wide => {
                let key = wide_arc(src, dst);
                self.out.write_all(&key.to_le_bytes())?;
            }
        }
        self.count += 1;
        Ok(())
    }

    fn finish(&mut self) -> PgResult<()> {
        self.out.commit()
    }
}

// ---------------------------------------------------------------------------
// Key helpers
// ---------------------------------------------------------------------------

/// Packs an arc into a `u128`: `(src << 64) | dst`. Always representable.
#[inline]
fn wide_arc(src: NodeId, dst: NodeId) -> u128 {
    ((src as u128) << 64) | (dst as u128)
}

/// Unpacks a merge key. Merge keys are always `u128` so that one heap serves
/// both codecs; a [`ArcCodec::Packed32`] key is the zero-extended `u64`.
#[inline]
fn unpack_key(codec: ArcCodec, key: u128) -> (usize, usize) {
    match codec {
        ArcCodec::Packed32 => (
            ((key >> 32) as u64 & 0xFFFF_FFFF) as usize,
            (key as u64 & 0xFFFF_FFFF) as usize,
        ),
        ArcCodec::Wide => ((key >> 64) as u64 as usize, (key as u64) as usize),
    }
}

// ---------------------------------------------------------------------------
// The in-memory run buffer
// ---------------------------------------------------------------------------

/// The buffer of not-yet-sorted arcs, in whichever width the codec asks for.
enum ArcBuf {
    Packed(Vec<u64>),
    Wide(Vec<u128>),
}

impl ArcBuf {
    fn with_capacity(codec: ArcCodec, cap: usize) -> Self {
        match codec {
            ArcCodec::Packed32 => ArcBuf::Packed(Vec::with_capacity(cap)),
            ArcCodec::Wide => ArcBuf::Wide(Vec::with_capacity(cap)),
        }
    }

    #[inline]
    fn len(&self) -> usize {
        match self {
            ArcBuf::Packed(v) => v.len(),
            ArcBuf::Wide(v) => v.len(),
        }
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Sorts, optionally deduplicates, and returns how many duplicates went.
    fn sort_dedup(&mut self, algo: SortAlgo, dedup: bool, pool: Option<&rayon::ThreadPool>) -> u64 {
        let before = self.len() as u64;
        match self {
            ArcBuf::Packed(v) => {
                sort_u64(v, algo, pool);
                if dedup {
                    v.dedup();
                }
            }
            ArcBuf::Wide(v) => {
                sort_u128(v, pool);
                if dedup {
                    v.dedup();
                }
            }
        }
        before - self.len() as u64
    }

    /// Writes every record little-endian. The buffer must already be sorted.
    fn write_to(&self, w: &mut impl Write) -> std::io::Result<u64> {
        match self {
            ArcBuf::Packed(v) => {
                for k in v {
                    w.write_all(&k.to_le_bytes())?;
                }
                Ok(v.len() as u64 * 8)
            }
            ArcBuf::Wide(v) => {
                for k in v {
                    w.write_all(&k.to_le_bytes())?;
                }
                Ok(v.len() as u64 * 16)
            }
        }
    }

    fn clear(&mut self) {
        match self {
            ArcBuf::Packed(v) => v.clear(),
            ArcBuf::Wide(v) => v.clear(),
        }
    }
}

/// Sorts `u64` keys with the requested backend.
fn sort_u64(v: &mut Vec<u64>, algo: SortAlgo, pool: Option<&rayon::ThreadPool>) {
    match algo {
        SortAlgo::Radix => {
            use rdst::RadixSort;
            v.radix_sort_unstable();
        }
        SortAlgo::Pdq => sort_pdq(v.as_mut_slice(), pool),
    }
}

/// Sorts `u128` keys. Always pattern-defeating quicksort: `rdst` does
/// implement `RadixKey` for `u128`, but that is sixteen LSD passes over
/// 16-byte records, which loses to a comparison sort on the only path that
/// ever uses this codec (a hypothetical future dataset with node ids past
/// `2^32`). See [`ArcSorter::effective_algo`].
fn sort_u128(v: &mut Vec<u128>, pool: Option<&rayon::ThreadPool>) {
    sort_pdq(v.as_mut_slice(), pool)
}

fn sort_pdq<T: Ord + Send>(slice: &mut [T], pool: Option<&rayon::ThreadPool>) {
    use rayon::slice::ParallelSliceMut;
    match pool {
        Some(p) => p.install(move || slice.par_sort_unstable()),
        None => slice.par_sort_unstable(),
    }
}

// ---------------------------------------------------------------------------
// ArcSorter
// ---------------------------------------------------------------------------

/// The [`ArcSink`] that replaces `sort | uniq`.
///
/// Arcs are packed into one integer, buffered up to `arcs_per_run`, sorted with
/// [`SortOpts::algo`], deduplicated in place, and spilled to a fixed-width
/// binary run. [`ArcSorter::into_sorted`] then either hands back the in-memory
/// buffer (the common case on this machine: at `N = 28` a 128 GiB budget holds
/// all `9.5e9` arcs in one run and nothing ever reaches the disk) or a k-way
/// merge over the runs.
pub struct ArcSorter {
    opts: SortOpts,
    algo: SortAlgo,
    arcs_per_run: usize,
    buf: ArcBuf,
    runs: Vec<PathBuf>,
    tmp: Option<tempfile::TempDir>,
    pool: Option<rayon::ThreadPool>,
    stats: SortStats,
    finished: bool,
}

impl ArcSorter {
    /// Creates a sorter.
    ///
    /// # Errors
    ///
    /// Fails with [`PgError::Other`] when `memory_bytes` is so small that a run
    /// would hold fewer than 1024 arcs, and with [`PgError::PlainIo`] when the
    /// `rayon` pool for [`SortAlgo::Pdq`] cannot be built.
    pub fn new(opts: SortOpts) -> PgResult<Self> {
        let record = opts.codec.record_size() as u64;
        // The factor two is the radix sort's scratch buffer. Dropping it turns
        // a nominal 128 GiB budget into 256 GiB of resident memory.
        let arcs_per_run = (opts.memory_bytes / (2 * record)) as usize;
        if arcs_per_run < MIN_ARCS_PER_RUN {
            return Err(PgError::other(format!(
                "memory budget of {} bytes only allows {} arcs per run (minimum {}); \
                 arcs_per_run = memory / (2 * {} bytes per arc), the factor 2 being the \
                 radix sort's scratch buffer",
                opts.memory_bytes, arcs_per_run, MIN_ARCS_PER_RUN, record
            )));
        }

        let algo = match (opts.algo, opts.codec) {
            (SortAlgo::Radix, ArcCodec::Wide) => {
                info!(
                    "arc codec is 'wide' (128-bit): downgrading the run sort from radix to \
                     pattern-defeating quicksort"
                );
                SortAlgo::Pdq
            }
            (a, _) => a,
        };

        // Only the comparison sort takes a pool: `sort_u64`/`sort_u128` pass it
        // to `sort_pdq` and nowhere else, because `rdst`'s radix sort follows
        // the ambient (global) `rayon` pool and cannot be told about a private
        // one. Building it unconditionally spawned a second full set of
        // threads that never ran a task — 225 OS threads at `--threads 112`.
        let pool = if opts.threads > 0 && algo == SortAlgo::Pdq {
            Some(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(opts.threads)
                    .build()
                    .map_err(|e| {
                        PgError::other(format!("could not build the sort thread pool: {e}"))
                    })?,
            )
        } else {
            None
        };

        debug!(
            "arc sorter: codec={:?} algo={:?} arcs_per_run={} memory={} bytes dedup={}",
            opts.codec, algo, arcs_per_run, opts.memory_bytes, opts.dedup
        );

        Ok(ArcSorter {
            buf: ArcBuf::with_capacity(opts.codec, arcs_per_run.min(INITIAL_ARCS_CAPACITY)),
            opts,
            algo,
            arcs_per_run,
            runs: Vec::new(),
            tmp: None,
            pool,
            stats: SortStats::default(),
            finished: false,
        })
    }

    /// The backend actually in use, after the [`ArcCodec::Wide`] downgrade.
    pub fn effective_algo(&self) -> SortAlgo {
        self.algo
    }

    /// Arcs buffered before a run is spilled: `memory / (2 * record_size)`.
    pub fn arcs_per_run(&self) -> usize {
        self.arcs_per_run
    }

    /// Statistics gathered so far.
    pub fn stats(&self) -> SortStats {
        self.stats
    }

    /// Directory the next run file will be created in, creating the private
    /// temporary directory on first use so that a fully in-memory sort never
    /// touches the filesystem at all.
    fn run_dir(&mut self) -> PgResult<PathBuf> {
        let base = &self.opts.tmp_dir;
        std::fs::create_dir_all(base).map_err(|e| PgError::io(base, e))?;
        if self.opts.keep_intermediate {
            return Ok(base.clone());
        }
        if self.tmp.is_none() {
            self.tmp = Some(
                tempfile::Builder::new()
                    // The pid is in the name on purpose: a run killed by
                    // SIGINT/SIGKILL never runs `TempDir`'s destructor, and
                    // `reap_stale_run_dirs` uses the pid to tell an orphan
                    // from a directory a concurrent run is still using.
                    .prefix(&format!("pgraph-sort-{}-", std::process::id()))
                    .tempdir_in(base)
                    .map_err(|e| PgError::io(base, e))?,
            );
        }
        Ok(self
            .tmp
            .as_ref()
            .expect("just created")
            .path()
            .to_path_buf())
    }

    /// Sorts, deduplicates and writes the buffer as one run file.
    fn spill(&mut self) -> PgResult<()> {
        if self.buf.is_empty() {
            return Ok(());
        }
        let removed = self
            .buf
            .sort_dedup(self.algo, self.opts.dedup, self.pool.as_ref());
        self.stats.duplicates_removed += removed;

        let dir = self.run_dir()?;
        let path = dir.join(format!("run-{:05}.arcs", self.runs.len()));
        let file = File::create(&path).map_err(|e| PgError::io(&path, e))?;
        let mut w = BufWriter::with_capacity(IO_BUF, file);
        let bytes = self
            .buf
            .write_to(&mut w)
            .map_err(|e| PgError::io(&path, e))?;
        w.flush().map_err(|e| PgError::io(&path, e))?;
        sync_if_durable(w.get_ref(), &path)?;
        drop(w);

        debug!(
            "spilled run {} ({} arcs, {} bytes) to {}",
            self.runs.len(),
            self.buf.len(),
            bytes,
            path.display()
        );
        self.stats.spilled_bytes += bytes;
        self.runs.push(path);
        self.stats.runs = self.runs.len();
        self.buf.clear();
        Ok(())
    }

    /// Finishes the sort and hands back the sorted arcs.
    ///
    /// When nothing was ever spilled this is a single in-memory sort. Otherwise
    /// the tail buffer becomes the last run and a k-way merge is set up.
    ///
    /// # Exactness of `distinct_arcs`
    ///
    /// With a single run (in memory or on disk) the count is exact by
    /// construction. With several runs *and* deduplication enabled, cross-run
    /// duplicates are only discovered during the merge, so this method performs
    /// one counting merge pass over the runs to make [`SortedArcs::num_arcs`]
    /// exact — the downstream BVGraph writer records an `arcs=` property and
    /// must not be told a guess. That extra pass is pure sequential I/O over
    /// binary records and never happens in the single-run case, which is what
    /// the production configuration on this machine always hits.
    pub fn into_sorted(mut self) -> PgResult<(SortedArcs, SortStats)> {
        if !self.finished {
            // Not fatal: `into_sorted` does everything `finish` would have.
            // Worth saying, because a caller that skipped `finish` on one sink
            // has probably skipped it on the others too, where it *is* fatal.
            debug!("ArcSorter::into_sorted called without a preceding finish()");
        }
        let codec = self.opts.codec;
        let dedup = self.opts.dedup;

        if self.runs.is_empty() {
            let removed = self.buf.sort_dedup(self.algo, dedup, self.pool.as_ref());
            self.stats.duplicates_removed += removed;
            self.stats.distinct_arcs = self.buf.len() as u64;
            let repr = match self.buf {
                ArcBuf::Packed(v) => Repr::MemoryPacked(v),
                ArcBuf::Wide(v) => Repr::MemoryWide(v),
            };
            let stats = self.stats;
            report_duplicates(&stats);
            return Ok((
                SortedArcs {
                    codec,
                    dedup,
                    num_arcs: stats.distinct_arcs,
                    max_node_id: stats.max_node_id,
                    repr,
                },
                stats,
            ));
        }

        self.spill()?;
        let runs = std::mem::take(&mut self.runs);
        let tmp = self.tmp.take();
        let record = codec.record_size() as u64;

        let num_arcs = if runs.len() == 1 {
            let len = std::fs::metadata(&runs[0])
                .map_err(|e| PgError::io(&runs[0], e))?
                .len();
            if len % record != 0 {
                return Err(PgError::TruncatedRun {
                    path: runs[0].clone(),
                    len,
                    record_size: record as usize,
                });
            }
            len / record
        } else if dedup {
            debug!(
                "{} runs: counting distinct arcs with one merge pass to keep num_arcs exact",
                runs.len()
            );
            count_merged(&runs, codec, true)?
        } else {
            self.stats.raw_arcs
        };

        self.stats.distinct_arcs = num_arcs;
        self.stats.duplicates_removed = self.stats.raw_arcs.saturating_sub(num_arcs);
        let stats = self.stats;
        report_duplicates(&stats);

        Ok((
            SortedArcs {
                codec,
                dedup,
                num_arcs,
                max_node_id: stats.max_node_id,
                repr: Repr::Runs {
                    paths: runs,
                    _tmp: tmp,
                },
            },
            stats,
        ))
    }
}

/// A non-zero duplicate count means the same `(prevTxId, prevOffset)` was an
/// input of the same transaction twice — never observed on this corpus, so it
/// deserves an operator's attention rather than a silent tally.
fn report_duplicates(stats: &SortStats) {
    if stats.duplicates_removed > 0 {
        warn!(
            "removed {} duplicate arcs out of {} ({} distinct); duplicates are not expected on \
             this dataset and may indicate a double-spent output or a concatenated input range",
            stats.duplicates_removed, stats.raw_arcs, stats.distinct_arcs
        );
    }
}

impl ArcSink for ArcSorter {
    #[inline]
    fn push(&mut self, src: NodeId, dst: NodeId) -> PgResult<()> {
        let hi = src.max(dst);
        if hi > self.stats.max_node_id {
            self.stats.max_node_id = hi;
        }
        match &mut self.buf {
            ArcBuf::Packed(v) => {
                let key = pack_arc(src, dst).ok_or(PgError::ArcCodecOverflow(hi))?;
                v.push(key);
            }
            ArcBuf::Wide(v) => v.push(wide_arc(src, dst)),
        }
        self.stats.raw_arcs += 1;
        if self.buf.len() >= self.arcs_per_run {
            self.spill()?;
        }
        Ok(())
    }

    /// Marks the sorter complete. Every run written so far has already been
    /// flushed and `fsync`ed; the buffered tail is turned into sorted output by
    /// [`ArcSorter::into_sorted`], which is what callers must invoke next.
    fn finish(&mut self) -> PgResult<()> {
        self.finished = true;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// SortedArcs
// ---------------------------------------------------------------------------

enum Repr {
    MemoryPacked(Vec<u64>),
    MemoryWide(Vec<u128>),
    Runs {
        paths: Vec<PathBuf>,
        /// Dropped last, deleting the run files, unless `keep_intermediate`.
        _tmp: Option<tempfile::TempDir>,
    },
}

/// A sorted, optionally deduplicated arc set: either a vector in memory or a
/// set of on-disk runs merged lazily.
///
/// This is exactly the contract `build_pg.sh` handed to WebGraph after
/// `sort -t$'\t' -k1,1n -k2,2n | uniq`: ascending by `(src, dst)`, free of
/// duplicates, self-loops preserved.
pub struct SortedArcs {
    codec: ArcCodec,
    dedup: bool,
    num_arcs: u64,
    max_node_id: NodeId,
    repr: Repr,
}

impl SortedArcs {
    /// Exact number of arcs this set will yield.
    pub fn num_arcs(&self) -> u64 {
        self.num_arcs
    }

    /// Largest node id seen on either endpoint. Note this is `max_id`, so a
    /// graph covering exactly these arcs has `max_id + 1` nodes — which is what
    /// Java's `ArcListASCIIGraph` inferred, and is generally *smaller* than the
    /// node map's `next_id` because unspent outputs never appear in an arc.
    pub fn max_node_id(&self) -> NodeId {
        self.max_node_id
    }

    /// The representation the arcs are stored in.
    pub fn codec(&self) -> ArcCodec {
        self.codec
    }

    /// Streams the arcs in ascending `(src, dst)` order.
    ///
    /// # The error side-channel
    ///
    /// `webgraph` demands a plain [`Iterator`], which cannot report failure, so
    /// an I/O error part-way through the k-way merge is stored in the returned
    /// [`ErrorSlot`] and the iterator simply ends. **Callers must call
    /// [`crate::take_iter_error`] on the slot once the iterator is exhausted**,
    /// otherwise a truncated graph looks like a successful one.
    pub fn iter(&self) -> PgResult<(SortedArcIter<'_>, ErrorSlot)> {
        let slot = new_error_slot();
        let inner = match &self.repr {
            Repr::MemoryPacked(v) => IterInner::Packed(v.iter()),
            Repr::MemoryWide(v) => IterInner::Wide(v.iter()),
            Repr::Runs { paths, .. } => IterInner::Merge(Box::new(Merger::open(
                paths,
                self.codec,
                self.dedup,
                slot.clone(),
            )?)),
        };
        Ok((
            SortedArcIter {
                codec: self.codec,
                inner,
            },
            slot,
        ))
    }

    /// Writes `graph/pg_el_N.tsv`: the sorted, deduplicated text edge list,
    /// byte-identical to what `(sort | uniq)` produced. Returns the line count.
    ///
    /// Progress is logged every [`ARC_PROGRESS_EVERY`] arcs: at `N = 28` this
    /// loop writes 195 GiB and used to take about ten minutes in total
    /// silence.
    pub fn write_tsv(&self, path: &Path) -> PgResult<u64> {
        let (iter, slot) = self.iter()?;
        let mut sink = TsvArcSink::create(path)?;
        let mut progress = ArcProgress::new("edge list", path);
        for (src, dst) in iter {
            sink.push(src as NodeId, dst as NodeId)?;
            progress.tick();
        }
        take_iter_error(&slot)?;
        sink.finish()?;
        progress.done(sink.count());
        Ok(sink.count())
    }

    /// Writes the sorted arcs as fixed-width little-endian binary records.
    /// Returns the arc count.
    pub fn write_binary(&self, path: &Path) -> PgResult<u64> {
        let (iter, slot) = self.iter()?;
        let mut sink = BinaryArcSink::create(path, self.codec)?;
        let mut progress = ArcProgress::new("binary arcs", path);
        for (src, dst) in iter {
            sink.push(src as NodeId, dst as NodeId)?;
            progress.tick();
        }
        take_iter_error(&slot)?;
        sink.finish()?;
        progress.done(sink.count());
        Ok(sink.count())
    }
}

enum IterInner<'a> {
    Packed(std::slice::Iter<'a, u64>),
    Wide(std::slice::Iter<'a, u128>),
    Merge(Box<Merger>),
}

/// Ascending iterator over a [`SortedArcs`]. See [`SortedArcs::iter`] for the
/// error side-channel.
///
/// This type is `Send` (it holds only `Vec` slices, `BufReader<File>`,
/// `BinaryHeap` and an `Arc<Mutex<..>>`), so `compress.rs` can move it into a
/// [`rayon::ThreadPool::install`] closure.
pub struct SortedArcIter<'a> {
    codec: ArcCodec,
    inner: IterInner<'a>,
}

impl Iterator for SortedArcIter<'_> {
    type Item = (usize, usize);

    #[inline]
    fn next(&mut self) -> Option<(usize, usize)> {
        match &mut self.inner {
            IterInner::Packed(it) => it.next().map(|&k| unpack_key(self.codec, k as u128)),
            IterInner::Wide(it) => it.next().map(|&k| unpack_key(self.codec, k)),
            IterInner::Merge(m) => m.next_arc(),
        }
    }
}

// ---------------------------------------------------------------------------
// k-way merge over run files
// ---------------------------------------------------------------------------

/// One run file, read sequentially through a 16 MiB buffer.
struct RunReader {
    reader: BufReader<File>,
    path: PathBuf,
    record_size: usize,
    remaining: u64,
}

impl RunReader {
    fn open(path: &Path, codec: ArcCodec) -> PgResult<Self> {
        let record_size = codec.record_size();
        let len = std::fs::metadata(path)
            .map_err(|e| PgError::io(path, e))?
            .len();
        if len % record_size as u64 != 0 {
            return Err(PgError::TruncatedRun {
                path: path.to_path_buf(),
                len,
                record_size,
            });
        }
        let file = File::open(path).map_err(|e| PgError::io(path, e))?;
        Ok(RunReader {
            reader: BufReader::with_capacity(IO_BUF, file),
            path: path.to_path_buf(),
            record_size,
            remaining: len / record_size as u64,
        })
    }

    /// Reads the next record, widened to a `u128` merge key.
    fn next_key(&mut self) -> PgResult<Option<u128>> {
        if self.remaining == 0 {
            return Ok(None);
        }
        let mut raw = [0u8; 16];
        self.reader
            .read_exact(&mut raw[..self.record_size])
            .map_err(|e| PgError::io(&self.path, e))?;
        self.remaining -= 1;
        Ok(Some(match self.record_size {
            8 => u64::from_le_bytes(raw[..8].try_into().expect("8 bytes")) as u128,
            _ => u128::from_le_bytes(raw),
        }))
    }
}

/// Streaming k-way merge with cross-run deduplication.
///
/// Every run was deduplicated before it was spilled, so only duplicates that
/// straddle a run boundary remain to be suppressed here.
struct Merger {
    readers: Vec<RunReader>,
    heap: BinaryHeap<Reverse<(u128, usize)>>,
    last: Option<u128>,
    dedup: bool,
    codec: ArcCodec,
    slot: ErrorSlot,
    done: bool,
}

impl Merger {
    fn open(paths: &[PathBuf], codec: ArcCodec, dedup: bool, slot: ErrorSlot) -> PgResult<Self> {
        let mut readers = Vec::with_capacity(paths.len());
        for p in paths {
            readers.push(RunReader::open(p, codec)?);
        }
        let mut heap = BinaryHeap::with_capacity(readers.len());
        for (i, r) in readers.iter_mut().enumerate() {
            if let Some(k) = r.next_key()? {
                heap.push(Reverse((k, i)));
            }
        }
        Ok(Merger {
            readers,
            heap,
            last: None,
            dedup,
            codec,
            slot,
            done: false,
        })
    }

    /// Records an error in the side-channel and ends iteration.
    fn fail(&mut self, e: PgError) -> Option<(usize, usize)> {
        self.done = true;
        if let Ok(mut g) = self.slot.lock() {
            if g.is_none() {
                *g = Some(e);
            }
        }
        None
    }

    fn next_arc(&mut self) -> Option<(usize, usize)> {
        if self.done {
            return None;
        }
        loop {
            let Reverse((key, idx)) = self.heap.pop()?;
            // Bound to a local: a `match` scrutinee keeps its temporaries (here
            // the `&mut self.readers[idx]` autoref) alive for the whole match,
            // which would collide with `self.heap` / `self.fail`.
            let advanced = self.readers[idx].next_key();
            match advanced {
                Ok(Some(next)) => self.heap.push(Reverse((next, idx))),
                Ok(None) => {}
                Err(e) => return self.fail(e),
            }
            if self.dedup && self.last == Some(key) {
                continue;
            }
            self.last = Some(key);
            return Some(unpack_key(self.codec, key));
        }
    }
}

/// Counts how many arcs a merge over `paths` would yield. Used to make
/// [`SortedArcs::num_arcs`] exact in the multi-run deduplicating case.
fn count_merged(paths: &[PathBuf], codec: ArcCodec, dedup: bool) -> PgResult<u64> {
    let slot = new_error_slot();
    let mut merger = Merger::open(paths, codec, dedup, slot.clone())?;
    let mut n = 0u64;
    let mut progress = ArcProgress::new("counting merge", Path::new("spill runs"));
    while merger.next_arc().is_some() {
        n += 1;
        progress.tick();
    }
    take_iter_error(&slot)?;
    progress.done(n);
    Ok(n)
}

/// Arcs between two progress lines from the long streaming loops.
///
/// 100 million arcs is about 2 GiB of text edge list, i.e. a line every few
/// seconds at full-scale throughput — enough to tell a stuck run from a slow
/// one, rare enough to keep `logs/` small.
pub const ARC_PROGRESS_EVERY: u64 = 100_000_000;

/// Periodic `INFO` line for a loop that streams many arcs.
pub struct ArcProgress {
    what: &'static str,
    path: String,
    start: std::time::Instant,
    seen: u64,
    next: u64,
}

impl ArcProgress {
    /// Starts a progress reporter for `what`, writing (or reading) `path`.
    pub fn new(what: &'static str, path: &Path) -> Self {
        ArcProgress {
            what,
            path: path.display().to_string(),
            start: std::time::Instant::now(),
            seen: 0,
            next: ARC_PROGRESS_EVERY,
        }
    }

    /// Counts one arc, logging when the threshold is reached.
    #[inline]
    pub fn tick(&mut self) {
        self.seen += 1;
        if self.seen == self.next {
            self.next += ARC_PROGRESS_EVERY;
            let secs = self.start.elapsed().as_secs_f64();
            info!(
                "{}: {} arcs in {:.0}s ({:.2e} arcs/s) -> {}",
                self.what,
                self.seen,
                secs,
                self.seen as f64 / secs.max(1e-9),
                self.path
            );
        }
    }

    /// Logs the final count, unless the loop was short enough to be silent.
    pub fn done(&self, total: u64) {
        if total >= ARC_PROGRESS_EVERY {
            info!(
                "{}: {} arcs in {:.0}s -> {}",
                self.what,
                total,
                self.start.elapsed().as_secs_f64(),
                self.path
            );
        }
    }
}

/// Removes `pgraph-sort-<pid>-*` spill directories left by processes that are
/// no longer alive.
///
/// A run killed by `SIGINT` or `SIGKILL` never runs `TempDir`'s destructor: a
/// four-second `timeout -s INT` left 391 run files behind, and `-s KILL` left
/// 549. Nothing ever removed them, so they accumulated and made the next run's
/// free-space preflight wrong. This is called once at startup, before the
/// space checks. Directories belonging to a live process — a concurrent
/// `pgraph` — are never touched.
///
/// Returns the number of directories removed. Errors are logged, never fatal:
/// reaping is hygiene, not correctness.
pub fn reap_stale_run_dirs(tmp_dir: &Path) -> usize {
    let me = std::process::id();
    let entries = match std::fs::read_dir(tmp_dir) {
        Ok(e) => e,
        Err(_) => return 0, // not created yet, or not readable: nothing to reap
    };
    let mut removed = 0usize;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let Some(rest) = name.strip_prefix("pgraph-sort-") else {
            continue;
        };
        // `pgraph-sort-<pid>-<random>`; anything else predates this naming.
        let Some((pid, _)) = rest.split_once('-') else {
            continue;
        };
        let Ok(pid) = pid.parse::<u32>() else {
            continue;
        };
        if pid == me || process_is_alive(pid) {
            continue;
        }
        if !entry.path().is_dir() {
            continue;
        }
        match std::fs::remove_dir_all(entry.path()) {
            Ok(()) => {
                info!(
                    "reaped the spill directory {} of dead process {pid}",
                    entry.path().display()
                );
                removed += 1;
            }
            Err(e) => warn!("could not reap {}: {e}", entry.path().display()),
        }
    }
    removed
}

/// Whether `/proc/<pid>` exists. Off Linux this answers "yes", so nothing is
/// ever reaped and the old behaviour (leak, but never delete a live run's
/// files) is preserved.
fn process_is_alive(pid: u32) -> bool {
    if cfg!(target_os = "linux") {
        Path::new(&format!("/proc/{pid}")).exists()
    } else {
        true
    }
}

// ---------------------------------------------------------------------------
// Readers for existing arc files
// ---------------------------------------------------------------------------

/// How an arc file on disk is encoded.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ArcFileFormat {
    /// `{src}\t{dst}\n` text, as produced by `TsvArcSink` and by Java's
    /// `PaymentGraphEdgeListBuilder`.
    Tsv,
    /// Fixed-width little-endian records.
    Binary(ArcCodec),
}

/// Sniffs *text versus binary*, and nothing else.
///
/// Reads the first 4 KiB. If every byte is an ASCII digit, a TAB or an LF the
/// file is reported as [`ArcFileFormat::Tsv`]; otherwise it is binary in
/// `codec`, whose record width the caller supplies.
///
/// # Why the width is a parameter and not a guess
///
/// A fixed-width binary file's record width is **not** inferable from its
/// length. The previous version tried (`len % 8 == 0` -> `Packed32`, else
/// `len % 16 == 0` -> `Wide`), but every multiple of 16 is also a multiple of
/// 8, so the `Wide` arm was unreachable and a 128-bit arc file was silently
/// decoded as twice as many bogus 64-bit arcs: a file holding the single wide
/// arc `(0, 2)` came back as the two arcs `0 -> 0` and `0 -> 2`, with no
/// warning and exit 0. There is no header to disambiguate, so the caller's
/// `--arc-codec` decides, and a length that is not a whole number of records
/// is an error.
///
/// **A zero-length file is reported as [`ArcFileFormat::Tsv`]**; the caller is
/// responsible for turning that into [`PgError::EmptyEdgeList`] rather than
/// letting it cascade. The Java pipeline did not: `logs/webgraph_builder.err`
/// records `IllegalArgumentException: Expected integer, found Token[EOF],
/// line 1`, thrown two stages downstream of the real problem.
pub fn detect_format(path: &Path, codec: ArcCodec) -> PgResult<ArcFileFormat> {
    let len = std::fs::metadata(path)
        .map_err(|e| PgError::io(path, e))?
        .len();
    if len == 0 {
        return Ok(ArcFileFormat::Tsv);
    }
    let mut file = File::open(path).map_err(|e| PgError::io(path, e))?;
    let mut head = vec![0u8; 4096.min(len as usize)];
    file.read_exact(&mut head)
        .map_err(|e| PgError::io(path, e))?;
    let textual = head
        .iter()
        .all(|&b| b.is_ascii_digit() || b == b'\t' || b == b'\n');
    if textual {
        return Ok(ArcFileFormat::Tsv);
    }
    let record = codec.record_size() as u64;
    if len % record != 0 {
        return Err(PgError::other(format!(
            "{}: not a TSV edge list, and its length {} is not a multiple of the {} bytes \
             per {:?} arc; pass --arc-codec for the width it was written with",
            path.display(),
            len,
            record,
            codec
        )));
    }
    Ok(ArcFileFormat::Binary(codec))
}

/// Reads a text edge list, yielding `(src, dst)`.
///
/// # Deliberately stricter than Java
///
/// Java's `ArcListASCIIGraph` used a `StreamTokenizer`, which treats every byte
/// in `0x00..=0x20` as a separator and *discards one token per record* — so
/// `0\t1\t9` parsed as the arc `0 -> 1` with the `9` silently eaten, and a
/// malformed three-column edge list produced a wrong graph with no error at
/// all. This parser requires exactly two TAB-separated decimal fields.
///
/// Blank lines are skipped, matching Java's tolerance of them. A malformed line
/// stores [`PgError::BadArcLine`] naming the file, the 1-based line number and
/// the offending text into the returned [`ErrorSlot`] and ends iteration;
/// callers must check the slot with [`crate::take_iter_error`].
pub fn read_tsv_arcs(path: &Path) -> PgResult<(TsvArcIter, ErrorSlot)> {
    let file = File::open(path).map_err(|e| PgError::io(path, e))?;
    let slot = new_error_slot();
    Ok((
        TsvArcIter {
            reader: BufReader::with_capacity(IO_BUF, file),
            path: path.to_path_buf(),
            line: 0,
            buf: String::with_capacity(64),
            slot: slot.clone(),
            done: false,
        },
        slot,
    ))
}

/// Iterator returned by [`read_tsv_arcs`].
pub struct TsvArcIter {
    reader: BufReader<File>,
    path: PathBuf,
    line: u64,
    buf: String,
    slot: ErrorSlot,
    done: bool,
}

impl TsvArcIter {
    fn fail(&mut self, e: PgError) -> Option<(usize, usize)> {
        self.done = true;
        if let Ok(mut g) = self.slot.lock() {
            if g.is_none() {
                *g = Some(e);
            }
        }
        None
    }
}

/// Parses a field of ASCII digits. Rejects an empty field, a sign and any
/// non-digit byte; leading zeros are accepted and normalised away, so a legacy
/// `007\t5` produced by some other tool becomes the arc `7 -> 5` rather than a
/// second, textually distinct copy of it.
fn parse_field(s: &str) -> Option<usize> {
    if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    s.parse::<u64>().ok().and_then(|v| usize::try_from(v).ok())
}

impl Iterator for TsvArcIter {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<(usize, usize)> {
        if self.done {
            return None;
        }
        loop {
            self.buf.clear();
            // Bound to a local so that the mutable borrows of `self.reader` and
            // `self.buf` end here rather than living for the whole `match`,
            // where they would collide with `self.fail`.
            let read = self.reader.read_line(&mut self.buf);
            match read {
                Ok(0) => {
                    self.done = true;
                    return None;
                }
                Ok(_) => {}
                Err(e) => {
                    let path = self.path.clone();
                    return self.fail(PgError::io(path, e));
                }
            }
            self.line += 1;
            let text = self.buf.trim_end_matches('\n').trim_end_matches('\r');
            if text.is_empty() {
                continue;
            }
            let parsed = text
                .split_once('\t')
                .and_then(|(a, b)| Some((parse_field(a)?, parse_field(b)?)));
            match parsed {
                Some(arc) => return Some(arc),
                None => {
                    let e = PgError::BadArcLine {
                        path: self.path.clone(),
                        line: self.line,
                        text: text.to_string(),
                    };
                    return self.fail(e);
                }
            }
        }
    }
}

/// Reads fixed-width little-endian arc records.
///
/// The file length is validated up front: a length that is not a multiple of
/// the record size means a torn write and yields [`PgError::TruncatedRun`]
/// immediately, before any arc is produced.
pub fn read_binary_arcs(path: &Path, codec: ArcCodec) -> PgResult<(BinaryArcIter, ErrorSlot)> {
    let record_size = codec.record_size();
    let len = std::fs::metadata(path)
        .map_err(|e| PgError::io(path, e))?
        .len();
    if len % record_size as u64 != 0 {
        return Err(PgError::TruncatedRun {
            path: path.to_path_buf(),
            len,
            record_size,
        });
    }
    let file = File::open(path).map_err(|e| PgError::io(path, e))?;
    let slot = new_error_slot();
    Ok((
        BinaryArcIter {
            reader: BufReader::with_capacity(IO_BUF, file),
            path: path.to_path_buf(),
            codec,
            remaining: len / record_size as u64,
            slot: slot.clone(),
            done: false,
        },
        slot,
    ))
}

/// Iterator returned by [`read_binary_arcs`].
#[derive(Debug)]
pub struct BinaryArcIter {
    reader: BufReader<File>,
    path: PathBuf,
    codec: ArcCodec,
    remaining: u64,
    slot: ErrorSlot,
    done: bool,
}

impl Iterator for BinaryArcIter {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<(usize, usize)> {
        if self.done || self.remaining == 0 {
            return None;
        }
        let rs = self.codec.record_size();
        let mut raw = [0u8; 16];
        if let Err(e) = self.reader.read_exact(&mut raw[..rs]) {
            self.done = true;
            if let Ok(mut g) = self.slot.lock() {
                if g.is_none() {
                    *g = Some(PgError::io(self.path.clone(), e));
                }
            }
            return None;
        }
        self.remaining -= 1;
        let key = match rs {
            8 => u64::from_le_bytes(raw[..8].try_into().expect("8 bytes")) as u128,
            _ => u128::from_le_bytes(raw),
        };
        Some(unpack_key(self.codec, key))
    }
}

/// Sorts an existing arc file, sniffing its format. This is the backend of
/// `pgraph sort-edges`.
pub fn sort_file(input: &Path, opts: SortOpts) -> PgResult<(SortedArcs, SortStats)> {
    let format = detect_format(input, opts.codec)?;
    info!(
        "sorting {} (detected {:?}) with codec {:?}",
        input.display(),
        format,
        opts.codec
    );
    let mut sorter = ArcSorter::new(opts)?;
    match format {
        ArcFileFormat::Tsv => {
            let (iter, slot) = read_tsv_arcs(input)?;
            for (src, dst) in iter {
                sorter.push(src as NodeId, dst as NodeId)?;
            }
            take_iter_error(&slot)?;
        }
        ArcFileFormat::Binary(codec) => {
            let (iter, slot) = read_binary_arcs(input, codec)?;
            for (src, dst) in iter {
                sorter.push(src as NodeId, dst as NodeId)?;
            }
            take_iter_error(&slot)?;
        }
    }
    sorter.finish()?;
    sorter.into_sorted()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{unpack_arc, ARC_ID_LIMIT};

    /// A fixed linear congruential generator, so the tests need no `rand`
    /// dependency and are bit-for-bit reproducible.
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 33
        }
    }

    fn tmpdir() -> tempfile::TempDir {
        tempfile::Builder::new()
            .prefix("pgraph-arcs-test-")
            .tempdir()
            .expect("temp dir")
    }

    /// `fsync(2)` on a character device returns EINVAL. A completed run used
    /// to be reported as `Error: I/O error on /dev/null: Invalid argument
    /// (os error 22)`, exit 1 — Java's `PrintWriter` never synced and handled
    /// `/dev/null`, fifos and process substitution fine.
    #[test]
    fn a_non_regular_destination_is_written_without_failing_on_fsync() {
        let dev_null = Path::new("/dev/null");
        if !dev_null.exists() {
            return; // not Unix; nothing to assert
        }
        let mut sink = TsvArcSink::create(dev_null).expect("create");
        sink.push(1, 2).expect("push");
        sink.push(3, 4).expect("push");
        sink.finish()
            .expect("a completed write to /dev/null is a success");
        assert_eq!(sink.count(), 2);

        let mut sink = BinaryArcSink::create(dev_null, ArcCodec::Packed32).expect("create");
        sink.push(1, 2).expect("push");
        sink.finish().expect("same for the binary sink");
    }

    /// A sink dropped without `finish` is a failed run: the partial file must
    /// go. It used to stay — 157 MB was measured after one dangling-reference
    /// error, and the name carries the pid, so every retry left another one.
    #[test]
    fn an_uncommitted_sink_leaves_no_temporary_file() {
        let dir = tmpdir();
        let path = dir.path().join("arcs.tsv");
        {
            let mut sink = TsvArcSink::create(&path).expect("create");
            for i in 0..10_000u64 {
                sink.push(i, i + 1).expect("push");
            }
            // No `finish()`: the run "failed" here.
        }
        let left: Vec<_> = std::fs::read_dir(dir.path())
            .expect("read_dir")
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        assert!(left.is_empty(), "an uncommitted sink left {left:?} behind");
        assert!(!path.exists(), "the final path must not appear either");
    }

    /// The reaper must remove a dead process's spill directory and keep this
    /// process's own.
    #[test]
    fn stale_spill_directories_are_reaped_and_live_ones_are_not() {
        let dir = tmpdir();
        // Pid 0 is never a live process on Linux.
        let dead = dir.path().join("pgraph-sort-0-abcdef");
        let mine = dir
            .path()
            .join(format!("pgraph-sort-{}-abcdef", std::process::id()));
        let other = dir.path().join("something-else");
        for d in [&dead, &mine, &other] {
            std::fs::create_dir_all(d).expect("mkdir");
        }
        std::fs::write(dead.join("run-00000.arcs"), [0u8; 8]).expect("write");

        let removed = reap_stale_run_dirs(dir.path());
        if cfg!(target_os = "linux") {
            assert_eq!(removed, 1);
            assert!(!dead.exists());
        }
        assert!(mine.exists(), "a live process's directory must survive");
        assert!(other.exists(), "unrelated directories are never touched");
    }

    #[test]
    fn pack_arc_round_trips() {
        for &(s, d) in &[
            (0u64, 0u64),
            (1, 2),
            (9, 171),
            (778_613_437, 914),
            (ARC_ID_LIMIT - 1, ARC_ID_LIMIT - 1),
        ] {
            let k = pack_arc(s, d).expect("representable");
            assert_eq!(unpack_arc(k), (s, d));
        }
        assert!(pack_arc(ARC_ID_LIMIT, 0).is_none());
        assert!(pack_arc(0, ARC_ID_LIMIT).is_none());
    }

    #[test]
    fn tsv_sink_is_canonical() {
        let dir = tmpdir();
        let path = dir.path().join("el.tsv");
        let mut sink = TsvArcSink::create(&path).expect("create");
        for &(s, d) in &[(9u64, 171u64), (9, 172), (172, 184)] {
            sink.push(s, d).expect("push");
        }
        sink.finish().expect("finish");
        let text = std::fs::read_to_string(&path).expect("read");
        assert_eq!(text, "9\t171\n9\t172\n172\t184\n");
        assert_eq!(sink.count(), 3);
    }

    /// 50 000 shuffled arcs with heavy duplication, a memory budget small
    /// enough to force several spill runs, compared against a naive sort.
    fn heavy_duplication_input() -> Vec<(u64, u64)> {
        let mut lcg = Lcg(0x5EED_1234);
        (0..50_000)
            .map(|_| (lcg.next() % 400, lcg.next() % 400))
            .collect()
    }

    #[test]
    fn spilling_sort_matches_naive_sort_and_dedups() {
        let dir = tmpdir();
        let input = heavy_duplication_input();

        let opts = SortOpts {
            memory_bytes: 12_000 * 2 * 8, // 12 000 arcs per run => 5 runs
            tmp_dir: dir.path().to_path_buf(),
            ..SortOpts::default()
        };
        let mut sorter = ArcSorter::new(opts).expect("sorter");
        assert_eq!(sorter.arcs_per_run(), 12_000);
        for &(s, d) in &input {
            sorter.push(s, d).expect("push");
        }
        sorter.finish().expect("finish");
        let (sorted, stats) = sorter.into_sorted().expect("into_sorted");
        assert!(stats.runs >= 3, "expected several runs, got {}", stats.runs);

        let mut expected: Vec<(u64, u64)> = input.clone();
        expected.sort_unstable();
        let raw = expected.len() as u64;
        expected.dedup();

        let (iter, slot) = sorted.iter().expect("iter");
        let got: Vec<(u64, u64)> = iter.map(|(s, d)| (s as u64, d as u64)).collect();
        take_iter_error(&slot).expect("no merge error");

        assert_eq!(got, expected);
        assert_eq!(sorted.num_arcs(), expected.len() as u64);
        assert_eq!(stats.raw_arcs, raw);
        assert_eq!(stats.distinct_arcs, expected.len() as u64);
        assert_eq!(stats.duplicates_removed, raw - expected.len() as u64);
        assert!(
            stats.duplicates_removed > 0,
            "the fixture must have duplicates"
        );
    }

    #[test]
    fn no_dedup_preserves_every_duplicate() {
        let dir = tmpdir();
        let input = heavy_duplication_input();

        let opts = SortOpts {
            dedup: false,
            memory_bytes: 12_000 * 2 * 8,
            tmp_dir: dir.path().to_path_buf(),
            ..SortOpts::default()
        };
        let mut sorter = ArcSorter::new(opts).expect("sorter");
        for &(s, d) in &input {
            sorter.push(s, d).expect("push");
        }
        sorter.finish().expect("finish");
        let (sorted, stats) = sorter.into_sorted().expect("into_sorted");

        let mut expected: Vec<(u64, u64)> = input.clone();
        expected.sort_unstable();

        let (iter, slot) = sorted.iter().expect("iter");
        let got: Vec<(u64, u64)> = iter.map(|(s, d)| (s as u64, d as u64)).collect();
        take_iter_error(&slot).expect("no merge error");

        assert_eq!(got, expected);
        assert_eq!(stats.duplicates_removed, 0);
        assert_eq!(sorted.num_arcs(), expected.len() as u64);
        assert!(got.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn wide_codec_handles_ids_above_2_32_and_downgrades_radix() {
        let dir = tmpdir();
        let opts = SortOpts {
            codec: ArcCodec::Wide,
            algo: SortAlgo::Radix,
            tmp_dir: dir.path().to_path_buf(),
            ..SortOpts::default()
        };
        let mut sorter = ArcSorter::new(opts).expect("sorter");
        assert_eq!(
            sorter.effective_algo(),
            SortAlgo::Pdq,
            "the wide codec must downgrade radix to pdq"
        );

        let big = ARC_ID_LIMIT + 7;
        let arcs = [(big, 5u64), (1u64, big), (big, big), (0, 1)];
        for &(s, d) in &arcs {
            sorter.push(s, d).expect("push");
        }
        sorter.finish().expect("finish");
        let (sorted, stats) = sorter.into_sorted().expect("into_sorted");
        assert_eq!(stats.max_node_id, big);

        let (iter, slot) = sorted.iter().expect("iter");
        let got: Vec<(u64, u64)> = iter.map(|(s, d)| (s as u64, d as u64)).collect();
        take_iter_error(&slot).expect("no error");
        let mut expected = arcs.to_vec();
        expected.sort_unstable();
        assert_eq!(got, expected);
    }

    #[test]
    fn packed_codec_rejects_ids_at_the_limit() {
        let dir = tmpdir();
        let opts = SortOpts {
            tmp_dir: dir.path().to_path_buf(),
            ..SortOpts::default()
        };
        let mut sorter = ArcSorter::new(opts).expect("sorter");
        let err = sorter.push(ARC_ID_LIMIT, 0).expect_err("must overflow");
        match err {
            PgError::ArcCodecOverflow(id) => assert_eq!(id, ARC_ID_LIMIT),
            other => panic!("expected ArcCodecOverflow, got {other:?}"),
        }
    }

    #[test]
    fn detect_format_distinguishes_tsv_binary_and_garbage() {
        let dir = tmpdir();

        let tsv = dir.path().join("a.tsv");
        std::fs::write(&tsv, b"0\t1\n2\t3\n").expect("write");
        assert_eq!(
            detect_format(&tsv, ArcCodec::Packed32).expect("detect"),
            ArcFileFormat::Tsv
        );

        let bin = dir.path().join("a.bin");
        std::fs::write(&bin, 1u64.to_le_bytes()).expect("write");
        assert_eq!(
            detect_format(&bin, ArcCodec::Packed32).expect("detect"),
            ArcFileFormat::Binary(ArcCodec::Packed32)
        );

        let odd = dir.path().join("a.odd");
        std::fs::write(&odd, [0xFFu8; 13]).expect("write");
        assert!(detect_format(&odd, ArcCodec::Packed32).is_err());

        let empty = dir.path().join("a.empty");
        std::fs::write(&empty, b"").expect("write");
        assert_eq!(
            detect_format(&empty, ArcCodec::Packed32).expect("detect"),
            ArcFileFormat::Tsv
        );
    }

    /// A 128-bit arc file is 16 bytes per record, which is also a multiple of
    /// 8: the width can only come from the caller.
    ///
    /// Before this, `--arc-codec wide` was ignored and the single wide arc
    /// `(0, 2)` was decoded as the TWO packed arcs `(0, 0)` and `(0, 2)`, with
    /// no warning and exit 0.
    #[test]
    fn a_wide_arc_file_is_not_mistaken_for_two_packed_arcs() {
        let dir = tmpdir();
        let path = dir.path().join("wide.bin");
        std::fs::write(&path, wide_arc(0, 2).to_le_bytes()).expect("write");

        assert_eq!(
            detect_format(&path, ArcCodec::Wide).expect("detect"),
            ArcFileFormat::Binary(ArcCodec::Wide)
        );
        let (iter, slot) = read_binary_arcs(&path, ArcCodec::Wide).expect("open");
        let arcs: Vec<_> = iter.collect();
        take_iter_error(&slot).expect("no error");
        assert_eq!(arcs, vec![(0, 2)]);

        // And the same file, read with the wrong width, is still two arcs —
        // which is why the width is never guessed.
        assert_eq!(
            detect_format(&path, ArcCodec::Packed32).expect("detect"),
            ArcFileFormat::Binary(ArcCodec::Packed32)
        );
    }

    #[test]
    fn malformed_tsv_line_names_its_line_number() {
        let dir = tmpdir();
        let path = dir.path().join("bad.tsv");
        std::fs::write(&path, b"0\t1\n2\t3\nnot an arc\n4\t5\n").expect("write");
        let (iter, slot) = read_tsv_arcs(&path).expect("open");
        let got: Vec<_> = iter.collect();
        assert_eq!(got, vec![(0, 1), (2, 3)]);
        match take_iter_error(&slot) {
            Err(PgError::BadArcLine { line, text, .. }) => {
                assert_eq!(line, 3);
                assert_eq!(text, "not an arc");
            }
            other => panic!("expected BadArcLine, got {other:?}"),
        }
    }

    #[test]
    fn run_file_round_trips_and_detects_truncation() {
        let dir = tmpdir();
        let opts = SortOpts {
            memory_bytes: 1024 * 2 * 8,
            tmp_dir: dir.path().to_path_buf(),
            keep_intermediate: true,
            ..SortOpts::default()
        };
        let mut sorter = ArcSorter::new(opts).expect("sorter");
        for i in 0..1024u64 {
            sorter.push(i % 17, i).expect("push");
        }
        // The 1024th push fills the buffer and spills exactly one run.
        sorter.finish().expect("finish");
        let (sorted, stats) = sorter.into_sorted().expect("into_sorted");
        assert_eq!(stats.runs, 1);

        let run = dir.path().join("run-00000.arcs");
        assert!(run.is_file(), "the run must survive keep_intermediate");

        let (iter, slot) = read_binary_arcs(&run, ArcCodec::Packed32).expect("open run");
        let from_file: Vec<(usize, usize)> = iter.collect();
        take_iter_error(&slot).expect("no error");

        let (iter, slot) = sorted.iter().expect("iter");
        let from_merge: Vec<(usize, usize)> = iter.collect();
        take_iter_error(&slot).expect("no error");
        assert_eq!(from_file, from_merge);
        assert_eq!(from_file.len(), 1024);

        // Lop off one byte: the length is no longer a multiple of 8.
        let truncated = dir.path().join("truncated.arcs");
        let mut bytes = std::fs::read(&run).expect("read run");
        bytes.pop();
        std::fs::write(&truncated, &bytes).expect("write");
        match read_binary_arcs(&truncated, ArcCodec::Packed32) {
            Err(PgError::TruncatedRun {
                len, record_size, ..
            }) => {
                assert_eq!(len, bytes.len() as u64);
                assert_eq!(record_size, 8);
            }
            other => panic!("expected TruncatedRun, got {other:?}"),
        }
    }

    #[test]
    fn sort_file_round_trips_through_tsv() {
        let dir = tmpdir();
        let raw = dir.path().join("raw.tsv");
        std::fs::write(&raw, b"5\t4\n1\t2\n5\t4\n0\t9\n").expect("write");
        let opts = SortOpts {
            tmp_dir: dir.path().to_path_buf(),
            ..SortOpts::default()
        };
        let (sorted, stats) = sort_file(&raw, opts).expect("sort");
        assert_eq!(stats.raw_arcs, 4);
        assert_eq!(sorted.num_arcs(), 3);

        let out = dir.path().join("el.tsv");
        assert_eq!(sorted.write_tsv(&out).expect("write_tsv"), 3);
        assert_eq!(
            std::fs::read_to_string(&out).expect("read"),
            "0\t9\n1\t2\n5\t4\n"
        );
    }

    #[test]
    fn memory_budget_below_one_run_is_rejected() {
        let opts = SortOpts {
            memory_bytes: 64,
            ..SortOpts::default()
        };
        assert!(ArcSorter::new(opts).is_err());
    }
}
