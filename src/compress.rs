//! BVGraph compression: the last stage of the pipeline, where the sorted arc
//! stream becomes `<basename>.{graph,offsets,properties}`.
//!
//! This is the only module in the crate that touches `webgraph`,
//! `dsi-bitstream` and `dsi-progress-logger`.
//!
//! # Attribution
//!
//! The pipeline this module terminates, and the graph-construction algorithm
//! behind it, are the work of **Matteo Loporchio**. This is an independent
//! reimplementation of that design; the algorithm is his.
//!
//! # Streamed, never materialised
//!
//! The reference implementation of this stage compressed a *text* edge list,
//! and did it by building the whole graph in memory first — one growable
//! integer list per node, every arc resident before a single bit was written.
//! That materialisation is what the 200 GB memory reservation in the historical
//! run scripts was paying for, and it put a hard floor under the machine the
//! last stage could run on.
//!
//! Here the sorted arcs are streamed straight into `webgraph-rs`: the graph is
//! never materialised, peak memory is the compression window rather than the
//! graph, and the 210 GB text edge list becomes optional — the arcs can go from
//! the sorter to the bitstream without ever being spelled out in ASCII.
//!
//! # Compression parameters
//!
//! [`CompFlags::default()`] is
//! `{ outdegrees: Gamma, references: Unary, blocks: Gamma, intervals: Gamma,
//! residuals: Zeta(3), min_interval_length: 4, compression_window: 7,
//! max_ref_count: 3 }` — **identical to the canonical `BVGraph` defaults**
//! (`windowsize=7`, `maxrefcount=3`, `minintervallength=4`, `zetak=3`), which
//! is what makes the bitstream interchangeable with graphs written by any other
//! BVGraph writer, and what makes the golden byte comparison meaningful at all.
//! Passing the flags explicitly is redundant, and is done here as
//! documentation: a future default change upstream would otherwise silently
//! move the output bits.
//!
//! `BvCompConf::bvgraphz()` and `BvCompConf::chunk_size()` are **forbidden**:
//! they switch to the Zuckerli compressor `BvCompZ`, whose reference selection
//! is not BVGraph's. The result is a different bitstream format, not a smaller
//! BVGraph, so nothing downstream that expects a BVGraph can read it.
//!
//! # Node count
//!
//! The historical stage inferred `numNodes = max(id over all sources AND all
//! targets) + 1`, filling the gaps with outdegree-0 nodes and renumbering
//! nothing; `--num-nodes from-arcs` reproduces exactly that. The node map's
//! `next_id`, by contrast, counts every UTXO slot, including the unspent
//! outputs that never appear in any arc. The two genuinely differ, `next_id` is
//! the larger, and [`CompressOpts::num_nodes`] is the caller's explicit
//! decision between them.
//!
//! Neither candidate has a ceiling here: `num_nodes` is a [`crate::NodeId`]
//! count, `usize` all the way through `webgraph-rs`, and a graph above
//! `u32::MAX` nodes is compressed and loaded like any other; the preflight
//! deliberately imposes no bound of its own.
//!
//! # `.properties` is not byte-reproducible
//!
//! The reference `.properties` files carry a `#<generation date>` comment line,
//! so that file can never match byte for byte — and should never be diffed.
//! `webgraph-rs` additionally writes a different key set, in insertion order,
//! adding `endianness=` and `length=` and omitting all the cosmetic
//! `bitsfor*`/`*expstats`/`*avggap` keys, every one of which a BVGraph loader
//! ignores. Only `.graph` and `.offsets` are valid byte-comparison targets, and
//! even there the files written here are 4-8 bytes longer, because
//! `BufBitWriter` pads the bitstream to a 64-bit word where the reference
//! writer padded it to a byte. Compare only the first `ceil(length / 8)` bytes
//! — the count `length=` declares; the rest is zero padding.

use std::panic::AssertUnwindSafe;
use std::path::Path;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use dsi_bitstream::prelude::BE;
use dsi_progress_logger::prelude::*;
use log::{debug, info, warn};
use webgraph::graphs::arc_list_graph;
use webgraph::prelude::*;

use crate::arcs::SortedArcs;
use crate::{new_error_slot, take_iter_error, ErrorSlot, Error, NodeId, Result};

// ---------------------------------------------------------------------------
// Options and statistics
// ---------------------------------------------------------------------------

/// Configuration for the BVGraph writers.
#[derive(Clone, Debug)]
pub struct CompressOpts {
    /// Number of nodes in the produced graph. Mandatory: nothing infers it.
    /// See the module documentation for `node-map` versus `from-arcs`.
    pub num_nodes: usize,
    /// Base directory for the parallel compressor's partial bitstreams.
    ///
    /// Note this does **not** control the parallel external sort's spill files:
    /// `ParSortIters`/`ParSortPairs` call bare `tempfile::tempdir()` and ignore
    /// `BvCompConf::tmp_dir`, so `main.rs` must set `TMPDIR` before any thread
    /// is spawned.
    pub tmp_dir: PathBuf,
    /// Threads for the parallel path. `0` means "let `rayon` decide".
    pub threads: usize,
    /// Memory budget handed to the parallel external sort.
    pub memory_bytes: u64,
    /// Use the parallel path. Not byte-identical to the sequential one; see
    /// [`compress_unsorted_iter`].
    pub parallel: bool,
    /// Also build `<basename>.ef`, required only for *random* access
    /// (`BvGraph`). Sequential access (`BvGraphSeq`) does not need it, and on a
    /// 2.2e9-node graph it is not free.
    pub build_ef: bool,
    /// Permit an edge list with no arcs instead of failing with
    /// [`Error::EmptyEdgeList`].
    pub allow_empty: bool,
    /// How thoroughly [`verify_graph`] re-reads the graph it just wrote.
    pub verify: VerifyLevel,
    /// How often the compressor's progress loggers report.
    ///
    /// This is the one place in the crate where the interval is carried in an
    /// options struct rather than threaded as a `&mut impl ProgressLog`. The
    /// loggers below are handed to webgraph's compressor and external sort,
    /// which own their lifecycle, so there is no loop here to take a logger.
    pub log_interval: Duration,
}

/// How much of the finished graph [`verify_graph`] reads back.
///
/// The full recount is a *second* complete decompression of the graph: at
/// `N = 28` that is `9.5e9` arcs over `2.2e9` nodes, several minutes added to
/// every run, on top of the arc count the `.properties` file already declares.
/// It used to be unconditional and unskippable.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum VerifyLevel {
    /// No verification at all. The compressor's own errors still apply.
    None,
    /// `.properties` exists, the node count matches and the *declared* arc
    /// count matches. O(1) reads. This is the default.
    #[default]
    Quick,
    /// `Quick`, plus a full decompression that counts every arc.
    Full,
}

impl Default for CompressOpts {
    fn default() -> Self {
        CompressOpts {
            num_nodes: 0,
            tmp_dir: PathBuf::from("./tmp"),
            threads: 0,
            memory_bytes: 256 << 20,
            parallel: false,
            build_ef: false,
            allow_empty: false,
            verify: VerifyLevel::default(),
            log_interval: Duration::from_secs(10),
        }
    }
}

/// What the compressor produced.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct CompressStats {
    /// Nodes in the graph, i.e. [`CompressOpts::num_nodes`].
    pub nodes: usize,
    /// Arcs actually written.
    pub arcs: u64,
    /// Length of the graph bitstream in bits, as recorded in `length=`.
    pub bits: u64,
}

// ---------------------------------------------------------------------------
// Basenames
// ---------------------------------------------------------------------------

/// Rejects a basename whose file name contains a `.`.
///
/// `webgraph-rs` derives its three output paths with [`Path::with_extension`],
/// which **replaces** an existing extension: a prefix such as `graph/pg_1.5`
/// would produce `graph/pg_1.graph`, silently clobbering a different graph.
/// A pipeline that builds its three output paths by plain string concatenation
/// never had this problem, so a prefix that was harmless in the historical run
/// scripts has to be rejected here rather than quietly writing over the
/// neighbouring graph.
pub fn sanitize_basename(p: &Path) -> Result<()> {
    let Some(name) = p.file_name() else {
        return Err(Error::BadBasename(p.to_path_buf()));
    };
    if name.to_string_lossy().contains('.') {
        return Err(Error::BadBasename(p.to_path_buf()));
    }
    Ok(())
}

/// The three files a successful compression leaves behind.
fn output_paths(basename: &Path) -> [PathBuf; 3] {
    [
        basename.with_extension(GRAPH_EXTENSION),
        basename.with_extension(OFFSETS_EXTENSION),
        basename.with_extension(PROPERTIES_EXTENSION),
    ]
}

/// Removes a half-written graph.
///
/// `.properties` is written *last*, after the whole bitstream, so a failed run
/// leaves a `.graph` with no `.properties`. Rather than leave that trap on
/// disk — where a later `verify_graph` would have to diagnose it — everything
/// is removed as soon as the compression is known to have failed.
fn remove_outputs(basename: &Path) {
    for p in output_paths(basename) {
        if p.exists() {
            if let Err(e) = std::fs::remove_file(&p) {
                warn!("could not remove the incomplete {}: {e}", p.display());
            }
        }
    }
}

/// Common preflight for every entry point: check the basename, then create the
/// output directory and the compressor's temporary directory.
///
/// # There is deliberately no node-count ceiling
///
/// Nothing here bounds [`CompressOpts::num_nodes`], and nothing should. The
/// whole compression path carries a node id as [`crate::NodeId`] — `usize`,
/// the type `webgraph-rs` itself uses: `SequentialGraph: SequentialLabeling<Label
/// = usize>`, `num_nodes: usize`, and a plain unbounded decimal in the
/// `.properties` `nodes=` line. A graph above `i32::MAX` or `u32::MAX` nodes is
/// therefore written and read back like any other.
///
/// An earlier version of this function rejected `num_nodes > i32::MAX` here,
/// refusing a real 2 181 021 971-node graph — the count an `N = 28` run
/// reaches — after the whole sort had already been paid for. Do not
/// reintroduce it: a ceiling that fires at the last stage, on work that is
/// already done, is the most expensive place to discover a limit.
///
/// # The one reader that cannot load such a graph
///
/// This is a real limitation and the deleted check was a clumsy way of
/// signalling it. [`output_paths`]'s `.properties` names a `graphclass` whose
/// reference implementation is 32-bit: it stores a node id in a signed 32-bit
/// integer and rejects `nodes > i32::MAX` at load. A graph above that count is
/// read back by `webgraph-rs` — including this crate's own
/// [`verify_graph`] — but **not** by that implementation.
///
/// The ceiling is therefore an *interoperability* property of the consumer,
/// not a property of the format or of this writer, and it is documented rather
/// than enforced. Refusing to write the graph did not make that reader able to
/// load a smaller one; it only meant the graph did not exist at all.
fn prepare(basename: &Path, opts: &CompressOpts) -> Result<()> {
    sanitize_basename(basename)?;
    if let Some(parent) = basename.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
        }
    }
    std::fs::create_dir_all(&opts.tmp_dir).map_err(|e| Error::io(&opts.tmp_dir, e))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// The validating arc adapter
// ---------------------------------------------------------------------------

/// Checks, as the arcs stream past, everything `webgraph-rs` does not.
///
/// The sequential path is fast precisely because it trusts its input:
///
/// * `arc_list_graph::NodeLabels` carries `unsafe impl SortedLender`. Nothing
///   verifies sortedness; unsorted-by-source input trips
///   `assert!(node == self.next_node - 1)` and **panics** rather than returning
///   an error.
/// * `NodeLabels::next` returns `None` once `next_node == num_nodes`, so an arc
///   with `src >= num_nodes` — and every arc after it — is **silently
///   dropped**.
/// * `dst` is not bounds-checked at all, so `dst >= num_nodes` writes successor
///   lists naming nodes the graph does not have — and any reader that indexes a
///   per-node array by destination goes out of bounds on it.
/// * `Compressor::write` computes `residuals[i] - residuals[i-1] - 1`, which
///   underflows on a duplicate or descending successor. Our release profile
///   sets `overflow-checks = true` so that panics loudly instead of wrapping
///   into a corrupt bitstream.
///
/// This adapter turns all of those into typed errors, reported through an
/// [`ErrorSlot`] because [`Iterator`] cannot return a `Result`.
struct Validate<I> {
    inner: I,
    num_nodes: usize,
    prev: Option<(NodeId, NodeId)>,
    count: Arc<AtomicU64>,
    slot: ErrorSlot,
    done: bool,
}

impl<I> Validate<I> {
    fn fail(&mut self, e: Error) -> Option<(NodeId, NodeId)> {
        self.done = true;
        if let Ok(mut g) = self.slot.lock() {
            if g.is_none() {
                *g = Some(e);
            }
        }
        None
    }
}

impl<I: Iterator<Item = (NodeId, NodeId)>> Iterator for Validate<I> {
    type Item = (NodeId, NodeId);

    #[inline]
    fn next(&mut self) -> Option<(NodeId, NodeId)> {
        if self.done {
            return None;
        }
        let (src, dst) = self.inner.next()?;
        if src >= self.num_nodes || dst >= self.num_nodes {
            return self.fail(Error::ArcOutOfBounds {
                src,
                dst,
                num_nodes: self.num_nodes,
            });
        }
        if let Some((p_src, p_dst)) = self.prev {
            if (src, dst) <= (p_src, p_dst) {
                return self.fail(Error::UnsortedArcs {
                    p_src,
                    p_dst,
                    c_src: src,
                    c_dst: dst,
                });
            }
        }
        self.prev = Some((src, dst));
        self.count.fetch_add(1, Ordering::Relaxed);
        Some((src, dst))
    }
}

/// Turns a caught panic payload into a message.
fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "panic with a non-string payload".to_string()
    }
}

// ---------------------------------------------------------------------------
// Sequential compression: the deterministic, reference-identical path
// ---------------------------------------------------------------------------

/// Compresses arcs that are already sorted by `(src, dst)` and deduplicated.
///
/// This is the **default** path, and the one verified to reproduce the
/// reference bitstream exactly: on the golden vector `(0,1) (0,2) (1,2) (2,0)
/// (3,4)` over five nodes it returns 40 bits and writes `7d c5 da f1 77` /
/// `8d 14 28 52`, which is the pinned reference output byte for byte.
///
/// `num_arcs` is used only for the empty-input check and for progress
/// forecasting; the returned [`CompressStats::arcs`] is the number actually
/// written, counted as the arcs go past.
///
/// # API notes
///
/// * [`LeftIterator`] is mandatory: `NodeLabels`'s `Label` is `(usize, ())`
///   while `comp_lender` demands `Label = usize`. Omitting it yields an
///   unreadable higher-ranked-trait-bound error.
/// * `arc_list_graph::NodeLabels::new` is used rather than `ArcListGraph::new`,
///   because `ArcListGraph` requires `I: Iterator + Clone` (its `iter_from`
///   clones and re-drives the iterator) and a file-backed merge iterator is not
///   `Clone`. `NodeLabels::new` has no such bound. Note `arc_list_graph` is not
///   re-exported by `webgraph::prelude` and must be imported by full path.
/// * The `comp_lender` call is wrapped in [`std::panic::catch_unwind`] so that
///   an assertion inside `webgraph` becomes [`Error::CompressorPanic`]
///   instead of aborting the process with a bare backtrace. This is also why
///   the release profile deliberately does **not** set `panic = "abort"`.
///
/// # Errors
///
/// On any failure the three output files are removed, so a half-written
/// bitstream can never be mistaken for a finished graph.
pub fn compress_sorted_iter<I>(
    arcs: I,
    num_arcs: u64,
    basename: &Path,
    opts: &CompressOpts,
) -> Result<CompressStats>
where
    I: Iterator<Item = (NodeId, NodeId)>,
{
    prepare(basename, opts)?;
    if num_arcs == 0 && !opts.allow_empty {
        // Fail before opening anything, and say what is actually wrong. A text
        // edge-list reader that primes its first line while constructing its
        // node iterator reports an empty input only as "expected an integer,
        // found end of file, line 1" — a parse error two stages downstream of
        // the real problem, which is exactly the cascade
        // `logs/webgraph_builder.err` records.
        return Err(Error::EmptyEdgeList);
    }

    let slot = new_error_slot();
    let count = Arc::new(AtomicU64::new(0));
    let validated = Validate {
        inner: arcs,
        num_nodes: opts.num_nodes,
        prev: None,
        count: count.clone(),
        slot: slot.clone(),
        done: false,
    };

    info!(
        "compressing {} nodes / {} arcs into {} (sequential, deterministic)",
        opts.num_nodes,
        num_arcs,
        basename.display()
    );

    let lender = LeftIterator(arc_list_graph::NodeLabels::new(
        opts.num_nodes,
        validated.map(|p| (p, ())),
    ));

    // `item_name` is inert: webgraph's compressor sets its own `item_name` and
    // `expected_updates` (comp/impls.rs), which is why upstream's
    // `cli/src/to/bvgraph.rs` passes only `display_memory` and `log_interval`.
    // Kept so the intent is readable if webgraph ever stops overriding it.
    let mut pl = progress_logger![
        display_memory = true,
        item_name = "node",
        log_interval = opts.log_interval
    ];
    let conf = BvCompConf::new(basename)
        // Redundant — this is what `BvCompConf::new` already installs — but it
        // is the line that documents "the canonical BVGraph parameters", and
        // the line that would have to change for the output bits to move.
        .comp_flags(CompFlags::default())
        .tmp_dir(&opts.tmp_dir);
    let mut conf = conf.progress_logger(&mut pl);

    let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| {
        conf.comp_lender::<BE, _>(lender, Some(opts.num_nodes))
    }));

    // The validator's error is the root cause, so report it first: a truncated
    // arc stream makes `comp_lender` succeed on a shorter graph.
    if let Err(e) = take_iter_error(&slot) {
        remove_outputs(basename);
        return Err(e);
    }
    let bits = match outcome {
        Err(payload) => {
            remove_outputs(basename);
            return Err(Error::CompressorPanic(panic_message(payload)));
        }
        Ok(Err(e)) => {
            remove_outputs(basename);
            return Err(Error::other(format!(
                "could not compress the graph: {e:#}"
            )));
        }
        Ok(Ok(bits)) => bits,
    };

    let arcs_written = count.load(Ordering::Relaxed);
    finish_outputs(basename, opts, arcs_written)?;
    Ok(CompressStats {
        nodes: opts.num_nodes,
        arcs: arcs_written,
        bits,
    })
}

/// Optional Elias-Fano index, plus a last sanity check that `.properties`
/// really landed.
fn finish_outputs(basename: &Path, opts: &CompressOpts, arcs: u64) -> Result<()> {
    let props = basename.with_extension(PROPERTIES_EXTENSION);
    if !props.is_file() {
        remove_outputs(basename);
        return Err(Error::other(format!(
            "{} was not written: the compression did not run to completion",
            props.display()
        )));
    }
    debug!("wrote {} arcs to {}", arcs, basename.display());
    if opts.build_ef {
        webgraph::graphs::bvgraph::store_ef_with_data(
            opts.num_nodes,
            basename.with_extension(GRAPH_EXTENSION),
            basename.with_extension(OFFSETS_EXTENSION),
            basename.with_extension(EF_EXTENSION),
            no_logging![],
        )
        .map_err(|e| Error::other(format!("could not build the Elias-Fano offsets: {e:#}")))?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Parallel compression
// ---------------------------------------------------------------------------

/// Compresses arcs that need not be sorted or deduplicated, using the parallel
/// external sort and the 112-way parallel compressor.
///
/// # Not byte-identical
///
/// Each parallel chunk restarts its compression window, so a node at a chunk
/// boundary cannot reference a node before it. The output is therefore
/// semantically identical but a few bits longer (measured: 237 354 versus
/// 237 337 bits on a 2 000-node graph); `webgraph::traits::graph::eq` passes.
/// This is inherent to chunked parallel compression rather than a quirk of this
/// implementation, which is why the sequential path is the default here.
///
/// Confirmed at full scale. `/data/bitcoin/2022/pg/pg.graph` was compressed in
/// parallel on a 112-core box and this pipeline's sequential `pg_28.graph` was
/// not; the two decode to the same graph (2 181 021 971 nodes and
/// 8 639 773 499 arcs walked in lockstep, zero differing successor lists), and
/// differ by 397 bits spread over 73 nodes in 20 clusters — 10 of them exactly
/// on a `ceil(nodes / 112)` boundary and all 20 within five nodes of one.
/// `docs/pg-graph-divergence.md` has the full account.
///
/// # API notes
///
/// * The sort **and** the compression must both run inside one
///   `pool.install`: `ParSortedGraphConf::default` reads
///   `rayon::current_num_threads()` at construction time to choose the
///   partition count, so building the sorted graph outside the pool silently
///   mismatches `par_comp`'s parallelism.
/// * `ThreadPool::install` requires `OP: FnOnce() -> R + Send`, hence the
///   `I: Send` bound, even though `sort_pairs` consumes the iterator on the
///   calling thread.
/// * `ParSortedGraph` memory-maps its batches. On Linux a low
///   `/proc/sys/vm/max_map_count` makes a large run die with `ENOMEM` at the
///   merge; `main.rs` warns about that at startup.
pub fn compress_unsorted_iter<I>(
    arcs: I,
    basename: &Path,
    opts: &CompressOpts,
) -> Result<CompressStats>
where
    I: Iterator<Item = (NodeId, NodeId)> + Send,
{
    prepare(basename, opts)?;

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(opts.threads)
        .build()
        .map_err(|e| Error::other(format!("could not build the compression thread pool: {e}")))?;

    info!(
        "compressing {} nodes into {} (parallel; output is NOT byte-identical to the sequential path)",
        opts.num_nodes,
        basename.display()
    );

    let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| {
        pool.install(move || -> Result<u64> {
            // `item_name` inert here too: `par_sort_pairs` sets it to "pair".
            let mut pls = progress_logger![
                display_memory = true,
                item_name = "arc",
                log_interval = opts.log_interval
            ];
            let sorted = ParSortedGraph::config()
                .dedup()
                .memory_usage(MemoryUsage::MemorySize(opts.memory_bytes as usize))
                .progress_logger(&mut pls)
                .sort_pairs(opts.num_nodes, arcs)
                .map_err(|e| Error::other(format!("could not sort the arcs: {e:#}")))?;

            let mut plc = progress_logger![
                display_memory = true,
                item_name = "node",
                log_interval = opts.log_interval
            ];
            let conf = BvCompConf::new(basename)
                .comp_flags(CompFlags::default())
                .tmp_dir(&opts.tmp_dir);
            let mut conf = conf.progress_logger(&mut plc);
            conf.par_comp::<BE, _>(sorted)
                .map_err(|e| Error::other(format!("could not compress the graph: {e:#}")))
        })
    }));

    let bits = match outcome {
        Err(payload) => {
            remove_outputs(basename);
            return Err(Error::CompressorPanic(panic_message(payload)));
        }
        Ok(Err(e)) => {
            remove_outputs(basename);
            return Err(e);
        }
        Ok(Ok(bits)) => bits,
    };

    // The parallel path never sees the arcs itself, so the authoritative count
    // is the one the compressor recorded.
    let arcs_written = read_property(basename, "arcs")?
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0);
    if arcs_written == 0 && !opts.allow_empty {
        remove_outputs(basename);
        return Err(Error::EmptyEdgeList);
    }
    finish_outputs(basename, opts, arcs_written)?;
    Ok(CompressStats {
        nodes: opts.num_nodes,
        arcs: arcs_written,
        bits,
    })
}

// ---------------------------------------------------------------------------
// The convenience entry point
// ---------------------------------------------------------------------------

/// Compresses a [`SortedArcs`] set into `<basename>.{graph,offsets,properties}`.
///
/// With [`CompressOpts::parallel`] this delegates to [`compress_unsorted_iter`],
/// which re-sorts what is already sorted — but buys 112-way compression.
pub fn compress_sorted(
    arcs: &SortedArcs,
    basename: &Path,
    opts: &CompressOpts,
) -> Result<CompressStats> {
    sanitize_basename(basename)?;
    if arcs.num_arcs() == 0 && !opts.allow_empty {
        return Err(Error::EmptyEdgeList);
    }
    if arcs.max_node_id() >= opts.num_nodes {
        // Caught here rather than mid-stream, so the operator learns it before
        // a multi-hour compression starts rather than after it. The exact
        // offending arc is reported by the streaming validator on the
        // sequential path; all we know up front is the maximum endpoint.
        return Err(Error::other(format!(
            "the arc set references node {} but the graph was asked for only {} nodes; \
             pass --num-nodes from-arcs (which would give {}) or a larger explicit value",
            arcs.max_node_id(),
            opts.num_nodes,
            arcs.max_node_id() + 1
        )));
    }

    let (iter, slot) = arcs.iter()?;
    let stats = if opts.parallel {
        compress_unsorted_iter(iter, basename, opts)?
    } else {
        compress_sorted_iter(iter, arcs.num_arcs(), basename, opts)?
    };
    // The merge iterator reports I/O failures out of band; an unchecked slot
    // would turn a truncated run into a silently truncated graph.
    if let Err(e) = take_iter_error(&slot) {
        remove_outputs(basename);
        return Err(e);
    }
    Ok(stats)
}

// ---------------------------------------------------------------------------
// Reading a graph back
// ---------------------------------------------------------------------------

/// Reads one `key=value` line out of `<basename>.properties`.
///
/// A trivial line scan: the file is a flat ISO-8859-1 `key=value` listing with
/// `#` and `!` comment lines, and every key read here is pure ASCII, so this
/// needs no properties-file crate — and no escape handling, since none of the
/// values this crate reads can contain one.
fn read_property(basename: &Path, key: &str) -> Result<Option<String>> {
    let path = basename.with_extension(PROPERTIES_EXTENSION);
    let bytes = std::fs::read(&path).map_err(|e| Error::io(&path, e))?;
    let text = String::from_utf8_lossy(&bytes);
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with('#') || line.starts_with('!') {
            continue;
        }
        if let Some((k, v)) = line.split_once('=') {
            if k.trim() == key {
                return Ok(Some(v.trim().to_string()));
            }
        }
    }
    Ok(None)
}

/// Returns the `length=` property: the number of meaningful bits in
/// `<basename>.graph`.
///
/// The golden test uses it to know how many bytes of the bitstream to compare;
/// everything past `ceil(length / 8)` is `BufBitWriter`'s zero padding to a
/// 64-bit word.
pub fn graph_length_bits(basename: &Path) -> Result<u64> {
    match read_property(basename, "length")? {
        Some(v) => v.parse::<u64>().map_err(|_| {
            Error::other(format!(
                "{}: 'length' is not an integer ({v:?})",
                basename.with_extension(PROPERTIES_EXTENSION).display()
            ))
        }),
        None => Err(Error::other(format!(
            "{}: no 'length' property",
            basename.with_extension(PROPERTIES_EXTENSION).display()
        ))),
    }
}

/// Loads the graph back and checks it against what was meant to be written.
///
/// Also asserts that `<basename>.properties` exists. It is written **last**,
/// after the whole bitstream, so a missing `.properties` means an incomplete
/// run that must be redone rather than trusted.
pub fn verify_graph(
    basename: &Path,
    expected_nodes: usize,
    expected_arcs: u64,
    level: VerifyLevel,
) -> Result<()> {
    if level == VerifyLevel::None {
        debug!(
            "verification of {} skipped (--verify none)",
            basename.display()
        );
        return Ok(());
    }
    let props = basename.with_extension(PROPERTIES_EXTENSION);
    if !props.is_file() {
        return Err(Error::other(format!(
            "{} is missing; it is written last, so the compression did not finish",
            props.display()
        )));
    }
    let graph = BvGraphSeq::with_basename(basename)
        .endianness::<BE>()
        .load()
        .map_err(|e| Error::other(format!("could not load {}: {e:#}", basename.display())))?;

    if graph.num_nodes() != expected_nodes {
        return Err(Error::other(format!(
            "{}: the graph has {} nodes, expected {}",
            basename.display(),
            graph.num_nodes(),
            expected_nodes
        )));
    }
    if let Some(declared) = graph.get_num_arcs() {
        if declared != expected_arcs {
            return Err(Error::other(format!(
                "{}: the graph declares {} arcs, expected {}",
                basename.display(),
                declared,
                expected_arcs
            )));
        }
    }
    if level == VerifyLevel::Full {
        // A full second decompression. Worth it only when the declared count
        // is not trusted; see [`VerifyLevel`].
        let counted = graph.iter().into_pairs().count() as u64;
        if counted != expected_arcs {
            return Err(Error::other(format!(
                "{}: the graph contains {} arcs, expected {}",
                basename.display(),
                counted,
                expected_arcs
            )));
        }
    }
    debug!(
        "verified {} ({level:?}): {} nodes, {} arcs",
        basename.display(),
        expected_nodes,
        expected_arcs
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir() -> tempfile::TempDir {
        tempfile::Builder::new()
            .prefix("utxo2webgraph-compress-test-")
            .tempdir()
            .expect("temp dir")
    }

    fn opts(dir: &tempfile::TempDir, num_nodes: usize) -> CompressOpts {
        CompressOpts {
            num_nodes,
            tmp_dir: dir.path().join("tmp"),
            ..CompressOpts::default()
        }
    }

    fn read_pairs(basename: &Path) -> (usize, Vec<(usize, usize)>) {
        let g = BvGraphSeq::with_basename(basename)
            .endianness::<BE>()
            .load()
            .expect("load");
        let n = g.num_nodes();
        (n, g.iter().into_pairs().collect())
    }

    /// The golden vector, pinned byte for byte against the reference BVGraph
    /// 3.6.10 bitstream. Every byte below was produced by the implementation
    /// this crate reproduces; if the compression parameters ever drift, this
    /// test is what notices.
    #[test]
    fn golden_vector_matches_the_reference_bit_for_bit() {
        let dir = tmpdir();
        let basename = dir.path().join("golden");
        let arcs = vec![(0, 1), (0, 2), (1, 2), (2, 0), (3, 4)];

        let stats =
            compress_sorted_iter(arcs.into_iter(), 5, &basename, &opts(&dir, 5)).expect("compress");
        assert_eq!(stats.bits, 40, "the reference bitstream is 40 bits long");
        assert_eq!(stats.nodes, 5);
        assert_eq!(stats.arcs, 5);
        assert_eq!(graph_length_bits(&basename).expect("length"), 40);

        let graph = std::fs::read(basename.with_extension("graph")).expect("graph");
        let offsets = std::fs::read(basename.with_extension("offsets")).expect("offsets");
        assert_eq!(&graph[..5], &[0x7d, 0xc5, 0xda, 0xf1, 0x77]);
        assert_eq!(&offsets[..4], &[0x8d, 0x14, 0x28, 0x52]);
        // Everything past ceil(40 / 8) is BufBitWriter's zero word-padding.
        assert!(
            graph[5..].iter().all(|&b| b == 0),
            "trailing .graph bytes must be zero padding"
        );
        assert!(
            offsets[4..].iter().all(|&b| b == 0),
            "trailing .offsets bytes must be zero padding"
        );

        let props = std::fs::read_to_string(basename.with_extension("properties")).expect("props");
        for expected in [
            "graphclass=it.unimi.dsi.webgraph.BVGraph",
            "version=0",
            "nodes=5",
            "arcs=5",
            "minintervallength=4",
            "maxrefcount=3",
            "windowsize=7",
            "zetak=3",
            "compressionflags=",
        ] {
            assert!(
                props.lines().any(|l| l.trim() == expected),
                "missing {expected:?} in\n{props}"
            );
        }

        verify_graph(&basename, 5, 5, VerifyLevel::Full).expect("verify");
    }

    #[test]
    fn isolated_and_trailing_nodes_survive() {
        let dir = tmpdir();
        let basename = dir.path().join("isolated");
        let arcs = vec![(0, 1), (1, 0)];
        let stats = compress_sorted_iter(arcs.clone().into_iter(), 2, &basename, &opts(&dir, 5))
            .expect("compress");
        assert_eq!(stats.nodes, 5);
        assert_eq!(stats.arcs, 2);

        let (n, pairs) = read_pairs(&basename);
        assert_eq!(n, 5, "nodes 2, 3 and 4 are real nodes with outdegree 0");
        assert_eq!(pairs, arcs);
        verify_graph(&basename, 5, 2, VerifyLevel::Full).expect("verify");
    }

    #[test]
    fn self_loops_survive() {
        let dir = tmpdir();
        let basename = dir.path().join("loops");
        let arcs = vec![(0, 0), (1, 1)];
        compress_sorted_iter(arcs.clone().into_iter(), 2, &basename, &opts(&dir, 2))
            .expect("compress");
        let (n, pairs) = read_pairs(&basename);
        assert_eq!(n, 2);
        assert_eq!(pairs, arcs);
    }

    #[test]
    fn basenames_with_a_dot_are_rejected() {
        assert!(sanitize_basename(Path::new("graph/pg_1.5")).is_err());
        assert!(sanitize_basename(Path::new("graph/pg_1")).is_ok());
        // A dot in a *directory* is harmless: `with_extension` only rewrites
        // the file name.
        assert!(sanitize_basename(Path::new("a.b/pg_1")).is_ok());
    }

    /// `prepare` is where a node-count ceiling used to live: it rejected
    /// `num_nodes > i32::MAX` and so refused the 2 181 021 971-node graph an
    /// `N = 28` run reaches, after the entire sort had already been paid for.
    /// Nothing downstream is 32-bit, nothing here allocates per node, and this
    /// test exists to keep the ceiling from coming back.
    #[test]
    fn there_is_no_node_count_ceiling() {
        let dir = tmpdir();
        let basename = dir.path().join("huge");
        for n in [
            2_181_021_971usize,
            i32::MAX as usize + 1,
            u32::MAX as usize + 1,
            usize::MAX,
        ] {
            prepare(&basename, &opts(&dir, n))
                .unwrap_or_else(|e| panic!("{n} nodes must be accepted, got {e}"));
        }
    }

    #[test]
    fn unsorted_input_is_an_error_not_a_panic() {
        let dir = tmpdir();
        let basename = dir.path().join("unsorted");
        let err = compress_sorted_iter(
            vec![(1, 0), (0, 1)].into_iter(),
            2,
            &basename,
            &opts(&dir, 2),
        )
        .expect_err("must reject");
        match err {
            Error::UnsortedArcs {
                p_src,
                p_dst,
                c_src,
                c_dst,
            } => {
                assert_eq!((p_src, p_dst), (1, 0));
                assert_eq!((c_src, c_dst), (0, 1));
            }
            other => panic!("expected UnsortedArcs, got {other:?}"),
        }
        assert!(
            !basename.with_extension("graph").exists(),
            "a failed compression must leave nothing behind"
        );
    }

    #[test]
    fn out_of_bounds_destination_is_rejected() {
        let dir = tmpdir();
        let basename = dir.path().join("oob");
        let err = compress_sorted_iter(vec![(0, 5)].into_iter(), 1, &basename, &opts(&dir, 2))
            .expect_err("must reject");
        match err {
            Error::ArcOutOfBounds {
                src,
                dst,
                num_nodes,
            } => {
                assert_eq!((src, dst, num_nodes), (0, 5, 2));
            }
            other => panic!("expected ArcOutOfBounds, got {other:?}"),
        }
    }

    #[test]
    fn empty_input_is_rejected_unless_allowed() {
        let dir = tmpdir();
        let basename = dir.path().join("empty");
        let err = compress_sorted_iter(std::iter::empty(), 0, &basename, &opts(&dir, 4))
            .expect_err("must reject");
        assert!(matches!(err, Error::EmptyEdgeList));
        assert!(!basename.with_extension("graph").exists());
    }

    /// A fixed linear congruential generator: reproducible without `rand`.
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

    fn fixture(num_nodes: usize, arcs: usize) -> Vec<(usize, usize)> {
        let mut lcg = Lcg(0xC0FFEE);
        let mut v: Vec<(usize, usize)> = (0..arcs)
            .map(|_| {
                (
                    (lcg.next() as usize) % num_nodes,
                    (lcg.next() as usize) % num_nodes,
                )
            })
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    #[test]
    fn parallel_and_sequential_agree() {
        let dir = tmpdir();
        let arcs = fixture(500, 2000);

        let seq = dir.path().join("seq");
        let par = dir.path().join("par");

        compress_sorted_iter(
            arcs.clone().into_iter(),
            arcs.len() as u64,
            &seq,
            &opts(&dir, 500),
        )
        .expect("sequential");

        let par_opts = CompressOpts {
            parallel: true,
            threads: 4,
            ..opts(&dir, 500)
        };
        compress_unsorted_iter(arcs.clone().into_iter(), &par, &par_opts).expect("parallel");

        let g_seq = BvGraphSeq::with_basename(&seq)
            .endianness::<BE>()
            .load()
            .expect("load seq");
        let g_par = BvGraphSeq::with_basename(&par)
            .endianness::<BE>()
            .load()
            .expect("load par");
        webgraph::traits::graph::eq(&g_seq, &g_par).expect("the two graphs must be equal");

        verify_graph(&seq, 500, arcs.len() as u64, VerifyLevel::Full).expect("verify seq");
        verify_graph(&par, 500, arcs.len() as u64, VerifyLevel::Full).expect("verify par");
    }
}
