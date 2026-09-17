//! BVGraph compression. Replaces `builder.GraphBuilder` in
//! `jar/WebgraphBuilder.jar` by **Matteo Loporchio**, which ran
//! `it.unimi.dsi.webgraph.BVGraph -g ArcListASCIIGraph <in> <out>` with argv
//! exactly `["-g", "ArcListASCIIGraph", <elFile>, <outputPrefix>]`.
//!
//! This is the only module in the crate that touches `webgraph`,
//! `dsi-bitstream` and `dsi-progress-logger`.
//!
//! # What the Java stage did, and what this does instead
//!
//! `GraphBuilder.main` built the string `"-g ArcListASCIIGraph " + inputFile +
//! " " + outputPrefix` and split it on spaces. (The constant-pool entry really
//! does end in a space; `javap` trims trailing whitespace from its own output,
//! which makes the literal look glued to the path.) `BVGraph.main` then loaded
//! the text edge list through `ArcListASCIIGraph.loadMapped`, which despite its
//! name memory-maps nothing: it reads the file sequentially into an
//! `ArrayListMutableGraph`, one `IntArrayList` per node, entirely on the JVM
//! heap. That materialisation is what `-Xmx200g` was paying for.
//!
//! Here the sorted arcs are streamed straight into `webgraph-rs`, so the graph
//! is never materialised and the 210 GB text edge list is optional.
//!
//! # Compression parameters
//!
//! [`CompFlags::default()`] is
//! `{ outdegrees: Gamma, references: Unary, blocks: Gamma, intervals: Gamma,
//! residuals: Zeta(3), min_interval_length: 4, compression_window: 7,
//! max_ref_count: 3 }` — **identical to the Java `BVGraph` defaults**
//! (`windowsize=7`, `maxrefcount=3`, `minintervallength=4`, `zetak=3`). Passing
//! it explicitly is redundant, and is done here as documentation.
//!
//! `BvCompConf::bvgraphz()` and `BvCompConf::chunk_size()` are **forbidden**:
//! they switch to the Zuckerli compressor `BvCompZ`, whose reference selection
//! does not match Java's.
//!
//! # Node count
//!
//! Java inferred `numNodes = max(id over all sources AND all targets) + 1`,
//! filling the gaps with outdegree-0 nodes and renumbering nothing. The node
//! map's `next_id`, by contrast, counts every UTXO slot, including the unspent
//! outputs that never appear in any arc. The two genuinely differ, `next_id` is
//! the larger, and [`CompressOpts::num_nodes`] is the caller's explicit
//! decision between them.
//!
//! # `.properties` is not byte-reproducible
//!
//! Java's `Properties.store` writes a `#<current date>` comment line, so the
//! file can never match byte for byte. `webgraph-rs` additionally writes a
//! different key set, in insertion order, adding `endianness=` and `length=`
//! and omitting all the cosmetic `bitsfor*`/`*expstats`/`*avggap` keys — all of
//! which `BVGraph.loadInternal` ignores. Only `.graph` and `.offsets` are valid
//! byte-comparison targets, and even there the Rust files are 4-8 bytes longer
//! because `BufBitWriter` pads to a 64-bit word while Java pads to a byte.
//! Compare only the first `ceil(length / 8)` bytes; the padding is all zeros.

use std::panic::AssertUnwindSafe;
use std::path::Path;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use dsi_bitstream::prelude::BE;
use dsi_progress_logger::prelude::*;
use log::{debug, info, warn};
use webgraph::graphs::arc_list_graph;
use webgraph::prelude::*;

use crate::arcs::SortedArcs;
use crate::{new_error_slot, take_iter_error, ErrorSlot, PgError, PgResult};

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
    /// [`PgError::EmptyEdgeList`].
    pub allow_empty: bool,
    /// How thoroughly [`verify_graph`] re-reads the graph it just wrote.
    pub verify: VerifyLevel,
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
/// Java built its paths by string concatenation and did not have this problem,
/// so a prefix that was harmless before must be rejected now rather than
/// quietly redirected.
pub fn sanitize_basename(p: &Path) -> PgResult<()> {
    let Some(name) = p.file_name() else {
        return Err(PgError::BadBasename(p.to_path_buf()));
    };
    if name.to_string_lossy().contains('.') {
        return Err(PgError::BadBasename(p.to_path_buf()));
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

/// Common preflight for every entry point.
fn prepare(basename: &Path, opts: &CompressOpts) -> PgResult<()> {
    sanitize_basename(basename)?;
    if opts.num_nodes > i32::MAX as usize {
        // `it.unimi.dsi.webgraph` (as opposed to `it.unimi.dsi.big.webgraph`)
        // is 32-bit throughout: `BVGraph.loadInternal` rejects `nodes >
        // Integer.MAX_VALUE`. At N = 28 the node-map estimate is ~2.21e9, so
        // this *will* fire and the message has to say plainly why.
        return Err(PgError::TooManyNodes(opts.num_nodes as u64));
    }
    if let Some(parent) = basename.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| PgError::io(parent, e))?;
        }
    }
    std::fs::create_dir_all(&opts.tmp_dir).map_err(|e| PgError::io(&opts.tmp_dir, e))?;
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
/// * `dst` is not bounds-checked at all, so `dst >= num_nodes` produces a graph
///   the Java reader will index out of bounds on.
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
    prev: Option<(usize, usize)>,
    count: Arc<AtomicU64>,
    slot: ErrorSlot,
    done: bool,
}

impl<I> Validate<I> {
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

impl<I: Iterator<Item = (usize, usize)>> Iterator for Validate<I> {
    type Item = (usize, usize);

    #[inline]
    fn next(&mut self) -> Option<(usize, usize)> {
        if self.done {
            return None;
        }
        let (src, dst) = self.inner.next()?;
        if src >= self.num_nodes || dst >= self.num_nodes {
            return self.fail(PgError::ArcOutOfBounds {
                src: src as u64,
                dst: dst as u64,
                num_nodes: self.num_nodes,
            });
        }
        if let Some((p_src, p_dst)) = self.prev {
            if (src, dst) <= (p_src, p_dst) {
                return self.fail(PgError::UnsortedArcs {
                    p_src: p_src as u64,
                    p_dst: p_dst as u64,
                    c_src: src as u64,
                    c_dst: dst as u64,
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
// Sequential compression: the deterministic, Java-identical path
// ---------------------------------------------------------------------------

/// Compresses arcs that are already sorted by `(src, dst)` and deduplicated.
///
/// This is the **default** path, and the one verified to reproduce the Java
/// bitstream exactly: on the golden vector `(0,1) (0,2) (1,2) (2,0) (3,4)` over
/// five nodes it returns 40 bits and writes `7d c5 da f1 77` / `8d 14 28 52`,
/// matching `it.unimi.dsi.webgraph.BVGraph` byte for byte.
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
///   an assertion inside `webgraph` becomes [`PgError::CompressorPanic`]
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
) -> PgResult<CompressStats>
where
    I: Iterator<Item = (usize, usize)>,
{
    prepare(basename, opts)?;
    if num_arcs == 0 && !opts.allow_empty {
        // Fail before opening anything. Java's `ArcListASCIIGraph` ran
        // `fillNextLine()` in its NodeIterator's instance initialiser, so a
        // 0-byte edge list threw `IllegalArgumentException: Expected integer,
        // found Token[EOF], line 1` two stages downstream of the real problem —
        // which is exactly what `logs/webgraph_builder.err` records.
        return Err(PgError::EmptyEdgeList);
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

    let mut pl = progress_logger![display_memory = true, item_name = "node"];
    let conf = BvCompConf::new(basename)
        // Redundant — this is what `BvCompConf::new` already installs — but it
        // is the line that documents "same parameters as the Java BVGraph".
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
            return Err(PgError::CompressorPanic(panic_message(payload)));
        }
        Ok(Err(e)) => {
            remove_outputs(basename);
            return Err(PgError::other(format!(
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
fn finish_outputs(basename: &Path, opts: &CompressOpts, arcs: u64) -> PgResult<()> {
    let props = basename.with_extension(PROPERTIES_EXTENSION);
    if !props.is_file() {
        remove_outputs(basename);
        return Err(PgError::other(format!(
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
        .map_err(|e| PgError::other(format!("could not build the Elias-Fano offsets: {e:#}")))?;
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
/// The Java compressor had exactly the same property — its thread count changed
/// its output bytes too — which is why the sequential path is the default here.
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
) -> PgResult<CompressStats>
where
    I: Iterator<Item = (usize, usize)> + Send,
{
    prepare(basename, opts)?;

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(opts.threads)
        .build()
        .map_err(|e| PgError::other(format!("could not build the compression thread pool: {e}")))?;

    info!(
        "compressing {} nodes into {} (parallel; output is NOT byte-identical to the sequential path)",
        opts.num_nodes,
        basename.display()
    );

    let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| {
        pool.install(move || -> PgResult<u64> {
            let mut pls = progress_logger![display_memory = true, item_name = "arc"];
            let sorted = ParSortedGraph::config()
                .dedup()
                .memory_usage(MemoryUsage::MemorySize(opts.memory_bytes as usize))
                .progress_logger(&mut pls)
                .sort_pairs(opts.num_nodes, arcs)
                .map_err(|e| PgError::other(format!("could not sort the arcs: {e:#}")))?;

            let mut plc = progress_logger![display_memory = true, item_name = "node"];
            let conf = BvCompConf::new(basename)
                .comp_flags(CompFlags::default())
                .tmp_dir(&opts.tmp_dir);
            let mut conf = conf.progress_logger(&mut plc);
            conf.par_comp::<BE, _>(sorted)
                .map_err(|e| PgError::other(format!("could not compress the graph: {e:#}")))
        })
    }));

    let bits = match outcome {
        Err(payload) => {
            remove_outputs(basename);
            return Err(PgError::CompressorPanic(panic_message(payload)));
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
        return Err(PgError::EmptyEdgeList);
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
) -> PgResult<CompressStats> {
    sanitize_basename(basename)?;
    if arcs.num_arcs() == 0 && !opts.allow_empty {
        return Err(PgError::EmptyEdgeList);
    }
    if arcs.max_node_id() >= opts.num_nodes as u64 {
        // Caught here rather than mid-stream, so the operator learns it before
        // a multi-hour compression starts rather than after it. The exact
        // offending arc is reported by the streaming validator on the
        // sequential path; all we know up front is the maximum endpoint.
        return Err(PgError::other(format!(
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
/// A trivial line scan: the file is ISO-8859-1 `java.util.Properties` text and
/// every key we care about is ASCII, so this needs no `java-properties`
/// dependency.
fn read_property(basename: &Path, key: &str) -> PgResult<Option<String>> {
    let path = basename.with_extension(PROPERTIES_EXTENSION);
    let bytes = std::fs::read(&path).map_err(|e| PgError::io(&path, e))?;
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
pub fn graph_length_bits(basename: &Path) -> PgResult<u64> {
    match read_property(basename, "length")? {
        Some(v) => v.parse::<u64>().map_err(|_| {
            PgError::other(format!(
                "{}: 'length' is not an integer ({v:?})",
                basename.with_extension(PROPERTIES_EXTENSION).display()
            ))
        }),
        None => Err(PgError::other(format!(
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
) -> PgResult<()> {
    if level == VerifyLevel::None {
        debug!(
            "verification of {} skipped (--verify none)",
            basename.display()
        );
        return Ok(());
    }
    let props = basename.with_extension(PROPERTIES_EXTENSION);
    if !props.is_file() {
        return Err(PgError::other(format!(
            "{} is missing; it is written last, so the compression did not finish",
            props.display()
        )));
    }
    let graph = BvGraphSeq::with_basename(basename)
        .endianness::<BE>()
        .load()
        .map_err(|e| PgError::other(format!("could not load {}: {e:#}", basename.display())))?;

    if graph.num_nodes() != expected_nodes {
        return Err(PgError::other(format!(
            "{}: the graph has {} nodes, expected {}",
            basename.display(),
            graph.num_nodes(),
            expected_nodes
        )));
    }
    if let Some(declared) = graph.get_num_arcs() {
        if declared != expected_arcs {
            return Err(PgError::other(format!(
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
            return Err(PgError::other(format!(
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
            .prefix("pgraph-compress-test-")
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

    /// The golden vector, reproduced byte for byte from
    /// `it.unimi.dsi.webgraph.BVGraph` 3.6.10 (the version inside
    /// `jar/WebgraphBuilder.jar`).
    #[test]
    fn golden_vector_matches_java_bit_for_bit() {
        let dir = tmpdir();
        let basename = dir.path().join("golden");
        let arcs = vec![(0, 1), (0, 2), (1, 2), (2, 0), (3, 4)];

        let stats =
            compress_sorted_iter(arcs.into_iter(), 5, &basename, &opts(&dir, 5)).expect("compress");
        assert_eq!(stats.bits, 40, "the Java bitstream is 40 bits long");
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
            PgError::UnsortedArcs {
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
            PgError::ArcOutOfBounds {
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
        assert!(matches!(err, PgError::EmptyEdgeList));
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
