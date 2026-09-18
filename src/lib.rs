//! `utxo2webgraph` — turns a Bitcoin transaction list into a compressed
//! WebGraph.
//!
//! The pipeline has four stages, each usable on its own:
//!
//! * [`split`] — cut a whole-chain transaction list into six-month chunks;
//! * [`edge_list`] — read chunks and emit a node map plus a raw arc stream;
//! * [`arcs`] — sort and deduplicate that stream, spilling to disk when it
//!   outgrows the memory budget;
//! * [`compress`] — feed the sorted arcs to `webgraph-rs` and write a BVGraph.
//!
//! # Attribution
//!
//! The pipeline and the graph-construction algorithm are the work of
//! **Matteo Loporchio**. This crate is an independent reimplementation of
//! that design; the algorithm is his. Every module that reimplements a piece
//! of it repeats the attribution in its own module doc.
//!
//! # What a node is
//!
//! A node is one transaction *output* — one UTXO — identified by the pair
//! `(txId, offset)` and assigned a dense [`NodeId`] by [`nodemap`]. An arc
//! runs from a spent output to each output of the transaction that spent it.
//!
//! Node ids are [`usize`], the same type `webgraph-rs` uses, and arcs are
//! packed into a `u128` ([`pack_arc`]). Neither has a 32-bit ceiling: a graph
//! with more than `u32::MAX` nodes is built and loaded like any other.
//!
//! # Guarantees that are easy to get wrong
//!
//! 1. **An empty output section emits zero edges.** A transaction that
//!    declares no outputs contributes no arcs at all, rather than a phantom
//!    edge into node `0`.
//! 2. **Blank and malformed lines are reported with a line number**, or
//!    skipped and tallied under [`Mode::Lenient`]. A single bad line in a
//!    132 GB corpus never costs a multi-hour run its diagnostics.
//! 3. **A dangling source reference produces a typed error** naming the input
//!    line, the current `txId`, the input index and the missing
//!    `(prevTxId, prevOffset)` — see [`Error::DanglingSource`]. The input must
//!    be a prefix of the transaction list starting at genesis, unless
//!    [`OnMissingSource::Skip`] or [`OnMissingSource::Create`] is chosen.
//! 4. **Every write error is propagated.** A disk-full during a multi-hour,
//!    210 GB write fails the run; it never truncates the edge list and reports
//!    success.
//! 5. **The node map is written in ascending `(txId, offset)` order**, so two
//!    runs over the same input produce byte-identical files. Compare node maps
//!    from other tools with `sort -t$'\t' -k1,1n -k2,2n | md5sum`.
//! 6. **A non-UTF-8 byte is a line-numbered error in strict mode**, naming the
//!    line and the byte offset within it. `--lenient` substitutes U+FFFD and
//!    tallies `non_utf8_lines` instead.
//!
//! # `--num-nodes`
//!
//! `utxo2webgraph build`, `build-pg` and `compress` all default to
//! `--num-nodes from-arcs`, i.e. `max(endpoint) + 1` (18 545 on `chunk_01`).
//! `--num-nodes node-map` is the opt-in richer graph in which every UTXO is a
//! node, including recent unspent outputs that appear in no arc (18 620 on
//! `chunk_01`, i.e. 75 extra isolated trailing nodes). Both candidates are
//! logged whenever they differ.

#![warn(missing_docs)]

pub mod arcs;
pub mod cli;
pub mod compress;
pub mod edge_list;
pub mod nodemap;
pub mod record;
pub mod split;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// A dense node identifier.
///
/// `usize` is the node type `webgraph-rs` itself uses at every layer of the
/// compression path — `SequentialGraph: SequentialLabeling<Label = usize>`,
/// `ArcListGraph::new(num_nodes: usize, _: impl Iterator<Item = (usize, usize)>)`,
/// `comp_lender(_, Some(num_nodes: usize))` — so a node id crosses the library
/// boundary without a cast, and there is no width at which this crate and the
/// compressor disagree about what a node is.
///
/// Nothing in that path, and nothing in this crate, bounds a node count: a
/// graph with more than `i32::MAX` or `u32::MAX` nodes is written and read
/// back by `webgraph-rs` like any other. Arc storage is likewise uncapped; see
/// [`pack_arc`].
///
/// The one caveat is a consumer, not a limit here: the `graphclass` recorded
/// in the `.properties` file has a 32-bit reference implementation that
/// rejects `nodes > i32::MAX` at load. Such a graph is fine for `webgraph-rs`
/// and unreadable by that one reader. See [`compress::prepare`].
pub type NodeId = usize;

/// This crate stores a [`NodeId`] pair in a `u128` and assumes a node id fits
/// in 64 bits. Both hold on every 64-bit target; a 16- or 32-bit target would
/// silently narrow [`unpack_arc`], so refuse to build there.
const _: () = assert!(
    usize::BITS >= 64,
    "utxo2webgraph requires a 64-bit target: NodeId is usize and arc packing assumes 64-bit ids"
);

/// Result alias used by every library function in this crate.
pub type Result<T> = std::result::Result<T, Error>;

/// Packs a `(txId, offset)` pair into the `u64` key the node map is addressed
/// by: `(txId as u32) << 32 | (offset as u32)`.
///
/// # Not a node id
///
/// This is a **lookup key**, not a [`NodeId`], and that is why it stays `u64`
/// while node ids are `usize`. It identifies an output in the *input's* own
/// coordinates; [`nodemap`] maps it to the dense node id the graph uses.
///
/// # Bijectivity
///
/// The round trip with [`unpack`] is **exact for all 2^64 `(i32, i32)` pairs,
/// negatives included**: the two 32-bit halves never overlap and both casts
/// are pure reinterpretation, so no information is lost. Two distinct
/// `(txId, offset)` pairs can therefore never collide on a key, which is what
/// keeps the node map free of aliasing — including for the BIP-30 duplicate
/// transaction ids, where the same `txId` legitimately appears twice.
///
/// The key is `u64` rather than `i64` deliberately: only equality and hashing
/// are ever performed on it, and an unsigned type makes it impossible to write
/// an arithmetic right shift by accident and sign-extend a negative `txId`
/// across the offset half.
#[inline]
pub fn pack(tx_id: i32, offset: i32) -> u64 {
    ((tx_id as u32 as u64) << 32) | (offset as u32 as u64)
}

/// Inverse of [`pack`].
///
/// Negative ids survive the round trip untouched: `pack(-5, 0)` is
/// `0xFFFFFFFB_00000000` and unpacks back to `(-5, 0)`, so a negative `txId`
/// reaches the node map's first column verbatim rather than as `4294967291`.
#[inline]
pub fn unpack(key: u64) -> (i32, i32) {
    (((key >> 32) as u32) as i32, (key as u32) as i32)
}

/// Packs an arc into one `u128`, `(src << 64) | dst`.
///
/// Sorting the packed values ascending is bit-for-bit equivalent to
/// `sort -t$'\t' -k1,1n -k2,2n`, and [`Vec::dedup`] on the sorted packed arcs
/// is exactly `uniq`.
///
/// # Why one wide integer and not two words
///
/// A `u128` holds a full [`NodeId`] pair with room to spare, so the packing is
/// **total**: it cannot fail, it has no ceiling to check, and no run can die
/// part-way through because the corpus outgrew the arc representation. That
/// matters more than the eight bytes it costs. The earlier scheme packed the
/// pair into a `u64` as `(src << 32) | dst`, which capped a node id at `2^32`
/// and returned `None` above it — a cap the corpus was already within a factor
/// of two of reaching, and one that could only be discovered hours into a run.
///
/// The width is also why sorting stays correct: the high 64 bits are the
/// source and the low 64 the destination, so a plain unsigned ascending sort
/// of the packed values orders arcs by source and then by destination, with no
/// comparator and no risk of the two halves interfering.
#[inline]
pub fn pack_arc(src: NodeId, dst: NodeId) -> u128 {
    ((src as u128) << 64) | dst as u128
}

/// Inverse of [`pack_arc`]. Round-trips exactly for every [`NodeId`] pair.
#[inline]
pub fn unpack_arc(a: u128) -> (NodeId, NodeId) {
    ((a >> 64) as NodeId, a as u64 as NodeId)
}

/// Bytes one packed arc occupies, in memory and on disk.
///
/// `size_of::<u128>()`, written out rather than derived from [`NodeId`], because
/// this is an **on-disk record width**. The binary run and sorted-arc formats
/// carry no header and no magic number, so every reader dispatches on this
/// number alone; deriving it from a type whose width could change would
/// silently reinterpret existing files.
pub const ARC_RECORD_SIZE: usize = 16;

/// How often the edge-list builder reports progress, in transactions.
pub const DEFAULT_PROGRESS_EVERY: u64 = 10_000_000;

/// How malformed input is treated.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum Mode {
    /// Any malformed record is a fatal, line-numbered error.
    ///
    /// The message names the line and the offending value, so a bad record in
    /// a multi-hour run is diagnosable without re-reading the corpus.
    #[default]
    Strict,
    /// Malformed records are skipped and tallied.
    ///
    /// Blank lines are skipped in *both* modes.
    Lenient,
}

/// What to do when an input references an output that was never registered.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum OnMissingSource {
    /// Abort with [`Error::DanglingSource`], naming the line and the pair.
    #[default]
    Fail,
    /// Emit no edges for that input; tally it.
    ///
    /// Makes single-chunk and sharded processing possible: a chunk that does
    /// not start at genesis necessarily refers back to outputs it lacks.
    Skip,
    /// Mint a node for it on the fly. CHANGES THE ID SPACE.
    ///
    /// Never the default: minting a node shifts every subsequent id, so the
    /// node map and edge list stop being comparable with a run that did not.
    Create,
}

/// Statistics output style.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum StatsStyle {
    /// Two lines on stdout: `Nodes:` and `Edges:`.
    ///
    /// `Nodes:` counts output *slots*, not distinct nodes — the two differ
    /// whenever a transaction declares an output that no input ever spends.
    /// The counter is reported this way for continuity with the historical
    /// pipeline logs in `logs/pg_el_builder.log`, so a naive diff against them
    /// still matches; [`StatsStyle::Extended`] prints the distinct-node count
    /// alongside it.
    #[default]
    Brief,
    /// The [`StatsStyle::Brief`] lines plus corrected and extra counters.
    ///
    /// Adds the distinct-node count that `Nodes:` is commonly mistaken for,
    /// plus the skipped-line, dangling-reference and duplicate-arc tallies.
    Extended,
}

/// Run-sorting backend.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum SortAlgo {
    /// `rdst` LSD radix sort.
    ///
    /// Replaces `sort --temporary-directory=./tmp -t$'\t' -k1,1n -k2,2n`,
    /// which at `N = 28` would push roughly 840 GB through the filesystem to
    /// re-parse integers this process already holds in registers. `rdst`
    /// implements `RadixKey` for `u128`, so this applies directly to a packed
    /// arc; it costs sixteen one-byte passes rather than eight.
    #[default]
    Radix,
    /// `rayon` pattern-defeating quicksort.
    ///
    /// The cross-check for [`SortAlgo::Radix`]: a comparison sort reaches the
    /// same order by an entirely different route, which is what makes it
    /// useful for confirming the radix sort on a new corpus.
    Pdq,
}

/// A sink for raw, unsorted arcs produced by the edge-list builder.
///
/// Implemented by `arcs::TsvArcSink`, `arcs::BinaryArcSink`,
/// `arcs::ArcSorter` and `arcs::NullArcSink`.
///
/// # Ordering contract
///
/// [`ArcSink::push`] is called in **emission order**: input file line order,
/// then input index ascending, then output offset ascending. Implementations
/// **must not** reorder, buffer out of order, or drop arcs — the raw TSV edge
/// list is pinned byte-for-byte by the golden tests (`el_01.tsv`, md5
/// `a3c31369f1a54bacfeb9b3cc2c953ebe`). Duplicates and self-loops are part of
/// that stream and must be preserved; deduplication happens later, in
/// `arcs::ArcSorter`.
pub trait ArcSink {
    /// Accepts one arc, in emission order.
    fn push(&mut self, src: NodeId, dst: NodeId) -> Result<()>;
    /// Flushes and durably commits. Must be called exactly once.
    ///
    /// Implementations that own a file must flush *and* `sync_all` here, and
    /// must return any error rather than swallow it. Swallowing one lets a
    /// disk-full silently truncate the output while the run reports success.
    fn finish(&mut self) -> Result<()>;
}

/// Error slot used by iterators that cannot return `Result` because
/// `webgraph` requires a plain `Iterator`. Callers MUST check it after the
/// iterator has been fully consumed, via [`take_iter_error`].
pub type ErrorSlot = Arc<Mutex<Option<Error>>>;

/// Creates a fresh, empty [`ErrorSlot`].
pub fn new_error_slot() -> ErrorSlot {
    Arc::new(Mutex::new(None))
}

/// Takes the error out of a slot, if any. Poisoned mutex => `Ok(())`.
pub fn take_iter_error(slot: &ErrorSlot) -> Result<()> {
    match slot.lock() {
        Ok(mut g) => match g.take() {
            Some(e) => Err(e),
            None => Ok(()),
        },
        Err(_) => Ok(()),
    }
}

/// `fsync(2)`s `file`, unless the destination is one for which durability is
/// meaningless.
///
/// `fsync` on a character device (`/dev/null`, `/dev/stdout`) or a pipe returns
/// `EINVAL`, and on some filesystems `ENOTSUP`/`EOPNOTSUPP`. Treating that as a
/// failure turned a completed run into `Error: I/O error on /dev/null: Invalid
/// argument (os error 22)` and exit 1, so `/dev/null`, fifos and process
/// substitution all became unusable as destinations.
///
/// Real durability errors (`ENOSPC`, `EIO`, …) are still propagated: that is
/// the whole point of syncing before reporting success.
pub fn sync_if_durable(file: &std::fs::File, path: &std::path::Path) -> Result<()> {
    // Cheap, exact answer first: only a regular file has to be durable.
    if let Ok(md) = file.metadata() {
        if !md.is_file() {
            return Ok(());
        }
    }
    match file.sync_all() {
        Ok(()) => Ok(()),
        // EINVAL (22) / ENOTSUP (95 on Linux) from a destination that cannot
        // be synced at all. Everything else is a genuine write failure.
        Err(e) if matches!(e.raw_os_error(), Some(22) | Some(95)) => {
            log::debug!(
                "{} does not support fsync ({e}); skipping the durability barrier",
                path.display()
            );
            Ok(())
        }
        Err(e) => Err(Error::io(path, e)),
    }
}

/// Every failure this library can report. Deliberately line-numbered, so a
/// failure names the input that caused it.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// An I/O error, with the path that caused it.
    ///
    /// The `Display` string deliberately does NOT interpolate `{source}`: the
    /// field is already `#[source]`, and `main.rs` prints the whole chain with
    /// `{e:#}`, which used to render the OS message twice
    /// ("I/O error on X: Permission denied: Permission denied").
    #[error("I/O error on {path}")]
    Io {
        /// The file or directory being operated on.
        path: PathBuf,
        /// The underlying operating-system error.
        #[source]
        source: std::io::Error,
    },

    /// An I/O error with no path context available.
    #[error("I/O error: {0}")]
    PlainIo(#[from] std::io::Error),

    /// A record did not have exactly three `':'`-separated sections.
    ///
    /// Both too few and too many sections are rejected: taking the first
    /// three of a longer line would silently reinterpret the record.
    #[error("line {line}: expected 3 ':'-separated sections, found {found}")]
    BadSectionCount {
        /// 1-based line number in the concatenated input.
        line: u64,
        /// The number of sections actually found.
        found: usize,
    },

    /// The info section had fewer than three `','`-separated fields, so
    /// `infos[2]` (the txId) does not exist.
    #[error("line {line}: info section has {found} ','-separated fields, expected at least 3")]
    BadInfoFields {
        /// 1-based line number in the concatenated input.
        line: u64,
        /// The number of fields actually found.
        found: usize,
    },

    /// An input descriptor had fewer than four `','`-separated fields, so
    /// `inputParts[2]` / `inputParts[3]` do not exist.
    #[error("line {line}: input #{index} has {found} ','-separated fields, expected at least 4")]
    BadInputFields {
        /// 1-based line number in the concatenated input.
        line: u64,
        /// Zero-based index of the input within its transaction.
        index: usize,
        /// The number of fields actually found.
        found: usize,
    },

    /// A transaction id or offset field is not an `i32`.
    ///
    /// `i32` is deliberate and is not a [`NodeId`]: these are the input
    /// format's own values, and widening them to `i64` would change the
    /// [`pack`] key for anything at or above 2^31 and rewrite the node map.
    #[error("line {line}, field {field}: cannot parse {value:?} as i32")]
    BadInteger {
        /// 1-based line number in the concatenated input.
        line: u64,
        /// Which field failed: `"txId"`, `"prevTxId"` or `"prevTxOffset"`.
        field: &'static str,
        /// The offending text, verbatim.
        value: String,
    },

    /// A negative txId or offset was seen in [`Mode::Strict`].
    ///
    /// A negative value round-trips through [`pack`] into a literal `-5` in
    /// the txId column of the node map, which no downstream consumer expects.
    /// [`Mode::Lenient`] accepts it anyway.
    #[error("line {line}, field {field}: negative value {value} rejected in strict mode")]
    NegativeId {
        /// 1-based line number in the concatenated input.
        line: u64,
        /// Which field was negative.
        field: &'static str,
        /// The offending value.
        value: i32,
    },

    /// An input referenced an output that was never registered.
    ///
    /// The input side of a transaction only ever *looks up* an output; it
    /// never registers one. The pipeline is therefore well-defined **only on a
    /// prefix of the transaction list starting at genesis** — which is what
    /// concatenating chunks `01..N` guarantees. Any partial range, any single
    /// mid-stream chunk, any sharded or resumed run hits this immediately.
    /// Measured incidence:
    ///
    /// | input | dangling refs / total input refs |
    /// |---|---|
    /// | `chunk_01` alone | 0 / 1033 |
    /// | `chunk_01 + chunk_02` | 0 / 2887 |
    /// | `chunk_02` alone | 134 / 1854 |
    /// | `chunk_05` alone | 18 145 / 1 168 956 |
    ///
    /// [`OnMissingSource::Fail`] aborts with exit 1, naming the line, the
    /// transaction, the input index and the missing pair.
    #[error(
        "line {line}, tx {tx_id}: input #{index} references unknown output ({prev_tx_id}, {prev_offset}); \
         the input must be a prefix of the transaction list starting at genesis, \
         or pass --on-missing-source skip|create"
    )]
    DanglingSource {
        /// 1-based line number in the concatenated input.
        line: u64,
        /// The transaction being processed when the reference failed.
        tx_id: i32,
        /// Zero-based index of the offending input within its transaction.
        index: usize,
        /// The referenced previous transaction id.
        prev_tx_id: i32,
        /// The referenced output offset within that transaction.
        prev_offset: i32,
    },

    /// A txId jumped *forward*, breaking the dense node map's "+1" invariant.
    ///
    /// Only a forward jump reaches this error. A txId *below* the cursor is a
    /// repeat — the BIP-30 duplicate-coinbase case — and the node map reuses
    /// the ids the first occurrence was given, in both modes; see
    /// [`crate::nodemap`].
    #[error(
        "line {line}: transaction id {found} skips ahead of {expected}, breaking the \
         node-map invariant that transaction ids are consecutive; re-run with --lenient \
         to bridge gaps of up to {gap} ids with zero-output placeholders, or repair the input",
        gap = crate::nodemap::MAX_TX_GAP
    )]
    NonDenseTxId {
        /// 1-based line number in the concatenated input.
        line: u64,
        /// The txId the dense map expected next.
        expected: i64,
        /// The txId actually found.
        found: i64,
    },

    /// A txId jumped forward by more than [`crate::nodemap::MAX_TX_GAP`], so
    /// even [`Mode::Lenient`] refuses to bridge it.
    ///
    /// Distinct from [`Error::NonDenseTxId`] for one reason: the operator is
    /// already running with `--lenient` when they see this, so telling them to
    /// re-run with `--lenient` would be advice they have taken. Bridging the
    /// gap would allocate `8 * gap` bytes of zero-output filler on the strength
    /// of a single suspicious line, which is how a two-line, 79-byte input once
    /// drove RSS to 16 GiB.
    #[error(
        "line {line}: transaction id {found} skips ahead of {expected} by {gap} ids, more \
         than the {ceiling} --lenient will bridge; filling it would allocate {bytes} bytes \
         of zero-output placeholders for one line. Repair the input, or split it so each \
         run starts at a transaction it actually contains",
        gap = found - expected,
        ceiling = crate::nodemap::MAX_TX_GAP,
        bytes = (found - expected) as u64 * 8
    )]
    TxIdGapTooLarge {
        /// 1-based line number in the concatenated input.
        line: u64,
        /// The txId the dense map expected next.
        expected: i64,
        /// The txId actually found.
        found: i64,
    },

    /// `--on-missing-source create` was asked to mint a node for an output
    /// whose transaction the run has **not read yet**.
    ///
    /// The dense node map lays a transaction's outputs out as one contiguous
    /// id run, addressed by a prefix sum over transaction ids
    /// ([`crate::nodemap::DenseNodeMap`]). A forced node, by contrast, lives in
    /// the side table, and the forward-walking cursor that mints dense ids
    /// never consults it — it has no reason to, because it only ever moves
    /// forwards over ids it has not yet written. So if a key were forced
    /// *ahead* of the cursor and the cursor later reached that transaction, the
    /// same `(txId, offset)` would receive a **second** id: two node-map rows
    /// for one output, out of ascending order, an inflated distinct-node count,
    /// and arcs that point at whichever of the two ids happened to be visible
    /// at the time. Refusing is the only answer that keeps the id space a
    /// bijection.
    ///
    /// Nothing legitimate is lost: a Bitcoin transaction can only spend an
    /// output that already exists, so a forward reference cannot occur in
    /// correctly ordered data (verified: zero forward references across
    /// `chunk_04` and `chunk_05`). The reachable, legitimate case — a chunk
    /// that starts above genesis and refers back to transactions it does not
    /// contain — is *behind* the cursor and is minted normally.
    #[error(
        "line {line}: --on-missing-source create cannot mint a node for output \
         ({tx_id}, {offset}), because transaction {tx_id} has not been read yet (the \
         node map has only reached transaction {reached}); minting it now would hand \
         that output a second id once the transaction is read. Use --on-missing-source \
         skip or fail, or feed the input in transaction order starting at genesis"
    )]
    ForwardForcedNode {
        /// 1-based line number in the concatenated input.
        line: u64,
        /// The referenced transaction id, at or ahead of the dense cursor.
        tx_id: i32,
        /// The referenced output offset within that transaction.
        offset: i32,
        /// The highest transaction id that already has a dense slot, or
        /// `min_tx - 1` when nothing has been registered at all.
        reached: i64,
    },

    /// A transaction line was not valid UTF-8.
    ///
    /// [`Mode::Lenient`] substitutes U+FFFD and parses the record from the
    /// lossy text. [`Mode::Strict`] reports this instead, because a stray byte
    /// in a 132 GB corpus is worth knowing about, and names the line and the
    /// byte offset within it rather than failing anonymously.
    #[error(
        "line {line}: invalid UTF-8 at byte offset {offset} within the line; \
         re-run with --lenient to replace it with U+FFFD and continue"
    )]
    InvalidUtf8 {
        /// 1-based line number in the concatenated input.
        line: u64,
        /// Byte offset of the first invalid byte, relative to the line start.
        offset: usize,
    },

    /// The edge list contained no arcs at all.
    ///
    /// Caught here rather than in the compressor, which reports an empty arc
    /// stream only as a parse failure on line 1 — the cascade recorded in
    /// `logs/webgraph_builder.err`.
    #[error("the edge list is empty; refusing to build a graph with no arcs (pass --allow-empty)")]
    EmptyEdgeList,

    /// A binary run file's length is not a whole number of arc records.
    #[error(
        "run file {path}: length {len} is not a multiple of {record_size} bytes (truncated write?)"
    )]
    TruncatedRun {
        /// The run file.
        path: PathBuf,
        /// Its length in bytes.
        len: u64,
        /// The expected record size, i.e. [`ARC_RECORD_SIZE`].
        record_size: usize,
    },

    /// A BVGraph basename contains a `.`, which `with_extension` would eat.
    #[error("invalid BVGraph basename {0}: it contains a '.', which webgraph-rs would strip from the extension")]
    BadBasename(PathBuf),

    /// A line of a text edge list was not two tab-separated integers.
    #[error("edge list {path} line {line}: expected two tab-separated integers, found {text:?}")]
    BadArcLine {
        /// The edge-list file.
        path: PathBuf,
        /// 1-based line number within it.
        line: u64,
        /// The offending line, verbatim.
        text: String,
    },

    /// Arcs reached the compressor out of `(src, dst)` ascending order.
    #[error("arcs are not sorted: ({c_src}, {c_dst}) follows ({p_src}, {p_dst})")]
    UnsortedArcs {
        /// Source of the previous arc.
        p_src: NodeId,
        /// Destination of the previous arc.
        p_dst: NodeId,
        /// Source of the offending arc.
        c_src: NodeId,
        /// Destination of the offending arc.
        c_dst: NodeId,
    },

    /// An arc endpoint is outside `0..num_nodes`.
    ///
    /// webgraph silently *drops* arcs whose source is out of bounds and does
    /// not bounds-check the destination at all, so this check is ours.
    #[error("arc ({src}, {dst}) is out of bounds for a {num_nodes}-node graph")]
    ArcOutOfBounds {
        /// The arc's source.
        src: NodeId,
        /// The arc's destination.
        dst: NodeId,
        /// The declared node count.
        num_nodes: usize,
    },

    /// The webgraph compressor panicked and the panic was caught.
    #[error("the webgraph compressor panicked: {0}")]
    CompressorPanic(String),

    /// The computed chunk boundaries do not end exactly at `--end`.
    ///
    /// `splitter.sh`'s `while (current < n && ...)` clamp is a no-op only
    /// because `END_TS == boundary[n]`. If the two ever diverge, the last
    /// chunk silently becomes a catch-all for everything after `--end`.
    #[error("boundary table ends at {last} but --end resolves to {end}; the last chunk would become a silent catch-all")]
    BoundaryMismatch {
        /// The last computed boundary.
        last: i64,
        /// The timestamp `--end` resolves to.
        end: i64,
    },

    /// `--tz` did not name an IANA timezone.
    #[error("unknown timezone {0:?}")]
    BadTimezone(String),

    /// `--start` or `--end` was not a `YYYY-MM-DD` date.
    #[error("invalid date {0:?}: expected YYYY-MM-DD")]
    BadDate(String),

    /// Anything that does not deserve its own variant.
    #[error("{0}")]
    Other(String),
}

impl Error {
    /// Convenience constructor for [`Error::Other`].
    pub fn other(msg: impl Into<String>) -> Self {
        Error::Other(msg.into())
    }
    /// Attaches a path to an `std::io::Error`.
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Error::Io {
            path: path.into(),
            source,
        }
    }
}

/// Fallback value of [`total_memory_bytes`] when `/proc/meminfo` is unusable.
const FALLBACK_TOTAL_MEMORY: u64 = 8 * 1024 * 1024 * 1024;

/// Total system RAM in bytes, read from `/proc/meminfo`. Falls back to 8 GiB.
///
/// Deliberately hand-rolled rather than pulling in `sysinfo`: the one number
/// this crate needs is the `MemTotal:` line, which is three lines of parsing.
///
/// On this machine it reports 527 938 072 kiB, i.e. about 503 GiB.
pub fn total_memory_bytes() -> u64 {
    let text = match std::fs::read_to_string("/proc/meminfo") {
        Ok(t) => t,
        Err(_) => return FALLBACK_TOTAL_MEMORY,
    };
    for line in text.lines() {
        // Format: "MemTotal:       527938072 kB"
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            let mut it = rest.split_whitespace();
            if let Some(value) = it.next() {
                if let Ok(kib) = value.parse::<u64>() {
                    return kib.saturating_mul(1024);
                }
            }
        }
    }
    FALLBACK_TOTAL_MEMORY
}

/// Parses `128GiB`, `200G`, `45%`, or a bare byte count into bytes.
///
/// Accepted forms, case-insensitively:
///
/// * a bare decimal byte count, e.g. `1024`;
/// * a decimal count with one of the suffixes `K KB KiB M MB MiB G GB GiB
///   T TB TiB`. **Every suffix is a binary multiple** — `200G` and `200GiB`
///   both mean `200 * 2^30`. This is deliberate: the value is a RAM budget,
///   and rounding it down to a decimal gigabyte would silently shrink the
///   sort run by 7%;
/// * a percentage of [`total_memory_bytes`], e.g. `45%`.
///
/// Zero is rejected, as is anything non-numeric. The error message names the
/// accepted forms, because this parser is wired straight into `clap` as the
/// `value_parser` for `--memory`.
pub fn parse_memory_spec(s: &str) -> std::result::Result<u64, String> {
    fn bad(s: &str) -> String {
        format!(
            "invalid memory specification {s:?}: expected a byte count (`1024`), \
             a size with a binary suffix (`128GiB`, `200G`, `64M`), \
             or a percentage of total RAM (`45%`)"
        )
    }

    let trimmed = s.trim();
    if trimmed.is_empty() {
        return Err(bad(s));
    }

    if let Some(pct_text) = trimmed.strip_suffix('%') {
        let pct: f64 = pct_text.trim().parse().map_err(|_| bad(s))?;
        if !(pct.is_finite() && pct > 0.0 && pct <= 100.0) {
            return Err(format!(
                "invalid memory percentage {s:?}: expected a value in (0, 100]"
            ));
        }
        let bytes = (total_memory_bytes() as f64 * pct / 100.0) as u64;
        if bytes == 0 {
            return Err(format!("memory specification {s:?} resolves to 0 bytes"));
        }
        return Ok(bytes);
    }

    // Split the leading number from the trailing unit.
    let split = trimmed
        .find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '_'))
        .unwrap_or(trimmed.len());
    let (number, unit) = trimmed.split_at(split);
    let number = number.replace('_', "");
    if number.is_empty() {
        return Err(bad(s));
    }
    let value: f64 = number.parse().map_err(|_| bad(s))?;
    if !value.is_finite() || value < 0.0 {
        return Err(bad(s));
    }

    // Every suffix is a BINARY multiple; see the doc comment.
    let multiplier: u64 = match unit.trim().to_ascii_lowercase().as_str() {
        "" | "b" => 1,
        "k" | "kb" | "kib" => 1u64 << 10,
        "m" | "mb" | "mib" => 1u64 << 20,
        "g" | "gb" | "gib" => 1u64 << 30,
        "t" | "tb" | "tib" => 1u64 << 40,
        _ => return Err(bad(s)),
    };

    let bytes = (value * multiplier as f64) as u64;
    if bytes == 0 {
        return Err(format!(
            "memory specification {s:?} resolves to 0 bytes; it must be positive"
        ));
    }
    Ok(bytes)
}

/// Free bytes on the filesystem containing `path`, via `statvfs(3)`.
/// Returns `None` if it cannot be determined.
///
/// Implemented with a local `extern "C"` declaration rather than a `libc`
/// dependency: one call with three interesting fields does not justify a
/// crate. Only the first eight `unsigned long` fields of `struct statvfs` are
/// named, which are stable across every 64-bit Linux ABI; the rest is opaque
/// padding sized generously so the kernel can never write past the buffer.
#[cfg(all(target_os = "linux", target_pointer_width = "64"))]
pub fn free_space_bytes(path: &std::path::Path) -> Option<u64> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    /// The prefix of `struct statvfs` that is identical on all 64-bit Linux
    /// ABIs, followed by enough padding to cover the whole structure
    /// (112 bytes on x86_64; 128 are reserved here).
    #[allow(dead_code)] // only three fields are read; the rest sizes the buffer
    #[repr(C)]
    struct StatVfs {
        f_bsize: u64,
        f_frsize: u64,
        f_blocks: u64,
        f_bfree: u64,
        f_bavail: u64,
        f_files: u64,
        f_ffree: u64,
        f_favail: u64,
        _tail: [u64; 8],
    }

    extern "C" {
        fn statvfs(path: *const std::os::raw::c_char, buf: *mut StatVfs) -> std::os::raw::c_int;
    }

    let c_path = CString::new(path.as_os_str().as_bytes()).ok()?;
    // `statvfs` needs an existing path; walk up to the nearest existing
    // ancestor so a not-yet-created output directory still answers.
    let c_path = if path.exists() {
        c_path
    } else {
        let mut cur = path.parent();
        loop {
            match cur {
                Some(p) if p.as_os_str().is_empty() => cur = None,
                Some(p) if p.exists() => break CString::new(p.as_os_str().as_bytes()).ok()?,
                Some(p) => cur = p.parent(),
                None => break CString::new(".").ok()?,
            }
        }
    };

    // SAFETY: `c_path` is a valid NUL-terminated string and `buf` is a
    // properly aligned, writable allocation at least as large as the
    // platform's `struct statvfs`.
    let mut buf = std::mem::MaybeUninit::<StatVfs>::zeroed();
    let rc = unsafe { statvfs(c_path.as_ptr(), buf.as_mut_ptr()) };
    if rc != 0 {
        return None;
    }
    // SAFETY: `statvfs` returned 0, so the buffer is fully initialised.
    let st = unsafe { buf.assume_init() };
    let block = if st.f_frsize != 0 {
        st.f_frsize
    } else {
        st.f_bsize
    };
    Some(st.f_bavail.saturating_mul(block))
}

/// Free bytes on the filesystem containing `path`. Always `None` off 64-bit
/// Linux: on a 32-bit Linux ABI `f_bsize`, `f_frsize`, `f_fsid`, `f_flag` and
/// `f_namemax` are 32-bit `unsigned long`, so the hand-rolled `StatVfs` above
/// would decode garbage. `require_space` treats `None` as "unknown" and only
/// warns.
#[cfg(not(all(target_os = "linux", target_pointer_width = "64")))]
pub fn free_space_bytes(_path: &std::path::Path) -> Option<u64> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_round_trips_including_negatives() {
        // The table includes the exact value pinned by the `edge/t6.txt`
        // golden fixture (txId = -5, offset = 0).
        let table: [(i32, i32, u64); 6] = [
            (-5, 0, 0xFFFF_FFFB_0000_0000),
            (0, 0, 0),
            (i32::MAX, i32::MAX, 0x7FFF_FFFF_7FFF_FFFF),
            (i32::MIN, -1, 0x8000_0000_FFFF_FFFF),
            (1, 2, 0x0000_0001_0000_0002),
            (778_613_437, 914, 0x2E68_B2BD_0000_0392),
        ];
        for (tx, off, key) in table {
            assert_eq!(pack(tx, off), key, "pack({tx}, {off})");
            assert_eq!(unpack(key), (tx, off), "unpack({key:#x})");
        }
    }

    #[test]
    fn pack_arc_round_trips_with_no_ceiling() {
        // The point of the `u128` packing is that there is no id it rejects.
        // The middle three rows are the ones the previous `(src << 32) | dst`
        // scheme could not represent at all: 2_181_021_971 is the node count a
        // 28-chunk run actually reaches, and the last row is the widest pair
        // a `NodeId` can hold.
        for (src, dst) in [
            (0usize, 0usize),
            (1, 2),
            (2_210_000_000, 7),
            (2_181_021_971, 2_181_021_970),
            (u32::MAX as usize + 1, u32::MAX as usize + 1),
            (usize::MAX, usize::MAX),
        ] {
            assert_eq!(unpack_arc(pack_arc(src, dst)), (src, dst));
        }

        // The two halves never bleed into each other: a maximal destination
        // must not disturb the source.
        assert_eq!(unpack_arc(pack_arc(1, usize::MAX)), (1, usize::MAX));
        assert_eq!(unpack_arc(pack_arc(usize::MAX, 0)), (usize::MAX, 0));

        // Sorting packed arcs ascending is `sort -k1,1n -k2,2n`, including
        // across the old 2^32 boundary.
        const HIGH: usize = 1 << 32;
        let mut packed: Vec<u128> = [(2, 1), (HIGH, 0), (1, 10), (1, 2), (2, 0)]
            .iter()
            .map(|&(s, d)| pack_arc(s, d))
            .collect();
        packed.sort_unstable();
        let unpacked: Vec<(usize, usize)> = packed.iter().copied().map(unpack_arc).collect();
        assert_eq!(
            unpacked,
            vec![(1, 2), (1, 10), (2, 0), (2, 1), (HIGH, 0)]
        );
    }

    #[test]
    fn arc_record_size_is_the_on_disk_width() {
        // Every binary run reader dispatches on this constant and the format
        // has no header, so it is a file-format promise, not a detail.
        assert_eq!(ARC_RECORD_SIZE, std::mem::size_of::<u128>());
        assert_eq!(ARC_RECORD_SIZE, 16);
    }

    #[test]
    fn memory_specs_parse_and_reject() {
        assert_eq!(parse_memory_spec("128GiB").unwrap(), 128 * (1u64 << 30));
        assert_eq!(parse_memory_spec("200G").unwrap(), 200 * (1u64 << 30));
        assert_eq!(parse_memory_spec("1024").unwrap(), 1024);
        // Percentages are relative to whatever this machine reports.
        let half = parse_memory_spec("50%").unwrap();
        let total = total_memory_bytes();
        assert!(half > 0 && half <= total);
        assert!((half as i128 - (total / 2) as i128).abs() <= 2);

        assert!(parse_memory_spec("").is_err());
        assert!(parse_memory_spec("lots").is_err());
        assert!(parse_memory_spec("0").is_err());
        assert!(parse_memory_spec("12PB").is_err());
        assert!(parse_memory_spec("150%").is_err());
    }
}
