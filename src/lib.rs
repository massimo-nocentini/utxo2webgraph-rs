//! `utxo2webgraph` — a Rust port of the Bitcoin Payment Graph pipeline.
//!
//! # Attribution
//!
//! The pipeline and the graph-construction algorithm are the work of
//! **Matteo Loporchio**. This crate is a reimplementation, in Rust, of
//!
//! * `PaymentGraphEdgeListBuilder.java` — the node-map and edge-list builder,
//! * `splitter.sh` — the six-month temporal splitter,
//! * `builder.sh` — the "first N chunks" driver,
//! * `build_pg.sh` — the three-stage pipeline script, and
//! * `builder.GraphBuilder` inside `jar/WebgraphBuilder.jar` — the thin wrapper
//!   that called `it.unimi.dsi.webgraph.BVGraph -g ArcListASCIIGraph`.
//!
//! The algorithm is his; this crate only reimplements it. Every module that
//! ports a piece of his logic repeats the attribution in its own module doc.
//!
//! # Deliberate behavioural divergences from the Java program
//!
//! All six of these are unreachable on the current corpus (measured over
//! `chunks/chunk_01..06` in full and over multi-million-line heads of
//! `chunk_10`, `chunk_20` and `chunk_28`), so none of them can perturb the
//! differential test against the Java reference output.
//!
//! 1. **An empty output section emits zero edges**, not a phantom edge into
//!    node `0`. Java sized `currentOutputNodeIds` as `outputs.length`
//!    (line 59), and `"".split(";")` returns a length-1 array, so the array
//!    stayed `{0}` while the `!parts[2].equals("")` guard (line 60) skipped
//!    only the *node creation*. Every input of such a transaction then emitted
//!    a bogus edge into the genesis output at lines 76-77. There is no reading
//!    under which that edge is correct, so it is fixed with no opt-out.
//! 2. **Blank and malformed lines are skipped or reported with a line
//!    number**, instead of throwing `ArrayIndexOutOfBoundsException` at Java
//!    lines 54/55. A blank line killed a multi-hour run with a message that
//!    named neither the line nor the file.
//! 3. **A dangling source reference produces a typed error** naming the input
//!    line, the current `txId`, the input index and the missing
//!    `(prevTxId, prevOffset)` — see [`PgError::DanglingSource`] — instead of
//!    the bare `NullPointerException` Java raised at line 74 when unboxing
//!    `nodes.get(sourceNodeKey)`. The default outcome (abort, exit 1) is
//!    unchanged; [`OnMissingSource::Skip`] and [`OnMissingSource::Create`] are
//!    new opt-ins.
//! 4. **Every write error is propagated.** Java used `PrintWriter`, which never
//!    throws and whose `checkError()` was never called, so a disk-full during a
//!    multi-hour, 210 GB write silently truncated the edge list while the
//!    program printed `Nodes:`/`Edges:` and exited 0.
//! 5. **The node map is written in ascending `(txId, offset)` order**, not in
//!    `java.util.HashMap` bucket order (Java lines 89-94). The *set* of triples
//!    is identical; the byte order is not. Compare node maps with
//!    `sort -t$'\t' -k1,1n -k2,2n | md5sum`. The edge list, whose order is
//!    fully determined by the input, is still byte-identical.
//! 6. **A non-UTF-8 byte is a line-numbered error in strict mode.** Java's
//!    `InputStreamReader` substituted U+FFFD and carried on; `--lenient` does
//!    exactly that (and tallies `non_utf8_lines`), while the default reports
//!    [`PgError::InvalidUtf8`] with the line and the byte offset. Neither mode
//!    aborts the way the port briefly did, with an `ErrorKind::InvalidData`
//!    that named neither the file nor the line.
//!
//! # Not a divergence: `--num-nodes`
//!
//! `pgraph build`, `build-pg` and `compress` all default to
//! `--num-nodes from-arcs`, which is `max(endpoint) + 1` — exactly what
//! `ArcListASCIIGraph` computed inside `WebgraphBuilder.jar`, and therefore
//! exactly the `nodes=` the shell pipeline produced (18 545 on `chunk_01`,
//! with a bitstream byte-identical to Java's). `--num-nodes node-map` is the
//! opt-in richer graph in which every UTXO is a node, including the recent
//! unspent outputs that appear in no arc (18 620 on `chunk_01`, i.e. 75 extra
//! isolated trailing nodes). Both candidates are logged whenever they differ.

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

/// A dense utxo2webgraph node identifier (Java: `long`, `nextId`).
pub type NodeId = u64;

/// Result alias used by every library function in this crate.
pub type PgResult<T> = std::result::Result<T, PgError>;

/// Java `pack(int txId, int offset)`. Bit-exact; bijective for all i32 pairs.
///
/// This is `PaymentGraphEdgeListBuilder.java` lines 28-30:
///
/// ```java
/// public static long pack(int txId, int offset) {
///     return ((long) txId << 32) | (offset & 0xFFFFFFFFL);
/// }
/// ```
///
/// `(long) txId` sign-extends and `<< 32` moves the pattern into the high
/// word; `offset & 0xFFFFFFFFL` is the unsigned 32-bit reinterpretation of
/// `offset`. The two halves never overlap, so the key is exactly
/// `(txId as u32) << 32 | (offset as u32)`.
///
/// Together with [`unpack`] (Java lines 90-91) the round trip is **exact and
/// bijective for all 2^64 `(i32, i32)` pairs, negatives included**: Java's
/// `>>>` is a logical shift and both narrowing casts are pure truncation, so
/// no information is ever lost. Two distinct `(txId, offset)` pairs therefore
/// can never collide on a key, and the node map has no aliasing.
///
/// The key type is `u64` and not `i64` on purpose: only equality and hashing
/// are ever performed on it, and `u64` makes it impossible to write an
/// arithmetic `>>` by accident where Java wrote `>>>`.
#[inline]
pub fn pack(tx_id: i32, offset: i32) -> u64 {
    ((tx_id as u32 as u64) << 32) | (offset as u32 as u64)
}

/// Inverse of [`pack`]. Java: `(int)(key >>> 32)`, `(int) key`.
///
/// Verified against the reference implementation with `txId = -5`:
/// `pack(-5, 0) == 0xFFFFFFFB_00000000` and the Java node map printed
/// `-5\t0\t0`, i.e. the negative value survived the round trip untouched.
#[inline]
pub fn unpack(key: u64) -> (i32, i32) {
    (((key >> 32) as u32) as i32, (key as u32) as i32)
}

/// Packs an arc into one `u64`: sorting these ascending is exactly
/// "sort by src, then by dst". Returns `None` if either id is >= 2^32.
///
/// Sorting the packed values ascending is bit-for-bit equivalent to
/// `sort -t$'\t' -k1,1n -k2,2n`, and [`Vec::dedup`] on the sorted packed arcs
/// is exactly `uniq` — the two shell stages `build_pg.sh` used.
///
/// # Warning
///
/// At `N = 28` chunks the estimated maximum node id is about `2.21e9` against
/// the [`ARC_ID_LIMIT`] of `4.29e9`: a margin of only **1.94x**, on an estimate
/// that itself carries roughly +/-20%. The `None` return must therefore never
/// be ignored — a caller that sees it must switch to [`ArcCodec::Wide`]
/// (`--arc-codec wide`) rather than truncate.
#[inline]
pub fn pack_arc(src: NodeId, dst: NodeId) -> Option<u64> {
    if src >= ARC_ID_LIMIT || dst >= ARC_ID_LIMIT {
        None
    } else {
        Some((src << 32) | dst)
    }
}

/// Inverse of [`pack_arc`].
///
/// Round-trips exactly for every pair below [`ARC_ID_LIMIT`].
#[inline]
pub fn unpack_arc(a: u64) -> (NodeId, NodeId) {
    (a >> 32, a & 0xFFFF_FFFF)
}

/// Exclusive upper bound on a node id storable in [`ArcCodec::Packed32`].
///
/// `2^32 == 4_294_967_296`. See the margin warning on [`pack_arc`].
pub const ARC_ID_LIMIT: u64 = 1u64 << 32;

/// Java's progress-report period.
///
/// `PaymentGraphEdgeListBuilder.java` line 83: `if (txCount % 10_000_000 == 0)`.
pub const DEFAULT_PROGRESS_EVERY: u64 = 10_000_000;

/// How malformed input is treated.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum Mode {
    /// Any malformed record is a fatal, line-numbered error.
    ///
    /// This is the closest analogue of the Java behaviour, which died with an
    /// `ArrayIndexOutOfBoundsException` or a `NumberFormatException` — only
    /// here the message names the line and the offending value.
    #[default]
    Strict,
    /// Malformed records are skipped and tallied.
    ///
    /// Has no Java counterpart: the Java program had no way to survive a bad
    /// line. Blank lines are skipped in *both* modes.
    Lenient,
}

/// What to do when an input references an output that was never registered
/// (Java: `NullPointerException` at line 74).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum OnMissingSource {
    /// Abort with [`PgError::DanglingSource`]. Matches Java's outcome.
    #[default]
    Fail,
    /// Emit no edges for that input; tally it.
    ///
    /// New. Makes single-chunk and sharded processing possible, which the Java
    /// program could not do at all.
    Skip,
    /// Mint a node for it on the fly. CHANGES THE ID SPACE.
    ///
    /// New, and never the default: minting a node shifts every subsequent id,
    /// so both output files stop being comparable with the Java reference.
    Create,
}

/// Statistics output style.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum StatsStyle {
    /// Byte-identical to the Java program's two stdout lines.
    ///
    /// Reproduces the mislabelled `Nodes:` counter (Java line 64 counts output
    /// *slots*, not distinct nodes) so that a naive diff of
    /// `logs/pg_el_builder.log` still matches.
    #[default]
    Java,
    /// The Java lines plus corrected/extra counters.
    ///
    /// Adds the distinct-node count that Java's `Nodes:` was supposed to be,
    /// plus the skipped-line, dangling-reference and duplicate-arc tallies.
    Extended,
}

/// On-disk/in-memory arc representation.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum ArcCodec {
    /// One `u64` per arc, `(src << 32) | dst`. Requires both ids < 2^32.
    ///
    /// Halves memory and I/O against a `(u64, u64)` pair and makes an LSD
    /// radix sort directly applicable. Guarded by [`pack_arc`].
    #[default]
    Packed32,
    /// One `u128` per arc, `(src << 64) | dst`. Always safe.
    ///
    /// The fallback for a dataset whose node count reaches [`ARC_ID_LIMIT`].
    Wide,
}

impl ArcCodec {
    /// Bytes per arc record on disk.
    #[inline]
    pub fn record_size(self) -> usize {
        match self {
            ArcCodec::Packed32 => 8,
            ArcCodec::Wide => 16,
        }
    }
}

/// Run-sorting backend.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum SortAlgo {
    /// `rdst` LSD radix sort. Only valid for [`ArcCodec::Packed32`].
    ///
    /// Replaces `sort --temporary-directory=./tmp -t$'\t' -k1,1n -k2,2n`,
    /// which at `N = 28` would push roughly 840 GB through the filesystem to
    /// re-parse integers this process already holds in registers.
    #[default]
    Radix,
    /// `rayon` pattern-defeating quicksort. Works for both codecs.
    ///
    /// The cross-check for [`SortAlgo::Radix`], and the only option for
    /// [`ArcCodec::Wide`].
    Pdq,
}

/// A sink for raw, unsorted arcs produced by the edge-list builder.
///
/// Implemented by `arcs::TsvArcSink`, `arcs::BinaryArcSink`,
/// `arcs::ArcSorter` and `arcs::NullArcSink`.
///
/// # Ordering contract
///
/// [`ArcSink::push`] is called in **exact Java emission order**: input file
/// line order, then input index ascending, then output offset ascending
/// (`PaymentGraphEdgeListBuilder.java` lines 68-80). Implementations **must
/// not** reorder, buffer out of order, or drop arcs, because the raw TSV edge
/// list is compared byte-for-byte against the Java reference output
/// (`el_01.tsv`, md5 `a3c31369f1a54bacfeb9b3cc2c953ebe`). Duplicates and
/// self-loops are part of that stream and must be preserved; deduplication
/// happens later, in `arcs::ArcSorter`.
pub trait ArcSink {
    /// Accepts one arc in Java emission order.
    fn push(&mut self, src: NodeId, dst: NodeId) -> PgResult<()>;
    /// Flushes and durably commits. Must be called exactly once.
    ///
    /// Implementations that own a file must flush *and* `sync_all` here, and
    /// must return any error rather than swallow it: Java's `PrintWriter` did
    /// the opposite, and a disk-full silently truncated the output while the
    /// program reported success.
    fn finish(&mut self) -> PgResult<()>;
}

/// Error slot used by iterators that cannot return `Result` because
/// `webgraph` requires a plain `Iterator`. Callers MUST check it after the
/// iterator has been fully consumed, via [`take_iter_error`].
pub type ErrorSlot = Arc<Mutex<Option<PgError>>>;

/// Creates a fresh, empty [`ErrorSlot`].
pub fn new_error_slot() -> ErrorSlot {
    Arc::new(Mutex::new(None))
}

/// Takes the error out of a slot, if any. Poisoned mutex => `Ok(())`.
pub fn take_iter_error(slot: &ErrorSlot) -> PgResult<()> {
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
/// argument (os error 22)` and exit 1 — the Java reference, which used a
/// `PrintWriter` and never synced, handled `/dev/null`, fifos and process
/// substitution fine.
///
/// Real durability errors (`ENOSPC`, `EIO`, …) are still propagated: that is
/// the whole point of syncing before reporting success.
pub fn sync_if_durable(file: &std::fs::File, path: &std::path::Path) -> PgResult<()> {
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
        Err(e) => Err(PgError::io(path, e)),
    }
}

/// Every failure this library can report. Deliberately line-numbered:
/// the Java program reported none of this.
#[derive(Debug, thiserror::Error)]
pub enum PgError {
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
    /// Java would either throw `ArrayIndexOutOfBoundsException` (fewer than
    /// three) or silently take `parts[0..3]` and fall into the node-0 phantom
    /// edge bug (more than three).
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

    /// A field that Java parsed with `Integer.parseInt` is not an `i32`.
    ///
    /// The port keeps `i32` semantics deliberately: widening to `i64` would
    /// change the packed key for any value at or above 2^31 and diverge from
    /// the reference node map.
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
    /// `Integer.parseInt("-5")` succeeded in Java and the value round-tripped
    /// through [`pack`] into a `-5` in the txId column of the node map, which
    /// no downstream consumer expects. [`Mode::Lenient`] accepts it,
    /// bug-compatibly.
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
    /// This is the central bug of the Java program. `nodes.get(sourceNodeKey)`
    /// at line 74 is a pure lookup — the input side never inserts — so a
    /// missing entry auto-unboxed `null` into a `long` and threw
    /// `NullPointerException`. The run died with *no* node map (it is written
    /// only after the read loop) and a truncated edge list, and the message
    /// named neither the line, nor the txId, nor the missing pair.
    ///
    /// The consequence is that the Java program is well-defined **only on a
    /// prefix of the transaction list starting at genesis** — exactly what
    /// `builder.sh` guaranteed by always concatenating chunks `01..N`. Any
    /// partial range, any single mid-stream chunk, any sharded or resumed run
    /// hits this immediately. Measured incidence:
    ///
    /// | input | dangling refs / total input refs |
    /// |---|---|
    /// | `chunk_01` alone | 0 / 1033 |
    /// | `chunk_01 + chunk_02` | 0 / 2887 |
    /// | `chunk_02` alone | 134 / 1854 |
    /// | `chunk_05` alone | 18 145 / 1 168 956 |
    ///
    /// [`OnMissingSource::Fail`] keeps the Java outcome (abort, exit 1) and
    /// only improves the diagnostic.
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
    /// Distinct from [`PgError::NonDenseTxId`] for one reason: the operator is
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
    /// Nothing legitimate is lost. Java had no analogue at all — line 74 threw
    /// `NullPointerException` on any missing source, forward or backwards — and
    /// a Bitcoin transaction can only spend an output that already exists, so a
    /// forward reference cannot occur in correctly ordered data (verified: zero
    /// forward references across `chunk_04` and `chunk_05`). The reachable,
    /// legitimate case — a chunk that starts above genesis and refers back to
    /// transactions it does not contain — is *behind* the cursor and is minted
    /// normally.
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
    /// Java's `InputStreamReader` substituted U+FFFD and carried on, so
    /// [`Mode::Lenient`] does the same (the record is then parsed from the
    /// lossy text, exactly as Java would have). [`Mode::Strict`] reports this
    /// instead, because a stray byte in a 132 GB corpus is worth knowing
    /// about — and, unlike the `read_line` error it replaces, it names the
    /// line and the byte offset within it.
    #[error(
        "line {line}: invalid UTF-8 at byte offset {offset} within the line; \
         re-run with --lenient to replace it with U+FFFD, as Java did"
    )]
    InvalidUtf8 {
        /// 1-based line number in the concatenated input.
        line: u64,
        /// Byte offset of the first invalid byte, relative to the line start.
        offset: usize,
    },

    /// A node id reached [`ARC_ID_LIMIT`] and can no longer be packed.
    #[error(
        "node id {0} reaches 2^32; packed 32-bit arcs cannot represent it, pass --arc-codec wide"
    )]
    ArcCodecOverflow(NodeId),

    /// The graph has more nodes than the 32-bit Java WebGraph reader accepts.
    #[error(
        "node count {0} exceeds i32::MAX; the Java WebGraph reader could never load this graph"
    )]
    TooManyNodes(u64),

    /// The edge list contained no arcs at all.
    ///
    /// Java's `ArcListASCIIGraph` threw `Expected integer, found Token[EOF],
    /// line 1` here — which is exactly the cascade recorded in
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
        /// The expected record size, from [`ArcCodec::record_size`].
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

impl PgError {
    /// Convenience constructor for [`PgError::Other`].
    pub fn other(msg: impl Into<String>) -> Self {
        PgError::Other(msg.into())
    }
    /// Attaches a path to an `std::io::Error`.
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        PgError::Io {
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
pub fn parse_memory_spec(s: &str) -> Result<u64, String> {
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
        // The table includes the exact value verified against the compiled
        // Java reference with `javaref/edge/t6.txt` (txId = -5, offset = 0).
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
    fn pack_arc_guards_the_32_bit_limit() {
        assert_eq!(pack_arc(ARC_ID_LIMIT, 0), None);
        assert_eq!(pack_arc(0, ARC_ID_LIMIT), None);
        assert_eq!(pack_arc(u64::MAX, u64::MAX), None);

        for (src, dst) in [
            (0u64, 0u64),
            (1, 2),
            (2_210_000_000, 7),
            (ARC_ID_LIMIT - 1, ARC_ID_LIMIT - 1),
        ] {
            let packed = pack_arc(src, dst).expect("below the limit");
            assert_eq!(unpack_arc(packed), (src, dst));
        }

        // Sorting packed arcs ascending is `sort -k1,1n -k2,2n`.
        let mut packed: Vec<u64> = [(2u64, 1u64), (1, 10), (1, 2), (2, 0)]
            .iter()
            .map(|&(s, d)| pack_arc(s, d).unwrap())
            .collect();
        packed.sort_unstable();
        let unpacked: Vec<(u64, u64)> = packed.iter().copied().map(unpack_arc).collect();
        assert_eq!(unpacked, vec![(1, 2), (1, 10), (2, 0), (2, 1)]);
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
