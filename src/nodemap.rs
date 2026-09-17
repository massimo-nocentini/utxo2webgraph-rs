//! Node map. Ported from `PaymentGraphEdgeListBuilder.java` by
//! **Matteo Loporchio** (lines 15-25), which used `HashMap<Long, Long>` keyed
//! on `pack(txId, offset)`:
//!
//! ```java
//! public static long getOrCreateId(long key) {
//!     long id = nodes.getOrDefault(key, -1L);
//!     if (id == -1) { id = nextId++; nodes.put(key, id); }
//!     return id;
//! }
//! ```
//!
//! Two implementations live here. [`DenseNodeMap`] is the default and exploits
//! the verified density of `txId`; [`HashNodeMap`] is a literal port of the
//! Java map and exists to cross-check the dense one (an integration test drives
//! both over all of `chunk_01.txt` and asserts identical output).

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::io::Write;

use crate::{pack, unpack, Mode, NodeId, NodeMapKind, PgError, PgResult};

/// Bytes buffered before flushing to the writer in [`NodeMap::write_tsv`].
const TSV_BUF: usize = 1 << 16;

/// Largest forward `txId` jump [`DenseNodeMap`] will bridge in
/// [`Mode::Lenient`] before giving up.
///
/// The fill costs one `u64` per skipped id. Unbounded, a single corrupt record
/// was catastrophic: a two-line, 79-byte input whose second `txId` is
/// `2_147_483_647` drove RSS to 16 GiB in 16 s, and under `ulimit -v` the
/// process *aborted* (`memory allocation of 67108864 bytes failed`, exit 134)
/// rather than returning the typed error the rest of the crate promises. Java's
/// `HashMap` simply inserted one entry, so that was the port being strictly
/// worse than the reference on the very input class `--lenient` exists to
/// survive.
///
/// 2^20 ids is 8 MiB of filler — far beyond any plausible gap in a corpus
/// whose txIds are globally consecutive (verified `0..778_613_437` across all
/// 28 chunks) — and anything larger is a corrupt file, not a gap:
/// [`PgError::NonDenseTxId`] already says to re-run with
/// `--node-map-kind hash`, which handles arbitrary txIds in `O(1)` memory per
/// record.
pub const MAX_TX_GAP: i64 = 1 << 20;

/// Assigns and resolves utxo2webgraph node ids.
///
/// **The contract both implementations obey**, taken from the Java:
///
/// * ids are dense, gapless and monotonically increasing from 0, assigned in
///   **first-appearance order** — input line order and, within a line, output
///   offset `0, 1, 2, …` ascending;
/// * **output nodes are the only nodes ever created.** Java line 63 is the one
///   and only `getOrCreateId` call site; line 74 (`nodes.get(sourceNodeKey)`)
///   is a pure lookup that inserts nothing, which is exactly why a dangling
///   input reference made the reference implementation throw
///   `NullPointerException` instead of minting a node;
/// * a line's outputs are registered **before** its inputs are resolved
///   (Java lines 59-65 precede 68-81), so a transaction spending its own output
///   resolves and yields a **self-loop**. Real data never does this (verified:
///   0 self-references and 0 forward references across all of `chunk_05`), but
///   the port must not "optimise" on that assumption — the synthetic Java test
///   `edge/t7.txt` really does emit `0\t0`.
pub trait NodeMap {
    /// Registers the outputs of one transaction, appending their node ids to
    /// `out` in ascending offset order.
    ///
    /// `out` is cleared first and then receives exactly `num_outputs` ids; it
    /// is Java's `long[] currentOutputNodeIds` (lines 59-65), passed in by the
    /// caller so one buffer can be reused for the whole run. `num_outputs == 0`
    /// leaves `out` empty — that is the fix for the phantom-edge bug described
    /// in [`crate::record::count_outputs`], where Java's `new long[1]` kept its
    /// default `{0}` and every input emitted an edge into the genesis output.
    ///
    /// `line` is only used to build [`PgError::NonDenseTxId`].
    fn register_tx(
        &mut self,
        tx_id: i32,
        num_outputs: u32,
        line: u64,
        out: &mut Vec<NodeId>,
    ) -> PgResult<()>;

    /// Resolves a previously registered output. **Never inserts.**
    ///
    /// `None` is the dangling reference that made Java throw at line 74.
    fn lookup(&self, tx_id: i32, offset: i32) -> Option<NodeId>;

    /// Mints a node for an unregistered `(txId, offset)`.
    ///
    /// Used **only** by [`crate::OnMissingSource::Create`]. Calling it
    /// **changes the id space**, and therefore every id in both output files:
    /// a run that used it is not comparable with a Java reference run, and the
    /// node map it produces describes a different graph labelling. Repeated
    /// calls with the same key return the same id.
    fn force_create(&mut self, tx_id: i32, offset: i32) -> PgResult<NodeId>;

    /// Java's `nextId`: one past the highest id handed out so far.
    fn next_id(&self) -> NodeId;

    /// Java's `nodes.size()`: the number of distinct `(txId, offset)` keys.
    ///
    /// Equal to [`NodeMap::next_id`] unless [`NodeMap::force_create`] was used
    /// or a `(txId, offset)` pair repeated.
    fn distinct_nodes(&self) -> u64;

    /// Writes the node map as `txId\toffset\tid`, LF-terminated, no header, in
    /// **ascending `(txId, offset)`** order.
    ///
    /// Java iterated `nodes.keySet()` (lines 89-94), i.e. `HashMap` bucket
    /// order. That is deterministic for a fixed JVM but is *not* a
    /// specification: `chunk_01`'s reference node map has 23 descents in the id
    /// column and 5 in the txId column, so it merely *looks* sorted. This
    /// ordering is deterministic, portable and streams straight out of the
    /// dense base array. Any consumer that byte-diffed the old `pg_nm_N.tsv`
    /// will see a different file with an identical *set* of triples; compare
    /// with `sort -t$'\t' -k1,1n -k2,2n | md5sum`
    /// (`4795d0ab4a84e775d16c7b57e26d2fc9` for `chunk_01`).
    fn write_tsv(&self, w: &mut dyn Write) -> PgResult<()>;
}

// ---------------------------------------------------------------------------
// DenseNodeMap
// ---------------------------------------------------------------------------

/// Node map backed by a single `Vec<u64>` indexed by transaction id.
///
/// **The invariant it relies on, verified over the whole corpus:** `txId` is
/// strictly increasing by exactly 1, starting at 0, with no gaps, globally
/// across all 28 chunks — it is the zero-based line number of the master
/// transaction list, and its maximum is 778 613 437. (Checked in full over
/// `chunk_01` and `chunk_05`, and across every chunk boundary: `chunk_01`
/// covers 0..18577, `chunk_02` starts at 18578, … `chunk_28` ends at
/// 778 613 437.)
///
/// Under that invariant the id of output `offset` of transaction `t` is simply
///
/// ```text
/// node_id(t, offset) = base[t - min_tx] + offset
/// ```
///
/// with `base` the prefix sum of the per-transaction output counts. Cost at
/// N=28: `778.6e6 * 8 = 6.23 GB`, versus 140-160 GB for the Java
/// `HashMap<Long, Long>` (a `Node` object plus two boxed `Long`s per entry) —
/// which is what `-Xmx200g` was actually paying for. Elias-Fano would compress
/// `base` to about 340 MB, but it is not incrementally appendable and 6.2 GB
/// out of 503 GB is free, so it is deliberately not done.
///
/// The element type is `u64`, not `u32`: the node count at N=28 is about
/// 2.21e9, which does fit in a `u32` but with only a 1.94x margin against an
/// estimate carrying +/-20% uncertainty. Three gigabytes are not worth that
/// cliff.
///
/// **The invariant is verified at runtime, never assumed.** A transaction id
/// that breaks it is [`PgError::NonDenseTxId`], which names the expected value
/// and tells the operator to re-run with `--node-map-kind hash`.
pub struct DenseNodeMap {
    /// Sentinel prefix sum: `base[i]` is the id of output 0 of transaction
    /// `min_tx + i`, and the final element is always equal to `next_id`.
    /// Hence `base.len() == number_of_transactions + 1` once started.
    base: Vec<u64>,
    /// Java's `nextId`.
    next_id: u64,
    /// The transaction id of `base[0]`. 0 for the standard pipeline.
    min_tx: i64,
    /// Strict/lenient behaviour for a non-dense transaction id.
    mode: Mode,
    /// Nodes minted by [`NodeMap::force_create`], keyed by [`pack`]. Empty on
    /// every path except `--on-missing-source create`.
    extra: HashMap<u64, NodeId, BuildMulShift>,
    /// For each minted node, the number of transactions registered at the time
    /// it was minted. A phantom recorded with value `k` occupies an id inside
    /// the interval `[base[k-1], base[k])`, after transaction `k-1`'s own
    /// outputs, so `num_outputs(k-1)` must discount it. Appended in order and
    /// therefore already sorted.
    phantom_at: Vec<usize>,
    /// Whether the lenient gap-filling warning has already been emitted.
    warned_gap: bool,
}

impl DenseNodeMap {
    /// Creates an empty dense map.
    pub fn new(mode: Mode) -> Self {
        DenseNodeMap {
            base: Vec::new(),
            next_id: 0,
            min_tx: 0,
            mode,
            extra: HashMap::default(),
            phantom_at: Vec::new(),
            warned_gap: false,
        }
    }

    /// Creates an empty dense map pre-sized for `txs` transactions.
    pub fn with_capacity(mode: Mode, txs: usize) -> Self {
        let mut m = Self::new(mode);
        m.base.reserve(txs.saturating_add(1));
        m
    }

    /// Number of outputs registered for `tx_id`, or `None` if that transaction
    /// was never registered.
    ///
    /// Derived from the prefix sum as `base[i + 1] - base[i]`, so it costs no
    /// extra storage. The `next_id` sentinel closes the last entry.
    pub fn num_outputs(&self, tx_id: i32) -> Option<u32> {
        let i = self.index_of(tx_id)?;
        Some(self.outputs_at(i) as u32)
    }

    /// Index into `base` for `tx_id`, if it names a registered transaction.
    #[inline]
    fn index_of(&self, tx_id: i32) -> Option<usize> {
        let ntx = self.base.len().checked_sub(1)?;
        let i = (tx_id as i64).checked_sub(self.min_tx)?;
        if i < 0 || i as u64 >= ntx as u64 {
            return None;
        }
        Some(i as usize)
    }

    /// Output count of the transaction at `base` index `i`, discounting any
    /// force-created nodes that were minted inside its id interval.
    #[inline]
    fn outputs_at(&self, i: usize) -> u64 {
        let raw = self.base[i + 1] - self.base[i];
        if self.phantom_at.is_empty() {
            return raw;
        }
        raw - self.phantoms_in(i)
    }

    /// How many force-created nodes fall inside transaction `i`'s id interval.
    #[inline]
    fn phantoms_in(&self, i: usize) -> u64 {
        let key = i + 1;
        let lo = self.phantom_at.partition_point(|v| *v < key);
        let hi = self.phantom_at.partition_point(|v| *v <= key);
        (hi - lo) as u64
    }
}

impl NodeMap for DenseNodeMap {
    fn register_tx(
        &mut self,
        tx_id: i32,
        num_outputs: u32,
        line: u64,
        out: &mut Vec<NodeId>,
    ) -> PgResult<()> {
        out.clear();

        if self.base.is_empty() {
            self.min_tx = tx_id as i64;
            self.base.push(self.next_id);
        } else {
            let expected = self.min_tx + (self.base.len() as i64 - 1);
            let found = tx_id as i64;
            if found != expected {
                // A backwards txId can never be repaired: the ids for the
                // transactions in between have already been handed out.
                if found < expected || self.mode == Mode::Strict {
                    return Err(PgError::NonDenseTxId {
                        line,
                        expected,
                        found,
                    });
                }
                let gap = found - expected;
                if gap > MAX_TX_GAP {
                    // Bridging this would allocate `8 * gap` bytes of filler
                    // for one suspicious line. Refuse, and point at the map
                    // that does not care about density.
                    return Err(PgError::NonDenseTxId {
                        line,
                        expected,
                        found,
                    });
                }
                if !self.warned_gap {
                    self.warned_gap = true;
                    log::warn!(
                        "line {line}: transaction id {found} skips ahead of {expected}; \
                         filling the gap with zero-output transactions (lenient mode). \
                         This warning is issued once."
                    );
                }
                // `try_reserve`, not `reserve`: an allocation failure must
                // surface as a typed error, not as an `abort()` that no
                // caller can catch.
                self.base.try_reserve(gap as usize).map_err(|_| {
                    PgError::other(format!(
                        "line {line}: transaction id {found} skips ahead of {expected}; \
                         filling the {gap}-id gap needs {} bytes, which could not be \
                         allocated; re-run with --node-map-kind hash",
                        gap as u64 * 8
                    ))
                })?;
                for _ in expected..found {
                    // A zero-output filler transaction: start == end.
                    self.base.push(self.next_id);
                }
            }
        }

        debug_assert_eq!(*self.base.last().unwrap(), self.next_id);
        let start = self.next_id;
        self.next_id = start + num_outputs as u64;
        self.base.push(self.next_id);

        out.reserve(num_outputs as usize);
        for k in 0..num_outputs as u64 {
            out.push(start + k);
        }
        Ok(())
    }

    fn lookup(&self, tx_id: i32, offset: i32) -> Option<NodeId> {
        if offset >= 0 {
            if let Some(i) = self.index_of(tx_id) {
                let n = self.outputs_at(i);
                if (offset as u64) < n {
                    return Some(self.base[i] + offset as u64);
                }
            }
        }
        if self.extra.is_empty() {
            return None;
        }
        self.extra.get(&pack(tx_id, offset)).copied()
    }

    fn force_create(&mut self, tx_id: i32, offset: i32) -> PgResult<NodeId> {
        if let Some(id) = self.lookup(tx_id, offset) {
            return Ok(id);
        }
        let id = self.next_id;
        self.next_id += 1;
        self.extra.insert(pack(tx_id, offset), id);
        self.phantom_at.push(self.base.len().saturating_sub(1));
        // Keep the `*base.last() == next_id` invariant, so the next
        // registration still starts where the prefix sum says it does.
        if let Some(last) = self.base.last_mut() {
            *last = self.next_id;
        }
        Ok(id)
    }

    fn next_id(&self) -> NodeId {
        self.next_id
    }

    fn distinct_nodes(&self) -> u64 {
        // Every allocation corresponds to a distinct key: output slots cannot
        // repeat under the dense invariant, and `force_create` de-duplicates.
        self.next_id
    }

    fn write_tsv(&self, w: &mut dyn Write) -> PgResult<()> {
        let mut extras: Vec<(i32, i32, NodeId)> = self
            .extra
            .iter()
            .map(|(k, v)| {
                let (t, o) = unpack(*k);
                (t, o, *v)
            })
            .collect();
        extras.sort_unstable();

        let mut buf: Vec<u8> = Vec::with_capacity(TSV_BUF + 64);
        let mut ei = 0usize;
        let ntx = self.base.len().saturating_sub(1);
        for i in 0..ntx {
            let tx = (self.min_tx + i as i64) as i32;
            let start = self.base[i];
            let n = self.outputs_at(i);
            for off in 0..n {
                let off_i = off as i32;
                while ei < extras.len() && (extras[ei].0, extras[ei].1) < (tx, off_i) {
                    push_row(&mut buf, extras[ei].0, extras[ei].1, extras[ei].2);
                    ei += 1;
                    flush_if_full(&mut buf, w)?;
                }
                push_row(&mut buf, tx, off_i, start + off);
                flush_if_full(&mut buf, w)?;
            }
        }
        while ei < extras.len() {
            push_row(&mut buf, extras[ei].0, extras[ei].1, extras[ei].2);
            ei += 1;
            flush_if_full(&mut buf, w)?;
        }
        if !buf.is_empty() {
            w.write_all(&buf)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// HashNodeMap
// ---------------------------------------------------------------------------

/// A multiply-shift hasher for `pack(txId, offset)` keys.
///
/// The keys carry high entropy in the upper 32 bits (the transaction id) and
/// almost none in the lower 32 (the output offset, virtually always < 10), so
/// the default SipHash roughly doubles the pass time while a single multiply
/// plus xor-shift mixes both words well enough for a power-of-two table.
#[derive(Default, Clone, Copy)]
struct MulShift {
    state: u64,
}

impl Hasher for MulShift {
    #[inline]
    fn finish(&self) -> u64 {
        self.state
    }
    #[inline]
    fn write_u64(&mut self, key: u64) {
        let mut s = self.state ^ key;
        s = s.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        self.state = s ^ (s >> 32);
    }
    fn write(&mut self, bytes: &[u8]) {
        debug_assert!(false, "MulShift is only used with u64 keys");
        for b in bytes {
            self.write_u64(*b as u64);
        }
    }
}

/// The `BuildHasher` used by [`HashNodeMap`] and by [`DenseNodeMap`]'s
/// force-created side table.
type BuildMulShift = BuildHasherDefault<MulShift>;

/// Literal port of the Java `HashMap<Long, Long>` node map.
///
/// Kept as the always-correct fallback for a dataset whose transaction ids are
/// not dense, and as the cross-check that proves [`DenseNodeMap`]: the unit
/// tests below and an integration test over `chunk_01.txt` drive both maps with
/// the same input and assert identical ids, counters and TSV output.
pub struct HashNodeMap {
    nodes: HashMap<u64, u64, BuildMulShift>,
    next_id: u64,
}

impl HashNodeMap {
    /// Creates an empty map.
    pub fn new() -> Self {
        HashNodeMap {
            nodes: HashMap::default(),
            next_id: 0,
        }
    }

    /// Creates a map pre-sized for `slots` output slots.
    pub fn with_capacity(slots: usize) -> Self {
        HashNodeMap {
            nodes: HashMap::with_capacity_and_hasher(slots, BuildMulShift::default()),
            next_id: 0,
        }
    }
}

impl Default for HashNodeMap {
    fn default() -> Self {
        Self::new()
    }
}

impl NodeMap for HashNodeMap {
    fn register_tx(
        &mut self,
        tx_id: i32,
        num_outputs: u32,
        _line: u64,
        out: &mut Vec<NodeId>,
    ) -> PgResult<()> {
        out.clear();
        out.reserve(num_outputs as usize);
        for offset in 0..num_outputs as i32 {
            // Java `getOrCreateId`, including the repeat case: a key that is
            // already present keeps its original id and `nextId` does not move.
            let next = &mut self.next_id;
            let id = *self.nodes.entry(pack(tx_id, offset)).or_insert_with(|| {
                let id = *next;
                *next += 1;
                id
            });
            out.push(id);
        }
        Ok(())
    }

    fn lookup(&self, tx_id: i32, offset: i32) -> Option<NodeId> {
        self.nodes.get(&pack(tx_id, offset)).copied()
    }

    fn force_create(&mut self, tx_id: i32, offset: i32) -> PgResult<NodeId> {
        let next = &mut self.next_id;
        let id = *self.nodes.entry(pack(tx_id, offset)).or_insert_with(|| {
            let id = *next;
            *next += 1;
            id
        });
        Ok(id)
    }

    fn next_id(&self) -> NodeId {
        self.next_id
    }

    fn distinct_nodes(&self) -> u64 {
        self.nodes.len() as u64
    }

    fn write_tsv(&self, w: &mut dyn Write) -> PgResult<()> {
        let mut rows: Vec<(i32, i32, NodeId)> = self
            .nodes
            .iter()
            .map(|(k, v)| {
                let (t, o) = unpack(*k);
                (t, o, *v)
            })
            .collect();
        rows.sort_unstable();

        let mut buf: Vec<u8> = Vec::with_capacity(TSV_BUF + 64);
        for (tx, off, id) in rows {
            push_row(&mut buf, tx, off, id);
            flush_if_full(&mut buf, w)?;
        }
        if !buf.is_empty() {
            w.write_all(&buf)?;
        }
        Ok(())
    }
}

/// Appends `txId\toffset\tid\n` to `buf`.
///
/// Uses `itoa` rather than `write!`/`format!`: at 2.21e9 lines the `std::fmt`
/// machinery is the difference between minutes and an hour.
#[inline]
fn push_row(buf: &mut Vec<u8>, tx: i32, offset: i32, id: NodeId) {
    let mut a = itoa::Buffer::new();
    buf.extend_from_slice(a.format(tx).as_bytes());
    buf.push(b'\t');
    let mut b = itoa::Buffer::new();
    buf.extend_from_slice(b.format(offset).as_bytes());
    buf.push(b'\t');
    let mut c = itoa::Buffer::new();
    buf.extend_from_slice(c.format(id).as_bytes());
    buf.push(b'\n');
}

/// Flushes `buf` into `w` once it has grown past [`TSV_BUF`].
#[inline]
fn flush_if_full(buf: &mut Vec<u8>, w: &mut dyn Write) -> PgResult<()> {
    if buf.len() >= TSV_BUF {
        w.write_all(buf)?;
        buf.clear();
    }
    Ok(())
}

/// Builds the node map selected by `kind`, pre-sized with `tx_capacity` when
/// one is given.
///
/// `tx_capacity` is a *transaction* count for [`NodeMapKind::Dense`] and is
/// used as an output-slot estimate for [`NodeMapKind::Hash`].
pub fn new_node_map(kind: NodeMapKind, mode: Mode, tx_capacity: Option<usize>) -> Box<dyn NodeMap> {
    match kind {
        NodeMapKind::Dense => match tx_capacity {
            Some(n) => Box::new(DenseNodeMap::with_capacity(mode, n)),
            None => Box::new(DenseNodeMap::new(mode)),
        },
        NodeMapKind::Hash => match tx_capacity {
            Some(n) => Box::new(HashNodeMap::with_capacity(n)),
            None => Box::new(HashNodeMap::new()),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn register(m: &mut dyn NodeMap, tx: i32, n: u32) -> Vec<NodeId> {
        let mut out = Vec::new();
        m.register_tx(tx, n, tx as u64 + 1, &mut out).unwrap();
        out
    }

    /// A single corrupt txId used to cost one `u64` of filler per skipped id,
    /// with no ceiling: a two-line, 79-byte input whose second txId is
    /// `2_147_483_647` drove RSS to 16 GiB and, under `ulimit -v`, *aborted*
    /// the process (exit 134) instead of returning an error. Java's `HashMap`
    /// inserted one entry.
    #[test]
    fn a_lenient_gap_beyond_the_ceiling_is_refused_rather_than_allocated() {
        let mut m = DenseNodeMap::new(Mode::Lenient);
        let mut out = Vec::new();
        m.register_tx(0, 1, 1, &mut out).expect("first tx");

        let err = m
            .register_tx(i32::MAX, 1, 2, &mut out)
            .expect_err("a 2.1e9-id gap must be refused");
        match err {
            PgError::NonDenseTxId {
                line,
                expected,
                found,
            } => {
                assert_eq!(line, 2);
                assert_eq!(expected, 1);
                assert_eq!(found, i32::MAX as i64);
            }
            other => panic!("expected NonDenseTxId, got {other:?}"),
        }

        // A gap inside the ceiling is still bridged, as before.
        let mut m = DenseNodeMap::new(Mode::Lenient);
        m.register_tx(0, 1, 1, &mut out).expect("first tx");
        m.register_tx(5, 2, 2, &mut out)
            .expect("small gap is filled");
        assert_eq!(out, vec![1, 2]);
        assert_eq!(m.lookup(0, 0), Some(0));
        assert_eq!(m.lookup(5, 1), Some(2));
        // The filler transactions have no outputs.
        assert_eq!(m.lookup(3, 0), None);
    }

    #[test]
    fn dense_assigns_contiguous_ids() {
        let mut m = DenseNodeMap::new(Mode::Strict);
        assert_eq!(register(&mut m, 0, 1), vec![0]);
        assert_eq!(register(&mut m, 1, 2), vec![1, 2]);
        assert_eq!(register(&mut m, 2, 0), Vec::<NodeId>::new());
        assert_eq!(register(&mut m, 3, 3), vec![3, 4, 5]);

        // bases: 0, 1, 3, 3
        assert_eq!(m.lookup(0, 0), Some(0));
        assert_eq!(m.lookup(1, 0), Some(1));
        assert_eq!(m.lookup(3, 0), Some(3));
        assert_eq!(m.next_id(), 6);
        assert_eq!(m.distinct_nodes(), 6);

        assert_eq!(m.lookup(1, 1), Some(2));
        assert_eq!(m.lookup(2, 0), None); // zero outputs
        assert_eq!(m.lookup(99, 0), None); // never registered
        assert_eq!(m.num_outputs(1), Some(2));
        assert_eq!(m.num_outputs(2), Some(0));
        assert_eq!(m.num_outputs(3), Some(3));
        assert_eq!(m.num_outputs(4), None);
    }

    #[test]
    fn dense_rejects_non_dense_tx_ids_in_strict_mode() {
        let mut m = DenseNodeMap::new(Mode::Strict);
        register(&mut m, 0, 1);
        let mut out = Vec::new();
        match m.register_tx(5, 1, 2, &mut out) {
            Err(PgError::NonDenseTxId {
                line: 2,
                expected: 1,
                found: 5,
            }) => {}
            other => panic!("expected NonDenseTxId, got {other:?}"),
        }
    }

    #[test]
    fn dense_fills_gaps_in_lenient_mode() {
        let mut m = DenseNodeMap::new(Mode::Lenient);
        register(&mut m, 0, 1);
        let ids = register(&mut m, 5, 1);
        assert_eq!(ids, vec![1]);
        assert_eq!(m.lookup(5, 0), Some(1));
        // The filler transactions exist with zero outputs.
        assert_eq!(m.num_outputs(3), Some(0));
        assert_eq!(m.lookup(3, 0), None);
        assert_eq!(m.next_id(), 2);
    }

    #[test]
    fn dense_backwards_tx_id_is_always_an_error() {
        let mut m = DenseNodeMap::new(Mode::Lenient);
        register(&mut m, 0, 1);
        register(&mut m, 1, 1);
        let mut out = Vec::new();
        assert!(m.register_tx(0, 1, 3, &mut out).is_err());
    }

    #[test]
    fn dense_force_create_keeps_the_prefix_sum_honest() {
        let mut m = DenseNodeMap::new(Mode::Strict);
        register(&mut m, 0, 2);
        let phantom = m.force_create(999, 7).unwrap();
        assert_eq!(phantom, 2);
        assert_eq!(m.lookup(999, 7), Some(2));
        // Re-creating the same key is idempotent.
        assert_eq!(m.force_create(999, 7).unwrap(), 2);
        // The next transaction still starts after the phantom, and the output
        // count of the previous one is unaffected.
        assert_eq!(register(&mut m, 1, 1), vec![3]);
        assert_eq!(m.num_outputs(0), Some(2));
        assert_eq!(m.num_outputs(1), Some(1));
        assert_eq!(m.lookup(0, 0), Some(0));
        assert_eq!(m.lookup(0, 1), Some(1));
        assert_eq!(m.lookup(1, 0), Some(3));
        assert_eq!(m.next_id(), 4);

        let mut tsv = Vec::new();
        m.write_tsv(&mut tsv).unwrap();
        assert_eq!(
            String::from_utf8(tsv).unwrap(),
            "0\t0\t0\n0\t1\t1\n1\t0\t3\n999\t7\t2\n"
        );
    }

    #[test]
    fn hash_repeat_registration_is_idempotent() {
        let mut m = HashNodeMap::new();
        let a = register(&mut m, 7, 2);
        let b = register(&mut m, 7, 2);
        assert_eq!(a, vec![0, 1]);
        assert_eq!(b, vec![0, 1]);
        assert_eq!(m.next_id(), 2);
        assert_eq!(m.distinct_nodes(), 2);
    }

    #[test]
    fn write_tsv_is_ascending_by_tx_and_offset() {
        let mut m = DenseNodeMap::new(Mode::Strict);
        register(&mut m, 0, 1);
        register(&mut m, 1, 2);
        register(&mut m, 2, 1);
        let mut tsv = Vec::new();
        m.write_tsv(&mut tsv).unwrap();
        assert_eq!(
            String::from_utf8(tsv).unwrap(),
            "0\t0\t0\n1\t0\t1\n1\t1\t2\n2\t0\t3\n"
        );
    }

    #[test]
    fn dense_and_hash_agree_on_a_long_random_run() {
        // A fixed LCG keeps this reproducible without a `rand` dependency.
        let mut state: u64 = 0x2545_F491_4F6C_DD1D;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };

        let mut dense = DenseNodeMap::new(Mode::Strict);
        let mut hash = HashNodeMap::new();
        let mut a = Vec::new();
        let mut b = Vec::new();
        let mut probes: Vec<(i32, i32)> = Vec::new();

        for tx in 0..5000i32 {
            let n = next() % 5; // 0..4 outputs, including zero-output txs
            dense.register_tx(tx, n, tx as u64 + 1, &mut a).unwrap();
            hash.register_tx(tx, n, tx as u64 + 1, &mut b).unwrap();
            assert_eq!(a, b, "output ids diverged at tx {tx}");
            if n > 0 {
                probes.push((tx, (next() % n) as i32));
            }
        }

        assert_eq!(dense.next_id(), hash.next_id());
        assert_eq!(dense.distinct_nodes(), hash.distinct_nodes());
        for (tx, off) in probes {
            assert_eq!(dense.lookup(tx, off), hash.lookup(tx, off));
            assert_eq!(dense.lookup(tx, off + 100), hash.lookup(tx, off + 100));
        }

        let mut da = Vec::new();
        let mut hb = Vec::new();
        dense.write_tsv(&mut da).unwrap();
        hash.write_tsv(&mut hb).unwrap();
        assert_eq!(da, hb);
    }
}
