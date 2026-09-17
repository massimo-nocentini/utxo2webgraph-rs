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
//! [`DenseNodeMap`] is the one implementation. It exploits the near-density of
//! `txId` — which is the zero-based line number of the master transaction list
//! — to replace Java's `HashMap<Long, Long>` with a single prefix-sum
//! `Vec<u64>`, and it handles every input the corpus actually contains,
//! including the repeats described below. A literal `HashMap` port used to live
//! here as a second, flag-selected map; it was deleted once the dense map
//! subsumed it, and the unit test
//! `dense_matches_a_java_getorcreateid_oracle_on_a_long_random_run` plus the
//! `chunk_01`-wide differential in `tests/golden.rs` keep the cross-check.
//!
//! # BIP-30: a `txId` may legitimately repeat, and may go *backwards*
//!
//! Bitcoin blocks 91812/91842 and 91722/91880 contain **duplicate coinbase
//! transactions**: the very same transaction id appears in two different
//! blocks. That is permanent, immutable consensus history (it is what BIP-30
//! was written to forbid *afterwards*), and the master transaction list
//! faithfully re-emits the record. In `chunk_04` of the 2022 corpus this shows
//! up as two backwards `txId`s — `142726` re-emitted 56 lines after its first
//! occurrence, and `142572` re-emitted 267 lines later — inside an otherwise
//! perfectly consecutive `0..778_613_437`.
//!
//! Java's `getOrCreateId` handled this without noticing: a key already in the
//! map keeps its original id and `nextId` does not move. [`DenseNodeMap`] now
//! does the same (see [`DenseNodeMap::register_tx`]), in **both**
//! [`Mode::Strict`] and [`Mode::Lenient`] — a BIP-30 duplicate is legitimate
//! data, not corruption, so it must not require `--lenient`. Only a *forward*
//! `txId` gap is still an anomaly, and that is where [`Mode`] still bites.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::io::Write;

use crate::{pack, unpack, Mode, NodeId, PgError, PgResult};

/// Bytes buffered before flushing to the writer in [`DenseNodeMap::write_tsv`].
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
/// 28 chunks, modulo the two BIP-30 *backwards* repeats, which are not gaps) —
/// and anything larger is a corrupt file, not a gap. Such a line raises
/// [`PgError::NonDenseTxId`] rather than being bridged.
pub const MAX_TX_GAP: i64 = 1 << 20;

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
/// **The invariant is verified at runtime, never assumed.** A *forward* jump
/// is [`PgError::NonDenseTxId`] in [`Mode::Strict`], and a bridged gap in
/// [`Mode::Lenient`] (up to [`MAX_TX_GAP`]). A *backwards* `txId` is not an
/// error at all: it is the BIP-30 duplicate-coinbase case described in the
/// module documentation, and [`DenseNodeMap::register_tx`] reuses the ids the
/// first occurrence was given, exactly as Java's `getOrCreateId` did.
///
/// # The contract it obeys
///
/// Taken from the Java:
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
    /// Nodes minted by [`DenseNodeMap::force_create`], or by a repeat that
    /// declared more outputs than the first occurrence did, keyed by [`pack`].
    /// Empty on every path except `--on-missing-source create`.
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

    /// Creates an empty dense map, pre-sized when the caller knows how many
    /// transactions to expect (`--max-tx-id`) and growing geometrically when
    /// it does not.
    pub fn sized(mode: Mode, txs: Option<usize>) -> Self {
        match txs {
            Some(n) => Self::with_capacity(mode, n),
            None => Self::new(mode),
        }
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

impl DenseNodeMap {
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
    /// A `tx_id` *below* the dense cursor is a repeat (BIP-30; see the module
    /// documentation) and is delegated to `register_repeat`,
    /// which mints nothing for the offsets the first occurrence already owns.
    ///
    /// `line` is only used to build [`PgError::NonDenseTxId`].
    pub fn register_tx(
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
            if found < expected {
                // The dense cursor has already passed this txId: it is a
                // REPEAT, not corruption — see the module docs on BIP-30. Java
                // silently kept the original ids, and so do we, in both modes.
                return self.register_repeat(tx_id, num_outputs, line, out);
            }
            if found != expected {
                if self.mode == Mode::Strict {
                    return Err(PgError::NonDenseTxId {
                        line,
                        expected,
                        found,
                    });
                }
                let gap = found - expected;
                if gap > MAX_TX_GAP {
                    // Bridging this would allocate `8 * gap` bytes of filler
                    // for one suspicious line. That is a corrupt file, not a
                    // gap: refuse rather than allocate. A *different* error
                    // from the strict-mode one above, because the operator is
                    // already running with `--lenient` and must not be told to
                    // re-run with it.
                    return Err(PgError::TxIdGapTooLarge {
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
                         allocated; the input's transaction ids are not consecutive",
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

    /// Re-registers a transaction the dense cursor has already passed.
    ///
    /// This is Java's `getOrCreateId` on a key that is already in the map: the
    /// original id comes back and `nextId` does **not** move. Two things follow
    /// from the layout, and both matter:
    ///
    /// * the offsets the first occurrence already owns — `0 .. k0`, where `k0`
    ///   is its output count read straight off the prefix sum — are answered
    ///   from `base` with **no allocation, no `extra` insert and no phantom**.
    ///   That is the entire BIP-30 case (`k0 == num_outputs == 1`), so it costs
    ///   nothing and, critically, leaves `phantom_at` empty: a single phantom
    ///   would permanently disable the [`DenseNodeMap::outputs_at`] fast path
    ///   and put two `partition_point` searches on every one of ~5.8e9
    ///   [`DenseNodeMap::lookup`] calls.
    /// * offsets `k0 ..` — only reachable when a repeat declares *more* outputs
    ///   than the original did, which the corpus never does — were never in the
    ///   Java map either, so they are minted through
    ///   [`DenseNodeMap::force_create`] into `extra`. That cannot alias a later
    ///   dense mint, because this path is only entered for a `tx_id` strictly
    ///   *behind* the cursor, which never revisits it; a txId *below* `min_tx`
    ///   takes the same route for every offset, for the same reason. (The
    ///   forward case is not this function's to worry about — `force_create`
    ///   refuses it, see its own documentation.)
    ///
    /// `next_id` and `base` are untouched except by `force_create` itself.
    fn register_repeat(
        &mut self,
        tx_id: i32,
        num_outputs: u32,
        line: u64,
        out: &mut Vec<NodeId>,
    ) -> PgResult<()> {
        out.reserve(num_outputs as usize);
        // `outputs_at`, not the raw `base[i+1] - base[i]`: force-created nodes
        // sit at the tail of a transaction's id interval, and they are not its
        // outputs. The two agree whenever `phantom_at` is empty, which is every
        // run that does not use `--on-missing-source create`.
        let (start, k0) = match self.index_of(tx_id) {
            Some(i) => (self.base[i], self.outputs_at(i)),
            // Below the rebase floor `min_tx`, so nothing to reuse.
            None => (0, 0),
        };
        let reused = k0.min(num_outputs as u64);
        for k in 0..reused {
            out.push(start + k);
        }
        for offset in reused..num_outputs as u64 {
            let id = self.force_create(tx_id, offset as i32, line)?;
            out.push(id);
        }
        Ok(())
    }

    /// Resolves a previously registered output. **Never inserts.**
    ///
    /// `None` is the dangling reference that made Java throw at line 74. A
    /// `tx_id` that appeared twice resolves to the ids of its **first**
    /// occurrence, because the dense hit is preferred over `extra`.
    pub fn lookup(&self, tx_id: i32, offset: i32) -> Option<NodeId> {
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

    /// Mints a node for an unregistered `(txId, offset)` that lies **behind**
    /// the dense cursor.
    ///
    /// Used by [`crate::OnMissingSource::Create`] and by
    /// `register_repeat`'s overflow path. Calling it from
    /// `OnMissingSource::Create` **changes the id space**, and therefore every
    /// id in both output files: a run that used it is not comparable with a
    /// Java reference run, and the node map it produces describes a different
    /// graph labelling. Repeated calls with the same key return the same id.
    ///
    /// # Why `tx_id` must be behind the cursor
    ///
    /// A forced node lives in `extra`; a dense node lives in the prefix sum.
    /// [`DenseNodeMap::register_tx`]'s mint path derives its ids from
    /// `next_id`/`base.len()` alone and never probes `extra` — deliberately,
    /// because probing a hash map once per transaction is pure cost on a path
    /// that runs ~7.8e8 times and, on every correct input, would never hit.
    /// The price of that is a rule: a key may only be forced into `extra` if
    /// the cursor can never reach it, i.e. if `tx_id` already has a dense slot
    /// (or sits below the rebase floor `min_tx`). Force a key *ahead* of the
    /// cursor and the cursor will later mint a **second** id for it —
    /// [`DenseNodeMap::lookup`] would then prefer the dense one, `write_tsv`
    /// would emit the key twice and out of order, and `distinct_nodes` would
    /// over-count. So a forward `tx_id` is [`PgError::ForwardForcedNode`],
    /// never a silent second id. `register_repeat` only ever calls this for a
    /// `tx_id` strictly below the cursor, so it can never trip the check.
    pub fn force_create(&mut self, tx_id: i32, offset: i32, line: u64) -> PgResult<NodeId> {
        // The highest txId that already owns a dense slot. `base.len() - 1` is
        // the transaction count, so this is `min_tx - 1` (i.e. "nothing") for
        // an empty map, where the very first `register_tx` is still free to
        // plant `min_tx` anywhere.
        let reached = self.min_tx + self.base.len().saturating_sub(1) as i64 - 1;
        if self.base.is_empty() || tx_id as i64 > reached {
            return Err(PgError::ForwardForcedNode {
                line,
                tx_id,
                offset,
                reached,
            });
        }
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

    /// Java's `nextId`: one past the highest id handed out so far.
    pub fn next_id(&self) -> NodeId {
        self.next_id
    }

    /// Java's `nodes.size()`: the number of distinct `(txId, offset)` keys.
    pub fn distinct_nodes(&self) -> u64 {
        // Every id ever handed out corresponds to a distinct key. Three
        // mint sites, and none of them can allocate twice for one key:
        //   * `register_tx`'s dense path mints `base[i]..base[i+1]` once per
        //     `i`, and `i` is only ever appended to — the cursor never goes
        //     back over an index it has already written;
        //   * `register_repeat` mints nothing for the offsets that index
        //     already owns (it hands back the ids it finds there), so a
        //     repeated `(txId, offset)` adds no id at all — which is exactly
        //     Java's `getOrCreateId` leaving `nextId` alone;
        //   * `force_create` de-duplicates against everything already minted,
        //     through `lookup`, and — this is the part that makes the dense
        //     path's refusal to probe `extra` sound — refuses outright any
        //     `txId` the cursor has not yet passed
        //     (`PgError::ForwardForcedNode`). So a forced key is always one
        //     the dense path is finished with and will never mint again.
        // Gap filler transactions have zero outputs and mint nothing.
        self.next_id
    }

    /// Writes the node map as `txId\toffset\tid`, LF-terminated, no header, in
    /// **ascending `(txId, offset)`** order, each key exactly once.
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
    ///
    /// A repeated `txId` contributes no extra row: the ids were emitted for its
    /// first occurrence, and only genuine overflow offsets live in `extra`.
    /// The "each key exactly once" half of that guarantee rests on
    /// [`DenseNodeMap::force_create`] refusing a `txId` ahead of the dense
    /// cursor: an `extra` key is therefore either below `min_tx` or an offset
    /// past the end of a transaction's own run, and in both cases the merge
    /// below interleaves it without ever colliding with a dense row.
    pub fn write_tsv(&self, w: &mut dyn Write) -> PgResult<()> {
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
// Hashing for the side table
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

/// The `BuildHasher` used by [`DenseNodeMap`]'s force-created side table.
type BuildMulShift = BuildHasherDefault<MulShift>;

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

#[cfg(test)]
mod tests {
    use super::*;

    fn register(m: &mut DenseNodeMap, tx: i32, n: u32) -> Vec<NodeId> {
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
        let text = err.to_string();
        match err {
            // Deliberately not `NonDenseTxId`: that one's remedy is
            // `--lenient`, which this caller is already using.
            PgError::TxIdGapTooLarge {
                line,
                expected,
                found,
            } => {
                assert_eq!(line, 2);
                assert_eq!(expected, 1);
                assert_eq!(found, i32::MAX as i64);
                assert!(
                    !text.contains("re-run with --lenient"),
                    "must not advise the flag the operator already passed: {text}"
                );
            }
            other => panic!("expected TxIdGapTooLarge, got {other:?}"),
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

    /// Was `dense_backwards_tx_id_is_always_an_error`. A backwards txId is no
    /// longer an error in either mode: BIP-30 makes it legitimate Bitcoin
    /// history, and Java's `getOrCreateId` accepted it silently. The test now
    /// asserts the id REUSE that replaced the refusal.
    #[test]
    fn dense_backwards_tx_id_reuses_the_original_ids() {
        for mode in [Mode::Strict, Mode::Lenient] {
            let mut m = DenseNodeMap::new(mode);
            assert_eq!(register(&mut m, 0, 1), vec![0]);
            assert_eq!(register(&mut m, 1, 1), vec![1]);
            let mut out = Vec::new();
            m.register_tx(0, 1, 3, &mut out)
                .unwrap_or_else(|e| panic!("{mode:?}: backwards txId must be accepted: {e}"));
            assert_eq!(out, vec![0], "{mode:?}: the original id comes back");
            assert_eq!(m.next_id(), 2, "{mode:?}: nextId must not move");
            assert_eq!(m.distinct_nodes(), 2);
        }
    }

    /// The real shape of the `chunk_04` anomaly: txId 142726 is re-emitted 56
    /// lines after its first occurrence, with one output both times (Bitcoin
    /// blocks 91812 and 91842 contain the same coinbase transaction). Scaled
    /// down to tx 100 repeated after tx 160.
    #[test]
    fn bip30_duplicate_tx_id_reuses_original_ids() {
        for mode in [Mode::Strict, Mode::Lenient] {
            let mut m = DenseNodeMap::new(mode);
            let first = register(&mut m, 100, 1);
            assert_eq!(first, vec![0]);
            for tx in 101..=160 {
                register(&mut m, tx, 2);
            }
            let next_id_before = m.next_id();
            let distinct_before = m.distinct_nodes();

            let again = register(&mut m, 100, 1);
            assert_eq!(
                again, first,
                "{mode:?}: the duplicate keeps the original id"
            );
            assert_eq!(
                m.next_id(),
                next_id_before,
                "{mode:?}: nextId must not move"
            );
            assert_eq!(
                m.distinct_nodes(),
                distinct_before,
                "{mode:?}: no new node was created"
            );
            assert_eq!(m.lookup(100, 0), Some(0), "{mode:?}: lookup sees the first");
            // The cheap path: no phantom, so `outputs_at` stays on its fast
            // branch and `lookup` does not pay two binary searches per call.
            assert!(
                m.phantom_at.is_empty(),
                "{mode:?}: a BIP-30 duplicate must not create a phantom"
            );
            assert!(m.extra.is_empty(), "{mode:?}: and must not touch `extra`");
            assert_eq!(m.num_outputs(100), Some(1));

            // One row per distinct key, still ascending.
            let mut tsv = Vec::new();
            m.write_tsv(&mut tsv).unwrap();
            let text = String::from_utf8(tsv).unwrap();
            let rows: Vec<&str> = text.lines().collect();
            assert_eq!(rows.len(), m.distinct_nodes() as usize);
            assert_eq!(rows[0], "100\t0\t0");
            let keys: Vec<(i64, i64)> = rows
                .iter()
                .map(|l| {
                    let mut f = l.split('\t');
                    (
                        f.next().unwrap().parse().unwrap(),
                        f.next().unwrap().parse().unwrap(),
                    )
                })
                .collect();
            let mut sorted = keys.clone();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(keys, sorted, "{mode:?}: ascending and duplicate-free");
        }
    }

    /// A repeat that declares MORE outputs than the first occurrence did. The
    /// corpus never does this, but the id space must stay sound if it ever
    /// happens: offsets the original owns come back unchanged, the surplus
    /// offsets are keys Java would also have minted, and nothing aliases.
    #[test]
    fn repeat_with_more_outputs_mints_only_the_overflow() {
        for mode in [Mode::Strict, Mode::Lenient] {
            let mut m = DenseNodeMap::new(mode);
            let first = register(&mut m, 100, 2);
            assert_eq!(first, vec![0, 1]);
            for tx in 101..=110 {
                register(&mut m, tx, 1);
            }
            let next_id_before = m.next_id();
            let distinct_before = m.distinct_nodes();

            let again = register(&mut m, 100, 4);
            assert_eq!(&again[..2], &first[..], "{mode:?}: offsets 0,1 are reused");
            assert_eq!(
                &again[2..],
                &[next_id_before, next_id_before + 1],
                "{mode:?}: offsets 2,3 are freshly minted"
            );
            assert_eq!(m.next_id(), next_id_before + 2);
            assert_eq!(
                m.distinct_nodes(),
                distinct_before + 2,
                "{mode:?}: exactly two new nodes"
            );
            assert_eq!(m.lookup(100, 0), Some(0));
            assert_eq!(m.lookup(100, 1), Some(1));
            assert_eq!(m.lookup(100, 2), Some(next_id_before));
            assert_eq!(m.lookup(100, 3), Some(next_id_before + 1));

            // Still one ascending row per distinct key, overflow included.
            let mut tsv = Vec::new();
            m.write_tsv(&mut tsv).unwrap();
            let text = String::from_utf8(tsv).unwrap();
            let mut keys: Vec<(i64, i64)> = Vec::new();
            for l in text.lines() {
                let mut f = l.split('\t');
                keys.push((
                    f.next().unwrap().parse().unwrap(),
                    f.next().unwrap().parse().unwrap(),
                ));
            }
            assert_eq!(keys.len(), m.distinct_nodes() as usize);
            let mut sorted = keys.clone();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(keys, sorted, "{mode:?}: ascending and duplicate-free");
            assert!(keys.contains(&(100, 2)) && keys.contains(&(100, 3)));
        }
    }

    /// A repeat of a txId that fell BELOW the rebase floor `min_tx`: every
    /// offset is minted through `force_create`, because there is no dense slot
    /// to reuse. This is `chunk_04`'s second anomaly, txId 142572 re-emitted
    /// 267 lines later, when the run starts above it.
    #[test]
    fn repeat_below_the_rebase_floor_is_force_created() {
        let mut m = DenseNodeMap::new(Mode::Strict);
        register(&mut m, 100, 1); // min_tx = 100
        register(&mut m, 101, 1);
        let ids = register(&mut m, 50, 1);
        assert_eq!(ids, vec![2], "a fresh id, since 50 has no dense slot");
        assert_eq!(m.lookup(50, 0), Some(2));
        assert_eq!(m.distinct_nodes(), 3);
        // Idempotent: a third sighting reuses the force-created id.
        assert_eq!(register(&mut m, 50, 1), vec![2]);
        assert_eq!(m.distinct_nodes(), 3);
    }

    /// A phantom minted for a key *behind* the cursor — the reachable
    /// `--on-missing-source create` case, a chunk that starts above genesis and
    /// refers back to a transaction it does not contain — must not disturb the
    /// prefix sum: the next transaction still starts after it, and the previous
    /// transaction's output count still excludes it.
    #[test]
    fn dense_force_create_keeps_the_prefix_sum_honest() {
        let mut m = DenseNodeMap::new(Mode::Strict);
        register(&mut m, 10, 2); // min_tx = 10
        let phantom = m.force_create(5, 7, 1).unwrap();
        assert_eq!(phantom, 2);
        assert_eq!(m.lookup(5, 7), Some(2));
        // Re-creating the same key is idempotent.
        assert_eq!(m.force_create(5, 7, 1).unwrap(), 2);
        // The next transaction still starts after the phantom, and the output
        // count of the previous one is unaffected.
        assert_eq!(register(&mut m, 11, 1), vec![3]);
        assert_eq!(m.num_outputs(10), Some(2));
        assert_eq!(m.num_outputs(11), Some(1));
        assert_eq!(m.lookup(10, 0), Some(0));
        assert_eq!(m.lookup(10, 1), Some(1));
        assert_eq!(m.lookup(11, 0), Some(3));
        assert_eq!(m.next_id(), 4);

        let mut tsv = Vec::new();
        m.write_tsv(&mut tsv).unwrap();
        assert_eq!(
            String::from_utf8(tsv).unwrap(),
            "5\t7\t2\n10\t0\t0\n10\t1\t1\n11\t0\t3\n"
        );
    }

    /// The aliasing this refusal exists to prevent, spelled out.
    ///
    /// `force_create` writes only `extra`; `register_tx`'s dense path reads
    /// only the prefix sum. Minting a key the cursor has not reached yet used
    /// to succeed, and the cursor then minted a **second** id for the very same
    /// `(txId, offset)`: two node-map rows for one output, emitted out of
    /// ascending order, `distinct_nodes()` over-counting, and `lookup`
    /// silently switching the node's identity mid-run. It is now a typed
    /// error, so neither `extra` nor the id space can ever hold the duplicate.
    #[test]
    fn force_create_refuses_a_txid_the_cursor_has_not_reached() {
        let mut m = DenseNodeMap::new(Mode::Strict);
        register(&mut m, 0, 1); // cursor has reached txId 0; txId 1 is next

        for forward in [1, 2, 999] {
            match m.force_create(forward, 0, 7) {
                Err(PgError::ForwardForcedNode {
                    line: 7,
                    tx_id,
                    offset: 0,
                    reached: 0,
                }) => assert_eq!(tx_id, forward),
                other => panic!("expected ForwardForcedNode for {forward}, got {other:?}"),
            }
        }

        // The refusal is total: nothing was minted, nothing was recorded, and
        // the cursor is free to mint those transactions itself, once each.
        assert_eq!(m.next_id(), 1);
        assert!(m.extra.is_empty() && m.phantom_at.is_empty());
        assert_eq!(register(&mut m, 1, 1), vec![1]);
        assert_eq!(m.lookup(1, 0), Some(1));
        assert_eq!(m.distinct_nodes(), 2);

        let mut tsv = Vec::new();
        m.write_tsv(&mut tsv).unwrap();
        assert_eq!(String::from_utf8(tsv).unwrap(), "0\t0\t0\n1\t0\t1\n");
    }

    /// The same refusal on an empty map, where `min_tx` is not yet pinned: the
    /// first `register_tx` may still plant the floor anywhere, so *every* key
    /// is ahead of the cursor and none may be forced.
    #[test]
    fn force_create_refuses_everything_before_the_first_transaction() {
        let mut m = DenseNodeMap::new(Mode::Strict);
        match m.force_create(0, 0, 1) {
            Err(PgError::ForwardForcedNode { reached: -1, .. }) => {}
            other => panic!("expected ForwardForcedNode, got {other:?}"),
        }
        assert_eq!(m.next_id(), 0);
        assert_eq!(register(&mut m, 0, 1), vec![0]);
    }

    /// Was `hash_repeat_registration_is_idempotent`, over the deleted
    /// hash-backed map. The assertions are unchanged; only the map is.
    #[test]
    fn repeat_registration_is_idempotent() {
        let mut m = DenseNodeMap::new(Mode::Strict);
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

    /// Java's `getOrCreateId`, transcribed (`PaymentGraphEdgeListBuilder.java`
    /// lines 18-22). The oracle the deleted hash-backed map used to be.
    #[derive(Default)]
    struct JavaOracle {
        nodes: HashMap<u64, u64>,
        next_id: u64,
    }

    impl JavaOracle {
        fn get_or_create(&mut self, tx: i32, offset: i32) -> u64 {
            let next = &mut self.next_id;
            *self.nodes.entry(pack(tx, offset)).or_insert_with(|| {
                let id = *next;
                *next += 1;
                id
            })
        }

        fn register(&mut self, tx: i32, n: u32, out: &mut Vec<NodeId>) {
            out.clear();
            for offset in 0..n as i32 {
                let id = self.get_or_create(tx, offset);
                out.push(id);
            }
        }

        fn lookup(&self, tx: i32, offset: i32) -> Option<NodeId> {
            self.nodes.get(&pack(tx, offset)).copied()
        }

        fn tsv(&self) -> String {
            let mut rows: Vec<(i32, i32, u64)> = self
                .nodes
                .iter()
                .map(|(k, v)| {
                    let (t, o) = unpack(*k);
                    (t, o, *v)
                })
                .collect();
            rows.sort_unstable();
            rows.iter()
                .map(|(t, o, i)| format!("{t}\t{o}\t{i}\n"))
                .collect()
        }
    }

    /// Was `dense_and_hash_agree_on_a_long_random_run`. The hash-backed map is
    /// gone, so the counterpart is now the inline `JavaOracle` above — the same
    /// cross-check against the same reference semantics, with the repeated and
    /// backwards txIds that BIP-30 forced us to support folded into the run.
    #[test]
    fn dense_matches_a_java_getorcreateid_oracle_on_a_long_random_run() {
        // A fixed LCG keeps this reproducible without a `rand` dependency.
        let mut state: u64 = 0x2545_F491_4F6C_DD1D;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };

        let mut dense = DenseNodeMap::new(Mode::Strict);
        let mut oracle = JavaOracle::default();
        let mut a = Vec::new();
        let mut b = Vec::new();
        let mut probes: Vec<(i32, i32)> = Vec::new();
        let mut counts: Vec<u32> = Vec::new();

        for tx in 0..5000i32 {
            let n = next() % 5; // 0..4 outputs, including zero-output txs
            counts.push(n);
            dense.register_tx(tx, n, tx as u64 + 1, &mut a).unwrap();
            oracle.register(tx, n, &mut b);
            assert_eq!(a, b, "output ids diverged at tx {tx}");
            if n > 0 {
                probes.push((tx, (next() % n) as i32));
            }
            // Every 250th transaction, re-emit an earlier one with the SAME
            // output count, the way BIP-30 re-emits a duplicate coinbase.
            if tx > 0 && tx % 250 == 0 {
                let back = tx - (next() % tx as u32) as i32 - 1;
                let back = back.max(0);
                let bn = counts[back as usize];
                dense
                    .register_tx(back, bn, tx as u64 + 1, &mut a)
                    .unwrap_or_else(|e| panic!("repeat of tx {back} at {tx} failed: {e}"));
                oracle.register(back, bn, &mut b);
                assert_eq!(a, b, "repeat of tx {back} diverged at tx {tx}");
            }
        }

        assert_eq!(dense.next_id(), oracle.next_id);
        assert_eq!(dense.distinct_nodes(), oracle.nodes.len() as u64);
        for (tx, off) in probes {
            assert_eq!(dense.lookup(tx, off), oracle.lookup(tx, off));
            assert_eq!(dense.lookup(tx, off + 100), oracle.lookup(tx, off + 100));
        }

        let mut da = Vec::new();
        dense.write_tsv(&mut da).unwrap();
        assert_eq!(String::from_utf8(da).unwrap(), oracle.tsv());
    }
}
