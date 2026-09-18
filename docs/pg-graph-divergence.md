# Why `pg.graph` and `pg_28.graph` differ in a handful of bytes

**Short answer: they do not differ as graphs. They are the same graph, arc for arc.
The 397 bits that differ are the cost of compressing in parallel, and have nothing to
do with the BIP-30 duplicate transactions.**

This note records the investigation, because the question ("is this the four-transaction
fix?") is a reasonable one and the answer is worth being able to reproduce.

| | reference | this pipeline |
|---|---|---|
| path | `/data/bitcoin/2022/pg/pg.graph` | `/data/bitcoin/2022/utxo-spllitting-pipeline/graph/pg_28.graph` |
| built | 2026-06-26 10:02 | 2026-09-18 13:45 |
| `nodes=` | 2 181 021 971 | 2 181 021 971 |
| `arcs=` | 8 639 773 499 | 8 639 773 499 |
| `avgref=` / `avgdist=` | 0.088 / 0.169 | 0.088 / 0.169 |
| `length=` | 69 003 145 030 bits | 69 003 144 **633** bits |
| file size | 8 625 393 136 B | 8 625 393 080 B |

---

## 1. The graphs are identical

Both `.graph` files were decoded in lockstep with `webgraph-rs` and their successor
lists compared node by node — the whole graph, not a sample:

```
A nodes = 2181021971
B nodes = 2181021971
SUMMARY: scanned nodes 0..2181021971 (2181021971 nodes), diffs=0 (degree-differing=0)
arcs seen: A=8639773499 B=8639773499 (delta=0)
elapsed 224.7s
```

Zero nodes differ. Not in content, not in order, not in degree. The run was repeated
independently and reproduced digit for digit (211.0 s the second time).

So whatever the byte differences are, they are **the same arcs encoded differently** —
not a permutation of node ids, and not a different arc set.

## 2. Where the bits go

Decoding both `.offsets` streams and comparing each node's *encoded length*:

```
nodes compared             : 2181021971
nodes with different length: 73
  A longer                 : 41
  B longer                 : 32
net (sum A - sum B) bits   : 397
total bits A=69003145030 B=69003144633 diff=397
```

73 nodes out of 2.18 billion — 0.0000033% — and they account for the difference
**exactly**, reproducing both `length=` fields to the bit.

Those 73 nodes fall into 20 clusters. That is the clue.

## 3. The cause: 112-way parallel compression

`ceil(2 181 021 971 / 112) = 19 473 411`, and this machine has 112 cores. Every one of
the 20 clusters sits at a multiple of that chunk size:

```
 116840466 = 19473411*  6 + 0      1090511016 = 19473411* 56 + 0
 155787293 = 19473411*  8 + 5      1246298304 = 19473411* 64 + 0
 175260699 = 19473411*  9 + 0      1285245129 = 19473411* 66 + 3
 214207521 = 19473411* 11 + 0      1324191948 = 19473411* 68 + 0
 233680932 = 19473411* 12 + 0      1343665360 = 19473411* 69 + 1
 253154344 = 19473411* 13 + 1      1518926058 = 19473411* 78 + 0
 292101166 = 19473411* 15 + 1      1538399469 = 19473411* 79 + 0
 467361864 = 19473411* 24 + 0      1674713347 = 19473411* 86 + 1
 778936444 = 19473411* 40 + 4      1947341101 = 19473411*100 + 1
 915250318 = 19473411* 47 + 1      2161548622 = 19473411*111 + 1
```

**10 of 20 land exactly on a chunk boundary; all 20 land within 5 nodes of one.**

The mechanism is the one this crate already documents in `src/compress.rs`:

> Each parallel chunk restarts its compression window, so a node at a chunk boundary
> cannot reference a node before it.

BVGraph encodes most adjacency lists as "copy node *k* positions back, with these
differences". A node at the start of a parallel chunk has no visible predecessor, so it
must be written out in full. That costs bits at the boundary, and perturbs a few
neighbours afterwards as the copy chain re-establishes itself.

You can see it directly. Node 116 840 466 (chunk 6, offset 0) has the 26-successor list
`[116840657 .. 116840682]`, identical in both files:

* reference: **39 bits** — a full encode (outdegree, no reference, one interval)
* this pipeline: **13 bits** — `gamma(26) + gamma(delta=2) + gamma(0 blocks)`, i.e. copy
  the list from node 116 840 464, two positions back

The reference could not reach back two nodes there. That is precisely a chunk boundary.

The direction of the total confirms it: the chunked side pays, and the reference is the
side that is 397 bits **longer**.

### Why the first difference "heals" and the second does not

The first cluster (nodes 116 840 466 … 116 840 498) nets **exactly zero bits**:

```
+26 −2 −4 −24 +4 +26 −2 −4 −24 +4 +26 −2 −24  =  0
```

Because no bits are gained or lost, the bitstream re-aligns and the files become
byte-identical again 106 bytes later (at byte 407 719 779). The second site, node
155 787 293, nets **+16 bits** — from there the two streams are permanently out of
phase, which is why every byte after ~556 MB compares as different even though the
graphs are the same.

That is the whole explanation for "equal except for a bunch of bytes".

## 4. The reference was not built by the Java pipeline

An incidental but important finding: `/data/bitcoin/2022/pg/pg.graph` was **not**
produced by `PaymentGraphEdgeListBuilder.java` + `WebgraphBuilder.jar`. It was written by
`webgraph-rs`, on 2026-06-26.

* Its `.properties` carries `endianness=big` and a `length=` key, and has **no**
  `java.util.Properties.store()` date comment and none of the Java stat block
  (`bitsforblocks`, `successorexpstats`, `residualavgloggap`, …). Every Java-built
  sibling in the same family — `ag`, `tg`, `ug`, `atg` — carries all of those.
* The key order and float formatting match `webgraph-rs`'s `flags.rs::to_properties`
  only within the commit window 2026-04-20 … 2026-07-06, which brackets the file's date.
* `jar/WebgraphBuilder.jar` calls `BVGraph -g ArcListASCIIGraph`, which runs through
  `ArrayListMutableGraph` and cannot represent 2 181 021 971 > 2³¹ nodes. `pg` is the
  only graph in the family above 2³¹ — and the only one in the Rust format.
* The delivered `PaymentGraphEdgeListBuilder.java.old` (and the copy inside
  `pipeline.zip`, dated 2026-09-14) uses a 32-bit `int nextId`, which overflows at
  2 147 483 647 — below the 2 181 021 973 output slots this corpus creates. It cannot
  have produced a 2.18-billion-node graph.

So the comparison was never Rust-versus-Java. It was **sequential webgraph-rs versus
parallel webgraph-rs**, which is exactly the difference measured.

*Confidence:* that the reference is webgraph-rs output is certain. The specific revision
(`1e9cc347`, via `webgraph from arcs`) is likely but not proven. No run log from the
June build survives.

## 5. It is not the BIP-30 fix

The corpus was scanned in full — all 28 chunks, 778 613 440 lines,
135 287 658 091 bytes, nothing sampled:

| property | count |
|---|---|
| duplicate txIds | **2** (142572, 142726; 4 lines, all in `chunk_04`) |
| zero-output transactions | 0 |
| forward txId gaps | 0 |
| chunk-boundary gaps/overlaps | 0 |
| non-canonical integers (leading zero, sign, whitespace) | 0 |
| lines without exactly 3 `:`-sections | 0 |

All four BIP-30 lines are **coinbases with no inputs and exactly one output**, and each
duplicated pair is byte-identical in its output field. They therefore:

* emit **zero arcs**, and
* reuse the id their first occurrence was given, under *both* implementations — Java's
  `getOrCreateId` returns the existing key without advancing `nextId`; the Rust
  `register_repeat` returns the same base id and mints nothing.

Arithmetically: 2 181 021 973 output slots − 2 re-emitted slots = **2 181 021 971**
nodes, the number both `.properties` declare. And Σ(inputs × outputs) over the corpus =
**8 639 773 499**, also exactly the declared arc count — meaning `sort | uniq` removed
precisely zero duplicate arcs in either pipeline.

Finally, the two BIP-30 transactions map to node ids **172 980** and **173 166**. The
lowest differing node is 116 840 466 — about 116 million ids later. Their adjacency
lists are identical in both graphs, as is everything else.

The four-transaction fix costs the dense node map a special case (without it the run
would abort with `NonDenseTxId`). It changes nothing about the output.

## 6. Reproducing the reference byte-for-byte

Nothing needs fixing. If you want bit-identical output to a given graph, match the
compression mode:

```sh
# sequential (default): deterministic, reproducible, what produced pg_28.graph
utxo2webgraph build 28 --chunk-dir chunks

# parallel: faster, restarts the window at each of N chunk boundaries
utxo2webgraph build 28 --chunk-dir chunks --parallel
```

`--parallel`'s output depends on the thread count, so it is only reproducible at a fixed
`--threads`. The sequential path has no such dependency, which is why it is the default.

To confirm two graphs are the same graph regardless of encoding, compare them
semantically rather than with `cmp` — `webgraph::traits::graph::eq`, or the lockstep
successor-list walk used here.

## 7. Staying in sync with `PaymentGraphEdgeListBuilder.java`

On this corpus the Rust pipeline and the current Java source assign **the same id to
every node**. Three independent measurements agree:

1. Corpus arithmetic: 2 181 021 973 slots − 2 BIP-30 reuses = 2 181 021 971 nodes.
2. A full 49.4 GB pass over `pg_nm_28.tsv`: 2 181 021 971 rows, `id == row_index` on
   every row, rows strictly ascending in `(txId, offset)`, first `0 0 0`, last
   `778613437 0 2181021970`. Ids are exactly 0 … 2 181 021 970 in ascending
   `(txId, offset)` order — which *is* first-seen order here, because txIds arrive
   consecutively with no gaps.
3. A literal re-implementation of Java's `getOrCreateId` replayed over chunks 01–04 (the
   chunks holding the anomaly) produced a node map **byte-identical** to
   `head -266443 pg_nm_28.tsv` (md5 `276b1146ba212c52622cb654d84bf205`).

So **no change is required to stay in sync**. What follows is the list of inputs where
the two *would* diverge, so the divergences are chosen rather than discovered later.

| # | input pattern | Java | this pipeline | occurs in corpus |
|---|---|---|---|---|
| 1 | repeated txId, repeat has ≤ outputs (BIP-30) | reuses existing id | reuses existing id | **2×** — no-op |
| 2 | repeated txId, repeat has *more* outputs | extra offsets get fresh ids at current `nextId` | same, via `force_create` | 0 |
| 3 | forward txId gap | mints nothing | strict: `NonDenseTxId`; lenient: zero-output filler, also mints nothing | 0 |
| 4 | zero-output transaction | **emits a phantom arc into node 0** for every input | emits nothing | 0 |
| 5 | dangling `(prevTxId, prevOffset)` | `NullPointerException`, run dies | typed error naming line/tx/index | 0 |
| 6 | `prevOffset` ≥ that transaction's output count | `NullPointerException` | `lookup` bounds-checks, returns `None` → `--on-missing-source` | 0 |
| 7 | duplicate `(prevTxId, prevOffset)` in one tx | duplicate arcs, removed by `uniq` | removed by `dedup` | 0 |
| 8 | leading zeros / signed integer text | *old* source keyed on raw text, so `007` ≠ `7`; current source parses | parses | 0 |
| 9 | node map file order | `HashMap` iteration order | ascending `(txId, offset)` | always — same *set*, different byte order |

Rows 4, 5 and 6 are deliberate divergences: each replaces a crash or a provably wrong
arc with defined behaviour, and none is reachable on this corpus. Row 9 is a byte-order
difference only; compare node maps with
`sort -t$'\t' -k1,1n -k2,2n | md5sum`.

**Two things worth knowing about the delivered source:**

* `PaymentGraphEdgeListBuilder.java.old` — and the copy inside `pipeline.zip` — is the
  `Map<String,Integer>` / `int nextId` version. It keys nodes on the **raw text**
  `txId + ":" + offset`, so `007` and `7` would be different nodes, and its 32-bit
  `nextId` overflows at 2 147 483 647. Do not treat it as the reference; the current
  `PaymentGraphEdgeListBuilder.java` (`Map<Long,Long>`, packed key, `long nextId`) is
  the one this pipeline matches.
* Neither version produced `pg.graph` (§4).

---

## Appendix: how to re-run any of this

```sh
# 1. do the graphs differ as graphs?  (lockstep successor-list walk, ~4 min)
#    open both with BvGraphSeq and compare Vec<usize> successor lists per node.

# 2. where do the bitstreams differ?
cmp /data/bitcoin/2022/pg/pg.graph \
    /data/bitcoin/2022/utxo-spllitting-pipeline/graph/pg_28.graph
# -> differ: byte 407719674

# 3. which nodes cost different bits?  decode both .offsets gamma streams and
#    compare per-node encoded lengths; expect 73 nodes, net 397 bits.

# 4. do the clusters sit on chunk boundaries?
python3 -c "import math; print(math.ceil(2181021971/112))"   # 19473411
nproc                                                        # 112
```

The throwaway tools written for this (`graphdiff`, `offdiff`, `whichnode`,
`dumpnodes`) are not part of the crate; they are three short binaries over the same
pinned `webgraph` revision the crate already depends on.
