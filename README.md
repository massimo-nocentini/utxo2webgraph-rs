# utxo2webgraph (`pgraph`)

A Rust port of the Bitcoin Payment Graph pipeline: transaction list ->
node map + edge list -> BVGraph.

---

## 1. Attribution

The pipeline and the graph-construction algorithm are the work of
**Matteo Loporchio**:

* `PaymentGraphEdgeListBuilder.java` — the node-map and edge-list builder,
* `splitter.sh` — the six-month temporal splitter,
* `builder.sh` — the "first N chunks" driver,
* `build_pg.sh` — the three-stage pipeline script,
* `builder.GraphBuilder` inside `jar/WebgraphBuilder.jar` — the wrapper around
  `it.unimi.dsi.webgraph.BVGraph -g ArcListASCIIGraph`.

This crate is a **reimplementation of his design in Rust**. The algorithm is
his; nothing here claims otherwise, and every ported module repeats the
attribution in its module documentation.

BVGraph compression uses **`webgraph-rs`** by Tommaso Fontana and Sebastiano
Vigna, consumed from a local read-only checkout at
`/home/mnocentini/Developer/working-copies/webgraph-rs`.
**Never edit it, never run `cargo` inside it, never `cargo update` it.** Cargo
builds it into *our* `target/` directory; it never writes into the checkout.

---

## 2. What it replaces

| shell / Java | `pgraph` |
|---|---|
| `./splitter.sh finalBCUTXO_2022` | `pgraph split finalBCUTXO_2022` |
| `./builder.sh 15` | `pgraph build 15` |
| `./build_pg.sh IN NM EL PFX` | `pgraph build-pg -i IN --node-map NM --edge-list EL --output-prefix PFX` |
| `java PaymentGraphEdgeListBuilder IN NM EL` | `pgraph edge-list -i IN --node-map NM --arcs EL --arc-format tsv` |
| `sort -t$'\t' -k1,1n -k2,2n EL \| uniq > OUT` | `pgraph sort-edges EL --edge-list OUT` |
| `java -jar jar/WebgraphBuilder.jar EL PFX` | `pgraph compress EL PFX` |

`--num-nodes` defaults to `from-arcs` in `build`, `build-pg` and `compress`,
which is what `ArcListASCIIGraph` computed (`max(endpoint) + 1`) and therefore
what the shell pipeline produced — the rows above are drop-in replacements with
no extra flags. `--num-nodes node-map` is the opt-in richer graph; see
section 6.

Two of these commands read very large files. `pgraph` refuses any input above
`--max-input-bytes` (default 16 MiB) and refuses `finalBCUTXO_2022` outright
unless `--i-know-this-is-big` is passed, so the first line of the table is
really

```
pgraph split finalBCUTXO_2022 --i-know-this-is-big --max-input-bytes 200GiB
```

That guard is deliberate: it lets `chunks/chunk_01.txt` (976 KB) through while
making a many-hour run impossible to start by a typo.

---

## 3. Quick start

```bash
cd utxo2webgraph-rs
cargo build --release          # -> target/release/pgraph
./target/release/pgraph build 1
cargo test                     # the full differential suite against the Java reference
cargo test -- --ignored        # any test later marked heavy (none are today)
```

Every differential test runs in well under a second on `chunk_01.txt`, so none
of them is `#[ignore]`d; `cargo test` is the full suite. The tests skip
themselves with a printed notice when the corpus or the Java reference
directory is absent, so they stay green on a machine without the data.

Useful first commands:

```bash
pgraph split --show-boundaries                 # prints the 28-row boundary table and exits
pgraph build 1 --dry-run                       # resolved plan, sizes, estimates; touches nothing
pgraph -vv build 1                             # trace-level logging on stderr
```

---

## 4. Deliberate divergences from the Java behaviour

Every one of these is **unreachable on the current corpus** (verified over
`chunks/chunk_01..06` in full and over multi-million-line heads of `chunk_10`,
`chunk_20` and `chunk_28`), so none of them can perturb the differential test.

| Java | `pgraph` | why the change is safe |
|---|---|---|
| empty output section => edge into node 0 (lines 59-60, 76-77: `"".split(";")` has length 1, so `currentOutputNodeIds` stayed `{0}` and `0` is a valid node id) | zero outputs => zero edges | silent corruption; needs a >=4-section line, of which the corpus has none |
| blank or short line => `ArrayIndexOutOfBoundsException` (lines 54/55) | blank lines skipped and tallied; other malformations are a line-numbered error under `--strict`, a tallied skip under `--lenient` | an opaque multi-hour-run killer; zero such lines exist |
| dangling source => `NullPointerException` (line 74), both output files left at 0 bytes | typed error naming line, txId, input index and the missing `(prevTxId, prevOffset)`; `--on-missing-source skip\|create` | same default outcome, usable diagnostics; 0/1033 dangling refs in chunk_01 |
| `PrintWriter` swallows I/O errors (`checkError()` never called) | every write checked, `sync_all` before the statistics, temp file renamed on success | a disk-full silently truncated a 210 GB edge list and still exited 0 |
| node map in `HashMap` bucket order (lines 89-94) | ascending `(txId, offset)` | deterministic and streamable; the *set* of triples is identical |
| ids `long`, output slots reported as `Nodes:` (line 64) | `u64` everywhere; both numbers under `--stats extended`, Java's number under `--stats java` | 2.2e9 nodes at N=28, above `i32::MAX` |
| `builder.sh` materialises a 135 GB temp file | `ChunkChain` streams the chunks | pure waste, plus an orphan `combined_chunks_XXXXXX.txt` in the project directory after a crash |
| a non-UTF-8 byte decodes to U+FFFD and is never reported (`InputStreamReader`, line 41) | `--lenient` does exactly that and tallies it; the default reports the line and the byte offset | the corpus is pure ASCII (`chunk_01`: 0 bytes >= 0x80), and either way a stray byte no longer kills a 132 GB pass with a message that names no line |
| `HashMap<Long,Long>` accepts any txId sequence | the default `--node-map-kind dense` requires txIds to be consecutive from the first one, and refuses with `NonDenseTxId` otherwise | **verified safe for this corpus**: txIds run consecutively `0..778_613_437` across all 28 chunks, every boundary contiguous. It is fail-fast with an actionable remedy (`--node-map-kind hash` is a literal port of the Java map and reproduces its output exactly), never a silent divergence. A future re-ingest producing sparse txIds is not a Rust bug — use `hash`. |

`--strict --on-missing-source fail --stats java` (the defaults) is the
bug-for-bug reference mode, modulo the fixes above.

---

## 5. Output formats

**Node map** (`pg_nm_N.tsv`): `txId \t offset \t id`, TAB-separated, LF, no
header, in **ascending `(txId, offset)`** order. Java iterated
`java.util.HashMap.keySet()`, i.e. bucket order, which merely *looks* sorted
(over chunk_01's 18 620 rows the id column has 23 descents and the txId column
5). The set of triples is identical, the byte order is not, so compare with:

```bash
sort -t$'\t' -k1,1n -k2,2n pg_nm_1.tsv | md5sum
```

**Edge list** (`pg_el_N.tsv`): `{src}\t{dst}\n`, decimal, no padding, no sign,
no leading zeros, sorted ascending by `(src, dst)` and deduplicated. The
canonical formatting matters: `sort | uniq` compares text, so `007\t5` and
`7\t5` would both survive and then crash the compressor. `pgraph` deduplicates
on parsed integers, so it cannot happen.

The *raw* arc stream out of `edge-list --arc-format tsv` is byte-identical to
Java's `tmp/edge_list.tsv`: unsorted, undeduplicated, in emission order.

**BVGraph**: `<prefix>.graph`, `.offsets`, `.properties`, with
`windowsize=7, maxrefcount=3, minintervallength=4, zetak=3, compressionflags=`
(empty) — identical parameters to the Java `BVGraph` defaults, because
`webgraph-rs`'s `CompFlags::default()` is exactly that set.

`.properties` is **never byte-reproducible**: Java writes a `#<date>` timestamp
as its second line, sorts keys (JDK 9+), and emits a dozen cosmetic statistics
keys; `webgraph-rs` writes keys in insertion order, adds `endianness=` and
`length=`, and omits the cosmetic ones. None of that matters — only
`graphclass`, `version`, `nodes`, `arcs`, `windowsize`, `maxrefcount`,
`minintervallength`, `zetak` and `compressionflags` are ever read back.

Only `.graph` and `.offsets` are meaningful byte targets, and even there the
Rust files carry 4-8 trailing **zero** pad bytes, because `BufBitWriter` pads
to a 64-bit word while Java pads to a byte. Compare the first
`ceil(length/8)` bytes, or use `webgraph::traits::graph::eq`.

---

## 6. The node-count decision

`--num-nodes` picks between two genuinely different graphs:

* **`from-arcs` (the default)** — `max(id over sources AND targets) + 1`,
  exactly what Java's `ArcListASCIIGraph` inferred inside
  `WebgraphBuilder.jar`, and therefore what `build_pg.sh` produced. This is the
  bit-compatible setting: on `chunk_01` the whole 36 019-bit payload of
  `.graph`, and the whole `.offsets` bitstream, are **byte-for-byte identical**
  to Java's.
* **`node-map`** — `next_id`, i.e. every UTXO is a node, including the recent
  unspent outputs that appear in no arc. Semantically complete, and **larger**:
  on `chunk_01` the two counts are 18 545 and 18 620, the difference being 75
  isolated trailing nodes (`isolatedNodes` 17 512 versus 17 587 as reported by
  the Java reader). The arc sets are identical either way. Needs `--node-map`
  in the standalone `compress`.
* **`<N>`** — a literal count, rejected if it is smaller than the arcs require
  (webgraph would silently drop the out-of-range arcs).

Both candidate values are logged whenever they differ. The choice changes the
`.properties` `nodes=` value and every node-indexed downstream array, which is
why the default is the one that matches the pipeline being replaced.

Note also that at N=28 the **node-map** count (~2.23e9) **exceeds `i32::MAX`**,
so the 32-bit `it.unimi.dsi.webgraph` reader could never load that graph.
`pgraph build`/`build-pg` refuse such a run **up front**, from the input-size
estimate, rather than after the read pass, the sort, a 51 GiB node map and a
195 GiB edge list.

### What `.properties` does and does not contain

With the node count held equal, `.graph` and `.offsets` are bit-identical to
Java's, but the **file sizes and md5s never match**: webgraph-rs flushes its bit
writer to a 64-bit word boundary, so `.graph` is 4504 bytes against Java's 4503
(4504 = 563*8) and `.offsets` 7672 against 7667 (7672 = 959*8). The extra bytes
are zero padding past the last used bit, which no reader ever reaches. **Compare
with `cmp -n $(( (length + 7) / 8 ))`, or by dumping the arcs — never by md5.**

`.properties` is informational and differs in shape: webgraph-rs writes
`endianness` and `length`, which Java does not, and does not write Java's 20
compression-statistics keys (`bitsfor*`, `avgbitsfor*`, `copiedarcs`,
`residualarcs`, `intervalisedarcs`, `*avggap`, `*avgloggap`, `*expstats`) or its
leading `#<date>` comment. Those numbers are accumulated inside Java's
`BVGraph.storeInternal`; webgraph-rs's compressor does not expose them, so
reproducing them would mean reimplementing its bit accounting. **Verified
harmless for interop**: `it.unimi.dsi.webgraph.BVGraph -o -O -L` loads the Rust
graph, regenerates a byte-identical `.offsets`, and `ImmutableGraph.load`
dumps exactly the same arcs. Only a script that scrapes those statistics keys
would notice; the three shared float keys agree in value and differ only in
formatting (Java rounds to 3 decimals, Rust prints the full `f64`).

---

## 7. Scale and resources at N=28

| quantity | value |
|---|---|
| input (`chunks/chunk_01..28`) | 135.3 GB |
| nodes (`nextId`) | ~2.21e9 |
| raw arcs | ~9.5e9 |
| arcs as packed `u64` | ~76 GB |
| `pg_el_28.tsv` (text) | ~210 GB |
| `pg_nm_28.tsv` (text) | ~55 GB |
| dense node map (`Vec<u64>`, 778 613 438 entries) | **6.23 GB** |
| the Java `HashMap<Long, Long>` it replaces | 140-160 GB (this is what `-Xmx200g` was paying for) |

Run sizing:

```
arcs_per_run = memory / (2 * arc_size)
```

The factor **2** is the radix sort's scratch buffer, which is the same size as
the data being sorted. Without it a nominal 128 GiB budget really costs
256 GiB and OOMs on the first full-scale run.

With the default budget — `min(45% of RAM, 192 GiB)`, i.e. 192 GiB on this
503 GB machine — `arcs_per_run` is about 1.29e10, comfortably above the 9.5e9
arcs of the full N=28 run: **the whole arc set sorts in RAM in a single run and
never touches the disk.** The external merge machinery exists for smaller
`--memory` settings, for `--arc-codec wide`, and for future growth.

`/data` has 1.8 TB free and `--tmp-dir` defaults to `./tmp`, on the same
filesystem as everything else. `pgraph` computes a space estimate from the
input size and **refuses to start** rather than filling the volume.

---

## 8. Operating notes

* **`TMPDIR` is set from `--tmp-dir` at startup**, before any thread is
  spawned. webgraph's `ParSortIters`/`ParSortPairs` call bare
  `tempfile::tempdir()` and ignore `BvCompConf::tmp_dir` (which only controls
  the parallel compressor's partial bitstreams). Without this a parallel N=28
  run spills tens of GB into `/tmp` on the system disk.
* **Check `/proc/sys/vm/max_map_count`** before a multi-hour parallel run;
  `pgraph --parallel` warns when it looks too low for the batch count.
* **Sequential compression is the default**, because it is the
  byte-identical-to-Java path (verified: 0 differing bytes over the first
  `ceil(length/8)` bytes of `.graph` and `.offsets` on a 2 000-node,
  19 944-arc cross-check against `webgraph-3.6.10.jar`). `--parallel` is faster
  but each chunk restarts its compression window, so the bytes differ.
  Semantically identical: `graph::eq` passes.
* **Overflow checks are ON in the release profile.** The webgraph workspace
  disables them, but `Compressor::write` computes
  `residuals[i] - residuals[i-1] - 1`, which wraps and writes a corrupt
  bitstream on a duplicate successor. A few percent of throughput is cheap
  insurance against a corrupt 210 GB graph; `--profile fast` turns them off.
* **`panic = "abort"` is deliberately NOT set**, so `compress.rs` can
  `catch_unwind` webgraph's sortedness assertions and turn them into typed
  errors.
* `.properties` is written **last**. A `.graph` with no `.properties` means an
  interrupted run — `pgraph` writes to a temporary basename and renames only
  after `verify_graph` passes, so this state never reaches the final names.
* **`--verify` controls how the finished graph is read back**: `quick` (the
  default) checks that `.properties` exists and that the node count and the
  *declared* arc count match; `full` adds a complete second decompression that
  recounts every arc (at N=28 that is another pass over 9.5e9 arcs, several
  minutes); `none` skips it.
* **`--log-dir` really is a log directory.** Everything that goes to stderr is
  teed into `<log-dir>/<stage>.log`, and the two Java statistics lines (which
  go to stdout, byte-for-byte) are echoed there too, so a finished run leaves
  the artefacts `build_pg.sh` used to leave in `logs/`.
* **Spill directories are pid-keyed and reaped at startup.** A run killed by
  `SIGINT`/`SIGKILL` never runs `TempDir`'s destructor; the next `pgraph`
  removes any `tmp/pgraph-sort-<pid>-*` whose pid is no longer alive, and never
  touches one belonging to a live process. Partial *outputs* are removed by the
  sink's own `Drop`, so a failed run leaves no `.<name>.pgraph-tmp-<pid>`
  either.
* **Non-regular outputs work.** `--arcs /dev/null`, a fifo or a process
  substitution are written through directly and are *not* `fsync`ed —
  `fsync(2)` returns `EINVAL` on a character device or a pipe, which used to
  turn a completed run into exit 1. Real durability errors (`ENOSPC`, `EIO`) on
  a regular file are still fatal, which is the entire point of syncing.
* **A binary arc file's width comes from `--arc-codec`, never from its
  length.** `--input-format auto` sniffs text versus binary only: every
  multiple of 16 is also a multiple of 8, so a 128-bit (`wide`) file cannot be
  told apart from twice as many 64-bit ones. Pass the same `--arc-codec` that
  wrote the file.

---

## 9. Why the old pipeline never ran here

This is the motivation for the fail-fast design, and it is all visible in the
checkout:

* `logs/pg_el_builder.err`:
  `Error: Could not find or load main class PaymentGraphEdgeListBuilder` —
  there is no `.class` in the project directory and `build_pg.sh` invokes
  `java -Xmx200g PaymentGraphEdgeListBuilder` with no `-cp`.
* `build_pg.sh` has **no `set -e` and no `set -o pipefail`** (unlike
  `splitter.sh` and `builder.sh`, which both have `set -euo pipefail`), so it
  went on to sort an empty file. `(sort | uniq) > $EL_FILE` truncates the
  target *before* `sort` runs.
* `logs/webgraph_builder.err`:
  `IllegalArgumentException: Expected integer, found Token[EOF], line 1` — the
  BVGraph stage dying on the empty edge list, two stages downstream of the real
  failure.
* `graph/pg_el_1.tsv`, `pg_el_10.tsv`, `pg_el_15.tsv` are all **0 bytes**.

`pgraph` exits non-zero on the first error, writes every output to a temporary
path and renames it only on success, and prints its statistics to stdout so
`logs/` is non-empty on success and diagnosable on failure.

**`build_pg.sh` must not be modified** (hard rule). `pgraph build-pg`
supersedes it.

---

## 10. Validation recipe

The Java reference outputs are **in the crate**, at
`tests/data/javaref` (3.0 MB, 35 files); `PGRAPH_JAVAREF` overrides the
location for a larger out-of-tree corpus, and a missing in-tree fixture is a
hard test failure, never a skip.

```bash
REF=utxo2webgraph-rs/tests/data/javaref

# raw edge list: BYTE-identical
pgraph edge-list -i chunks/chunk_01.txt --node-map rust_nm_01.tsv \
                 --arcs rust_el_01.tsv --arc-format tsv
cmp rust_el_01.tsv $REF/el_01.tsv                                  # a3c31369f1a54bacfeb9b3cc2c953ebe

# node map: compared as a SET (HashMap order is not portable)
sort -t$'\t' -k1,1n -k2,2n rust_nm_01.tsv | md5sum                 # 4795d0ab4a84e775d16c7b57e26d2fc9

# stdout under --stats java (the seconds field will differ):
#   Processed: 18578 transactions (after 0 seconds).
#   Nodes: 18620<TAB>Edges: 1094

# chunks 1+2 (cat chunk_01.txt chunk_02.txt > c12.txt):
#   el 92439127b404b877889776a816e3602c, nm 2e9de9a09685f5dc3157f231053715af
#   Processed: 32705 ... / Nodes: 32777<TAB>Edges: 3628

# canonical (sort | uniq) edge list: 941a3876f734b9d75c36475d94266c66, 1094 lines

# failure parity: chunk_02 alone must exit 1, exactly as the Java NPE did
#   (the error is the dangling reference at line 59, NOT the output target:
#    `--arcs /dev/null` on chunk_01 exits 0 and prints the statistics)
pgraph edge-list -i chunks/chunk_02.txt --arcs /dev/null ; echo $?  # 1
pgraph edge-list -i chunks/chunk_01.txt --arcs /dev/null ; echo $?  # 0

# BVGraph parity, default flags (--num-nodes from-arcs):
#   java -jar jar/WebgraphBuilder.jar $REF/el_01.sorted.tsv jv_pg_01
#   pgraph compress $REF/el_01.sorted.tsv rs_pg_01
#   both report 18545 nodes / 1094 arcs / 36019 bits, and
#   cmp -n 4503 jv_pg_01.graph rs_pg_01.graph     -> identical
#   cmp -n 7667 jv_pg_01.offsets rs_pg_01.offsets -> identical
#   (the Rust files are a few zero-padding bytes longer; see section 6)
```

`cargo test` runs all of the above as `tests/golden.rs`, plus the eight-case
edge corpus in `$REF/edge/` and the dense-versus-hash node-map cross-check.

---

## 11. Future work

* **CSR counting sort.** The arcs are emitted in order of the *consuming*
  transaction and a UTXO is spent at most once, so for a fixed `src` the `dst`
  values are already strictly increasing; only the grouping by `src` is
  missing. That is a counting sort: O(E), no comparisons, ~62 GB total at N=28
  (`base` 6.23 GB + `off` 17.7 GB + `adj` 38.0 GB) against 76 GB + 76 GB of
  scratch for the radix sort.
  **Mandatory caveat:** it must *verify* that every bucket is strictly
  increasing after the scatter, in a linear pass, and sort any that is not —
  never assume the invariant. It was left out of v1 because the radix sort
  already handles the whole N=28 arc set in a single in-RAM run on this box, so
  the win is unmeasured.
* **Elias-Fano node map.** `base` is monotone with max ~2.21e9 over 778.6e6
  entries, so `sux::dict::EliasFano` would store it in ~3.5 bits/entry, i.e.
  ~340 MB instead of 6.23 GB. Not incrementally appendable during the build
  pass, and 6.23 GB out of 503 GB is free, so it is not worth doing today.

---

## 12. Hard rules for contributors

1. **Do not modify any existing file** in
   `/data/bitcoin/2022/utxo-spllitting-pipeline` — `builder.sh`, `build_pg.sh`,
   `splitter.sh`, the `.java` files, `jar/`, `chunks/`, `finalBCUTXO_2022`.
   Only create or edit files under `utxo2webgraph-rs/`.
2. **Never compile the `.java` in place.** Compile it into a scratch directory.
3. **Never run the pipeline on anything larger than `chunk_01.txt` (976 KB)**
   without a deliberate decision. `chunk_05.txt` (84 MB) is acceptable for a
   correctness cross-check, but not for timing claims. `finalBCUTXO_2022` is
   135 GB and is guarded in code.
4. **Never edit, and never run `cargo` inside, the `webgraph-rs` checkout.**
   Depend on it by path only.
