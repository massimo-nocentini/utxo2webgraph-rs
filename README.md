# utxo2webgraph

A Rust tool that turns a Bitcoin transaction list into a compressed WebGraph:
transaction list -> node map + edge list -> BVGraph.

A node is one transaction *output* — one UTXO — identified by `(txId, offset)`.
An arc runs from a spent output to every output of the transaction that spent
it.

---

## 1. Attribution

The pipeline and the graph-construction algorithm are the work of
**Matteo Loporchio**: the six-month temporal splitter, the node-map and
edge-list construction, the "first N chunks" driver, the three-stage
build/sort/compress pipeline, and the BVGraph wrapper around it.

This crate is an **independent reimplementation of his design in Rust**. The
algorithm is his; nothing here claims otherwise, and every module that
reimplements a piece of it repeats the attribution in its module documentation.

BVGraph compression uses **`webgraph-rs`** by Tommaso Fontana and Sebastiano
Vigna, taken straight from upstream and pinned to the exact commit this crate
was verified against:

```toml
webgraph = { git = "https://github.com/vigna/webgraph-rs.git", rev = "f8698a7bdda2c4e171017548307179cd5c7a3166" }
```

That is `vigna/webgraph-rs` `main` @ `f8698a7` (crate version 0.6.1). The rev is
pinned deliberately: the golden suite pins this crate's outputs against recorded
reference fixtures and reads the compressed graph back arc for arc, so an
unpinned upstream bump could change the compressed bytes under the tests.
`ssh://git@github.com/vigna/webgraph-rs.git` is the same repository if you would
rather authenticate over SSH.

---

## 2. Subcommand reference

| subcommand | what it does | key flags |
|---|---|---|
| `split` | cuts a master transaction list into six-month `chunk_NN.txt` files | `-o/--output-dir`, `--start`, `--end`, `--months`, `--tz`, `--append`, `--show-boundaries` |
| `edge-list` | reads transactions, writes the node map and the **raw, unsorted** arc stream | `-i/--input` \| `--chunk-dir -n`, `--node-map`, `--arcs`, `--arc-format tsv\|binary`, `--on-missing-source`, `--max-tx-id`, `--progress-every` |
| `sort-edges` | sorts arcs by `(src, dst)` and removes duplicates | `--input-format auto\|tsv\|binary`, `--edge-list`, `--sorted-arcs`, `--sort-algo radix\|pdq`, `--no-dedup` |
| `compress` | compresses a sorted edge list into `<prefix>.{graph,offsets,properties}` | `--input-format`, `--num-nodes`, `--node-map`, `--parallel`, `--build-ef`, `--allow-empty`, `--verify quick\|full\|none` |
| `build-pg` | `edge-list` + `sort-edges` + `compress` over one input, fused | `--node-map`, `--edge-list`, `--output-prefix`, `--no-text-edge-list`, `--no-node-map`, `--num-nodes`, `--parallel`, `--verify` |
| `build N` | `build-pg` over chunks `1..N`, streamed | `--chunk-dir`, `--graph-dir`, `--from-stage`, plus every `build-pg` flag |

`build-pg` is the payment-graph build: "pg" is the domain term, and it is also
what names the artefacts `build N` writes — `graph/pg_nm_N.tsv`,
`graph/pg_el_N.tsv`, `graph/pg_N.{graph,offsets,properties}`.

Global flags apply to every stage and may be given before *or* after the
subcommand (`utxo2webgraph --threads 64 build 15` and `utxo2webgraph build 15
--threads 64` are the same command): `--threads`, `--memory`, `--tmp-dir`,
`--log-dir`, `--log-interval`, `--keep-intermediate`, `--strict`/`--lenient`,
`--stats brief|extended`, `-v`/`-vv`, `--dry-run`.

`--num-nodes` defaults to `from-arcs`, i.e. `max(endpoint) + 1`, in `build`,
`build-pg` and `compress`; `--num-nodes node-map` is the opt-in richer graph.
See section 6.

Some of these commands read very large files, and nothing caps the input size:
`utxo2webgraph split finalBCUTXO_2022` is literally what you type. What the
commands do instead is estimate their output from the input size and refuse to
start a run that cannot finish. `split`, `edge-list` and `build`/`build-pg`
check the free space on every volume they are about to write to and stop rather
than fill it (section 7). `sort-edges` and `compress` run no size preflight at
all.

There is **no node-count ceiling**, and no preflight for one. Node ids are
`usize` and an arc is a `u128`, so a graph with more than `2^32` nodes builds,
compresses and loads like any other; see section 6.

`--dry-run` (see the quick start below) prints the resolved plan, the input
sizes and those estimates for any stage without touching anything; run it first
when the input is large.

---

## 3. Quick start

```bash
cd utxo2webgraph-rs
cargo build --release          # -> target/release/utxo2webgraph
./target/release/utxo2webgraph build 1
cargo test                     # the golden suite against the recorded reference fixtures
cargo test -- --ignored        # any test later marked heavy (none are today)
```

Every golden test runs in well under a second on `chunk_01.txt`, so none of them
is `#[ignore]`d; `cargo test` is the full suite. The tests skip themselves with
a printed notice when the corpus is absent, so they stay green on a machine
without the data — but a missing *in-tree* fixture is a hard failure, never a
skip.

Useful first commands:

```bash
utxo2webgraph split --show-boundaries          # prints the 28-row boundary table and exits
utxo2webgraph build 1 --dry-run                # resolved plan, sizes, estimates; touches nothing
utxo2webgraph -vv build 1                      # trace-level logging on stderr
```

---

## 4. Guarantees and edge-case handling

These are the behaviours that are easy to get wrong and that this tool pins
deliberately. Every one of them concerns input the current corpus does not
actually contain (verified over `chunks/chunk_01..06` in full and over
multi-million-line heads of `chunk_10`, `chunk_20` and `chunk_28`), so none of
them can perturb the golden tests — they exist so that a 132 GB, multi-hour run
fails *diagnosably* instead of quietly.

| case | what happens | why |
|---|---|---|
| a transaction declares **no outputs** | zero edges, not a phantom edge into node `0` | an empty output section is genuinely empty. Emitting an arc into node `0` — a perfectly valid node id — is silent corruption that no later stage can detect. It needs a `>=4`-section line, of which the corpus has none |
| a **blank** line | skipped and tallied, in both modes | a blank line carries no record; killing a multi-hour run over one would be an expensive way to say nothing. Zero such lines exist in the corpus |
| a **malformed** line (wrong section count, short input descriptor, unparsable integer) | a line-numbered error under `--strict`, a tallied skip under `--lenient` | the message names the line and the offending value, so a bad record in a 132 GB pass is diagnosable without re-reading the corpus |
| an input references an output that was **never registered** | typed error naming the line, the txId, the input index and the missing `(prevTxId, prevOffset)`; `--on-missing-source skip\|create` to continue | the input side of a transaction only ever *looks up* an output, so the pipeline is well-defined only on a prefix of the transaction list starting at genesis. Measured: 0/1033 dangling refs in `chunk_01`, 134/1854 in `chunk_02` alone, 18 145/1 168 956 in `chunk_05` alone |
| a **write error** (disk full, EIO) | fatal: every write is checked, `sync_all` runs before the statistics, and the temp file is renamed only on success | the failure mode this prevents is a truncated 210 GB edge list next to exit status 0 |
| the **node map order** | ascending `(txId, offset)`, always | deterministic and streamable: two runs over the same input produce byte-identical files. The *set* of triples is what matters, so compare other tools' node maps as a set (section 5) |
| a **non-UTF-8 byte** | `--strict` reports the line *and the byte offset within it*; `--lenient` substitutes U+FFFD and tallies `non_utf8_lines` | the corpus is pure ASCII (`chunk_01`: 0 bytes `>= 0x80`), and either way a stray byte does not kill a 132 GB pass with a message that names no line |
| a txId that jumps **forward** | `NonDenseTxId`, bridged with zero-output placeholders under `--lenient` up to 2^20 ids | the single node map is a prefix-sum `Vec<u64>` indexed by txId, so a gap is real memory. Bridging without a ceiling is how a two-line, 79-byte input once drove RSS to 16 GiB |
| a txId that goes **backwards** | not an error: it is a repeat, and the ids of its first occurrence come back | **required for this corpus, not merely safe.** txIds run consecutively `0..778_613_437` across all 28 chunks, but `chunk_04` re-emits txId 142726 and txId 142572 — the BIP-30 duplicate coinbase transactions of blocks 91812/91842 and 91722/91880, permanent consensus history. Refusing them would refuse the real data; a second, hash-backed node map behind a flag used to be the workaround and has been deleted. `tests/data/bip30/c04anomaly.txt` is the regression |
| a trailing separator, e.g. `a;b;` | two fields, not three | a trailing run of separators is not significant in this format |

`--strict --on-missing-source fail --stats brief` are the defaults.

---

## 5. Output formats

**Node map** (`pg_nm_N.tsv`): `txId \t offset \t id`, TAB-separated, LF, no
header, in **ascending `(txId, offset)`** order. Any tool that materialises the
map in a hash table's iteration order produces the same *set* of triples in a
different byte order — over `chunk_01`'s 18 620 rows such an order merely
*looks* sorted (the id column has 23 descents and the txId column 5) — so
compare as a set:

```bash
sort -t$'\t' -k1,1n -k2,2n pg_nm_1.tsv | md5sum
```

**Edge list** (`pg_el_N.tsv`): `{src}\t{dst}\n`, decimal, no padding, no sign,
no leading zeros, sorted ascending by `(src, dst)` and deduplicated. The
canonical formatting matters: a textual `sort | uniq` compares bytes, so `007\t5`
and `7\t5` would both survive and then crash the graph writer on a duplicate
arc. `utxo2webgraph` deduplicates on parsed integers, so it cannot happen.

The *raw* arc stream out of `edge-list --arc-format tsv` is unsorted,
undeduplicated and in emission order — input line order, then input index
ascending, then output offset ascending — and is pinned byte-for-byte by the
golden tests (`el_01.tsv`, md5 `a3c31369f1a54bacfeb9b3cc2c953ebe`).

**Binary arcs** (`--arc-format binary`, `--sorted-arcs`): a headerless stream of
little-endian `u128` records, `(src << 64) | dst`, **16 bytes each, always**.
There is exactly one binary width and no flag to choose it, because the format
carries no header: a file whose length is a multiple of both 8 and 16 has two
readings, and a reader that guesses turns one wide arc `(0, 2)` into the two
bogus arcs `(0, 0)` and `(0, 2)` while exiting 0. Sixteen bytes also means the
packing is **total** — no ceiling, no arc the representation can refuse.

**BVGraph**: `<prefix>.graph`, `.offsets`, `.properties`, with
`windowsize=7, maxrefcount=3, minintervallength=4, zetak=3, compressionflags=`
(empty), which is `webgraph-rs`'s `CompFlags::default()` and the standard
BVGraph parameter set.

`.properties` is **informational and not byte-reproducible** across writers: it
carries a `length=` and an `endianness=` key, key order follows insertion, and
compression-statistics keys (`bitsfor*`, `avgbitsfor*`, `copiedarcs`,
`residualarcs`, `intervalisedarcs`, `*avggap`, `*avgloggap`, `*expstats`) are
not emitted at all — those numbers are accumulated inside the compressor's bit
accounting, which `webgraph-rs` does not expose. Only `graphclass`, `version`,
`nodes`, `arcs`, `windowsize`, `maxrefcount`, `minintervallength`, `zetak` and
`compressionflags` are ever read back, and **that has been verified for
interop**: a stock 32-bit `BVGraph -o -O -L` reader loads the graph, regenerates
a byte-identical `.offsets`, and dumps exactly the same arcs. Only a script that
scrapes the statistics keys would notice their absence.

Only `.graph` and `.offsets` are meaningful byte targets, and even there the
files carry 4-8 trailing **zero** pad bytes, because `BufBitWriter` flushes to a
64-bit word boundary rather than to a byte. Compare the first
`ceil(length/8)` bytes, or use `webgraph::traits::graph::eq` — **never an md5
of the whole file.**

---

## 6. The node-count decision

`--num-nodes` picks between two genuinely different graphs:

* **`from-arcs` (the default)** — `max(id over sources AND targets) + 1`,
  inferred from the arcs alone, exactly as a bare arc-list reader would infer
  it. This is the byte-compatible setting: on `chunk_01` the whole 36 019-bit
  payload of `.graph`, and the whole `.offsets` bitstream, are byte-for-byte
  identical to what an independent BVGraph writer produces from the same arcs
  and the same node count (section 8).
* **`node-map`** — `next_id`, i.e. every UTXO is a node, including the recent
  unspent outputs that appear in no arc. Semantically complete, and **larger**:
  on `chunk_01` the two counts are 18 545 and 18 620, the difference being 75
  isolated trailing nodes (17 512 isolated nodes versus 17 587). The arc sets
  are identical either way. Needs `--node-map` in the standalone `compress`.
* **`<N>`** — a literal count, rejected if it is smaller than the arcs require
  (webgraph would silently drop the out-of-range arcs).

Pass `--num-nodes node-map` (with `--node-map`) and both candidate values are
logged, so the difference is visible. The default `from-arcs` is computed from
the arcs alone and does not open the node map, so it logs only the number it
picked. The choice changes the `.properties` `nodes=` value and every
node-indexed downstream array, which is why the default is the one that
reproduces the historical pipeline's output.

### There is no node-count ceiling

At N=28 the `node-map` count is about **2.23e9**, above `i32::MAX` (2.147e9) and
within a factor of two of `u32::MAX`. Nothing in this crate cares. A node
id is a `usize` — the same type `webgraph-rs` uses at every layer of the
compression path, so an id crosses the library boundary without a cast — and an
arc is a `u128` holding a full `(src, dst)` pair with room to spare, which makes
the arc packing **total**: it cannot fail and has no ceiling to check.

That is what makes N=28 work. The earlier scheme packed an arc into a `u64` as
`(src << 32) | dst`, which capped a node id at `2^32` — a cap the corpus was
already within a factor of two of reaching, and one that could only be
discovered hours into a run, after the read pass, the sort, a 51 GiB node map
and a 195 GiB edge list. There was also a preflight that rejected a projected
node count above `i32::MAX` under `--num-nodes node-map`. Both are gone. The
price is exactly the eight extra bytes an arc costs on disk and in the sort
buffer (section 7).

#### One consumer still has a 32-bit limit

This is a property of a *reader*, not of the format or of this tool. The
`graphclass` recorded in `.properties` has a 32-bit reference implementation
that stores a node id in a signed 32-bit integer and refuses `nodes > i32::MAX`
at load. A graph above that count is written here, read back by `webgraph-rs`,
and verified by `--verify` — but that one reader cannot open it.

So if the graph has to be consumed by that reader, keep the node count under
2 147 483 647: at N=28 use the default `--num-nodes from-arcs` rather than
`node-map`, or build fewer chunks. Refusing to write the graph, as this tool
used to, never made that reader able to load it.

---

## 7. Scale and resources at N=28

| quantity | value |
|---|---|
| input (`chunks/chunk_01..28`) | 135.3 GB |
| nodes (`nextId`) | ~2.21e9 |
| raw arcs | ~9.5e9 |
| arcs as packed `u128` (16 B each) | ~152 GB |
| `pg_el_28.tsv` (text) | ~210 GB |
| `pg_nm_28.tsv` (text) | ~55 GB |
| dense node map (`Vec<u64>`, 778 613 438 entries) | **6.23 GB** |
| the hash map it replaces (`(txId,offset) -> id`, 2.2e9 entries) | 140-160 GB |

Run sizing:

```
arcs_per_run = memory / (2 * ARC_RECORD_SIZE)      # ARC_RECORD_SIZE = 16
```

The factor **2** is the radix sort's scratch buffer, which is the same size as
the data being sorted. Without it a nominal 128 GiB budget really costs
256 GiB and OOMs on the first full-scale run.

With the default budget — `min(45% of RAM, 192 GiB)`, i.e. 192 GiB on this
503 GB machine — `arcs_per_run` is `192 * 2^30 / 32`, about **6.44e9 arcs**,
against the ~9.5e9 arcs of a full N=28 run. So N=28 sorts in **two runs and a
merge**, not one: holding the whole arc set in RAM would need
`2 * 9.5e9 * 16` bytes, about **283 GiB** of `--memory`. The external merge path
is therefore the normal path at full scale, not a fallback — which is the price
of the 16-byte arc, and is why the merge machinery is tested rather than
vestigial. (At the old 8-byte width the same budget bought 1.29e10 arcs per run;
that width is what the `2^32` node ceiling bought, and it was not worth it.)

`/data` has 1.8 TB free and `--tmp-dir` defaults to `./tmp`, on the same
filesystem as everything else. `utxo2webgraph` computes a space estimate from
the input size and **refuses to start** rather than filling the volume.

---

## 8. Operating notes

* **`TMPDIR` is set from `--tmp-dir` at startup**, before any thread is
  spawned. webgraph's `ParSortIters`/`ParSortPairs` call bare
  `tempfile::tempdir()` and ignore `BvCompConf::tmp_dir` (which only controls
  the parallel compressor's partial bitstreams). Without this a parallel N=28
  run spills tens of GB into `/tmp` on the system disk.
* **Check `/proc/sys/vm/max_map_count`** before a multi-hour parallel run;
  `utxo2webgraph --parallel` warns when it looks too low for the batch count.
* **Sequential compression is the default**, because it is the reproducible
  path: the same arcs always produce the same bytes (verified: 0 differing
  bytes over the first `ceil(length/8)` bytes of `.graph` and `.offsets` on a
  2 000-node, 19 944-arc cross-check against an independent BVGraph writer).
  `--parallel` is faster, but each chunk restarts its compression window, so the
  bytes differ. Semantically identical: `graph::eq` passes.
* **Overflow checks are ON in the release profile.** The webgraph workspace
  disables them, but `Compressor::write` computes
  `residuals[i] - residuals[i-1] - 1`, which wraps and writes a corrupt
  bitstream on a duplicate successor. A few percent of throughput is cheap
  insurance against a corrupt 210 GB graph; `--profile fast` turns them off.
* **`panic = "abort"` is deliberately NOT set**, so `compress.rs` can
  `catch_unwind` webgraph's sortedness assertions and turn them into typed
  errors.
* `.properties` is written **last**. A `.graph` with no `.properties` means an
  interrupted run — `utxo2webgraph` writes to a temporary basename and renames
  only after `verify_graph` passes, so this state never reaches the final names.
* **`--verify` controls how the finished graph is read back**: `quick` (the
  default) checks that `.properties` exists and that the node count and the
  *declared* arc count match; `full` adds a complete second decompression that
  recounts every arc (at N=28 that is another pass over 9.5e9 arcs, several
  minutes); `none` skips it.
* **`--log-dir` really is a log directory.** Everything that goes to stderr is
  teed into `<log-dir>/<stage>.log`, and the two statistics lines (which go to
  stdout, byte-for-byte) are echoed there too, so a finished run leaves the same
  artefacts in `logs/` the historical pipeline did. Because the tee makes the
  sink a pipe, **colour is disabled** (`WriteStyle::Never` next to
  `Target::Pipe`): the log file can never pick up ANSI escapes, and neither can
  stderr while the tee is active.
* **The log line format is webgraph-rs's**, so a `utxo2webgraph` stage and a
  `webgraph` stage in the same pipeline read alike:

  ```text
  <utc ts> <elapsed> <LEVEL> [<ThreadId>] <target> - <message>
  2026-09-17 10:58:38.895 14ms INFO [ThreadId(1)] utxo2webgraph - threads=112 …
  ```

  The timestamp is UTC (`jiff::Timestamp::strftime`, the same instant
  `env_logger` used to print with a trailing `Z`); the second field is the time
  elapsed since the logger was installed.
* **`--log-interval` (global, default `10s`)** sets how often the long loops
  report. Suffixes `s`/`m`/`h`/`d`, and a bare number is **milliseconds**, which
  is how the historical pipeline's intervals were written:
  `--log-interval 1d2h3m4s567` parses. Every large loop — split, edge-list
  build, node-map write, arc read, arc sort, counting merge, edge-list write,
  arc scan, BVGraph compression — is driven by a `dsi-progress-logger`, which
  prints a count, a rate, resident memory and, where the total is known exactly,
  a percentage and an ETA.
* **`RUST_LOG` layers on top of `-v`/`-vv`, it does not replace it.** The
  builder is `filter_level(level).parse_default_env()`, which installs a
  catch-all directive first, so `RUST_LOG=utxo2webgraph::arcs=debug` raises that
  one module and leaves everything else at the `-v` level. (Switching to
  `Builder::from_env(..default_filter_or("info"))` would silence every target
  `RUST_LOG` does not name and make `-vv` a no-op.) Targets are module paths:
  `utxo2webgraph::arcs`, `utxo2webgraph::split`, `utxo2webgraph::compress`, and
  plain `utxo2webgraph` for the binary. Note that a progress line carries the
  target of the module that *built* the logger, which for the pipeline loops is
  `utxo2webgraph` (`main.rs` owns them) and for the compressor's own loggers is
  `utxo2webgraph::compress`.
* **Spill directories are pid-keyed and reaped at startup.** A run killed by
  `SIGINT`/`SIGKILL` never runs `TempDir`'s destructor; the next
  `utxo2webgraph` removes any `tmp/utxo2webgraph-sort-<pid>-*` whose pid is no
  longer alive, and never touches one belonging to a live process. Partial
  *outputs* are removed by the sink's own `Drop`, so a failed run leaves no
  `.<name>.utxo2webgraph-tmp-<pid>` either.
* **Non-regular outputs work.** `--arcs /dev/null`, a fifo or a process
  substitution are written through directly and are *not* `fsync`ed —
  `fsync(2)` returns `EINVAL` on a character device or a pipe, which used to
  turn a completed run into exit 1. Real durability errors (`ENOSPC`, `EIO`) on
  a regular file are still fatal, which is the entire point of syncing.
* **A binary arc file has exactly one reading.** Every record is 16 bytes, so
  `--input-format auto` only has to sniff text versus binary; there is no width
  to pass in and no way to pass in the wrong one. A binary file whose length is
  not a multiple of 16 is a `TruncatedRun` error naming the file and its length,
  not a stream of plausible-looking garbage.

---

## 9. Why the design fails fast

Every preflight, every checked write and every typed error above exists because
the un-fused three-stage pipeline this crate replaces could fail in the first
stage and still run to completion, and the evidence is still in the checkout:

* the first stage died before writing anything, leaving its two outputs at
  **0 bytes**;
* the driver script had **no `set -e` and no `set -o pipefail`** (unlike the
  splitter and the "first N chunks" driver, which both have `set -euo
  pipefail`), so it went on to sort an empty file — and
  `(sort | uniq) > $EL_FILE` truncates the target *before* `sort` even runs;
* the BVGraph stage then died on the empty edge list with
  `Expected integer, found Token[EOF], line 1` — a parse failure on line 1, two
  stages downstream of the real problem and naming nothing about it;
* `graph/pg_el_1.tsv`, `pg_el_10.tsv` and `pg_el_15.tsv` are all **0 bytes**.

`utxo2webgraph` exits non-zero on the first error, writes every output to a
temporary path and renames it only on success, refuses to build a graph from an
empty edge list at all (`EmptyEdgeList`, `--allow-empty` to override), and
prints its statistics to stdout, so `logs/` is non-empty on success and
diagnosable on failure. `build-pg` fuses the three stages so the 210 GB
intermediate text edge list never exists unless you ask for it.

---

## 10. Validation recipe

The reference outputs are **in the crate**, at `tests/data/reference` (3.0 MB,
35 files); `UTXO2WEBGRAPH_REFERENCE` overrides the location for a larger
out-of-tree corpus, and a missing in-tree fixture is a hard test failure, never
a skip.

```bash
REF=utxo2webgraph-rs/tests/data/reference

# raw edge list: BYTE-identical
utxo2webgraph edge-list -i chunks/chunk_01.txt --node-map rust_nm_01.tsv \
                        --arcs rust_el_01.tsv --arc-format tsv
cmp rust_el_01.tsv $REF/el_01.tsv                                  # a3c31369f1a54bacfeb9b3cc2c953ebe

# node map: compared as a SET (a hash-table iteration order is not portable)
sort -t$'\t' -k1,1n -k2,2n rust_nm_01.tsv | md5sum                 # 4795d0ab4a84e775d16c7b57e26d2fc9

# stdout under --stats brief (the seconds field will differ):
#   Processed: 18578 transactions (after 0 seconds).
#   Nodes: 18620<TAB>Edges: 1094

# chunks 1+2 (cat chunk_01.txt chunk_02.txt > c12.txt):
#   el 92439127b404b877889776a816e3602c, nm 2e9de9a09685f5dc3157f231053715af
#   Processed: 32705 ... / Nodes: 32777<TAB>Edges: 3628

# canonical (sort | uniq) edge list: 941a3876f734b9d75c36475d94266c66, 1094 lines

# failure parity: chunk_02 alone must exit 1
#   (the error is the dangling reference at line 59, NOT the output target:
#    `--arcs /dev/null` on chunk_01 exits 0 and prints the statistics)
utxo2webgraph edge-list -i chunks/chunk_02.txt --arcs /dev/null ; echo $?  # 1
utxo2webgraph edge-list -i chunks/chunk_01.txt --arcs /dev/null ; echo $?  # 0

# BVGraph, default flags (--num-nodes from-arcs):
utxo2webgraph compress $REF/el_01.sorted.tsv rs_pg_01
#   reports 18545 nodes / 1094 arcs / 36019 bits, and reading the graph back
#   reproduces $REF/el_01.sorted.tsv arc for arc.
#   Against an independent BVGraph writer given the same 18545 nodes, the
#   payloads agree bit for bit:
#     cmp -n 4503 other.graph   rs_pg_01.graph     -> identical
#     cmp -n 7667 other.offsets rs_pg_01.offsets   -> identical
#   The written files are a few zero-padding bytes longer (4504 = 563*8 and
#   7672 = 959*8), so compare the first ceil(length/8) bytes, never an md5 of
#   the whole file; see section 5.
```

`cargo test` runs all of the above as `tests/golden.rs` — the graph stage as an
arc-for-arc round trip through `BvGraphSeq`, since `.graph` bytes are not kept
as a fixture — plus the eight-case
edge corpus in `$REF/edge/`, the node-map id-assignment cross-check over all of
`chunk_01`, and the BIP-30 duplicate-coinbase regression.

---

## 11. Future work

* **CSR counting sort.** The arcs are emitted in order of the *consuming*
  transaction and a UTXO is spent at most once, so for a fixed `src` the `dst`
  values are already strictly increasing; only the grouping by `src` is
  missing. That is a counting sort: O(E), no comparisons, ~62 GB total at N=28
  (`base` 6.23 GB + `off` 17.7 GB + `adj` 38.0 GB) against 152 GB plus 152 GB of
  scratch for the radix sort. (That `adj` estimate assumes a 32-bit
  destination, which the N=28 node count still fits and the radix path does not
  assume; a CSR that wants the same no-ceiling guarantee would pay more.)
  **Mandatory caveat:** it must *verify* that every bucket is strictly
  increasing after the scatter, in a linear pass, and sort any that is not —
  never assume the invariant. It was left out of v1 because the radix sort
  already handles the whole N=28 arc set, and the win is unmeasured.
* **Elias-Fano node map.** `base` is monotone with max ~2.21e9 over 778.6e6
  entries, so `sux::dict::EliasFano` would store it in ~3.5 bits/entry, i.e.
  ~340 MB instead of 6.23 GB. Not incrementally appendable during the build
  pass, and 6.23 GB out of 503 GB is free, so it is not worth doing today.

---

## 12. Hard rules for contributors

1. **Leave the original pipeline alone.** Where this crate sits next to the
   original scripts and their recorded outputs (`builder.sh`, `build_pg.sh`,
   `splitter.sh`, `chunks/`, the master transaction list), treat all of it as
   read-only reference. It is the oracle the golden suite compares against;
   editing it invalidates the comparison. In particular, **`build_pg.sh` must
   not be modified**: `utxo2webgraph build-pg` supersedes it without touching
   it.
2. **Never build a reference artefact in place.** Anything regenerated from the
   original sources goes into a scratch directory, so no build output lands next
   to them.
3. **Never run the pipeline on anything larger than `chunk_01.txt` (976 KB)**
   without a deliberate decision. `chunk_05.txt` (84 MB) is acceptable for a
   correctness cross-check, but not for timing claims; the full
   `finalBCUTXO_2022` is 135 GB. This is a contributor convention and nothing
   more: no code enforces it, the 84 MB boundary has no successor in the
   binary, and `utxo2webgraph` will start whatever you point it at. The only
   automatic brake is the disk-space preflight of section 7, which stops a run
   that cannot finish rather than one that is merely large. Run `--dry-run`
   first and read the plan.
4. **Keep the `webgraph` dependency pinned to a rev.** Bumping it is fine, but
   re-run the validation recipe in section 10 in the same commit: the pin is
   what keeps the compressed output reproducible.
