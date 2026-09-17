//! Differential test of the whole pipeline against the compiled Java
//! reference, on the real `chunks/chunk_01.txt`.
//!
//! The algorithm under test is **Matteo Loporchio**'s
//! `PaymentGraphEdgeListBuilder.java`; these tests assert that the Rust port
//! reproduces it exactly, except at the points where the port deliberately
//! fixes a bug (each such point is asserted explicitly, with the Java
//! behaviour named in a comment).
//!
//! Everything here drives the library API directly, so no `assert_cmd` or
//! compiled binary is needed. Every test starts with a guard: if the corpus or
//! the reference directory is missing, the test prints a skip notice and
//! returns, so the suite stays green on a machine without the data.
//!
//! HARD RULE, encoded here: no test may touch anything larger than
//! `chunk_01.txt` (976 KB) or `chunks/chunk_02.txt` (796 KB). Never
//! `finalBCUTXO_2022`.

use std::fs;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use dsi_bitstream::prelude::BE;
use lender::for_;
use webgraph::graphs::bvgraph::BvGraphSeq;
use webgraph::prelude::*;

use utxo2webgraph::arcs::{ArcSorter, NullArcSink, SortOpts, TsvArcSink};
use utxo2webgraph::compress::{self, CompressOpts, VerifyLevel};
use utxo2webgraph::edge_list::{build_edge_list, write_node_map, EdgeListOpts, EdgeListStats};
use utxo2webgraph::nodemap::{new_node_map, NodeMap};
use utxo2webgraph::{
    ArcCodec, ArcSink, Mode, NodeId, NodeMapKind, OnMissingSource, PgError, SortAlgo, StatsStyle,
};

const CHUNK_01: &str = "/data/bitcoin/2022/utxo-spllitting-pipeline/chunks/chunk_01.txt";
const CHUNK_02: &str = "/data/bitcoin/2022/utxo-spllitting-pipeline/chunks/chunk_02.txt";

/// Where the outputs of the compiled Java reference live.
///
/// **In the crate**, under `tests/data/javaref` (3.0 MB, 35 files: the two
/// chunk_01/chunk_02 node maps and edge lists, their concatenation, and the
/// eight-case `edge/` corpus). They used to live in a per-session scratchpad,
/// and `available()` turned a missing fixture into a green skip — so once that
/// directory was reaped `cargo test` would still have reported "8 passed"
/// while every Java-parity assertion had silently stopped running.
///
/// `PGRAPH_JAVAREF` overrides the location, for a larger out-of-tree corpus.
/// A missing *in-tree* fixture is a hard failure.
fn javaref_dir() -> PathBuf {
    match std::env::var_os("PGRAPH_JAVAREF") {
        Some(dir) => PathBuf::from(dir),
        None => Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("data")
            .join("javaref"),
    }
}

// ---------------------------------------------------------------- fixtures

/// Returns `false` (and prints a skip notice) when a *corpus* file is missing.
///
/// Only the real chunk files may be absent — they are 976 KB and 796 KB of
/// production data that does not belong in the repository. The Java reference
/// outputs are in-tree and are asserted to exist.
fn available(paths: &[&str]) -> bool {
    let dir = javaref_dir();
    assert!(
        dir.is_dir(),
        "the Java reference fixtures are missing at {}; they are part of the crate \
         (tests/data/javaref), so this is a broken checkout, not a machine without the data",
        dir.display()
    );
    for p in paths {
        if !Path::new(p).exists() {
            eprintln!("SKIP: fixture {p} is not present on this machine");
            return false;
        }
    }
    true
}

fn javaref(name: &str) -> PathBuf {
    let p = javaref_dir().join(name);
    assert!(
        p.exists(),
        "missing in-tree Java reference fixture {}",
        p.display()
    );
    p
}

// --------------------------------------------------------------------- md5

/// Minimal MD5, inlined so the crate needs no `md-5` dependency.
///
/// Only the tests need it, and only to compare against digests that were
/// computed with `md5sum` when the Java reference was produced.
fn md5_hex(bytes: &[u8]) -> String {
    #[rustfmt::skip]
    const S: [u32; 64] = [
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
        5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20,
        4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
        6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
    ];
    #[rustfmt::skip]
    const K: [u32; 64] = [
        0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee,
        0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
        0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
        0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
        0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa,
        0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
        0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
        0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
        0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
        0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
        0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05,
        0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
        0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039,
        0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
        0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
        0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391,
    ];

    let mut msg = bytes.to_vec();
    let bit_len = (bytes.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_le_bytes());

    let (mut a0, mut b0, mut c0, mut d0) = (
        0x6745_2301u32,
        0xefcd_ab89u32,
        0x98ba_dcfeu32,
        0x1032_5476u32,
    );

    for chunk in msg.chunks_exact(64) {
        let mut m = [0u32; 16];
        for (i, word) in chunk.chunks_exact(4).enumerate() {
            m[i] = u32::from_le_bytes([word[0], word[1], word[2], word[3]]);
        }
        let (mut a, mut b, mut c, mut d) = (a0, b0, c0, d0);
        for i in 0..64 {
            let (f, g) = match i {
                0..=15 => ((b & c) | ((!b) & d), i),
                16..=31 => ((d & b) | ((!d) & c), (5 * i + 1) % 16),
                32..=47 => (b ^ c ^ d, (3 * i + 5) % 16),
                _ => (c ^ (b | (!d)), (7 * i) % 16),
            };
            let f = f.wrapping_add(a).wrapping_add(K[i]).wrapping_add(m[g]);
            a = d;
            d = c;
            c = b;
            b = b.wrapping_add(f.rotate_left(S[i]));
        }
        a0 = a0.wrapping_add(a);
        b0 = b0.wrapping_add(b);
        c0 = c0.wrapping_add(c);
        d0 = d0.wrapping_add(d);
    }

    let mut out = String::with_capacity(32);
    for word in [a0, b0, c0, d0] {
        for byte in word.to_le_bytes() {
            out.push_str(&format!("{byte:02x}"));
        }
    }
    out
}

#[test]
fn md5_matches_known_vectors() {
    assert_eq!(md5_hex(b""), "d41d8cd98f00b204e9800998ecf8427e");
    assert_eq!(md5_hex(b"abc"), "900150983cd24fb0d6963f7d28e17f72");
    assert_eq!(
        md5_hex(b"The quick brown fox jumps over the lazy dog"),
        "9e107d9d372bb6826bd81d3542a419d6"
    );
    // A message longer than one 64-byte block, exercising the padding path.
    assert_eq!(
        md5_hex(&b"a".repeat(1000)),
        "cabe45dcc9ae5b66ba86600cca6b8ba8"
    );
}

// ----------------------------------------------------------- tiny helpers

fn default_opts(mode: Mode, on_missing: OnMissingSource) -> EdgeListOpts {
    EdgeListOpts {
        mode,
        on_missing_source: on_missing,
        progress_every: utxo2webgraph::DEFAULT_PROGRESS_EVERY,
        // `quiet_stats` keeps the Java stdout lines out of the test output;
        // `print_stats` is exercised by its own unit test in `edge_list.rs`.
        stats: StatsStyle::Java,
        quiet_stats: true,
    }
}

/// Runs the builder over `input` into a text arc file, returning the stats,
/// the arc bytes and the node-map bytes.
fn run_builder(
    input: &Path,
    dir: &Path,
    kind: NodeMapKind,
    mode: Mode,
    on_missing: OnMissingSource,
    tag: &str,
) -> (EdgeListStats, Vec<u8>, Vec<u8>) {
    let arcs_path = dir.join(format!("{tag}_el.tsv"));
    let nm_path = dir.join(format!("{tag}_nm.tsv"));

    let mut node_map = new_node_map(kind, mode, None);
    let mut sink = TsvArcSink::create(&arcs_path).expect("create arc sink");
    let reader = BufReader::new(fs::File::open(input).expect("open input"));
    let stats = build_edge_list(
        reader,
        &mut *node_map,
        &mut sink,
        default_opts(mode, on_missing),
    )
    .expect("build_edge_list");
    drop(sink);

    write_node_map(&*node_map, &nm_path).expect("write node map");
    (
        stats,
        fs::read(&arcs_path).expect("read arcs"),
        fs::read(&nm_path).expect("read node map"),
    )
}

/// Canonicalises a node map the way `sort -t$'\t' -k1,1n -k2,2n` does.
///
/// The Java node map is written by iterating `java.util.HashMap.keySet()`,
/// i.e. **bucket order**. It merely LOOKS sorted: over chunk_01's 18 620 rows
/// the id column has 23 descents and the txId column 5. That order is
/// deterministic for a fixed JVM but is not a specification, so the node map
/// is compared as a SET and never byte-for-byte.
fn canonical_node_map(bytes: &[u8]) -> (String, usize) {
    let text = std::str::from_utf8(bytes).expect("node map is ASCII");
    let mut rows: Vec<(i64, i64, u64)> = text
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| {
            let mut f = l.split('\t');
            let tx: i64 = f.next().unwrap().parse().unwrap();
            let off: i64 = f.next().unwrap().parse().unwrap();
            let id: u64 = f.next().unwrap().parse().unwrap();
            (tx, off, id)
        })
        .collect();
    rows.sort_by_key(|&(tx, off, _)| (tx, off));
    let mut out = String::new();
    for (tx, off, id) in &rows {
        out.push_str(&format!("{tx}\t{off}\t{id}\n"));
    }
    let n = rows.len();
    (out, n)
}

/// Canonicalises an edge list the way `sort -t$'\t' -k1,1n -k2,2n | uniq` does.
fn canonical_edge_list(bytes: &[u8]) -> (String, usize) {
    let text = std::str::from_utf8(bytes).expect("edge list is ASCII");
    let mut arcs: Vec<(u64, u64)> = text
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| {
            let mut f = l.split('\t');
            (
                f.next().unwrap().parse().unwrap(),
                f.next().unwrap().parse().unwrap(),
            )
        })
        .collect();
    arcs.sort_unstable();
    arcs.dedup();
    let mut out = String::new();
    for (s, d) in &arcs {
        out.push_str(&format!("{s}\t{d}\n"));
    }
    let n = arcs.len();
    (out, n)
}

// ------------------------------------------------------- the central test

#[test]
fn test_chunk01_edge_list_byte_identical() {
    if !available(&[CHUNK_01]) {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let (stats, arcs, node_map) = run_builder(
        Path::new(CHUNK_01),
        dir.path(),
        NodeMapKind::Dense,
        Mode::Strict,
        OnMissingSource::Fail,
        "c1",
    );

    // Java printed `Processed: 18578 transactions` and `Nodes: 18620 Edges: 1094`.
    assert_eq!(stats.tx_count, 18578, "one transaction per input line");
    assert_eq!(
        stats.node_slots, 18620,
        "Java's `Nodes:` counter (output slots)"
    );
    assert_eq!(
        stats.edge_count, 1094,
        "sum over lines of #inputs * #outputs"
    );
    assert_eq!(
        stats.distinct_nodes, 18620,
        "txIds are globally unique here, so slots == distinct nodes"
    );
    assert_eq!(stats.dangling_refs, 0, "chunk_01 is a prefix from genesis");
    assert_eq!(stats.skipped_lines, 0, "the corpus has no malformed lines");

    // The edge list is compared BYTE-FOR-BYTE: its order is fully determined
    // by the input (line order, then input index, then output offset).
    let reference = fs::read(javaref("el_01.tsv")).expect("read el_01.tsv");
    assert_eq!(
        md5_hex(&arcs),
        "a3c31369f1a54bacfeb9b3cc2c953ebe",
        "raw edge list md5"
    );
    assert_eq!(arcs.len(), 11_623, "el_01.tsv is 11623 bytes");
    assert_eq!(
        arcs, reference,
        "raw edge list must be byte-identical to Java"
    );
    assert_eq!(arcs.iter().filter(|&&b| b == b'\n').count(), 1094);

    let first_five: Vec<&str> = std::str::from_utf8(&arcs)
        .unwrap()
        .lines()
        .take(5)
        .collect();
    assert_eq!(
        first_five,
        vec!["9\t171", "9\t172", "172\t184", "172\t185", "185\t187"]
    );

    // The node map is compared as a SET; see `canonical_node_map`.
    let (canon_nm, nm_rows) = canonical_node_map(&node_map);
    assert_eq!(nm_rows, 18620, "one row per output slot");
    assert_eq!(
        md5_hex(canon_nm.as_bytes()),
        "4795d0ab4a84e775d16c7b57e26d2fc9",
        "sorted node map md5"
    );
    let (canon_ref_nm, ref_rows) =
        canonical_node_map(&fs::read(javaref("nm_01.tsv")).expect("read nm_01.tsv"));
    assert_eq!(ref_rows, nm_rows);
    assert_eq!(canon_nm, canon_ref_nm, "the set of node triples must match");

    // What `build_pg.sh` actually fed to WebGraph.
    let (canon_el, el_rows) = canonical_edge_list(&arcs);
    assert_eq!(
        md5_hex(canon_el.as_bytes()),
        "941a3876f734b9d75c36475d94266c66",
        "canonical (sort | uniq) edge list md5"
    );
    assert_eq!(
        el_rows, 1094,
        "chunk_01 contains zero duplicate arcs and zero self-loops"
    );
}

#[test]
fn test_chunk01_plus_02() {
    // In-tree fixture: `javaref` asserts it exists.
    let c12 = javaref("c12.txt");
    let dir = tempfile::tempdir().expect("tempdir");
    let (stats, arcs, node_map) = run_builder(
        &c12,
        dir.path(),
        NodeMapKind::Dense,
        Mode::Strict,
        OnMissingSource::Fail,
        "c12",
    );

    // Java: `Processed: 32705 transactions` / `Nodes: 32777  Edges: 3628`.
    assert_eq!(stats.tx_count, 32705);
    assert_eq!(stats.node_slots, 32777);
    assert_eq!(stats.edge_count, 3628);
    assert_eq!(
        stats.dangling_refs, 0,
        "chunks 1+2 together are still a prefix from genesis (0/2887 dangling)"
    );

    assert_eq!(md5_hex(&arcs), "92439127b404b877889776a816e3602c");
    assert_eq!(
        arcs,
        fs::read(javaref("el_12.tsv")).expect("read el_12.tsv"),
        "raw edge list must be byte-identical to Java"
    );

    let (canon_nm, nm_rows) = canonical_node_map(&node_map);
    assert_eq!(nm_rows, 32777);
    assert_eq!(
        md5_hex(canon_nm.as_bytes()),
        "2e9de9a09685f5dc3157f231053715af"
    );

    let (canon_el, el_rows) = canonical_edge_list(&arcs);
    assert_eq!(
        md5_hex(canon_el.as_bytes()),
        "424155f903415496255cc6e155d1e95f"
    );
    assert_eq!(el_rows, 3628);
}

// ----------------------------------------------------------- failure parity

#[test]
fn test_chunk02_alone_fails() {
    if !available(&[CHUNK_02]) {
        return;
    }
    // Java died here with
    //   Cannot invoke "java.lang.Long.longValue()" because the return value of
    //   "java.util.Map.get(Object)" is null
    // at PaymentGraphEdgeListBuilder.java:74, leaving BOTH output files 0
    // bytes (the node map is written only after the read loop). 134 of the
    // chunk's 1854 input references are dangling.
    let mut node_map = new_node_map(NodeMapKind::Dense, Mode::Strict, None);
    let mut sink = NullArcSink::new();
    let reader = BufReader::new(fs::File::open(CHUNK_02).expect("open chunk_02"));
    let err = build_edge_list(
        reader,
        &mut *node_map,
        &mut sink,
        default_opts(Mode::Strict, OnMissingSource::Fail),
    )
    .expect_err("chunk_02 alone must fail, exactly as the Java NPE did");
    match err {
        PgError::DanglingSource {
            line,
            tx_id,
            prev_tx_id,
            ..
        } => {
            // The improvement over Java: the error names the line, the
            // transaction and the missing pair.
            assert!(line > 0, "the error must name the input line");
            assert!(tx_id >= 0);
            assert!(prev_tx_id >= 0);
        }
        other => panic!("expected PgError::DanglingSource, got {other:?}"),
    }

    // With `skip`, the same input succeeds and tallies every dangling ref.
    let mut node_map = new_node_map(NodeMapKind::Dense, Mode::Lenient, None);
    let mut sink = NullArcSink::new();
    let reader = BufReader::new(fs::File::open(CHUNK_02).expect("open chunk_02"));
    let stats = build_edge_list(
        reader,
        &mut *node_map,
        &mut sink,
        default_opts(Mode::Lenient, OnMissingSource::Skip),
    )
    .expect("--on-missing-source skip must succeed");
    assert_eq!(
        stats.dangling_refs, 134,
        "chunk_02 alone has 134 dangling input references out of 1854"
    );
}

// ------------------------------------------------------------- edge corpus

/// Runs one `javaref/edge/tN.txt` case and returns its arcs and node map.
fn run_edge_case(tag: &str, dir: &Path, mode: Mode) -> Option<(EdgeListStats, Vec<u8>, Vec<u8>)> {
    let input = javaref(&format!("edge/{tag}.txt"));
    if !input.exists() {
        eprintln!("SKIP: {} is not present", input.display());
        return None;
    }
    let arcs_path = dir.join(format!("{tag}_el.tsv"));
    let nm_path = dir.join(format!("{tag}_nm.tsv"));
    let mut node_map = new_node_map(NodeMapKind::Dense, mode, None);
    let mut sink = TsvArcSink::create(&arcs_path).expect("create sink");
    let reader = BufReader::new(fs::File::open(&input).expect("open case"));
    let stats = build_edge_list(
        reader,
        &mut *node_map,
        &mut sink,
        default_opts(mode, OnMissingSource::Fail),
    )
    .unwrap_or_else(|e| panic!("edge case {tag} failed in {mode:?} mode: {e}"));
    drop(sink);
    write_node_map(&*node_map, &nm_path).expect("write node map");
    Some((
        stats,
        fs::read(&arcs_path).unwrap(),
        fs::read(&nm_path).unwrap(),
    ))
}

fn reference_case(tag: &str, ext: &str) -> Vec<u8> {
    fs::read(javaref(&format!("edge/{tag}.{ext}"))).unwrap_or_default()
}

#[test]
fn test_edge_case_corpus() {
    if !available(&[]) {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");

    // --- t1: a zero-output line (`info:inputs:`). Java's split(":") returned
    // only 2 sections, so `parts[2]` threw ArrayIndexOutOfBoundsException at
    // line 55 and BOTH reference files are 0 bytes. The port skips the line in
    // Lenient mode, so the arcs still match (both empty) but the node map
    // holds the first line's output, which Java never got to write.
    if let Some((stats, arcs, nm)) = run_edge_case("t1", dir.path(), Mode::Lenient) {
        assert!(
            arcs.is_empty(),
            "t1: no edges, as in the (crashed) Java run"
        );
        assert_eq!(reference_case("t1", "el"), arcs);
        assert_eq!(
            stats.skipped_lines, 1,
            "t1: the zero-output line is skipped"
        );
        let (_, rows) = canonical_node_map(&nm);
        assert_eq!(rows, 1, "t1: line 1's single output is still registered");
        assert!(
            reference_case("t1", "nm").is_empty(),
            "t1: Java wrote nothing because it crashed before the node-map loop"
        );
    }

    // --- t2: a 4-colon-section line, so `parts[2] == ""`. Java sized
    // `currentOutputNodeIds` as `"".split(";").length == 1`, left it at its
    // default `{0}`, and emitted the phantom edge `0\t0`: it printed
    // `Nodes: 1  Edges: 1`. THE PORT MUST EMIT ZERO EDGES.
    if let Some((stats, arcs, nm)) = run_edge_case("t2", dir.path(), Mode::Lenient) {
        assert_eq!(reference_case("t2", "el"), b"0\t0\n".to_vec());
        assert!(
            arcs.is_empty(),
            "t2: the phantom edge into node 0 is a deliberate fix, not reproduced"
        );
        assert_eq!(stats.edge_count, 0);
        let (canon, rows) = canonical_node_map(&nm);
        assert_eq!(rows, 1);
        assert_eq!(canon, "0\t0\t0\n", "t2: the node map still matches Java");
    }

    // --- t3: `info::` -> one section. Java threw AIOOBE at line 54; both
    // reference files are empty. The port skips and matches exactly.
    if let Some((stats, arcs, nm)) = run_edge_case("t3", dir.path(), Mode::Lenient) {
        assert_eq!(arcs, reference_case("t3", "el"));
        assert_eq!(nm, reference_case("t3", "nm"));
        assert!(arcs.is_empty() && nm.is_empty());
        assert_eq!(stats.skipped_lines, 1);
    }

    // --- t4: a trailing blank line. Java threw AIOOBE at line 54 and wrote
    // nothing. The port skips blank lines in BOTH modes and keeps going, so
    // the edge list matches while the node map holds line 1's output.
    if let Some((stats, arcs, nm)) = run_edge_case("t4", dir.path(), Mode::Lenient) {
        assert!(arcs.is_empty());
        assert_eq!(arcs, reference_case("t4", "el"));
        assert_eq!(
            stats.blank_lines, 1,
            "t4: the blank line is counted, not fatal"
        );
        let (_, rows) = canonical_node_map(&nm);
        assert_eq!(rows, 1);
        assert!(
            reference_case("t4", "nm").is_empty(),
            "t4: Java crashed first"
        );
    }

    // --- t5: txId = 3000000000, above i32::MAX. Java threw
    // `NumberFormatException: For input string: "3000000000"`. The port keeps
    // i32 semantics and reports the line and the offending text; in Lenient
    // mode the line is skipped, so both files match Java's (empty) output.
    if let Some((stats, arcs, nm)) = run_edge_case("t5", dir.path(), Mode::Lenient) {
        assert_eq!(arcs, reference_case("t5", "el"));
        assert_eq!(nm, reference_case("t5", "nm"));
        assert_eq!(stats.skipped_lines, 1);
    }
    // In Strict mode it is a typed, line-numbered error.
    {
        let input = javaref("edge/t5.txt");
        if input.exists() {
            let mut node_map = new_node_map(NodeMapKind::Dense, Mode::Strict, None);
            let mut sink = NullArcSink::new();
            let reader = BufReader::new(fs::File::open(&input).unwrap());
            let err = build_edge_list(
                reader,
                &mut *node_map,
                &mut sink,
                default_opts(Mode::Strict, OnMissingSource::Fail),
            )
            .expect_err("t5 must fail in strict mode");
            assert!(
                matches!(err, PgError::BadInteger { .. }),
                "t5: expected BadInteger, got {err:?}"
            );
        }
    }

    // --- t6: txId = -5. `Integer.parseInt("-5")` succeeded in Java, the value
    // round-tripped through pack/unpack, and the node map printed `-5 0 0`.
    // Lenient is bug-compatible; Strict rejects it.
    if let Some((_, arcs, nm)) = run_edge_case("t6", dir.path(), Mode::Lenient) {
        assert_eq!(arcs, reference_case("t6", "el"), "t6: Java emitted `0\t1`");
        let (canon, rows) = canonical_node_map(&nm);
        let (ref_canon, ref_rows) = canonical_node_map(&reference_case("t6", "nm"));
        assert_eq!(rows, ref_rows);
        assert_eq!(
            canon, ref_canon,
            "t6: negative txId survives the round trip"
        );
    }
    {
        let input = javaref("edge/t6.txt");
        if input.exists() {
            let mut node_map = new_node_map(NodeMapKind::Dense, Mode::Strict, None);
            let mut sink = NullArcSink::new();
            let reader = BufReader::new(fs::File::open(&input).unwrap());
            let err = build_edge_list(
                reader,
                &mut *node_map,
                &mut sink,
                default_opts(Mode::Strict, OnMissingSource::Fail),
            )
            .expect_err("t6 must fail in strict mode");
            assert!(
                matches!(err, PgError::NegativeId { .. }),
                "t6: expected NegativeId, got {err:?}"
            );
        }
    }

    // --- t7: a transaction spending its own output. Outputs are registered
    // BEFORE the input loop (Java lines 59-67 precede 68-80), so the lookup
    // succeeds and a SELF-LOOP is emitted. MUST match Java exactly.
    if let Some((stats, arcs, nm)) = run_edge_case("t7", dir.path(), Mode::Lenient) {
        assert_eq!(arcs, reference_case("t7", "el"), "t7: self-loop `0\t0`");
        assert_eq!(arcs, b"0\t0\n".to_vec());
        assert_eq!(stats.edge_count, 1);
        let (canon, _) = canonical_node_map(&nm);
        let (ref_canon, _) = canonical_node_map(&reference_case("t7", "nm"));
        assert_eq!(canon, ref_canon);
    }

    // --- t8: two inputs of the same transaction resolving to the same source
    // node, so the m x n join emits `0\t2` twice. Duplicates are NOT filtered
    // by the builder (`build_pg.sh` deduped afterwards). MUST match Java.
    if let Some((stats, arcs, nm)) = run_edge_case("t8", dir.path(), Mode::Lenient) {
        assert_eq!(arcs, reference_case("t8", "el"), "t8: `0\t2` emitted twice");
        assert_eq!(arcs, b"0\t2\n0\t2\n".to_vec());
        assert_eq!(stats.edge_count, 2);
        let (canon, rows) = canonical_node_map(&nm);
        let (ref_canon, ref_rows) = canonical_node_map(&reference_case("t8", "nm"));
        assert_eq!(rows, ref_rows);
        assert_eq!(canon, ref_canon);
    }
}

// ------------------------------------------------------------ full pipeline

fn sort_opts(tmp: &Path) -> SortOpts {
    SortOpts {
        algo: SortAlgo::Radix,
        codec: ArcCodec::Packed32,
        dedup: true,
        memory_bytes: 64 * 1024 * 1024,
        tmp_dir: tmp.to_path_buf(),
        threads: 1,
        keep_intermediate: false,
    }
}

fn compress_opts(num_nodes: usize, tmp: &Path) -> CompressOpts {
    CompressOpts {
        num_nodes,
        tmp_dir: tmp.to_path_buf(),
        threads: 1,
        memory_bytes: 64 * 1024 * 1024,
        // Sequential is the byte-identical-to-Java path.
        parallel: false,
        build_ef: false,
        allow_empty: false,
        // The differential test recounts every arc on purpose.
        verify: VerifyLevel::Full,
    }
}

/// Reads a BVGraph back and returns its arcs in `(src, dst)` order.
fn read_back(basename: &Path) -> (usize, Vec<(usize, usize)>) {
    let graph = BvGraphSeq::with_basename(basename)
        .endianness::<BE>()
        .load()
        .expect("load the compressed graph");
    let num_nodes = graph.num_nodes();
    let mut arcs = Vec::new();
    for_!((src, succ) in graph.iter() {
        for dst in succ {
            arcs.push((src, dst));
        }
    });
    (num_nodes, arcs)
}

#[test]
fn test_full_pipeline_chunk01() {
    if !available(&[CHUNK_01]) {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let tmp = dir.path().join("tmp");
    fs::create_dir_all(&tmp).unwrap();

    let mut node_map = new_node_map(NodeMapKind::Dense, Mode::Strict, None);
    let mut sorter = ArcSorter::new(sort_opts(&tmp)).expect("ArcSorter");
    let reader = BufReader::new(fs::File::open(CHUNK_01).expect("open chunk_01"));
    let stats = build_edge_list(
        reader,
        &mut *node_map,
        &mut sorter,
        default_opts(Mode::Strict, OnMissingSource::Fail),
    )
    .expect("build_edge_list");
    assert_eq!(stats.edge_count, 1094);
    let next_id = node_map.next_id();
    assert_eq!(next_id, 18620);

    let (sorted, sort_stats) = sorter.into_sorted().expect("into_sorted");
    assert_eq!(sort_stats.raw_arcs, 1094);
    assert_eq!(
        sort_stats.duplicates_removed, 0,
        "chunk_01 contains no duplicate arcs"
    );
    assert_eq!(sorted.num_arcs(), 1094);
    assert_eq!(sorted.max_node_id(), 18544, "the largest endpoint in el_01");

    // The canonical `sort | uniq` artefact that `build_pg.sh` produced.
    let el_path = dir.path().join("pg_el.tsv");
    let written = sorted.write_tsv(&el_path).expect("write_tsv");
    assert_eq!(written, 1094);
    let el_bytes = fs::read(&el_path).unwrap();
    assert_eq!(
        md5_hex(&el_bytes),
        "941a3876f734b9d75c36475d94266c66",
        "sorted, deduplicated edge list md5"
    );

    // --- from-arcs: what Java's ArcListASCIIGraph inferred, max(id) + 1.
    let from_arcs = sorted.max_node_id() as usize + 1;
    assert_eq!(from_arcs, 18545);
    let base_a = dir.path().join("pg_a");
    let cstats = compress::compress_sorted(&sorted, &base_a, &compress_opts(from_arcs, &tmp))
        .expect("compress_sorted");
    assert_eq!(cstats.nodes, from_arcs);
    assert_eq!(cstats.arcs, 1094);
    assert!(cstats.bits > 0);
    compress::verify_graph(&base_a, from_arcs, 1094, VerifyLevel::Full).expect("verify_graph");
    for ext in ["graph", "offsets", "properties"] {
        let p = base_a.with_extension(ext);
        assert!(p.exists(), "{} must exist", p.display());
        assert!(fs::metadata(&p).unwrap().len() > 0);
    }

    // Reading the graph back must reproduce the exact sorted arc list.
    let expected: Vec<(usize, usize)> = std::str::from_utf8(&el_bytes)
        .unwrap()
        .lines()
        .map(|l| {
            let mut f = l.split('\t');
            (
                f.next().unwrap().parse::<usize>().unwrap(),
                f.next().unwrap().parse::<usize>().unwrap(),
            )
        })
        .collect();
    let (nodes_a, arcs_a) = read_back(&base_a);
    assert_eq!(nodes_a, from_arcs);
    assert_eq!(arcs_a, expected, "the BVGraph must round-trip the arc list");

    // --- node-map: every UTXO is a node. The counts DIFFER, and that
    // difference is the semantic divergence the README calls out: the 75
    // highest-numbered outputs of chunk_01 are unspent and appear in no arc.
    let base_b = dir.path().join("pg_b");
    let bstats =
        compress::compress_sorted(&sorted, &base_b, &compress_opts(next_id as usize, &tmp))
            .expect("compress_sorted with the node-map count");
    assert_eq!(bstats.nodes, 18620);
    compress::verify_graph(&base_b, 18620, 1094, VerifyLevel::Full).expect("verify_graph");
    let (nodes_b, arcs_b) = read_back(&base_b);
    assert_eq!(nodes_b, 18620);
    assert_ne!(
        nodes_b, nodes_a,
        "`--num-nodes node-map` and `from-arcs` must disagree here (18620 vs 18545)"
    );
    assert_eq!(nodes_b - nodes_a, 75);
    assert_eq!(
        arcs_b, expected,
        "the extra nodes are isolated; the arc set is unchanged"
    );
}

// ---------------------------------------------------- dense vs hash node map

#[test]
fn test_dense_vs_hash_agree() {
    if !available(&[CHUNK_01]) {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let (dense_stats, dense_arcs, dense_nm) = run_builder(
        Path::new(CHUNK_01),
        dir.path(),
        NodeMapKind::Dense,
        Mode::Strict,
        OnMissingSource::Fail,
        "dense",
    );
    let (hash_stats, hash_arcs, hash_nm) = run_builder(
        Path::new(CHUNK_01),
        dir.path(),
        NodeMapKind::Hash,
        Mode::Strict,
        OnMissingSource::Fail,
        "hash",
    );

    // The cheap, decisive validation that the dense optimisation is sound.
    assert_eq!(
        dense_stats,
        hash_stats_without_time(&hash_stats, &dense_stats)
    );
    assert_eq!(dense_arcs, hash_arcs, "identical arc bytes");
    assert_eq!(
        canonical_node_map(&dense_nm),
        canonical_node_map(&hash_nm),
        "identical node-map sets"
    );
}

/// `elapsed_secs` is wall-clock and may differ between two runs; everything
/// else must be identical.
fn hash_stats_without_time(hash: &EdgeListStats, dense: &EdgeListStats) -> EdgeListStats {
    EdgeListStats {
        elapsed_secs: dense.elapsed_secs,
        ..*hash
    }
}

// ------------------------------------------------------------ sanity checks

#[test]
fn node_map_and_sink_traits_are_object_safe() {
    // A compile-time check that the trait objects `main.rs` relies on really
    // are object-safe, so a refactor of either trait breaks here first.
    let mut map = new_node_map(NodeMapKind::Dense, Mode::Strict, Some(4));
    let map_ref: &mut dyn NodeMap = &mut *map;
    let mut out = Vec::<NodeId>::new();
    map_ref.register_tx(0, 2, 1, &mut out).expect("register");
    assert_eq!(out, vec![0, 1]);
    assert_eq!(map_ref.lookup(0, 1), Some(1));

    let mut sink = NullArcSink::new();
    let sink_ref: &mut dyn ArcSink = &mut sink;
    sink_ref.push(0, 1).expect("push");
    sink_ref.finish().expect("finish");
    assert_eq!(sink.count, 1);
}
