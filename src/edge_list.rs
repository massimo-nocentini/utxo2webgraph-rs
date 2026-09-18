//! Edge-list builder: the streaming pass that turns a transaction list into
//! node ids and a raw arc stream.
//!
//! # Attribution
//!
//! The pipeline this stage belongs to, and the graph-construction algorithm it
//! implements, are the work of **Matteo Loporchio**; what follows is an
//! independent reimplementation of that design.
//!
//! The streaming loop registers a transaction's output nodes, resolves its
//! inputs against the node map, emits the complete bipartite join of
//! `inputs x outputs`, and prints two summary lines on stdout.
//!
//! # The output contract
//!
//! * the `m x n` complete bipartite join: `m` inputs and `n` outputs emit
//!   exactly `m * n` arcs, with no pair omitted and none invented;
//! * a fixed emission order — line, then input index, then output offset — so
//!   the raw TSV edge list is byte-reproducible without any sorting, and is
//!   pinned as such by the golden fixtures under `tests/data/reference`;
//! * duplicate arcs and self-loops are emitted, never filtered (dedup happens
//!   later, in `crate::arcs`);
//! * `Nodes:` counts output **slots**, not distinct nodes; see [`print_stats`];
//! * the final `Processed:` line duplicates the last progress line at exact
//!   multiples of `--progress-every`;
//! * elapsed seconds are measured from before the read loop and include
//!   node-map writing.
//!
//! # The failure modes this stage refuses to have
//!
//! Every row is a way a streaming pass over 132 GB can lose or corrupt a
//! multi-hour run without saying so, and the rule that prevents it:
//!
//! | input | what happens here |
//! |---|---|
//! | a transaction with an empty output section | zero outputs ⇒ zero edges, never a phantom edge into node `0` |
//! | a blank or truncated line | skipped and tallied, or a line-numbered error — never an unchecked index into a short field list |
//! | an input referencing an unregistered output | [`Error::DanglingSource`] naming the line, the transaction and the pair; `--on-missing-source skip\|create` to continue |
//! | a disk-full while writing | every write checked and `sync_all`'d before the statistics, so a truncated file can never be reported as a success |
//! | the node-map dump | ascending `(txId, offset)`, so two runs over one input produce byte-identical files |
//! | a non-UTF-8 byte | reported with its line and byte offset in strict mode; U+FFFD and a tally under `--lenient` |
//!
//! This module deliberately depends on nothing outside `std` and this crate:
//! no `webgraph`, no `rayon`, no `clap`, no `anyhow`.

use std::borrow::Cow;
use std::fs::File;
use std::io::{BufRead, BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use dsi_progress_logger::prelude::*;

use crate::nodemap::DenseNodeMap;
use crate::record::{self, ParseOpts};
use crate::{
    sync_if_durable, ArcSink, Error, Mode, NodeId, OnMissingSource, Result, StatsStyle,
    DEFAULT_PROGRESS_EVERY,
};

/// Buffer size for the node-map writer.
const NODE_MAP_BUF: usize = 8 << 20;

/// Options for [`build_edge_list`].
#[derive(Copy, Clone, Debug)]
pub struct EdgeListOpts {
    /// Strict or lenient handling of malformed records.
    pub mode: Mode,
    /// What to do with an input that references an unregistered output.
    pub on_missing_source: OnMissingSource,
    /// Transactions between progress lines; `0` disables them.
    pub progress_every: u64,
    /// Which statistics [`print_stats`] emits.
    pub stats: StatsStyle,
    /// Suppress all stdout output, including progress lines.
    pub quiet_stats: bool,
}

impl Default for EdgeListOpts {
    fn default() -> Self {
        EdgeListOpts {
            mode: Mode::default(),
            on_missing_source: OnMissingSource::default(),
            progress_every: DEFAULT_PROGRESS_EVERY,
            stats: StatsStyle::default(),
            quiet_stats: false,
        }
    }
}

/// Counters collected by [`build_edge_list`].
///
/// `node_slots` and `edge_count` are the two the `Nodes:`/`Edges:` line
/// reports; every other field exists so that a run which did something
/// unexpected can say what, without being re-read.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct EdgeListStats {
    /// Transactions processed: every line that parsed.
    pub tx_count: u64,
    /// Output **slots**: incremented once per declared output, inside the
    /// per-output loop, without checking whether that output actually created
    /// a node. This is the number `Nodes:` prints; see [`print_stats`].
    pub node_slots: u64,
    /// Distinct `(txId, offset)` nodes the node map actually holds.
    pub distinct_nodes: u64,
    /// Emitted arcs, including duplicates and self-loops.
    pub edge_count: u64,
    /// Non-blank lines skipped because they were malformed (lenient mode).
    pub skipped_lines: u64,
    /// Blank or whitespace-only lines skipped (both modes).
    pub blank_lines: u64,
    /// Inputs whose source node was unknown, under `skip` or `create`.
    pub dangling_refs: u64,
    /// Transactions with an empty output section.
    pub zero_output_txs: u64,
    /// Lines that were not valid UTF-8 and were decoded with U+FFFD
    /// substitution (lenient mode only; strict mode reports
    /// [`Error::InvalidUtf8`]).
    pub non_utf8_lines: u64,
    /// Wall-clock seconds, truncated, measured from before the read loop.
    pub elapsed_secs: u64,
}

/// Runs the edge-list pass over `input`.
///
/// Node ids are assigned through `node_map` and arcs are pushed into `sink` in
/// emission order; `sink.finish()` is called exactly once, on success.
///
/// # Correctness properties
///
/// * The graph is the **complete bipartite join**: a transaction with `m`
///   inputs and `n` outputs emits exactly `m * n` arcs. Verified on
///   `chunk_01.txt`: the sum over lines of `#inputs * #outputs` is 1094, which
///   is exactly what this pass reports as `Edges:`.
/// * Emission order is fully deterministic — line order, then input index,
///   then output offset — so the raw edge list is byte-reproducible without
///   any sorting. That is why the TSV sink's output can be compared
///   byte-for-byte against the stored reference edge list (`chunk_01` md5
///   `a3c31369f1a54bacfeb9b3cc2c953ebe`).
/// * Duplicates and self-loops are emitted, not filtered.
/// * A coinbase emits no arcs but still creates its output nodes — 18 445 of
///   `chunk_01`'s 18 578 lines are coinbases.
///
/// # Timing
///
/// The returned `elapsed_secs` covers the read loop **only**, but the figure
/// the `Processed:` line is expected to carry covers the node-map dump as
/// well — at `N = 28` that dump is 55 GB and is not a rounding error. The
/// caller must therefore write the node map first and only then call
/// [`print_stats`], with `elapsed_secs` refreshed from a clock it owns.
/// `main.rs` does exactly that and overwrites the field.
///
/// # Progress
///
/// `pl` is ticked once per transaction and is entirely separate from the
/// `--progress-every` stdout line, which is part of the stdout format contract
/// and is not a log. Pass `no_logging!()` to suppress it.
pub fn build_edge_list<R: BufRead>(
    mut input: R,
    node_map: &mut DenseNodeMap,
    sink: &mut dyn ArcSink,
    opts: EdgeListOpts,
    pl: &mut impl ProgressLog,
) -> Result<EdgeListStats> {
    let start = Instant::now();
    // No `expected_updates`: the transaction count is one line per input line
    // and nothing has counted the lines. An estimate here would put a wrong
    // percentage and a wrong ETA on every progress line.
    pl.item_name("transaction");
    pl.start("Building the edge list...");
    let parse_opts = ParseOpts { mode: opts.mode };
    // Reused across the whole run: never reallocated per line.
    let mut out_ids: Vec<NodeId> = Vec::with_capacity(1024);
    // Bytes, not a `String`: `BufRead::read_line` validates UTF-8 and turns a
    // single stray byte anywhere in 132 GB into `ErrorKind::InvalidData`,
    // which named neither the file nor the line and which `--lenient` could
    // not skip. Decoding by hand is what lets strict mode report the line and
    // the byte offset, and lenient mode substitute U+FFFD and carry on. Every
    // field this parser reads is ASCII digits, so the validation pass is not
    // needed for correctness either.
    let mut line_buf: Vec<u8> = Vec::with_capacity(256);
    let mut stats = EdgeListStats::default();
    let mut line_no: u64 = 0;

    loop {
        line_buf.clear();
        let read = input.read_until(b'\n', &mut line_buf)?;
        if read == 0 {
            break;
        }
        line_no += 1;

        let line: Cow<'_, str> = match std::str::from_utf8(&line_buf) {
            Ok(text) => Cow::Borrowed(text),
            Err(e) => {
                if opts.mode == Mode::Strict {
                    return Err(Error::InvalidUtf8 {
                        line: line_no,
                        offset: e.valid_up_to(),
                    });
                }
                // Lenient: substitute and keep going. The offending bytes
                // are almost always in a label field nobody parses.
                stats.non_utf8_lines += 1;
                String::from_utf8_lossy(&line_buf)
            }
        };

        let rec = match record::parse_line(&line, line_no, parse_opts)? {
            Some(rec) => rec,
            None => {
                if line.trim().is_empty() {
                    stats.blank_lines += 1;
                } else {
                    stats.skipped_lines += 1;
                }
                continue;
            }
        };

        // The outputs are registered BEFORE the inputs are resolved, which is
        // what makes a transaction that spends one of its own outputs produce
        // a self-loop rather than a dangling reference.
        node_map.register_tx(rec.tx_id, rec.num_outputs, line_no, &mut out_ids)?;
        // Incremented per output slot, unconditionally — including for a slot
        // a repeated transaction id already owns. See `node_slots`.
        stats.node_slots += rec.num_outputs as u64;
        if rec.num_outputs == 0 {
            stats.zero_output_txs += 1;
        }

        if !rec.is_coinbase() {
            for (index, item) in rec.inputs().enumerate() {
                let (prev_tx, prev_off) = item?;
                let src = match node_map.lookup(prev_tx, prev_off) {
                    Some(id) => id,
                    None => match opts.on_missing_source {
                        // The lookup found nothing. Reported as a typed error
                        // naming the line, the transaction, the input index
                        // and the missing pair: the alternative is a run that
                        // dies with both output files at 0 bytes behind a
                        // message that names neither the line nor the
                        // transaction.
                        OnMissingSource::Fail => {
                            return Err(Error::DanglingSource {
                                line: line_no,
                                tx_id: rec.tx_id,
                                index,
                                prev_tx_id: prev_tx,
                                prev_offset: prev_off,
                            })
                        }
                        OnMissingSource::Skip => {
                            stats.dangling_refs += 1;
                            continue;
                        }
                        OnMissingSource::Create => {
                            stats.dangling_refs += 1;
                            node_map.force_create(prev_tx, prev_off, line_no)?
                        }
                    },
                };
                // Ascending output offset, so the emission order is fixed.
                for &dst in out_ids.iter() {
                    sink.push(src, dst)?;
                    stats.edge_count += 1;
                }
            }
        }

        stats.tx_count += 1;
        pl.light_update();
        if opts.progress_every != 0
            && stats.tx_count % opts.progress_every == 0
            && !opts.quiet_stats
        {
            print_progress(stats.tx_count, start.elapsed().as_secs());
        }
    }

    sink.finish()?;
    stats.distinct_nodes = node_map.distinct_nodes();
    stats.elapsed_secs = start.elapsed().as_secs();
    pl.done_with_count(stats.tx_count as usize);
    Ok(stats)
}

/// Writes the node map to `path` and returns the number of bytes written.
///
/// The trap this function exists to avoid is a writer whose errors are never
/// looked at: a disk-full during a multi-hour run over 132 GB of chunks then
/// truncates the node map silently, while the program goes on to print
/// `Nodes: … Edges: …` and exit 0 — a successful-looking run whose output is
/// short by however much did not fit. Here every write is checked, and the
/// file is flushed **and** `sync_all`'d before the caller prints its
/// statistics.
///
/// `pl` counts **bytes**, because that is the only quantity this function sees
/// incrementally: the node map is rendered by `DenseNodeMap::write_tsv` in one
/// call, so there is no row-by-row hook to tick. At `N = 28` this writes 55 GB
/// and used to be a single silent phase.
pub fn write_node_map(
    node_map: &DenseNodeMap,
    path: &Path,
    pl: &mut impl ProgressLog,
) -> Result<u64> {
    let file = File::create(path).map_err(|e| Error::io(path, e))?;
    pl.item_name("byte");
    // No path in the message: `path` here is the caller's atomic *temporary*
    // file, and every caller has already logged the destination it will be
    // renamed to. Naming both logged the same phase twice, under two different
    // paths, one of which never survives the run.
    pl.start("Writing the node map...");
    let mut w = CountingWriter {
        inner: BufWriter::with_capacity(NODE_MAP_BUF, file),
        written: 0,
        pl,
    };
    node_map.write_tsv(&mut w)?;
    let CountingWriter {
        mut inner,
        written,
        pl,
    } = w;
    inner.flush().map_err(|e| Error::io(path, e))?;
    // `fsync` on `/dev/null` or a fifo returns EINVAL; a fully written node
    // map must not be reported as a failure because of that.
    sync_if_durable(inner.get_ref(), path)?;
    pl.done_with_count(written as usize);
    Ok(written)
}

/// A `Write` that tallies the bytes it forwards and reports them to a
/// [`ProgressLog`].
///
/// `update_with_count`, not `light_update`: one call here covers a whole
/// buffer, and `light_update` only consults the clock once every 2^20 calls,
/// so a writer that makes one call per 8 KiB block would log roughly once per
/// 8 GiB regardless of `--log-interval`.
struct CountingWriter<'a, P: ProgressLog> {
    inner: BufWriter<File>,
    written: u64,
    pl: &'a mut P,
}

impl<P: ProgressLog> Write for CountingWriter<'_, P> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.written += n as u64;
        self.pl.update_with_count(n);
        Ok(n)
    }
    fn write_all(&mut self, buf: &[u8]) -> std::io::Result<()> {
        self.inner.write_all(buf)?;
        self.written += buf.len() as u64;
        self.pl.update_with_count(buf.len());
        Ok(())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// Writes one line to stdout, ignoring a broken pipe.
///
/// stdout is a compatibility contract here — `logs/pg_el_builder.log` is
/// expected to stay diffable — so the statistics never go through the logger.
fn emit(line: &str) {
    let stdout = std::io::stdout();
    let mut lock = stdout.lock();
    let _ = lock.write_all(line.as_bytes());
    let _ = lock.flush();
}

/// Prints the periodic progress line to **stdout**:
///
/// ```text
/// Processed: {tx_count} transactions (after {elapsed_secs} seconds).
/// ```
pub fn print_progress(tx_count: u64, elapsed_secs: u64) {
    emit(&format!(
        "Processed: {tx_count} transactions (after {elapsed_secs} seconds).\n"
    ));
}

/// Prints the final statistics to **stdout**.
///
/// Under [`StatsStyle::Brief`] exactly two lines:
///
/// ```text
/// Processed: {tx_count} transactions (after {elapsed_secs} seconds).
/// Nodes: {node_slots}\tEdges: {edge_count}
/// ```
///
/// The first line is unconditional and therefore **duplicates** the last
/// progress line whenever `tx_count` is an exact multiple of the progress
/// period — with the same count and a possibly larger elapsed time. It is
/// emitted anyway, because these two lines are a stdout format contract: the
/// historical pipeline logs in `logs/pg_el_builder.log` are expected to stay
/// diffable against a fresh run, and suppressing the duplicate would put a
/// spurious deletion in every such diff.
///
/// `Nodes:` counts output **slots**, not distinct nodes: the counter is
/// bumped once per declared output, without checking whether that output
/// created a node. The two differ by exactly the number of output slots
/// belonging to a re-emitted transaction id — two, over the whole corpus, for
/// the BIP-30 duplicate coinbases described in [`crate::nodemap`] — plus any
/// node minted by [`crate::OnMissingSource::Create`]. The label is kept as it
/// is for continuity with those logs, and nothing is hidden by it:
/// [`StatsStyle::Extended`] prints **both** counts, the distinct-node count
/// beside the slot count, so the discrepancy is visible on demand.
///
/// Under [`StatsStyle::Extended`] two further lines follow with the corrected
/// counters.
pub fn print_stats(stats: &EdgeListStats, style: StatsStyle) {
    print_progress(stats.tx_count, stats.elapsed_secs);
    emit(&format!(
        "Nodes: {}\tEdges: {}\n",
        stats.node_slots, stats.edge_count
    ));
    if style == StatsStyle::Extended {
        emit(&format!(
            "Distinct nodes: {}\tOutput slots: {}\n",
            stats.distinct_nodes, stats.node_slots
        ));
        emit(&format!(
            "Skipped lines: {}\tBlank lines: {}\tDangling refs: {}\tZero-output txs: {}\n",
            stats.skipped_lines, stats.blank_lines, stats.dangling_refs, stats.zero_output_txs
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Collects arcs in emission order.
    struct VecSink {
        arcs: Vec<(NodeId, NodeId)>,
        finished: bool,
    }

    impl VecSink {
        fn new() -> Self {
            VecSink {
                arcs: Vec::new(),
                finished: false,
            }
        }
    }

    impl ArcSink for VecSink {
        fn push(&mut self, src: NodeId, dst: NodeId) -> Result<()> {
            self.arcs.push((src, dst));
            Ok(())
        }
        fn finish(&mut self) -> Result<()> {
            assert!(!self.finished, "finish() must be called exactly once");
            self.finished = true;
            Ok(())
        }
    }

    /// The third element is the node map's `next_id`, i.e. a [`NodeId`] and
    /// not a count — hence `NodeId` rather than the `u64` the statistics use.
    fn run(input: &str, opts: EdgeListOpts) -> (EdgeListStats, Vec<(NodeId, NodeId)>, NodeId) {
        let mut map = DenseNodeMap::new(opts.mode);
        let mut sink = VecSink::new();
        let stats =
            build_edge_list(input.as_bytes(), &mut map, &mut sink, opts, no_logging!()).unwrap();
        assert!(sink.finished);
        (stats, sink.arcs, map.next_id())
    }

    fn quiet() -> EdgeListOpts {
        EdgeListOpts {
            quiet_stats: true,
            ..Default::default()
        }
    }

    /// A lenient run substitutes U+FFFD and carries on. This pass used to
    /// abort the whole run instead with `stream did not contain valid UTF-8`,
    /// naming neither the file nor the line, and `--lenient` could not skip
    /// it; strict mode now names both, and lenient mode only tallies it.
    #[test]
    fn invalid_utf8_is_lossy_when_lenient_and_line_numbered_when_strict() {
        let mut input: Vec<u8> = Vec::new();
        input.extend_from_slice(b"1231006505,0,0,1,0,204,0::0,5000000000,1\n");
        // A second, well-formed record whose trailing output label is garbage.
        input.extend_from_slice(b"1231006600,1,1,0,0,0,0:0,50,0,0:");
        input.extend_from_slice(&[0xFF, 0xFE]);
        input.extend_from_slice(b",1,1\n");

        // Lenient: the bad bytes are in a label field nobody parses, so the
        // record still yields its arc.
        let mut map = DenseNodeMap::new(Mode::Lenient);
        let mut sink = VecSink::new();
        let stats = build_edge_list(
            input.as_slice(),
            &mut map,
            &mut sink,
            EdgeListOpts {
                mode: Mode::Lenient,
                quiet_stats: true,
                ..Default::default()
            },
            no_logging!(),
        )
        .expect("a lenient run survives a stray byte");
        assert_eq!(stats.tx_count, 2);
        assert_eq!(stats.non_utf8_lines, 1);
        assert_eq!(sink.arcs, vec![(0, 1)]);

        // Strict: a typed error that names the line and the byte offset.
        let mut map = DenseNodeMap::new(Mode::Strict);
        let mut sink = VecSink::new();
        let err = build_edge_list(
            input.as_slice(),
            &mut map,
            &mut sink,
            EdgeListOpts {
                quiet_stats: true,
                ..Default::default()
            },
            no_logging!(),
        )
        .expect_err("strict mode reports it");
        match err {
            Error::InvalidUtf8 { line, offset } => {
                assert_eq!(line, 2);
                assert_eq!(offset, 32);
            }
            other => panic!("expected InvalidUtf8, got {other:?}"),
        }
    }

    #[test]
    fn genesis_only() {
        let (stats, arcs, next_id) = run("1231006505,0,0,1,0,204,0::0,5000000000,1\n", quiet());
        assert_eq!(stats.tx_count, 1);
        assert_eq!(stats.node_slots, 1);
        assert_eq!(stats.edge_count, 0);
        assert_eq!(stats.distinct_nodes, 1);
        assert_eq!(next_id, 1);
        assert!(arcs.is_empty());
    }

    #[test]
    fn two_line_chain() {
        let input = "1231006505,0,0,1,0,204,0::0,5000000000,1\n\
                     1231006600,1,1,0,0,0,0:0,50,0,0:a,1,1;b,2,2\n";
        let (stats, arcs, _) = run(input, quiet());
        assert_eq!(arcs, vec![(0, 1), (0, 2)]);
        assert_eq!(stats.node_slots, 3);
        assert_eq!(stats.edge_count, 2);
        assert_eq!(stats.tx_count, 2);
    }

    #[test]
    fn complete_bipartite_join_is_input_major() {
        let input = "1,0,0,1,0,0,0::a,1,1;b,2,2\n\
                     2,1,1,0,0,0,0:x,5,0,0;y,5,0,1:p,1,1;q,2,2;r,3,3\n";
        let (stats, arcs, _) = run(input, quiet());
        assert_eq!(arcs, vec![(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4)]);
        assert_eq!(stats.edge_count, 6);
        assert_eq!(stats.node_slots, 5);
    }

    #[test]
    fn self_spend_yields_a_self_loop() {
        // The outputs are registered before the inputs are resolved, so the
        // lookup succeeds — the behaviour pinned by the `edge/t7.txt`
        // fixture, whose expected edge list is the single arc `0\t0`.
        let input = "1,0,0,0,0,0,0:h,5,0,0:a,1,1\n";
        let (stats, arcs, _) = run(input, quiet());
        assert_eq!(arcs, vec![(0, 0)]);
        assert_eq!(stats.edge_count, 1);
    }

    #[test]
    fn duplicate_inputs_emit_duplicate_arcs() {
        // The `edge/t8.txt` fixture contains the same pair twice; dedup is
        // the sorter's job, not this stage's.
        let input = "1,0,0,1,0,0,0::a,1,1\n\
                     2,1,1,0,0,0,0:x,5,0,0;y,5,0,0:b,2,2\n";
        let (_, arcs, _) = run(input, quiet());
        assert_eq!(arcs, vec![(0, 1), (0, 1)]);
    }

    #[test]
    fn zero_outputs_emit_no_edge_into_node_zero() {
        // The trap: an empty output section still splits into one (empty)
        // field, so sizing the output-id buffer from that field count leaves
        // a leftover `0` in it, and every input of such a line then emits a
        // phantom edge into the genesis output. The buffer is cleared and
        // filled by `register_tx` instead, so zero outputs means zero edges.
        let input = "1,0,0,1,0,0,0::a,1,1\n\
                     2,1,1,0,0,0,0:x,5,0,0::trailing\n";
        let opts = EdgeListOpts {
            mode: Mode::Lenient,
            ..quiet()
        };
        let (stats, arcs, _) = run(input, opts);
        assert!(arcs.is_empty(), "no phantom edge into node 0");
        assert_eq!(stats.zero_output_txs, 1);
        assert_eq!(stats.edge_count, 0);
        assert_eq!(stats.node_slots, 1);
    }

    #[test]
    fn blank_lines_are_tallied_not_fatal() {
        let input = "1,0,0,1,0,0,0::a,1,1\n\n   \n";
        let (stats, _, _) = run(input, quiet());
        assert_eq!(stats.tx_count, 1);
        assert_eq!(stats.blank_lines, 2);
        assert_eq!(stats.skipped_lines, 0);
    }

    #[test]
    fn dangling_source_fails_by_default() {
        let input = "1,0,0,1,0,0,0::a,1,1\n\
                     2,1,1,0,0,0,0:x,5,999,0:b,2,2\n";
        let mut map = DenseNodeMap::new(Mode::Strict);
        let mut sink = VecSink::new();
        match build_edge_list(
            input.as_bytes(),
            &mut map,
            &mut sink,
            quiet(),
            no_logging!(),
        ) {
            Err(Error::DanglingSource {
                line: 2,
                tx_id: 1,
                index: 0,
                prev_tx_id: 999,
                prev_offset: 0,
            }) => {}
            other => panic!("expected DanglingSource, got {other:?}"),
        }
    }

    #[test]
    fn dangling_source_can_be_skipped() {
        let input = "1,0,0,1,0,0,0::a,1,1\n\
                     2,1,1,0,0,0,0:x,5,999,0:b,2,2\n";
        let opts = EdgeListOpts {
            on_missing_source: OnMissingSource::Skip,
            ..quiet()
        };
        let (stats, arcs, next_id) = run(input, opts);
        assert!(arcs.is_empty());
        assert_eq!(stats.dangling_refs, 1);
        assert_eq!(next_id, 2);
    }

    /// The reachable `create` case: a chunk that starts above genesis (txIds
    /// 10, 11) and spends an output of transaction 5, which lives in an earlier
    /// chunk this run does not have. The reference is *behind* the cursor, so a
    /// node can be minted for it without ever colliding with a dense id.
    #[test]
    fn dangling_source_can_be_created() {
        let input = "1,0,10,1,0,0,0::a,1,1\n\
                     2,1,11,0,0,0,0:x,0,5,0:b,2,2\n";
        let opts = EdgeListOpts {
            on_missing_source: OnMissingSource::Create,
            ..quiet()
        };
        let (stats, arcs, next_id) = run(input, opts);
        // tx 11's own output took id 1, so the minted node is id 2.
        assert_eq!(arcs, vec![(2, 1)]);
        assert_eq!(stats.dangling_refs, 1);
        assert_eq!(next_id, 3);
    }

    /// A dangling reference to a transaction the run has **not read yet**
    /// cannot be minted: the dense cursor would reach that transaction later
    /// and hand the same output a second id. `create` refuses rather than
    /// corrupt the node map; `skip` and `fail` are unaffected because neither
    /// mints anything.
    #[test]
    fn forward_dangling_source_cannot_be_created() {
        // tx 0 spends (5, 0); transaction 5 arrives three lines later.
        let input = "1,0,0,1,0,0,0:x,0,5,0:a,1,1\n\
                     1,0,1,1,0,0,0::a,1,1\n\
                     1,0,2,1,0,0,0::a,1,1\n\
                     1,0,3,1,0,0,0::a,1,1\n\
                     1,0,4,1,0,0,0::a,1,1\n\
                     1,0,5,1,0,0,0::a,1,1\n";
        let opts = EdgeListOpts {
            on_missing_source: OnMissingSource::Create,
            ..quiet()
        };
        let mut map = DenseNodeMap::new(Mode::Strict);
        let mut sink = VecSink::new();
        match build_edge_list(input.as_bytes(), &mut map, &mut sink, opts, no_logging!()) {
            Err(Error::ForwardForcedNode {
                line: 1,
                tx_id: 5,
                offset: 0,
                reached: 0,
            }) => {}
            other => panic!("expected ForwardForcedNode, got {other:?}"),
        }

        // And the same input is fine under the two non-minting policies.
        let (stats, arcs, next_id) = run(
            input,
            EdgeListOpts {
                on_missing_source: OnMissingSource::Skip,
                ..quiet()
            },
        );
        assert!(arcs.is_empty());
        assert_eq!(stats.dangling_refs, 1);
        assert_eq!(next_id, 6);
    }

    #[test]
    fn strict_mode_reports_the_offending_line() {
        let input = "1,0,0,1,0,0,0::a,1,1\n\
                     not-a-record\n";
        let mut map = DenseNodeMap::new(Mode::Strict);
        let mut sink = VecSink::new();
        match build_edge_list(
            input.as_bytes(),
            &mut map,
            &mut sink,
            quiet(),
            no_logging!(),
        ) {
            Err(Error::BadSectionCount { line: 2, found: 1 }) => {}
            other => panic!("expected BadSectionCount, got {other:?}"),
        }
    }
}
