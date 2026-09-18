//! `utxo2webgraph` command line: one subcommand per pipeline stage, plus the
//! two fused stages (`build-pg`, `build`) that run the whole thing end to end.
//!
//! # Attribution
//!
//! The pipeline these subcommands drive, and the graph-construction algorithm
//! at its centre, are the work of **Matteo Loporchio**; this crate is an
//! independent reimplementation of that design.
//!
//! This module is pure declaration: the `clap` derive types, a handful of
//! `ValueEnum` mirrors of the shared-contract enums, and two tiny parsers.
//! No I/O and no business logic live here; dispatch is in `main.rs`.
//!
//! The mirrors exist so that `clap` never has to appear in the dependency
//! surface of the library's core and graph modules: deriving `ValueEnum`
//! directly on [`Mode`], [`SortAlgo`] and friends would force every module
//! that names them to compile against `clap`.

use std::path::PathBuf;
use std::time::Duration;

use clap::{ArgAction, Args, Parser, Subcommand, ValueEnum};

use crate::compress::VerifyLevel;
use crate::{Mode, OnMissingSource, SortAlgo, StatsStyle};

/// The subcommand overview, shown by `utxo2webgraph --help`.
const LONG_ABOUT: &str = "\
Build the Bitcoin Payment Graph from a transaction list.

The pipeline is four stages, one subcommand each. Every stage reads what the
previous one wrote, so any of them can be run on its own:

  split        cut a whole-chain transaction list into six-month chunk_NN.txt
  edge-list    read chunks; write the node map and the raw, unsorted arc stream
  sort-edges   sort those arcs by (src, dst) ascending and drop duplicates
  compress     turn a sorted arc list into PREFIX.{graph,offsets,properties}

Two subcommands fuse the last three, so the arcs never have to be written out
as decimal text at all:

  build-pg     edge-list + sort-edges + compress over one input
  build N      the same over chunks 1..N, streamed, producing
               graph/pg_nm_N.tsv, graph/pg_el_N.tsv and graph/pg_N.*

`--num-nodes` defaults to `from-arcs` everywhere: max(endpoint) + 1. That is
the smallest node count the arcs require, not a count of the outputs an arc
touches - every id below the maximum is a node whether or not an arc reaches
it, so isolated nodes in the middle are kept and only unspent outputs numbered
*above* the maximum are dropped. `--num-nodes node-map` is the opt-in richer
graph in which every UTXO, spent or not, is a node; pass it with `--node-map`
and both counts are logged so the difference is visible.

stdout carries the statistics; all logging goes to stderr.";

/// The `utxo2webgraph` command line.
#[derive(Parser, Debug)]
#[command(name = "utxo2webgraph", version, propagate_version = true,
          about = "Build the Bitcoin Payment Graph from a transaction list",
          long_about = LONG_ABOUT)]
pub struct Cli {
    /// Options shared by every stage.
    #[command(flatten)]
    pub common: CommonOpts,
    /// The stage to run.
    #[command(subcommand)]
    pub command: Command,
}

/// Options that apply to every stage.
///
/// All of them are `global`, so they may be given before *or* after the
/// subcommand: `utxo2webgraph --threads 64 build 15` and `utxo2webgraph build 15
/// --threads 64` are the same command.
#[derive(Args, Debug, Clone)]
pub struct CommonOpts {
    /// Worker threads for sorting and compression. Default: all cores.
    #[arg(long, global = true, value_name = "N")]
    pub threads: Option<usize>,

    /// Memory budget for in-memory sort runs, e.g. `128GiB`, `200G`, `45%`.
    ///
    /// This is the TOTAL budget and it INCLUDES the radix sort's scratch
    /// buffer, which is the same size as the data being sorted. The usable
    /// run size is therefore HALF this value: `arcs_per_run = memory / (2 *
    /// 16)`, an arc being one 16-byte packed record. Computing it without the
    /// factor 2 would make a nominal 128 GiB budget cost 256 GiB of RSS and
    /// OOM on the first full-scale run.
    ///
    /// Default: min(45% of total RAM, 192GiB).
    #[arg(long, global = true, value_name = "SIZE", value_parser = crate::parse_memory_spec)]
    pub memory: Option<u64>,

    /// Directory for spill runs and other intermediates.
    ///
    /// Also exported as `TMPDIR` before any thread starts, because webgraph's
    /// external sort calls bare `tempfile::tempdir()` and ignores its own
    /// `tmp_dir` setting.
    #[arg(long, global = true, value_name = "DIR", default_value = "./tmp")]
    pub tmp_dir: PathBuf,

    /// Directory for run logs.
    #[arg(long, global = true, value_name = "DIR", default_value = "./logs")]
    pub log_dir: PathBuf,

    /// Keep spill runs and intermediate binary arc files after a successful run.
    #[arg(long, global = true)]
    pub keep_intermediate: bool,

    /// Abort on the first malformed record (the default).
    #[arg(long, global = true, conflicts_with = "lenient")]
    pub strict: bool,

    /// Skip malformed records and tally them instead of aborting.
    #[arg(long, global = true, conflicts_with = "strict")]
    pub lenient: bool,

    /// Log verbosity: -v = debug, -vv = trace. `RUST_LOG` layers on top.
    #[arg(short, long, global = true, action = ArgAction::Count)]
    pub verbose: u8,

    /// How often the long loops log their progress.
    ///
    /// Suffixes `s`, `m`, `h`, `d`; a bare number is milliseconds. They
    /// accumulate: `1d2h3m4s567`.
    // Upstream webgraph-rs flattens a per-subcommand `LogIntervalArg` instead.
    // `global = true` is a deliberate divergence: every other field of this
    // struct is global, so `utxo2webgraph --log-interval 1m build 15` and `utxo2webgraph
    // build 15 --log-interval 1m` must mean the same thing, and there is no
    // stage whose progress interval would sensibly differ from the run's.
    #[arg(long, global = true, value_name = "DURATION", value_parser = parse_duration, default_value = "10s")]
    pub log_interval: Duration,

    /// Report the resolved plan and touch nothing.
    #[arg(long, global = true)]
    pub dry_run: bool,

    /// Statistics style on stdout.
    ///
    /// `brief` prints a `Processed:` line and then one tab-separated
    /// `Nodes: N\tEdges: M` line, and nothing else. `extended` adds two more
    /// tab-separated lines: the distinct-node count that `Nodes:` is commonly
    /// mistaken for, and the skipped-line, blank-line, dangling-reference and
    /// zero-output-transaction tallies.
    #[arg(long, global = true, value_enum, default_value_t = StatsArg::Brief)]
    pub stats: StatsArg,
}

impl CommonOpts {
    /// The parsing mode these flags select. Strict unless `--lenient`.
    pub fn mode(&self) -> Mode {
        if self.lenient {
            Mode::Lenient
        } else {
            Mode::Strict
        }
    }

    /// The statistics style these flags select.
    pub fn stats_style(&self) -> StatsStyle {
        self.stats.into()
    }
}

/// The pipeline stage to run.
#[derive(Subcommand, Debug)]
pub enum Command {
    /// Split a transaction list into six-month chunks. Replaces `splitter.sh`.
    Split(SplitArgs),

    /// Build the node map and the raw (unsorted) arc list from one or more
    /// transaction lists.
    EdgeList(EdgeListArgs),

    /// Sort arcs by `(src, dst)` and remove duplicates.
    /// Replaces `sort -t$'\t' -k1,1n -k2,2n | uniq`.
    SortEdges(SortEdgesArgs),

    /// Compress a sorted edge list into the BVGraph representation:
    /// `PREFIX.graph`, `PREFIX.offsets` and `PREFIX.properties`.
    Compress(CompressArgs),

    /// edge-list + sort-edges + compress over one input. Replaces
    /// `build_pg.sh`, but fused: the arcs never hit the disk as text unless
    /// asked, so the 210 GB intermediate `tmp/edge_list.tsv` disappears.
    BuildPg(BuildPgArgs),

    /// build-pg over chunks `1..N`, streamed. Replaces `builder.sh`.
    ///
    /// `utxo2webgraph build 15` reads exactly like `./builder.sh 15` and produces the
    /// same names `build_pg.sh` derived — `graph/pg_nm_15.tsv`,
    /// `graph/pg_el_15.tsv`, `graph/pg_15.{graph,offsets,properties}` — but it
    /// streams the chunks instead of materialising a 23.9 GB (at N=15) or
    /// 135.3 GB (at N=28) temporary file in the project directory.
    Build(BuildArgs),
}

// ------------------------------------------------------------------ split

/// Arguments of `utxo2webgraph split`. Replaces `splitter.sh`.
#[derive(Args, Debug, Clone)]
pub struct SplitArgs {
    /// Path to the master transaction list (e.g. `finalBCUTXO_2022`).
    #[arg(value_name = "TRANSACTION_LIST")]
    pub input: PathBuf,

    /// Directory to write `chunk_NN.txt` into.
    #[arg(short, long, value_name = "DIR", default_value = "chunks")]
    pub output_dir: PathBuf,

    /// First date, inclusive (YYYY-MM-DD).
    #[arg(long, value_name = "DATE", default_value = "2009-01-01")]
    pub start: String,

    /// Last date, exclusive (YYYY-MM-DD).
    #[arg(long, value_name = "DATE", default_value = "2023-01-01")]
    pub end: String,

    /// Months per chunk.
    #[arg(long, value_name = "N", default_value_t = 6)]
    pub months: u32,

    /// Timezone the chunk boundaries are computed in.
    ///
    /// `Europe/Rome` reproduces the existing `chunks/` directory: `splitter.sh`
    /// called `date -d ... +%s` with no `-u` on a Europe/Rome machine, so every
    /// boundary is a LOCAL midnight (UTC-1h in January, UTC-2h in July).
    /// Passing `utc` would move thousands of records between chunks and break
    /// any comparison against the 132 GB already on disk. The literal name is
    /// pinned rather than read from `$TZ` so a cron job cannot silently produce
    /// different chunks from an interactive run.
    #[arg(long, value_name = "TZ", default_value = "Europe/Rome")]
    pub tz: String,

    /// Append to existing chunk files instead of truncating them.
    ///
    /// This is what `splitter.sh` did (`>>`), and it silently DOUBLED every
    /// chunk when the script was re-run without clearing `chunks/`.
    #[arg(long)]
    pub append: bool,

    /// Do not create empty chunk files for intervals with no transactions.
    ///
    /// The awk splitter created no file at all for such an interval, after
    /// which `builder.sh` failed with "chunk file not found".
    #[arg(long)]
    pub no_create_empty: bool,

    /// Keep scanning past the first record at or after `--end`.
    ///
    /// The awk used `exit`, not `next`, so one stray future timestamp early in
    /// the master file would silently drop everything after it.
    #[arg(long)]
    pub skip_out_of_range: bool,

    /// Print the boundary table and exit.
    #[arg(long)]
    pub show_boundaries: bool,
}

// -------------------------------------------------------------- edge-list

/// Either `-i FILE...` or `--chunk-dir DIR --chunks N`.
#[derive(Args, Debug, Clone)]
#[group(required = true, multiple = false)]
pub struct InputSelector {
    /// One or more transaction-list files, concatenated in the order given.
    #[arg(short, long, value_name = "FILE", num_args = 1..)]
    pub input: Vec<PathBuf>,

    /// Use `chunk_01.txt .. chunk_NN.txt` from this directory.
    ///
    /// Streamed through a concatenating reader; never copied to disk, unlike
    /// `builder.sh`'s `combined_chunks_XXXXXX.txt`.
    #[arg(long, value_name = "DIR", requires = "chunks")]
    pub chunk_dir: Option<PathBuf>,
}

/// Arguments of `utxo2webgraph edge-list`.
#[derive(Args, Debug, Clone)]
pub struct EdgeListArgs {
    /// Where the transactions come from.
    #[command(flatten)]
    pub inputs: InputSelector,

    /// Number of leading chunks to read, with `--chunk-dir`.
    #[arg(short = 'n', long, value_name = "N")]
    pub chunks: Option<usize>,

    /// Node map output: `txId \t offset \t nodeId`, ascending `(txId, offset)`.
    #[arg(long, value_name = "FILE")]
    pub node_map: Option<PathBuf>,

    /// Raw arc output. Format follows `--arc-format`.
    #[arg(long, value_name = "FILE")]
    pub arcs: PathBuf,

    /// `binary` = little-endian 16-byte packed arcs, fixed width, read back
    /// without re-parsing a digit; `tsv` = `src \t dst` decimal text,
    /// byte-compatible with `tmp/edge_list.tsv`.
    #[arg(long, value_enum, default_value_t = ArcFormatArg::Binary)]
    pub arc_format: ArcFormatArg,

    /// Pre-size the dense node map (default: grow geometrically).
    #[arg(long, value_name = "N")]
    pub max_tx_id: Option<usize>,

    /// What to do when an input references an output that was never seen.
    #[arg(long, value_enum, default_value_t = OnMissingSourceArg::Fail)]
    pub on_missing_source: OnMissingSourceArg,

    /// Emit a `Processed:` line every N transactions.
    #[arg(long, value_name = "N", default_value_t = crate::DEFAULT_PROGRESS_EVERY)]
    pub progress_every: u64,
}

// ------------------------------------------------------------- sort-edges

/// Arguments of `utxo2webgraph sort-edges`. Replaces `sort | uniq`.
#[derive(Args, Debug, Clone)]
pub struct SortEdgesArgs {
    /// Raw arc file produced by `edge-list`.
    #[arg(value_name = "ARCS")]
    pub input: PathBuf,

    /// Input format; `auto` sniffs binary versus text.
    #[arg(long, value_enum, default_value_t = InputFormatArg::Auto)]
    pub input_format: InputFormatArg,

    /// Sorted, deduplicated text edge list (`graph/pg_el_N.tsv`).
    #[arg(long, value_name = "FILE")]
    pub edge_list: Option<PathBuf>,

    /// Sorted, deduplicated binary arcs, for feeding `compress` without
    /// re-parsing 210 GB of decimal text.
    #[arg(long, value_name = "FILE")]
    pub sorted_arcs: Option<PathBuf>,

    /// Run-sorting backend.
    #[arg(long, value_enum, default_value_t = SortAlgoArg::Radix)]
    pub sort_algo: SortAlgoArg,

    /// Keep duplicate arcs. The shell pipeline always removed them, and a
    /// duplicate arc is FATAL to the compressor, so this is for diagnosis only.
    #[arg(long)]
    pub no_dedup: bool,
}

// --------------------------------------------------------------- compress

/// Arguments of `utxo2webgraph compress`.
#[derive(Args, Debug, Clone)]
pub struct CompressArgs {
    /// Sorted edge list: `.tsv` text or binary packed arcs.
    #[arg(value_name = "EDGE_LIST")]
    pub input: PathBuf,

    /// Basename for `<prefix>.graph`, `.offsets` and `.properties`.
    ///
    /// Must not contain a `.`: webgraph-rs derives the three names with
    /// `with_extension`, which would eat it.
    #[arg(value_name = "OUTPUT_PREFIX")]
    pub output_prefix: PathBuf,

    /// Input format; `auto` sniffs binary versus text.
    #[arg(long, value_enum, default_value_t = InputFormatArg::Auto)]
    pub input_format: InputFormatArg,

    /// How to determine the node count: `from-arcs`, `node-map`, or a literal
    /// integer.
    ///
    /// `from-arcs` (the default) is `max(id over sources AND targets) + 1`:
    /// the smallest node count the arcs themselves justify, and the count the
    /// historical shell pipeline produced, so a graph built this way is
    /// directly comparable with the ones already on disk. `node-map` is the
    /// semantically complete answer — every UTXO is a node, including the
    /// unspent ones that appear in no arc — and needs `--node-map FILE`. Both
    /// values are logged whenever they differ.
    #[arg(long, value_name = "SPEC", default_value = "from-arcs",
          value_parser = parse_num_nodes)]
    pub num_nodes: NumNodesSpec,

    /// Node map, required by `--num-nodes node-map`.
    #[arg(long, value_name = "FILE")]
    pub node_map: Option<PathBuf>,

    /// Compress in parallel.
    ///
    /// Faster, but each chunk restarts its compression window, so the bytes
    /// differ from the sequential output, which is the one that reproduces the
    /// graphs already on disk. Semantically identical: `graph::eq` passes.
    #[arg(long)]
    pub parallel: bool,

    /// Also build `<prefix>.ef`, needed only for random access.
    #[arg(long)]
    pub build_ef: bool,

    /// Emit a 0-arc graph instead of failing on an empty edge list.
    #[arg(long)]
    pub allow_empty: bool,

    /// How thoroughly the finished graph is read back.
    #[arg(long, value_enum, default_value_t = VerifyArg::Quick)]
    pub verify: VerifyArg,
}

// --------------------------------------------------------------- build-pg

/// Arguments of `utxo2webgraph build-pg`. Replaces `build_pg.sh`, fused.
#[derive(Args, Debug, Clone)]
pub struct BuildPgArgs {
    /// Where the transactions come from.
    #[command(flatten)]
    pub inputs: InputSelector,

    /// Number of leading chunks to read, with `--chunk-dir`.
    #[arg(short = 'n', long, value_name = "N")]
    pub chunks: Option<usize>,

    /// Node map output (`build_pg.sh` argument 2).
    #[arg(long, value_name = "FILE")]
    pub node_map: PathBuf,

    /// Sorted edge list output (`build_pg.sh` argument 3).
    #[arg(long, value_name = "FILE")]
    pub edge_list: PathBuf,

    /// BVGraph basename (`build_pg.sh` argument 4).
    #[arg(long, value_name = "PREFIX")]
    pub output_prefix: PathBuf,

    /// Skip the text edge list; go straight from sorted arcs to BVGraph.
    #[arg(long)]
    pub no_text_edge_list: bool,

    /// Skip the node map file (~55 GB at N=28).
    #[arg(long)]
    pub no_node_map: bool,

    /// Run-sorting backend.
    #[arg(long, value_enum, default_value_t = SortAlgoArg::Radix)]
    pub sort_algo: SortAlgoArg,

    /// Pre-size the dense node map.
    #[arg(long, value_name = "N")]
    pub max_tx_id: Option<usize>,

    /// What to do when an input references an output that was never seen.
    #[arg(long, value_enum, default_value_t = OnMissingSourceArg::Fail)]
    pub on_missing_source: OnMissingSourceArg,

    /// How to determine the node count. `from-arcs` (the default) is what
    /// `build_pg.sh`'s third stage computed; `node-map` also counts unspent
    /// outputs and produces a graph with extra isolated trailing nodes.
    #[arg(long, value_name = "SPEC", default_value = "from-arcs",
          value_parser = parse_num_nodes)]
    pub num_nodes: NumNodesSpec,

    /// Emit a `Processed:` line every N transactions.
    #[arg(long, value_name = "N", default_value_t = crate::DEFAULT_PROGRESS_EVERY)]
    pub progress_every: u64,

    /// Compress in parallel (non-deterministic bytes; see `compress`).
    #[arg(long)]
    pub parallel: bool,

    /// Emit a 0-arc graph instead of failing on an empty edge list.
    #[arg(long)]
    pub allow_empty: bool,

    /// How thoroughly the finished graph is read back.
    #[arg(long, value_enum, default_value_t = VerifyArg::Quick)]
    pub verify: VerifyArg,
}

// ------------------------------------------------------------------ build

/// Arguments of `utxo2webgraph build N`. Replaces `builder.sh N`.
#[derive(Args, Debug, Clone)]
pub struct BuildArgs {
    /// Number of leading chunks, exactly like `./builder.sh N`.
    #[arg(value_name = "N")]
    pub chunks: usize,

    /// Directory holding `chunk_01.txt .. chunk_NN.txt`.
    #[arg(long, value_name = "DIR", default_value = "chunks")]
    pub chunk_dir: PathBuf,

    /// Output directory; artifacts are `pg_nm_N.tsv`, `pg_el_N.tsv`, `pg_N`.
    #[arg(long, value_name = "DIR", default_value = "graph")]
    pub graph_dir: PathBuf,

    /// Skip the text edge list.
    #[arg(long)]
    pub no_text_edge_list: bool,

    /// Skip the node map file.
    #[arg(long)]
    pub no_node_map: bool,

    /// Run-sorting backend.
    #[arg(long, value_enum, default_value_t = SortAlgoArg::Radix)]
    pub sort_algo: SortAlgoArg,

    /// Pre-size the dense node map to N transactions (default: grow
    /// geometrically, which costs about 1.35x the steady-state array while it
    /// doubles).
    #[arg(long, value_name = "N")]
    pub max_tx_id: Option<usize>,

    /// What to do when an input references an output that was never seen.
    #[arg(long, value_enum, default_value_t = OnMissingSourceArg::Fail)]
    pub on_missing_source: OnMissingSourceArg,

    /// How to determine the node count. `from-arcs` (the default) is what
    /// `build_pg.sh`'s third stage computed; `node-map` also counts unspent
    /// outputs and produces a graph with extra isolated trailing nodes.
    #[arg(long, value_name = "SPEC", default_value = "from-arcs",
          value_parser = parse_num_nodes)]
    pub num_nodes: NumNodesSpec,

    /// Emit a `Processed:` line every N transactions.
    #[arg(long, value_name = "N", default_value_t = crate::DEFAULT_PROGRESS_EVERY)]
    pub progress_every: u64,

    /// Compress in parallel (non-deterministic bytes; see `compress`).
    #[arg(long)]
    pub parallel: bool,

    /// Emit a 0-arc graph instead of failing on an empty edge list.
    #[arg(long)]
    pub allow_empty: bool,

    /// How thoroughly the finished graph is read back.
    #[arg(long, value_enum, default_value_t = VerifyArg::Quick)]
    pub verify: VerifyArg,

    /// Resume from a stage, reusing intermediates left by
    /// `--keep-intermediate`.
    #[arg(long, value_enum)]
    pub from_stage: Option<StageArg>,
}

// ------------------------------------------------------------ value enums

/// CLI mirror of [`Mode`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum ModeArg {
    /// Any malformed record is a fatal, line-numbered error (the default).
    Strict,
    /// Malformed records are skipped and tallied.
    Lenient,
}

impl From<ModeArg> for Mode {
    fn from(a: ModeArg) -> Mode {
        match a {
            ModeArg::Strict => Mode::Strict,
            ModeArg::Lenient => Mode::Lenient,
        }
    }
}

/// CLI mirror of [`OnMissingSource`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum OnMissingSourceArg {
    /// Abort, naming the line, the transaction, the input index and the
    /// missing `(prevTxId, prevOffset)` pair.
    Fail,
    /// Emit no edges for that input and tally it.
    Skip,
    /// Mint a node on the fly. Changes every id in both output files.
    Create,
}

impl From<OnMissingSourceArg> for OnMissingSource {
    fn from(a: OnMissingSourceArg) -> OnMissingSource {
        match a {
            OnMissingSourceArg::Fail => OnMissingSource::Fail,
            OnMissingSourceArg::Skip => OnMissingSource::Skip,
            OnMissingSourceArg::Create => OnMissingSource::Create,
        }
    }
}

/// CLI mirror of [`StatsStyle`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum StatsArg {
    /// A `Processed:` line, then `Nodes: N\tEdges: M`, and nothing else.
    ///
    /// `Nodes:` counts output *slots* rather than distinct nodes — the two
    /// differ whenever a transaction declares an output nothing ever spends —
    /// and is reported that way for continuity with the historical pipeline
    /// logs in `logs/pg_el_builder.log`, so a naive diff against them still
    /// matches. Both numbers are on one tab-separated line, so parse the line
    /// rather than expecting `Edges:` at the start of one.
    Brief,
    /// The `brief` lines plus two more, also tab-separated:
    /// `Distinct nodes:`/`Output slots:`, and `Skipped lines:`/`Blank lines:`/
    /// `Dangling refs:`/`Zero-output txs:`.
    ///
    /// The duplicate-arc tally is *not* here: deduplication happens in the
    /// sorter, after these statistics are produced, and is reported on stderr
    /// at `WARN` when it is ever non-zero.
    Extended,
}

impl From<StatsArg> for StatsStyle {
    fn from(a: StatsArg) -> StatsStyle {
        match a {
            StatsArg::Brief => StatsStyle::Brief,
            StatsArg::Extended => StatsStyle::Extended,
        }
    }
}

/// CLI mirror of [`VerifyLevel`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum VerifyArg {
    /// Do not read the finished graph back at all.
    None,
    /// Check `.properties`, the node count and the declared arc count (O(1)).
    Quick,
    /// Also decompress the whole graph and recount every arc. At N=28 that is
    /// a second full pass over 9.5e9 arcs.
    Full,
}

impl From<VerifyArg> for VerifyLevel {
    fn from(a: VerifyArg) -> VerifyLevel {
        match a {
            VerifyArg::None => VerifyLevel::None,
            VerifyArg::Quick => VerifyLevel::Quick,
            VerifyArg::Full => VerifyLevel::Full,
        }
    }
}

/// CLI mirror of [`SortAlgo`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum SortAlgoArg {
    /// `rdst` LSD radix sort over packed `u128` arcs.
    Radix,
    /// `rayon` pattern-defeating quicksort.
    Pdq,
}

impl From<SortAlgoArg> for SortAlgo {
    fn from(a: SortAlgoArg) -> SortAlgo {
        match a {
            SortAlgoArg::Radix => SortAlgo::Radix,
            SortAlgoArg::Pdq => SortAlgo::Pdq,
        }
    }
}

/// How `edge-list` writes its raw arcs.
///
/// There is no shared-contract enum for this: it selects between
/// `arcs::TsvArcSink` and `arcs::BinaryArcSink`, which `main.rs` does directly.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum ArcFormatArg {
    /// Fixed-width 16-byte binary records, read back without re-parsing.
    Binary,
    /// `src \t dst` decimal text, byte-compatible with `tmp/edge_list.tsv`.
    Tsv,
}

/// How `sort-edges` and `compress` read an existing arc file.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum InputFormatArg {
    /// Sniff the file with `arcs::detect_format`.
    Auto,
    /// Fixed-width binary records.
    Binary,
    /// `src \t dst` decimal text.
    Tsv,
}

/// The stage `utxo2webgraph build --from-stage` resumes at.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum StageArg {
    /// Parse the transactions from scratch (the default).
    EdgeList,
    /// Reuse the kept binary arc file and sort it.
    SortEdges,
    /// Reuse the kept text edge list and compress it.
    Compress,
}

// ------------------------------------------------------------- num-nodes

/// How the BVGraph node count is determined.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum NumNodesSpec {
    /// `node_map.next_id()`: every UTXO is a node. Semantically complete.
    NodeMap,
    /// `max(src, dst) + 1`: only the outputs some arc touches. Smaller than
    /// [`NumNodesSpec::NodeMap`], because unspent outputs appear in no arc.
    FromArcs,
    /// A literal node count.
    Explicit(usize),
}

/// Parses `--num-nodes`: `node-map`, `from-arcs`, or a literal integer.
///
/// Wired into `clap` as a `value_parser`, so the error message is what the
/// user sees on a bad value.
pub fn parse_num_nodes(s: &str) -> Result<NumNodesSpec, String> {
    match s.trim() {
        "node-map" | "node_map" | "nodemap" => Ok(NumNodesSpec::NodeMap),
        "from-arcs" | "from_arcs" | "fromarcs" => Ok(NumNodesSpec::FromArcs),
        other => other
            .parse::<usize>()
            .map(NumNodesSpec::Explicit)
            .map_err(|_| {
                format!(
                    "invalid node count {s:?}: expected `node-map`, `from-arcs`, \
                 or a non-negative integer"
                )
            }),
    }
}

/// Parses `--log-interval`.
///
/// A port of webgraph-rs `cli/src/lib.rs`, so a duration accepted by
/// `webgraph` is accepted here and means the same thing — which is why a bare
/// number is **milliseconds** and not seconds: keeping upstream's convention
/// is the only way `--log-interval 500` cannot mean two different things to
/// the two tools. The suffixes are `s` (seconds), `m` (minutes), `h` (hours)
/// and `d` (days), and they accumulate: `1d2h3m4s567` is one day, two hours,
/// three minutes, four seconds and 567 milliseconds.
///
/// Private, exactly as upstream has it: `clap`'s `value_parser` does not need
/// a public function, and widening the library's API would oblige it to carry
/// a doc under `#![warn(missing_docs)]` for something nothing outside this
/// file calls. Upstream's two `anyhow` bails become `Err(String)`, which is
/// what `clap` renders as the user-facing message.
fn parse_duration(value: &str) -> Result<Duration, String> {
    if value.is_empty() {
        return Err("empty duration string; for every 0 milliseconds use `0`".to_string());
    }
    let mut duration = Duration::from_secs(0);
    let mut acc = String::new();
    for c in value.chars() {
        if c.is_ascii_digit() {
            acc.push(c);
        } else if c.is_whitespace() {
            continue;
        } else {
            let dur = acc
                .parse::<u64>()
                .map_err(|e| format!("invalid duration {value:?}: {e}"))?;
            match c {
                's' => duration += Duration::from_secs(dur),
                'm' => duration += Duration::from_secs(dur * 60),
                'h' => duration += Duration::from_secs(dur * 60 * 60),
                'd' => duration += Duration::from_secs(dur * 60 * 60 * 24),
                _ => return Err(format!("invalid duration suffix: {c}")),
            }
            acc.clear();
        }
    }
    if !acc.is_empty() {
        let dur = acc
            .parse::<u64>()
            .map_err(|e| format!("invalid duration {value:?}: {e}"))?;
        duration += Duration::from_millis(dur);
    }
    Ok(duration)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn cli_definition_is_consistent() {
        // clap's own structural check: duplicated ids, dangling `requires`,
        // groups naming unknown arguments, and so on.
        Cli::command().debug_assert();
    }

    #[test]
    fn num_nodes_spec_parses() {
        assert_eq!(parse_num_nodes("node-map").unwrap(), NumNodesSpec::NodeMap);
        assert_eq!(
            parse_num_nodes("from-arcs").unwrap(),
            NumNodesSpec::FromArcs
        );
        assert_eq!(
            parse_num_nodes("2147483647").unwrap(),
            NumNodesSpec::Explicit(2_147_483_647)
        );
        assert!(parse_num_nodes("nope").is_err());
    }

    #[test]
    fn log_interval_parses_like_webgraph() {
        // The example from upstream's own doc comment.
        assert_eq!(
            parse_duration("1d2h3m4s567").unwrap(),
            Duration::from_secs(((24 + 2) * 60 + 3) * 60 + 4) + Duration::from_millis(567)
        );
        // A bare number is MILLISECONDS, not seconds, as upstream has it.
        assert_eq!(parse_duration("500").unwrap(), Duration::from_millis(500));
        assert_eq!(parse_duration("10s").unwrap(), Duration::from_secs(10));
        assert_eq!(parse_duration("0").unwrap(), Duration::ZERO);
        // Whitespace is ignored, as upstream does.
        assert_eq!(parse_duration("1m 30s").unwrap(), Duration::from_secs(90));
        assert!(parse_duration("").is_err());
        assert!(parse_duration("3x").is_err());
    }

    #[test]
    fn log_interval_is_global_and_defaults_to_ten_seconds() {
        let before = Cli::try_parse_from(["utxo2webgraph", "--log-interval", "1m5s", "build", "1"])
            .expect("--log-interval before the subcommand");
        let after = Cli::try_parse_from(["utxo2webgraph", "build", "1", "--log-interval", "1m5s"])
            .expect("--log-interval after the subcommand");
        assert_eq!(before.common.log_interval, Duration::from_secs(65));
        assert_eq!(after.common.log_interval, before.common.log_interval);

        let default = Cli::try_parse_from(["utxo2webgraph", "build", "1"]).expect("defaults");
        assert_eq!(default.common.log_interval, Duration::from_secs(10));
    }

    #[test]
    fn strict_and_lenient_conflict() {
        let err = Cli::try_parse_from(["utxo2webgraph", "--strict", "--lenient", "build", "1"]);
        assert!(err.is_err(), "--strict --lenient must be rejected");
    }

    #[test]
    fn chunk_dir_requires_chunks() {
        let err = Cli::try_parse_from([
            "utxo2webgraph",
            "edge-list",
            "--chunk-dir",
            "chunks",
            "--arcs",
            "out.bin",
        ]);
        assert!(
            err.is_err(),
            "--chunk-dir without --chunks must be rejected"
        );

        let ok = Cli::try_parse_from([
            "utxo2webgraph",
            "edge-list",
            "--chunk-dir",
            "chunks",
            "--chunks",
            "2",
            "--arcs",
            "out.bin",
        ]);
        assert!(ok.is_ok(), "--chunk-dir with --chunks must parse");
    }

    #[test]
    fn global_options_may_precede_the_subcommand() {
        let cli = Cli::try_parse_from(["utxo2webgraph", "--threads", "64", "build", "15"])
            .expect("global option before the subcommand must parse");
        assert_eq!(cli.common.threads, Some(64));
        match cli.command {
            Command::Build(ref a) => assert_eq!(a.chunks, 15),
            ref other => panic!("expected `build`, got {other:?}"),
        }
        assert_eq!(cli.common.mode(), Mode::Strict);
        assert_eq!(cli.common.stats_style(), StatsStyle::Brief);
    }
}
