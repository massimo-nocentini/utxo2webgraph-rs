//! Record parser for the master transaction list.
//!
//! # Attribution
//!
//! The record format, the pipeline it feeds and the graph-construction
//! algorithm behind it are the work of **Matteo Loporchio**. This module is an
//! independent reimplementation of that design.
//!
//! One input record is a single line of the master transaction list:
//!
//! ```text
//! <info>:<inputs>:<outputs>
//! info    := f0 "," f1 "," f2 "," ...      ; f2 = txId, the ONLY field read
//! inputs  := "" | input (";" input)*
//! input   := i0 "," i1 "," i2 "," i3       ; i2 = prevTxId, i3 = prevTxOffset
//! outputs := "" | output (";" output)*     ; ONLY THE COUNT MATTERS
//! ```
//!
//! Real examples, verbatim from the corpus:
//!
//! ```text
//! 1231006505,0,0,1,0,204,0::0,5000000000,1
//! 1293837540,100406,217999,0,0,273,0:175742,5000000000,217481,0;175802,5000000000,217536,0:103753,10000000000,2
//! ```
//!
//! # Separator semantics
//!
//! Three separators nest: `':'` cuts a line into sections, `';'` cuts a
//! section into groups, `','` cuts a group into fields. All three obey the
//! same splitting rule, and every function in this module implements that one
//! rule:
//!
//! 1. **A trailing run of separators is not significant.** `"a;b;"` has two
//!    fields, and so does `"a;b;;;"`. A string that is nothing but separators
//!    has *none*.
//! 2. **Leading and interior empty fields are significant and are kept.**
//!    `";a;b"` has three fields, the first of them empty — which is how a
//!    coinbase's empty input section survives as `parts[1]`.
//! 3. **A string in which the separator never occurs is one field**, even when
//!    it is empty: `""` splits into a single empty field, not into none.
//!
//! Rule 1 is the surprising one, and it is not a detail this module is free to
//! change: it is the rule the corpus was written against and the rule the
//! golden fixtures pin, so flipping it would silently reinterpret every record
//! that happens to end in a separator.
//!
//! Everything here is zero-copy: the parser borrows from the caller's line
//! buffer and allocates nothing on the happy path. The splitting rules above
//! are reproduced exactly where they matter and corrected where they produced
//! wrong edges; every correction is called out in the item documentation,
//! together with the failure it prevents.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::{Error, Mode, Result};

/// Parsing options. Currently only the strict/lenient switch.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub struct ParseOpts {
    /// How malformed records are treated.
    pub mode: Mode,
}

/// One parsed transaction record, borrowing from the caller's line buffer.
///
/// Exactly two things are extracted: `infos[2]` (the transaction id) and the
/// *number* of output groups. Amounts, addresses, the block height and the
/// `isCoinbase` flag at `infos[3]` are never read — a transaction is treated
/// as a coinbase precisely when its input section is empty, which is the
/// condition that actually drives edge generation. See
/// [`TxRecord::is_coinbase`].
#[derive(Copy, Clone, Debug)]
pub struct TxRecord<'a> {
    /// `infos[2]`, the transaction id, parsed as `i32`.
    ///
    /// `i32` is the input format's own width and is deliberate; see
    /// [`parse_line`] for why it is neither widened nor made unsigned.
    pub tx_id: i32,
    /// Number of output slots: the count of `';'`-separated groups in the
    /// third section, with an empty section counting as **zero**, not one.
    /// See [`count_outputs`].
    pub num_outputs: u32,
    /// The raw, unsplit inputs section (`parts[1]`).
    pub inputs_section: &'a str,
    /// 1-based input line number, used for diagnostics.
    pub line: u64,
    /// The options this record was parsed with; inherited by [`TxRecord::inputs`].
    pub opts: ParseOpts,
}

impl<'a> TxRecord<'a> {
    /// True when the transaction has no inputs (a coinbase).
    ///
    /// Emptiness of the input section is the whole test. The `isCoinbase` flag
    /// at `infos[3]` is ignored on purpose: the two agree everywhere on the
    /// corpus, but it is the input section that generates arcs — a record with
    /// no inputs emits none whatever the flag claims — so consulting the flag
    /// could only introduce a disagreement between what a record says about
    /// itself and what the pipeline does with it.
    #[inline]
    pub fn is_coinbase(&self) -> bool {
        self.inputs_section.is_empty()
    }

    /// Iterates the `(prevTxId, prevTxOffset)` pairs of this transaction, in
    /// file order.
    #[inline]
    pub fn inputs(&self) -> InputRefs<'a> {
        InputRefs::new(self.inputs_section, self.line, self.opts)
    }
}

/// Iterator over a record's inputs, yielding `(prevTxId, prevTxOffset)`.
///
/// Splits the input section on `';'` and each resulting group on `','`, taking
/// fields 2 and 3 of the group. The module's splitting rules apply verbatim: a
/// trailing `;` run is swallowed, so `"a;b;"` is two inputs; an empty section
/// yields zero items; and a *leading* `;` produces an empty leading group,
/// which then fails the four-field check and surfaces as a line-numbered
/// [`Error::BadInputFields`] naming the index of the offending input, rather
/// than as an out-of-bounds panic with no line number attached.
pub struct InputRefs<'a> {
    /// `None` once the section is known to hold no pieces at all.
    inner: Option<std::str::Split<'a, char>>,
    index: usize,
    line: u64,
    opts: ParseOpts,
}

impl<'a> InputRefs<'a> {
    fn new(section: &'a str, line: u64, opts: ParseOpts) -> Self {
        // Trailing separators are not significant, so a section that is empty
        // or made of nothing but `;` holds no usable inputs at all.
        let trimmed = section.trim_end_matches(';');
        let inner = if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.split(';'))
        };
        InputRefs {
            inner,
            index: 0,
            line,
            opts,
        }
    }
}

impl<'a> Iterator for InputRefs<'a> {
    type Item = Result<(i32, i32)>;

    fn next(&mut self) -> Option<Self::Item> {
        let piece = self.inner.as_mut()?.next()?;
        let index = self.index;
        self.index += 1;

        // Only fields 2 and 3 of an input are read. Fields 0 and 1 (the
        // previous transaction's hash and the amount) are never touched, but
        // they must still be *present*: a group with fewer than four fields is
        // a malformed record, not an input with defaults.
        let mut fields = piece.split(',');
        let f0 = fields.next();
        let f1 = fields.next();
        let prev_tx = fields.next();
        let prev_off = fields.next();
        let (prev_tx, prev_off) = match (f0, f1, prev_tx, prev_off) {
            (Some(_), Some(_), Some(a), Some(b)) => (a, b),
            _ => {
                let found = piece.split(',').count();
                return Some(Err(Error::BadInputFields {
                    line: self.line,
                    index,
                    found,
                }));
            }
        };

        let prev_tx_id = match prev_tx.parse::<i32>() {
            Ok(v) => v,
            Err(_) => {
                return Some(Err(Error::BadInteger {
                    line: self.line,
                    field: "prevTxId",
                    value: prev_tx.to_string(),
                }))
            }
        };
        let prev_tx_offset = match prev_off.parse::<i32>() {
            Ok(v) => v,
            Err(_) => {
                return Some(Err(Error::BadInteger {
                    line: self.line,
                    field: "prevTxOffset",
                    value: prev_off.to_string(),
                }))
            }
        };

        if self.opts.mode == Mode::Strict {
            // A negative id is not a parse failure: `"-5"` parses cleanly as an
            // `i32` and `crate::pack` round-trips it exactly, so without this
            // check a negative reference would reach the node map and print as
            // a literal `-5\t0\t0` row. Nothing downstream expects that, so
            // strict mode refuses it and lenient mode lets it through verbatim.
            if prev_tx_id < 0 {
                return Some(Err(Error::NegativeId {
                    line: self.line,
                    field: "prevTxId",
                    value: prev_tx_id,
                }));
            }
            if prev_tx_offset < 0 {
                return Some(Err(Error::NegativeId {
                    line: self.line,
                    field: "prevTxOffset",
                    value: prev_tx_offset,
                }));
            }
        }

        Some(Ok((prev_tx_id, prev_tx_offset)))
    }
}

/// Strips one trailing `\n` and then one trailing `\r`.
///
/// `BufRead::read_line`/`lines()` keep (respectively drop) the `\n`, but
/// neither strips a lone `\r` from a CRLF file, and such a byte would end up
/// inside the last output field — where it would not break the *count* of
/// outputs, and so would corrupt the graph silently rather than loudly. The
/// corpus contains zero CR bytes (`grep -c $'\r'` = 0 on 100 000-line heads of
/// `chunk_01` and `chunk_05`), so this is a no-op today; it exists so the
/// parser still reads the data unchanged if the corpus is ever regenerated on
/// a machine that writes CRLF.
pub fn strip_eol(line: &str) -> &str {
    let line = line.strip_suffix('\n').unwrap_or(line);
    line.strip_suffix('\r').unwrap_or(line)
}

/// Splits a line on `':'`, returning the first three pieces and the
/// *post-trimming* piece count.
///
/// The rules, in full — the module-level splitting rules, restated here
/// because this is where they are implemented:
///
/// 1. all trailing empty pieces are dropped, repeatedly;
/// 2. leading and interior empty pieces are kept;
/// 3. when the separator does not occur at all the whole input is returned as
///    a single piece — so `""` has length **1**, not 0.
///
/// The resulting table, which the unit tests assert verbatim:
///
/// | line | count | outcome |
/// |---|---|---|
/// | `info:inputs:outputs` | 3 | normal path |
/// | `info::outputs` (coinbase) | 3 | normal, `parts[1]` empty |
/// | `info:inputs:` (zero outputs) | **2** | too few sections; rejected |
/// | `info::` | **1** | too few sections; rejected |
/// | `""` (blank line) | **1** | skipped as blank, in both modes |
/// | `info:inputs::x` | 4 | too many sections; `parts[2]` is empty, the phantom-edge shape |
///
/// The last two rows are why the count is returned at all: `""` and
/// `info:inputs::x` both produce a usable-looking three-element array, and
/// only the count distinguishes them from a well-formed record.
///
/// Missing pieces come back as `""`, so the caller can index the array
/// unconditionally after checking the count.
pub fn split_sections(line: &str) -> ([&str; 3], usize) {
    let mut parts = [""; 3];
    let mut it = line.split(':');
    for slot in parts.iter_mut() {
        match it.next() {
            Some(p) => *slot = p,
            None => break,
        }
    }

    let count = if !line.as_bytes().contains(&b':') {
        // Rule 3: no match at all, the input itself is the single piece.
        1
    } else {
        let trimmed = line.trim_end_matches(':');
        if trimmed.is_empty() {
            // The line was nothing but separators: every piece is empty, and
            // rule 1 drops them all, so `":"` has zero pieces.
            0
        } else {
            1 + trimmed.as_bytes().iter().filter(|b| **b == b':').count()
        }
    };

    (parts, count)
}

/// Counts the output slots of a record's third section.
///
/// The number of `';'`-separated groups, **with an empty section counting as
/// zero** rather than one.
///
/// # Why an empty section must count as zero
///
/// The obvious way to write this stage is to split the section, allocate one
/// node-id slot per group, and fill those slots only when the section is
/// non-empty:
///
/// ```text
/// outputs   = split(section, ';')      // "" splits into one empty piece!
/// outputIds = new array[len(outputs)]  // so: one slot, default-initialised 0
/// if section != "":
///     fill outputIds                   // skipped for an empty section
/// ...
/// for id in outputIds:
///     emit(sourceId, id)               // still runs once: emits (source, 0)
/// ```
///
/// The guard blocks node *creation* but not edge *emission*, and the default
/// slot value `0` is a perfectly valid node id — the id of output 0 of the
/// genesis coinbase. Every input of such a line therefore emitted a silent,
/// bogus edge into the genesis output (reproduced against the original
/// implementation of the pipeline: `Nodes: 1  Edges: 1`, edge list `0\t0`).
/// Zero lines in the corpus can reach it, and there is no reading under which
/// that edge is correct, so it is fixed unconditionally and without a flag: an
/// empty output section means zero outputs, and zero outputs mean zero edges.
///
/// The count is a byte scan, never an allocating split, and it applies the
/// module's trailing-separator rule: `"a;b;"` counts two outputs, `";"` none.
pub fn count_outputs(section: &str) -> u32 {
    if section.is_empty() {
        return 0;
    }
    if !section.as_bytes().contains(&b';') {
        return 1;
    }
    let trimmed = section.trim_end_matches(';');
    if trimmed.is_empty() {
        return 0;
    }
    (1 + trimmed.as_bytes().iter().filter(|b| **b == b';').count()) as u32
}

/// Counts how many over-long (>3 section) lines have been seen, so the warning
/// can be rate-limited to one per 10 000 occurrences.
static EXTRA_SECTION_WARNINGS: AtomicU64 = AtomicU64::new(0);

/// Counts records dropped by [`Mode::Lenient`], so the log is not flooded when
/// a whole file is malformed.
static SKIPPED_WARNINGS: AtomicU64 = AtomicU64::new(0);

/// Rate-limited "this record was dropped" warning: one line per 10 000
/// occurrences, the same policy [`EXTRA_SECTION_WARNINGS`] uses.
fn warn_skipped(line_no: u64, why: &str) {
    let seen = SKIPPED_WARNINGS.fetch_add(1, Ordering::Relaxed);
    if seen % 10_000 == 0 {
        log::warn!("line {line_no}: skipped, {why} (occurrence {})", seen + 1);
    }
}

/// Parses one line into a [`TxRecord`], or `Ok(None)` when the line carries no
/// transaction (blank, or malformed and skipped under [`Mode::Lenient`]).
///
/// # Deliberately strict
///
/// Each of these cases is unreachable on the current corpus. They are checked
/// anyway because every one of them, left unchecked, corrupts the graph or
/// kills a multi-hour run without naming the line that did it:
///
/// * a blank or whitespace-only line is skipped in **both** modes. A parser
///   that indexes the section array straight away dies on the first blank line
///   with an out-of-bounds panic that names neither the line number nor the
///   file — an opaque way to lose hours of work to a stray newline left behind
///   by a `cat` or an editor;
/// * fewer than three sections is a line-numbered [`Error::BadSectionCount`]
///   under [`Mode::Strict`], naming the count actually found;
/// * *more* than three sections is rejected under [`Mode::Strict`] as well.
///   Quietly taking the first three of a longer line looks harmless, but a
///   fourth section means the third one is empty, and an empty third section
///   is exactly the phantom-edge shape [`count_outputs`] documents. Under
///   [`Mode::Lenient`] the first three are used and a rate-limited warning is
///   logged.
///
/// # Why the transaction id is an `i32`
///
/// The id is parsed as `i32`, never `u32` and never `i64`. It is the input
/// format's own value and **not** a [`crate::NodeId`]: [`crate::pack`] builds
/// the node-map key by reinterpreting the two 32-bit halves of the
/// `(txId, offset)` pair, so widening this field would change the key for
/// every value at or above 2^31 and rewrite the node map. Making it unsigned
/// would be worse still — a negative id would stop being detectable, and the
/// strict-mode rejection below would have nothing left to test. The largest
/// txId in the corpus is 778 613 437, comfortably inside the range.
pub fn parse_line(line: &str, line_no: u64, opts: ParseOpts) -> Result<Option<TxRecord<'_>>> {
    let line = strip_eol(line);

    // Skipping blanks here is what keeps a stray empty line from killing the
    // run. It cannot change behaviour on the corpus, which has none, but a
    // blank introduced by a future `cat` or an editor would otherwise abort
    // the whole pass on a line that carries no transaction at all.
    if line.trim().is_empty() {
        return Ok(None);
    }

    let (parts, n) = split_sections(line);
    if n < 3 {
        return match opts.mode {
            Mode::Strict => Err(Error::BadSectionCount {
                line: line_no,
                found: n,
            }),
            Mode::Lenient => Ok(None),
        };
    }
    if n > 3 {
        if opts.mode == Mode::Strict {
            return Err(Error::BadSectionCount {
                line: line_no,
                found: n,
            });
        }
        let seen = EXTRA_SECTION_WARNINGS.fetch_add(1, Ordering::Relaxed);
        if seen % 10_000 == 0 {
            log::warn!(
                "line {line_no}: {n} ':'-separated sections, using the first three \
                 (occurrence {}); the extra section means the output section is empty, \
                 which a naive parser would turn into phantom edges into node 0",
                seen + 1
            );
        }
    }

    // The info section: of its ','-separated fields, only field 2 -- the
    // transaction id -- is ever read. The first two need only exist.
    let mut infos = parts[0].split(',');
    let f0 = infos.next();
    let f1 = infos.next();
    let f2 = infos.next();
    let tx_field = match (f0, f1, f2) {
        (Some(_), Some(_), Some(v)) => v,
        _ => {
            // There is no field 2 to read. Strict reports the line and the
            // count found; Lenient skips the record, per the `Mode::Lenient`
            // contract, with a rate-limited warning.
            if opts.mode == Mode::Lenient {
                warn_skipped(line_no, "the info section has fewer than 3 fields");
                return Ok(None);
            }
            return Err(Error::BadInfoFields {
                line: line_no,
                found: parts[0].split(',').count(),
            });
        }
    };
    let tx_id = match tx_field.parse::<i32>() {
        Ok(v) => v,
        Err(_) => {
            // The bare parse failure says only that the number does not fit;
            // it names neither the line nor the file, and propagating it as-is
            // would end the run on an anonymous message. Strict reports the
            // line number and the offending text verbatim; Lenient skips the
            // record, as `Mode::Lenient` documents.
            if opts.mode == Mode::Lenient {
                warn_skipped(line_no, "txId does not fit in an i32");
                return Ok(None);
            }
            return Err(Error::BadInteger {
                line: line_no,
                field: "txId",
                value: tx_field.to_string(),
            });
        }
    };
    if tx_id < 0 && opts.mode == Mode::Strict {
        return Err(Error::NegativeId {
            line: line_no,
            field: "txId",
            value: tx_id,
        });
    }

    Ok(Some(TxRecord {
        tx_id,
        num_outputs: count_outputs(parts[2]),
        inputs_section: parts[1],
        line: line_no,
        opts,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    const STRICT: ParseOpts = ParseOpts { mode: Mode::Strict };
    const LENIENT: ParseOpts = ParseOpts {
        mode: Mode::Lenient,
    };

    #[test]
    fn split_sections_matches_the_documented_table() {
        assert_eq!(
            split_sections("info:inputs:outputs"),
            (["info", "inputs", "outputs"], 3)
        );
        assert_eq!(
            split_sections("info::outputs"),
            (["info", "", "outputs"], 3)
        );
        assert_eq!(split_sections("info:inputs:").1, 2);
        assert_eq!(split_sections("info::").1, 1);
        assert_eq!(split_sections("").1, 1);
        assert_eq!(
            split_sections("info:inputs::x"),
            (["info", "inputs", ""], 4)
        );
        // Extra rows required by the spec.
        assert_eq!(split_sections("a:b:c:").1, 3);
        assert_eq!(split_sections("a:::").1, 1);
        // Degenerate: nothing but separators drops every piece.
        assert_eq!(split_sections(":").1, 0);
        // Leading and interior empties are kept.
        assert_eq!(split_sections(":a:b"), (["", "a", "b"], 3));
    }

    #[test]
    fn count_outputs_ignores_trailing_separators() {
        assert_eq!(count_outputs(""), 0);
        assert_eq!(count_outputs("0,5000000000,1"), 1);
        assert_eq!(count_outputs("a;b"), 2);
        assert_eq!(count_outputs("a;b;"), 2);
        assert_eq!(count_outputs("802635047,635073896,2;1050047673,0,4"), 2);
        assert_eq!(count_outputs(";"), 0);
    }

    #[test]
    fn genesis_line_parses() {
        let rec = parse_line("1231006505,0,0,1,0,204,0::0,5000000000,1", 1, STRICT)
            .unwrap()
            .unwrap();
        assert_eq!(rec.tx_id, 0);
        assert_eq!(rec.num_outputs, 1);
        assert!(rec.is_coinbase());
        assert_eq!(rec.inputs().count(), 0);
    }

    #[test]
    fn two_input_line_parses() {
        let line = "1293837540,100406,217999,0,0,273,0:175742,5000000000,217481,0;\
                    175802,5000000000,217536,0:103753,10000000000,2";
        let rec = parse_line(line, 7, STRICT).unwrap().unwrap();
        assert_eq!(rec.tx_id, 217999);
        // The output section is `103753,10000000000,2`: a SINGLE `;`-separated
        // group, hence one output. The trailing `2` is the third `,`-separated
        // field of that one output, not an output count -- reading it as a
        // count is the classic way to mint two phantom nodes for this line.
        assert_eq!(rec.num_outputs, 1);
        assert!(!rec.is_coinbase());
        let inputs: Vec<(i32, i32)> = rec.inputs().map(|r| r.unwrap()).collect();
        assert_eq!(inputs, vec![(217481, 0), (217536, 0)]);
    }

    #[test]
    fn late_coinbase_line_parses() {
        let line = "1656626446,743071,745466762,1,0,190,1::802635047,635073896,2;1050047673,0,4";
        let rec = parse_line(line, 1, STRICT).unwrap().unwrap();
        assert_eq!(rec.tx_id, 745466762);
        assert_eq!(rec.num_outputs, 2);
        assert!(rec.is_coinbase());
    }

    #[test]
    fn blank_lines_are_skipped_in_both_modes() {
        for opts in [STRICT, LENIENT] {
            assert!(parse_line("", 1, opts).unwrap().is_none());
            assert!(parse_line("   \t ", 2, opts).unwrap().is_none());
            assert!(parse_line("\n", 3, opts).unwrap().is_none());
        }
    }

    #[test]
    fn short_lines_are_errors_in_strict_and_skips_in_lenient() {
        for line in ["a:b:", "a::"] {
            match parse_line(line, 42, STRICT) {
                Err(Error::BadSectionCount { line: 42, .. }) => {}
                other => panic!("expected BadSectionCount, got {other:?}"),
            }
            assert!(parse_line(line, 42, LENIENT).unwrap().is_none());
        }
    }

    #[test]
    fn over_long_lines_are_errors_in_strict_and_parse_in_lenient() {
        let line = "1,2,3,0,0,0,0:b:c:d";
        match parse_line(line, 9, STRICT) {
            Err(Error::BadSectionCount { line: 9, found: 4 }) => {}
            other => panic!("expected BadSectionCount, got {other:?}"),
        }
        let rec = parse_line(line, 9, LENIENT).unwrap().unwrap();
        assert_eq!(rec.tx_id, 3);
        assert_eq!(rec.num_outputs, 1);
    }

    #[test]
    fn crlf_is_stripped() {
        assert_eq!(strip_eol("abc\r\n"), "abc");
        assert_eq!(strip_eol("abc\n"), "abc");
        assert_eq!(strip_eol("abc"), "abc");
        let rec = parse_line("1231006505,0,0,1,0,204,0::0,5000000000,1\r\n", 1, STRICT)
            .unwrap()
            .unwrap();
        assert_eq!(rec.num_outputs, 1);
    }

    #[test]
    fn tx_id_above_i32_max_is_an_error() {
        match parse_line("1,2,3000000000,0,0,0,0::a", 5, STRICT) {
            Err(Error::BadInteger {
                line: 5,
                field: "txId",
                value,
            }) => assert_eq!(value, "3000000000"),
            other => panic!("expected BadInteger, got {other:?}"),
        }
        // Lenient drops the record instead, per the `Mode::Lenient` contract;
        // without such a mode the only option would be to abort on the spot.
        assert!(parse_line("1,2,3000000000,0,0,0,0::a", 5, LENIENT)
            .unwrap()
            .is_none());
    }

    #[test]
    fn short_info_section_is_fatal_only_in_strict_mode() {
        match parse_line("1,2::a", 9, STRICT) {
            Err(Error::BadInfoFields { line: 9, found: 2 }) => {}
            other => panic!("expected BadInfoFields, got {other:?}"),
        }
        assert!(parse_line("1,2::a", 9, LENIENT).unwrap().is_none());
    }

    #[test]
    fn negative_tx_id_is_rejected_only_in_strict_mode() {
        match parse_line("1,2,-5,0,0,0,0::a", 3, STRICT) {
            Err(Error::NegativeId {
                line: 3,
                field: "txId",
                value: -5,
            }) => {}
            other => panic!("expected NegativeId, got {other:?}"),
        }
        let rec = parse_line("1,2,-5,0,0,0,0::a", 3, LENIENT)
            .unwrap()
            .unwrap();
        // The negative id survives verbatim: `crate::pack` round-trips it, so
        // it reaches the node map's first column as `-5`, not as 4294967291.
        assert_eq!(rec.tx_id, -5);
    }

    #[test]
    fn malformed_input_field_counts_are_reported() {
        let rec = parse_line("1,2,3,0,0,0,0:1,2:a", 11, STRICT)
            .unwrap()
            .unwrap();
        match rec.inputs().next().unwrap() {
            Err(Error::BadInputFields {
                line: 11,
                index: 0,
                found: 2,
            }) => {}
            other => panic!("expected BadInputFields, got {other:?}"),
        }
    }

    #[test]
    fn trailing_input_separator_is_swallowed() {
        let rec = parse_line("1,2,3,0,0,0,0:0,0,1,0;:a", 1, STRICT)
            .unwrap()
            .unwrap();
        let inputs: Vec<(i32, i32)> = rec.inputs().map(|r| r.unwrap()).collect();
        assert_eq!(inputs, vec![(1, 0)]);
    }
}
