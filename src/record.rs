//! Record parser. Ported from `PaymentGraphEdgeListBuilder.java` by
//! **Matteo Loporchio** (lines 50-72).
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
//! Everything here is zero-copy: the parser borrows from the caller's line
//! buffer and allocates nothing on the happy path. Java's `String.split`
//! semantics are reproduced exactly where they matter and fixed where they are
//! bugs; every divergence is called out in the item documentation with the
//! Java line number it refers to.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::{Mode, PgError, PgResult};

/// Parsing options. Currently only the strict/lenient switch.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub struct ParseOpts {
    /// How malformed records are treated.
    pub mode: Mode,
}

/// One parsed transaction record, borrowing from the caller's line buffer.
///
/// Only the fields the Java program actually consumes are extracted:
/// `infos[2]` (the transaction id) and the *number* of output sections.
/// Amounts, addresses, the block height and the `isCoinbase` flag at
/// `infos[3]` are never read — Java detects a coinbase purely with
/// `parts[1].equals("")` (line 68), and so do we.
#[derive(Copy, Clone, Debug)]
pub struct TxRecord<'a> {
    /// `infos[2]`, parsed as `i32` exactly like `Integer.parseInt` (line 57).
    pub tx_id: i32,
    /// Number of output slots, i.e. `outputs.length` in Java (line 55) with
    /// the empty-section bug fixed. See [`count_outputs`].
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
    /// Java line 68: `if (!parts[1].equals(""))`. The `isCoinbase` flag in
    /// `infos[3]` is ignored by the reference implementation, so it is ignored
    /// here too — the two agree on the corpus, but the emptiness of the input
    /// section is what actually drives edge generation.
    #[inline]
    pub fn is_coinbase(&self) -> bool {
        self.inputs_section.is_empty()
    }

    /// Iterates the `(prevTxId, prevTxOffset)` pairs of this transaction, in
    /// file order — Java lines 69-72.
    #[inline]
    pub fn inputs(&self) -> InputRefs<'a> {
        InputRefs::new(self.inputs_section, self.line, self.opts)
    }
}

/// Iterator over a record's inputs, yielding `(prevTxId, prevTxOffset)`.
///
/// Reproduces `parts[1].split(";")` followed by `inputs[k].split(",")`:
/// a trailing `;` run is swallowed (Java: `"a;b;".split(";")` has length 2),
/// an empty section yields zero items, and a leading `;` produces an empty
/// piece which then fails the four-field check (Java threw
/// `ArrayIndexOutOfBoundsException` there).
pub struct InputRefs<'a> {
    /// `None` once the section is known to hold no pieces at all.
    inner: Option<std::str::Split<'a, char>>,
    index: usize,
    line: u64,
    opts: ParseOpts,
}

impl<'a> InputRefs<'a> {
    fn new(section: &'a str, line: u64, opts: ParseOpts) -> Self {
        // Java drops *all* trailing empty pieces, and `"".split(";")` /
        // `";".split(";")` yield a zero-length list of usable inputs.
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
    type Item = PgResult<(i32, i32)>;

    fn next(&mut self) -> Option<Self::Item> {
        let piece = self.inner.as_mut()?.next()?;
        let index = self.index;
        self.index += 1;

        // Java: inputs[offset].split(",") then inputParts[2] / inputParts[3].
        // Fields 0 and 1 (the previous transaction's hash and the amount) are
        // never touched.
        let mut fields = piece.split(',');
        let f0 = fields.next();
        let f1 = fields.next();
        let prev_tx = fields.next();
        let prev_off = fields.next();
        let (prev_tx, prev_off) = match (f0, f1, prev_tx, prev_off) {
            (Some(_), Some(_), Some(a), Some(b)) => (a, b),
            _ => {
                let found = piece.split(',').count();
                return Some(Err(PgError::BadInputFields {
                    line: self.line,
                    index,
                    found,
                }));
            }
        };

        let prev_tx_id = match prev_tx.parse::<i32>() {
            Ok(v) => v,
            Err(_) => {
                return Some(Err(PgError::BadInteger {
                    line: self.line,
                    field: "prevTxId",
                    value: prev_tx.to_string(),
                }))
            }
        };
        let prev_tx_offset = match prev_off.parse::<i32>() {
            Ok(v) => v,
            Err(_) => {
                return Some(Err(PgError::BadInteger {
                    line: self.line,
                    field: "prevTxOffset",
                    value: prev_off.to_string(),
                }))
            }
        };

        if self.opts.mode == Mode::Strict {
            // Java accepted negatives silently: `Integer.parseInt("-5")`
            // succeeds and `pack` round-trips it, so the node map could print
            // `-5\t0\t0`. Nothing downstream expects that.
            if prev_tx_id < 0 {
                return Some(Err(PgError::NegativeId {
                    line: self.line,
                    field: "prevTxId",
                    value: prev_tx_id,
                }));
            }
            if prev_tx_offset < 0 {
                return Some(Err(PgError::NegativeId {
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
/// inside the last output field. The corpus contains zero CR bytes
/// (`grep -c $'\r'` = 0 on 100 000-line heads of `chunk_01` and `chunk_05`),
/// so this is a no-op today; it exists so the port stays byte-identical to
/// `BufferedReader.readLine()` if the data is ever regenerated on a machine
/// that writes CRLF.
pub fn strip_eol(line: &str) -> &str {
    let line = line.strip_suffix('\n').unwrap_or(line);
    line.strip_suffix('\r').unwrap_or(line)
}

/// Splits a line on `':'` with Java's `String.split(":")` semantics
/// (default limit 0), returning the first three pieces and the *post-trimming*
/// piece count.
///
/// Java's rules, in full:
///
/// 1. all trailing empty pieces are dropped, repeatedly;
/// 2. leading and interior empty pieces are kept;
/// 3. when the separator does not occur at all the whole input is returned as
///    a single piece — so `"".split(":")` has length **1**, not 0.
///
/// The resulting table, which the unit tests assert verbatim:
///
/// | line | count | Java outcome |
/// |---|---|---|
/// | `info:inputs:outputs` | 3 | normal path |
/// | `info::outputs` (coinbase) | 3 | normal, `parts[1]` empty |
/// | `info:inputs:` (zero outputs) | **2** | AIOOBE at line 55 |
/// | `info::` | **1** | AIOOBE at line 54 |
/// | `""` (blank line) | **1** | AIOOBE at line 54 |
/// | `info:inputs::x` | 4 | `parts[2]` empty → the node-0 bug |
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
            // The line was nothing but separators: every piece is empty and
            // Java drops them all (`":".split(":")` has length 0).
            0
        } else {
            1 + trimmed.as_bytes().iter().filter(|b| **b == b':').count()
        }
    };

    (parts, count)
}

/// Counts the output slots of a record's third section.
///
/// Equivalent to `parts[2].split(";").length` **with the phantom-edge bug
/// fixed**: an empty section means *zero* outputs here.
///
/// Java, lines 55/59/60 and 76-77:
///
/// ```java
/// String[] outputs = parts[2].split(";");                   // "" -> [""], length 1 !
/// long[] currentOutputNodeIds = new long[outputs.length];   // new long[1] == {0}
/// if (!parts[2].equals("")) { ...fill... }                  // skipped: array stays {0}
/// ...
/// for (int i = 0; i < currentOutputNodeIds.length; i++)
///     edgeWriter.printf("%d\t%d\n", sourceNodeId, currentOutputNodeIds[i]);
/// ```
///
/// The guard blocks node *creation* but not edge *emission*, and Java's
/// default array value `0` is a perfectly valid node id — the id of output 0
/// of the genesis coinbase. Every input of such a line therefore emitted a
/// silent, bogus edge into the genesis output (reproduced with the compiled
/// class: `Nodes: 1  Edges: 1`, edge list `0\t0`). Zero lines in the corpus
/// can reach it, and there is no reading under which that edge is correct, so
/// it is fixed unconditionally and without a flag.
///
/// The count is a byte scan, never an allocating split: a trailing `;` run is
/// dropped exactly as `"a;b;".split(";")` yields length 2.
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
/// The Java equivalent is lines 50-57 plus the `outputs.length` of line 55.
/// Deliberate divergences, each unreachable on the current corpus:
///
/// * a blank or whitespace-only line is skipped in **both** modes; Java threw
///   `ArrayIndexOutOfBoundsException: Index 1 out of bounds for length 1` at
///   line 54 and destroyed a multi-hour run with an opaque message;
/// * fewer than three sections is a line-numbered [`PgError::BadSectionCount`]
///   under [`Mode::Strict`] (Java: AIOOBE at line 54 or 55);
/// * *more* than three sections is also rejected under [`Mode::Strict`]; Java
///   silently used `parts[0..3]`, and since the fourth section pushes an empty
///   `parts[2]`, that is precisely the path into the node-0 phantom-edge bug.
///
/// The transaction id is parsed as `i32`, never `u32` and never `i64`:
/// widening would change `crate::pack` for any value `>= 2^31` and diverge from
/// the reference. The largest txId in the corpus is 778 613 437.
pub fn parse_line(line: &str, line_no: u64, opts: ParseOpts) -> PgResult<Option<TxRecord<'_>>> {
    let line = strip_eol(line);

    // Fix for the AIOOBE at Java line 54. Cannot change behaviour on the
    // corpus (it has no blank lines), but a stray blank line introduced by a
    // future `cat` or an editor would otherwise kill the whole run.
    if line.trim().is_empty() {
        return Ok(None);
    }

    let (parts, n) = split_sections(line);
    if n < 3 {
        return match opts.mode {
            Mode::Strict => Err(PgError::BadSectionCount {
                line: line_no,
                found: n,
            }),
            Mode::Lenient => Ok(None),
        };
    }
    if n > 3 {
        if opts.mode == Mode::Strict {
            return Err(PgError::BadSectionCount {
                line: line_no,
                found: n,
            });
        }
        let seen = EXTRA_SECTION_WARNINGS.fetch_add(1, Ordering::Relaxed);
        if seen % 10_000 == 0 {
            log::warn!(
                "line {line_no}: {n} ':'-separated sections, using the first three \
                 (occurrence {}); Java would have taken the same three and emitted \
                 phantom edges into node 0",
                seen + 1
            );
        }
    }

    // Java line 53/57: infos = parts[0].split(","), txId = parseInt(infos[2]).
    let mut infos = parts[0].split(',');
    let f0 = infos.next();
    let f1 = infos.next();
    let f2 = infos.next();
    let tx_field = match (f0, f1, f2) {
        (Some(_), Some(_), Some(v)) => v,
        _ => {
            // Java: AIOOBE on `infos[2]`. Strict reports the line; Lenient
            // skips it, per the `Mode::Lenient` contract.
            if opts.mode == Mode::Lenient {
                warn_skipped(line_no, "the info section has fewer than 3 fields");
                return Ok(None);
            }
            return Err(PgError::BadInfoFields {
                line: line_no,
                found: parts[0].split(',').count(),
            });
        }
    };
    let tx_id = match tx_field.parse::<i32>() {
        Ok(v) => v,
        Err(_) => {
            // Java: `NumberFormatException: For input string: "3000000000"`,
            // with no indication of which line it came from -- and it killed
            // the whole run. Strict names the line and the offending text;
            // Lenient skips the record, as `Mode::Lenient` documents.
            if opts.mode == Mode::Lenient {
                warn_skipped(line_no, "txId does not fit in an i32");
                return Ok(None);
            }
            return Err(PgError::BadInteger {
                line: line_no,
                field: "txId",
                value: tx_field.to_string(),
            });
        }
    };
    if tx_id < 0 && opts.mode == Mode::Strict {
        return Err(PgError::NegativeId {
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
    fn split_sections_matches_java_table() {
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
    fn count_outputs_matches_java_split() {
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
        // group, hence one output. Java agrees -- `parts[2].split(";")` has
        // length 1 (PaymentGraphEdgeListBuilder.java:55). The trailing `2` is a
        // field of that one output, not an output count.
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
                Err(PgError::BadSectionCount { line: 42, .. }) => {}
                other => panic!("expected BadSectionCount, got {other:?}"),
            }
            assert!(parse_line(line, 42, LENIENT).unwrap().is_none());
        }
    }

    #[test]
    fn over_long_lines_are_errors_in_strict_and_parse_in_lenient() {
        let line = "1,2,3,0,0,0,0:b:c:d";
        match parse_line(line, 9, STRICT) {
            Err(PgError::BadSectionCount { line: 9, found: 4 }) => {}
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
            Err(PgError::BadInteger {
                line: 5,
                field: "txId",
                value,
            }) => assert_eq!(value, "3000000000"),
            other => panic!("expected BadInteger, got {other:?}"),
        }
        // Lenient drops the record instead, per the `Mode::Lenient` contract.
        // Java had no such mode: it died on the spot.
        assert!(parse_line("1,2,3000000000,0,0,0,0::a", 5, LENIENT)
            .unwrap()
            .is_none());
    }

    #[test]
    fn short_info_section_is_fatal_only_in_strict_mode() {
        match parse_line("1,2::a", 9, STRICT) {
            Err(PgError::BadInfoFields { line: 9, found: 2 }) => {}
            other => panic!("expected BadInfoFields, got {other:?}"),
        }
        assert!(parse_line("1,2::a", 9, LENIENT).unwrap().is_none());
    }

    #[test]
    fn negative_tx_id_is_rejected_only_in_strict_mode() {
        match parse_line("1,2,-5,0,0,0,0::a", 3, STRICT) {
            Err(PgError::NegativeId {
                line: 3,
                field: "txId",
                value: -5,
            }) => {}
            other => panic!("expected NegativeId, got {other:?}"),
        }
        let rec = parse_line("1,2,-5,0,0,0,0::a", 3, LENIENT)
            .unwrap()
            .unwrap();
        assert_eq!(rec.tx_id, -5);
    }

    #[test]
    fn malformed_input_field_counts_are_reported() {
        let rec = parse_line("1,2,3,0,0,0,0:1,2:a", 11, STRICT)
            .unwrap()
            .unwrap();
        match rec.inputs().next().unwrap() {
            Err(PgError::BadInputFields {
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
