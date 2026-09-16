"""Structured lab results from OCR output.

Turns the rows of a lab report such as

    Hemoglobin ........ 9.9      L  g/dL     12 - 15

into database-ready records, in the shape the lab system inserts::

    {"name": "Hemoglobin", "value": 9.9, "flag": "L", "unit": "g/dL",
     "ref_range": "12 - 15", "category": "COMPLETE BLOOD COUNT"}

Differential rows ("Neutrophils (%) 65.7% 7.15 H x10^9/L 2 - 7") also carry
``percent``, placed after ``name`` as in the database format.

Alongside the records comes a review list: per-record OCR confidence and
consistency checks that catch misread digits -- the H/L flag against the
reference range, differential percentages against WBC, and the red-cell
indices against RBC/Hb/Hct. OCR is never perfect, and a wrong digit in a lab
value is worse than a missing one, so every record says whether it needs a
human look before it is inserted.

Test names, categories and canonical units come from ``lab_catalog.json`` (the
lab's test list); tests not in the catalog are still extracted, with the
nearest heading on the report as their category.

Pure functions over the engine's page dicts. Nothing here logs document
content.
"""

from __future__ import annotations

import difflib
import json
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from . import ocr

CATALOG_PATH = Path(os.getenv("OCR_LAB_CATALOG", str(Path(__file__).with_name("lab_catalog.json"))))

# A value read below this confidence gets a low_confidence note -- unless a
# cross-check (red-cell indices, differential x WBC) already confirms it.
# Surya scores bold print low even when it reads it right, and this lab's
# reports print every flagged value in bold, so the cross-checks matter.
MIN_VALUE_CONFIDENCE = float(os.getenv("OCR_LAB_MIN_CONFIDENCE", "80"))

# --------------------------------------------------------------------------
# Patterns
# --------------------------------------------------------------------------

_NUM = r"\d+(?:[.,]\d+)?"
_RANGE = rf"(?:[<>≤≥]=?\s*{_NUM}|{_NUM}\s*-\s*{_NUM})"
_FLAG = r"(?:HH|LL|H|L|\*|↑|↓)"
_TAIL = (
    rf"(?P<value>(?:[<>≤≥]\s*)?{_NUM})"
    rf"(?:\s+(?P<flag>{_FLAG}))?"
    r"(?:\s+(?P<unit>.+?))??"
    rf"(?:\s+(?P<range>{_RANGE}))?$"
)
# "9.9 L g/dL 12 - 15"
_PLAIN = re.compile(rf"^{_TAIL}")
# "65.7% 7.15 H x10^9/L 2 - 7" -- a differential row: percentage, then count.
_PERCENT = re.compile(rf"^(?P<percent>{_NUM})\s*%\s+{_TAIL}")
# Same, for a row whose name says "(%)" but whose % sign OCR dropped.
_PERCENT_NO_SIGN = re.compile(rf"^(?P<percent>{_NUM})\s*%?\s+{_TAIL}")

# Dot leaders between a test name and its value: "WBC .......... 10.88".
# OCR reads them as runs of dots and dashes ("-.- .----"); a range's
# single " - " never has three in a row.
_LEADERS = re.compile(r"(?:\s*[.·•…_\-‹›,]){3,}")
_KHMER = re.compile(r"[ក-៿]")
# Not one Unicode range: ¹²³ are Latin-1 (U+00B9/B2/B3), the rest U+2070-2079.
_SUPER_DIGITS = "⁰¹²³⁴⁵⁶⁷⁸⁹"
_SUPERSCRIPTS = str.maketrans(_SUPER_DIGITS, "0123456789")
# "x10⁹/L", "x10¹²/L", "x10^9/L", and OCR's flattened "x109/L" / "x1012/L".
_POWER_UNIT = re.compile(rf"^[x×*]\s*10\s*\^?\s*([0-9{_SUPER_DIGITS}]{{1,2}})\s*/\s*([A-Za-zµμ]+)$")

# Where the results table starts and stops on a page.
_TABLE_HEADER = re.compile(r"\bt[eo]st\s*n[a-z]{2,3}\b", re.I)
_DEPARTMENT = re.compile(
    r"^(?:ha?ea?matology|hematology|biochemistry|immunology|serology|microbiology|"
    r"parasitology|urinalysis|hormones?|endocrinology|coagulation)\b",
    re.I,
)
_FOOTER = re.compile(r"ថ្ងៃពិសោធន៍|ថ្ងៃចេញលទ្ធផល|ត្រួតពិនិត្យ|\b(?:validated|verified|approved)\s+by\b", re.I)

# Report header: what links the results to a patient and a sample. The name
# and address are deliberately not extracted -- the patient code is enough
# to join on, and it keeps identifying details out of the results database.
_PATIENT_CODE = re.compile(r"(?:លេខអ្នកជំងឺ|patient\s*(?:id|no\.?|code))\s*[:៖]?\s*([A-Z0-9][A-Z0-9-]{3,})", re.I)
_SAMPLE_NO = re.compile(r"(?<![\d-])(\d{3,5}-\d{6,8})(?![\d-])")
_DATETIME = re.compile(r"\b(\d{1,2})-([A-Za-z]{3})-(\d{4})\s+(\d{1,2}):(\d{2})\b")
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}


# --------------------------------------------------------------------------
# Catalog
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CatalogTest:
    name: str
    category: str | None
    unit: str | None
    kind: str  # "number" | "text"
    aliases: tuple[str, ...]
    percent: bool  # a differential row: "(%)" in the printed name


def _alias_pattern(alias: str) -> re.Pattern[str]:
    """Match ``alias`` at the start of a row, spacing-insensitive, whole word."""
    tokens = re.findall(r"\(|\)|[^\s()]+", alias)
    return re.compile(r"^" + r"\s*".join(re.escape(t) for t in tokens) + r"(?=$|[\s:])", re.I)


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9%]+", "", text.lower())


@lru_cache(maxsize=1)
def catalog() -> tuple[CatalogTest, ...]:
    data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    tests = []
    for entry in data["tests"]:
        aliases = (entry["name"], *entry.get("aliases", ()))
        tests.append(
            CatalogTest(
                name=entry["name"],
                category=entry.get("category"),
                unit=entry.get("unit"),
                kind=entry.get("kind", "number"),
                aliases=aliases,
                percent=any("(%)" in alias for alias in aliases),
            )
        )
    return tuple(tests)


@lru_cache(maxsize=1)
def _alias_index() -> tuple[tuple[re.Pattern[str], CatalogTest], ...]:
    """Every alias, longest first, so "Neutrophils (%)" wins over "Neutrophils"
    and "MCHC" is never read as "MCH"."""
    pairs = [(alias, test) for test in catalog() for alias in test.aliases]
    pairs.sort(key=lambda pair: len(pair[0]), reverse=True)
    return tuple((_alias_pattern(alias), test) for alias, test in pairs)


@lru_cache(maxsize=1)
def _fuzzy_index() -> dict[str, CatalogTest]:
    # Short names ("Hb", "MCH") are too close to each other to guess at.
    return {_norm(alias): test for test in catalog() for alias in test.aliases if len(_norm(alias)) >= 5}


# --------------------------------------------------------------------------
# Row parsing
# --------------------------------------------------------------------------


@dataclass
class Row:
    text: str
    page: int
    words: list[tuple[str, float]] = field(default_factory=list)  # (text, conf)

    def confidence_of(self, tokens: list[str]) -> float | None:
        """Lowest confidence among the words that carry ``tokens``.

        Only the value (and its percentage) matter: a catalog-matched name is
        already verified, and units are normalised.
        """
        wanted = {token.rstrip("%") for token in tokens if token}
        confs = [conf for text, conf in self.words if text.rstrip("%") in wanted]
        if not confs:  # e.g. OCR glued "10.88H" into one word
            confs = [conf for text, conf in self.words if any(t in text for t in wanted)]
        return min(confs) if confs else None


@dataclass
class Parsed:
    record: dict[str, Any]
    notes: list[dict[str, Any]] = field(default_factory=list)
    tokens: list[str] = field(default_factory=list)  # value text as printed


def _clean(text: str) -> str:
    text = _LEADERS.sub(" ", text)
    text = text.replace("×", "x").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text).strip()


def _number(text: str) -> int | float | str:
    """"0.60" -> 0.6, "744" -> 744, "< 0.5" stays text (a censored value)."""
    text = text.replace(",", ".").strip()
    if not re.fullmatch(_NUM, text):
        return re.sub(r"^([<>≤≥])\s*", r"\1 ", text)
    number = float(text)
    return int(number) if number.is_integer() else number


def normalize_unit(unit: str | None) -> str | None:
    if not unit:
        return None
    unit = unit.strip().replace("×", "x")
    match = _POWER_UNIT.match(unit)
    if match:
        return f"x10^{match.group(1).translate(_SUPERSCRIPTS)}/{match.group(2)}"
    # Any other superscript power: "mm³" -> "mm^3".
    return re.sub(f"[{_SUPER_DIGITS}]+", lambda m: "^" + m.group(0).translate(_SUPERSCRIPTS), unit)


def normalize_range(text: str | None) -> str | None:
    if not text:
        return None
    text = re.sub(r"\s*-\s*", " - ", text.strip())
    return re.sub(r"^([<>≤≥]=?)\s*", r"\1 ", text)


def _plausible_unit(unit: str) -> bool:
    return (
        len(unit) <= 15
        and unit.count(" ") <= 1
        and not unit.startswith("-")
        and not _KHMER.search(unit)
        and bool(re.search(r"[A-Za-zµμ%]", unit))
    )


def _parse_tail(tail: str, percent_expected: bool) -> dict[str, Any] | None:
    """Split "65.7% 7.15 H x109/L 2 - 7" into its parts, or None."""
    patterns = [_PERCENT_NO_SIGN if percent_expected else _PERCENT, _PLAIN]
    if not percent_expected:
        patterns.reverse()
    for pattern in patterns:
        match = pattern.match(tail)
        if not match:
            continue
        unit = match.group("unit")
        if unit is not None and not _plausible_unit(unit):
            continue
        flag = match.group("flag")
        percent_text = match.group("percent") if "percent" in pattern.groupindex else None
        return {
            "tokens": [match.group("value"), percent_text],
            "percent_sign": percent_text is not None and bool(re.match(rf"^{_NUM}\s*%", tail)),
            "percent": _number(percent_text) if percent_text else None,
            "value": _number(match.group("value")),
            "flag": {"↑": "H", "↓": "L"}.get(flag, flag),
            "unit": normalize_unit(unit),
            "ref_range": normalize_range(match.group("range")),
        }
    return None


def _match_catalog(text: str) -> tuple[CatalogTest | None, str, str, dict[str, Any] | None]:
    """Find the test name at the start of ``text``.

    Returns (catalog test or None, name as printed, the rest of the row,
    a fuzzy-match note or None).
    """
    for pattern, test in _alias_index():
        match = pattern.match(text)
        if match:
            return test, match.group(0), text[match.end():].lstrip(" :"), None

    # Not a known spelling: the name is everything before the first number.
    match = re.search(r"(?:^|\s)(?=[<>≤≥]?\s*\d)", text)
    name = (text[: match.start()] if match else text).strip(" :")
    rest = text[match.start():].strip() if match else ""
    guess = difflib.get_close_matches(_norm(name), _fuzzy_index(), n=1, cutoff=0.85) if len(_norm(name)) >= 5 else []
    if guess:
        test = _fuzzy_index()[guess[0]]
        note = {"code": "name_fuzzy", "params": {"read": name, "matched": test.name}}
        return test, name, rest, note
    return None, name, rest, None


def _reconcile_unit(unit: str | None, test: CatalogTest, notes: list[dict[str, Any]]) -> str | None:
    if not test.unit:
        return unit
    if unit is None:
        notes.append({"code": "unit_missing", "params": {"unit": test.unit}})
        return test.unit
    if _norm(unit) == _norm(test.unit):
        return test.unit  # same unit, canonical spelling ("fL" -> "fl")
    notes.append({"code": "unit_differs", "params": {"unit": unit, "expected": test.unit}})
    return unit


def parse_row(text: str, heading: str | None) -> Parsed | None:
    """One report row -> a record, or None if it is not a result row."""
    text = _clean(text)
    if not text:
        return None
    test, name, rest, fuzzy = _match_catalog(text)
    notes = [fuzzy] if fuzzy else []

    if test is not None and test.kind == "text":
        value = re.sub(r"\s*:\s*", ": ", rest).strip(" :")
        # OCR glues the ABO group to "Rh": "ORh (D): Positive".
        value = re.sub(r"^(AB|A|B|O)\s*(Rh\b)", r"\1 \2", value)
        if not value:
            return None
        record = {"name": test.name, "value": value, "flag": None, "unit": None,
                  "ref_range": None, "category": test.category or heading}
        return Parsed(record, notes, value.split())

    percent_expected = (test.percent if test else False) or name.rstrip().endswith("(%)")
    parts = _parse_tail(rest, percent_expected)
    if parts is None:
        return None

    if test is None:
        # Unknown test: demand enough structure that a stray line with a
        # number in it (a date, a phone number) or a name OCR turned to
        # noise ("ឯ]ខ169 , -.-") cannot pass as a result.
        compact = name.replace(" ", "")
        if not compact or sum(ch.isalpha() for ch in compact) < 0.6 * len(compact):
            return None
        if not (parts["unit"] or parts["ref_range"]):
            return None
        if not isinstance(parts["value"], (int, float)) and not parts["ref_range"]:
            return None
        resolved = re.sub(r"\s*\(%\)$", "", name) if parts["percent"] is not None else name
        unit, category = parts["unit"], heading
    else:
        resolved = test.name
        unit = _reconcile_unit(parts["unit"], test, notes)
        category = test.category or heading

    record: dict[str, Any] = {"name": resolved}
    if parts["percent"] is not None:
        record["percent"] = parts["percent"]
        if not parts["percent_sign"]:
            # OCR reads a "%" as "96" or "9" as easily as it drops it:
            # "0.2%" -> "0.296". _cross_checks may suggest the reading.
            notes.append({"code": "percent_sign_missing", "params": {"read": parts["tokens"][1]}})
    record.update(value=parts["value"], flag=parts["flag"], unit=unit,
                  ref_range=parts["ref_range"], category=category)
    return Parsed(record, notes, [t for t in parts["tokens"] if t])


def _is_heading(text: str) -> bool:
    text = _clean(text)
    letters = sum(ch.isalpha() for ch in text)
    return (
        0 < len(text) <= 60
        and not re.search(r"\d|:", text)
        and letters >= 0.6 * len(text.replace(" ", ""))
    )


# --------------------------------------------------------------------------
# Pages -> rows
# --------------------------------------------------------------------------


def rows_from_page(page: dict[str, Any]) -> list[Row]:
    """Rebuild the page's visual rows from its word boxes.

    The engine can put a table's name and value columns in different blocks,
    which its own line numbering would keep apart. Falls back to the page
    text when words were not returned.
    """
    words = page.get("words")
    if not words:
        return [Row(line, page["page"]) for line in page["text"].splitlines() if line.strip()]

    boxes = [
        {**word, "bbox": [word["bbox"][0], word["bbox"][1],
                          word["bbox"][0] + word["bbox"][2], word["bbox"][1] + word["bbox"][3]]}
        for word in words
    ]
    return [
        Row(" ".join(w["text"] for w in row), page["page"], [(w["text"], float(w["conf"])) for w in row])
        for row in ocr.group_rows(boxes)
    ]


# --------------------------------------------------------------------------
# Consistency checks
# --------------------------------------------------------------------------


def _range_bounds(ref_range: str | None) -> tuple[float | None, float | None] | None:
    if not ref_range:
        return None
    match = re.fullmatch(rf"({_NUM}) - ({_NUM})", ref_range)
    if match:
        return float(match.group(1).replace(",", ".")), float(match.group(2).replace(",", "."))
    match = re.fullmatch(rf"([<>≤≥])=? ({_NUM})", ref_range)
    if match:
        bound = float(match.group(2).replace(",", "."))
        return (None, bound) if match.group(1) in "<≤" else (bound, None)
    return None


def _check_flag(record: dict[str, Any]) -> dict[str, Any] | None:
    value, bounds = record["value"], _range_bounds(record["ref_range"])
    if not isinstance(value, (int, float)) or bounds is None:
        return None
    low, high = bounds
    expected = "H" if high is not None and value > high else "L" if low is not None and value < low else None
    flag = record["flag"]
    actual = {"HH": "H", "LL": "L"}.get(flag, flag)
    params = {"value": value, "flag": flag, "range": record["ref_range"]}
    if expected and actual is None:
        return {"code": "flag_not_set", "params": {**params, "expected": expected}}
    if actual in ("H", "L") and expected is None:
        return {"code": "flag_unexpected", "params": params}
    if actual in ("H", "L") and expected and actual != expected:
        return {"code": "flag_wrong_direction", "params": {**params, "expected": expected}}
    if actual == "*" and expected is None:
        return {"code": "flag_unexpected", "params": params}
    return None


def _value_of(records: list[dict[str, Any]], name: str, unit: str | None = None) -> float | None:
    for record in records:
        if record["name"] == name and isinstance(record["value"], (int, float)):
            if unit is None or record["unit"] == unit:
                return float(record["value"])
    return None


def _cross_checks(records: list[dict[str, Any]], notes: list[list[dict[str, Any]]]) -> dict[int, list[str]]:
    """Relationships between results that a single misread digit breaks.

    Adds a note to every record involved in a failed check, and returns the
    checks each record *passed* (record index -> formulas): a value that
    satisfies one of these relations is confirmed by the others, whatever
    confidence OCR gave it.
    """
    index = {record["name"]: i for i, record in enumerate(records)}
    passed: dict[int, list[str]] = {}

    def confirm(names: list[str], formula: str) -> None:
        for name in names:
            if name in index:
                passed.setdefault(index[name], []).append(formula)

    # Differential: absolute count = percentage x WBC.
    wbc = _value_of(records, "WBC")
    differential = [i for i, r in enumerate(records) if "percent" in r and isinstance(r["percent"], (int, float))]
    all_pairs_hold = bool(differential) and wbc is not None
    def pair_holds(percent: float, value: float) -> bool:
        expected = wbc * percent / 100
        return abs(value - expected) <= 0.02 * expected + 0.011

    for i in differential:
        record = records[i]
        if wbc is None or not isinstance(record["value"], (int, float)):
            all_pairs_hold = False
            continue
        formula = f"{record['name']} % × WBC"
        if pair_holds(record["percent"], record["value"]):
            passed.setdefault(i, []).append(formula)
            continue
        all_pairs_hold = False
        notes[i].append({"code": "differential_mismatch", "params": {
            "percent": record["percent"], "wbc": wbc,
            "expected": round(wbc * record["percent"] / 100, 2), "value": record["value"]}})
        # A "%" misread as trailing digits: offer the reading that fits WBC.
        for note in notes[i]:
            read = note["params"].get("read", "") if note["code"] == "percent_sign_missing" else ""
            for suffix in ("96", "9", "6"):
                candidate = read[: -len(suffix)].rstrip(".") if read.endswith(suffix) else ""
                if candidate and re.fullmatch(_NUM, candidate):
                    number = _number(candidate)
                    if pair_holds(float(number), record["value"]):
                        note["params"]["suggested"] = number
                        break
    if all_pairs_hold and len(differential) >= 3:
        confirm(["WBC"], "differential % × WBC")
    if len(differential) >= 5:
        total = sum(records[i]["percent"] for i in differential)
        if abs(total - 100) > 1.0:
            for i in differential:
                notes[i].append({"code": "differential_sum", "params": {"total": round(total, 1)}})

    # Red-cell indices are computed from RBC, Hb and Hct by the analyser, so
    # they agree to rounding -- unless OCR changed a digit somewhere.
    rbc = _value_of(records, "RBC", "x10^12/L")
    hb = _value_of(records, "Hemoglobin", "g/dL")
    hct = _value_of(records, "Hematocrit", "%")
    formulas = {
        "MCV": (hct, rbc, lambda: hct / rbc * 10, "Hct / RBC × 10", ["Hematocrit", "RBC"]),
        "MCH": (hb, rbc, lambda: hb / rbc * 10, "Hb / RBC × 10", ["Hemoglobin", "RBC"]),
        "MCHC": (hb, hct, lambda: hb / hct * 100, "Hb / Hct × 100", ["Hemoglobin", "Hematocrit"]),
    }
    for name, (a, b, compute, formula, inputs) in formulas.items():
        value = _value_of(records, name)
        if name not in index or value is None or a is None or not b:
            continue
        expected = compute()
        if abs(value - expected) > 0.03 * expected + 0.1:
            notes[index[name]].append({"code": "index_mismatch", "params": {
                "index": name, "value": value, "expected": round(expected, 1), "formula": formula}})
        else:
            confirm([name, *inputs], f"{name} = {formula}")
    return passed


# --------------------------------------------------------------------------
# Report header
# --------------------------------------------------------------------------


def _iso_datetime(match: re.Match[str]) -> str | None:
    day, month, year, hour, minute = match.groups()
    number = _MONTHS.get(month.lower())
    if number is None:
        return None
    return f"{int(year):04d}-{number:02d}-{int(day):02d}T{int(hour):02d}:{minute}:00"


def _scan_header(text: str, header: dict[str, Any]) -> None:
    """Pick up the patient code, sample number and sample times from a row.

    The first value found wins; a different value later (a second page for
    another patient in the same upload) is recorded under ``conflicts``.
    """
    found: dict[str, str | None] = {}
    match = _PATIENT_CODE.search(text)
    if match:
        found["patient_code"] = match.group(1)
    # The specimen row: "Blood-EDTA 0007-10092026 ... 10-Sep-2026 04:45 10-Sep-2026 05:43"
    times = list(_DATETIME.finditer(text))
    sample = _SAMPLE_NO.search(text)
    if sample and times:
        found["sample_no"] = sample.group(1)
        found["collected_at"] = _iso_datetime(times[0])
        if len(times) > 1:
            found["received_at"] = _iso_datetime(times[1])
    for key, value in found.items():
        if value is None:
            continue
        if header.get(key) is None:
            header[key] = value
        elif header[key] != value and key not in header["conflicts"]:
            header["conflicts"].append(key)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def extract(pages: list[dict[str, Any]]) -> dict[str, Any]:
    """OCR pages (with ``words``) -> ``{"report", "results", "review", "unparsed"}``.

    ``report`` is the header that links the results to a patient and sample:
    ``patient_code``, ``sample_no``, ``collected_at``, ``received_at`` (ISO
    8601, or None when not found) and ``conflicts`` (fields whose value
    differed between pages). ``results`` is the database-ready list. ``review[i]`` describes
    ``results[i]``: its page, the OCR confidence of its value, the source
    row, the cross-checks it passed and its notes. Any note means a person
    should check that value against the document before it is inserted.
    ``unparsed`` lists rows inside a results table that contain a number but
    could not be read as a result, so nothing disappears silently.
    """
    records: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    notes: list[list[dict[str, Any]]] = []
    unparsed: list[dict[str, Any]] = []
    confidences: list[float | None] = []
    header: dict[str, Any] = {"patient_code": None, "sample_no": None, "collected_at": None,
                              "received_at": None, "conflicts": []}

    for page in pages:
        rows = rows_from_page(page)
        # If OCR missed the "Test Name" header, read the whole page.
        in_table = not any(_TABLE_HEADER.search(row.text) for row in rows)
        heading: str | None = None

        for row in rows:
            text = row.text.strip()
            _scan_header(text, header)
            if _TABLE_HEADER.search(text):
                in_table, heading = True, None
                continue
            if _DEPARTMENT.match(text) or _FOOTER.search(text):
                in_table = False
                continue
            if not in_table:
                continue

            parsed = parse_row(text, heading)
            if parsed is not None:
                row_notes = list(parsed.notes)
                flag_note = _check_flag(parsed.record)
                if flag_note:
                    row_notes.append(flag_note)
                confidence = row.confidence_of(parsed.tokens)
                records.append(parsed.record)
                notes.append(row_notes)
                confidences.append(confidence)
                review.append({"page": row.page, "confidence": None if confidence is None else round(confidence, 1),
                               "source": _clean(text)})
            elif _is_heading(text):
                heading = _clean(text)
            elif re.search(r"\d", text):
                unparsed.append({"page": row.page, "text": _clean(text)})

    passed = _cross_checks(records, notes)
    for i, (entry, row_notes, confidence) in enumerate(zip(review, notes, confidences)):
        entry["cross_checked"] = passed.get(i, [])
        if confidence is not None and confidence < MIN_VALUE_CONFIDENCE and not entry["cross_checked"]:
            row_notes.insert(0, {"code": "low_confidence", "params": {"confidence": round(confidence, 1)}})
        entry["notes"] = row_notes
        entry["needs_review"] = bool(row_notes)
    return {"report": header, "results": records, "review": review, "unparsed": unparsed}
