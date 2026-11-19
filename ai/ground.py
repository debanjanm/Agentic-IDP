"""Is an extracted value actually on the page?

The check a model cannot fake: its own "I'm 90% sure" tells you little, but a
value that appears nowhere in the page text is almost certainly invented or
misread. Numbers are looked for in every common notation (1,240.55 / 1.240,55 /
1 240,55), dates in common written forms, other strings by their words.

Returns None - "can't tell" - when there is no page text (a page the OCR could
not read), the value is empty, or a string is under 4 characters (currency codes
and the like are inferred, not printed), so a missing signal is never a failed one.
"""

import re
from datetime import date

DATE_FORMATS = ("%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%d/%m/%Y",
                "%m/%d/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d/%m/%y", "%Y/%m/%d", "%Y-%m-%d")
WORD_SHARE = 0.8  # a value with a few OCR-damaged words still counts as on the page


def grounded(value, text):
    """True if `value` is on the page, False if not, None if that can't be checked."""
    text = (text or "").replace(" ", " ")
    if value in (None, "") or isinstance(value, bool) or not text.strip():
        return None
    if isinstance(value, (int, float)):
        return bool(re.search(_number_pattern(value), text))
    value = str(value).strip()
    if len(value) < 4:  # "GBP" is inferred from a symbol or an address, and "A" is on every page
        return None
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        try:
            return _has_date(date.fromisoformat(value), text)
        except ValueError:
            pass
    return _has_words(value, text)


def _number_pattern(n) -> str:
    whole, frac = f"{abs(n):.2f}".split(".")
    forms = set()
    for decimals in {frac, frac.rstrip("0")}:
        for group in ("", ",", ".", " "):
            for point in (".", ","):
                if group == point:
                    continue
                digits = f"{int(whole):,}".replace(",", group)
                forms.add(digits + (point + decimals if decimals else ""))
    # not glued to more digits, so "6.00" is not found inside "16.00" or "6.005"
    return r"(?<![\d.,])(?:" + "|".join(re.escape(f) for f in sorted(forms, key=len, reverse=True)) + r")(?!\d|[.,]\d)"


def _has_date(d, text) -> bool:
    forms = {d.strftime(f) for f in DATE_FORMATS}
    forms |= {re.sub(r"\b0(\d)", r"\1", f) for f in forms}  # 05/03/2024 and 5/3/2024
    return any(re.search(rf"(?<!\d){re.escape(f)}(?!\d)", text, re.I) for f in forms)


def _has_words(value, text) -> bool:
    words = re.findall(r"\w+", value.casefold())
    if not words:
        return None
    page = re.findall(r"\w+", text.casefold())
    if " ".join(words) in " ".join(page):
        return True
    have = set(page)
    return sum(w in have for w in words) / len(words) >= WORD_SHARE
