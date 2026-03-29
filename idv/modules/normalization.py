"""Text normalization and field validation utilities."""
from __future__ import annotations

import re
from datetime import datetime
from typing import Dict, Tuple

ARABIC_TO_LATIN_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def normalize_text_whitespace(text: str) -> str:
    """Normalize whitespace while preserving Arabic content."""
    return re.sub(r"\s+", " ", text).strip()


def normalize_digits(text: str) -> str:
    """Normalize Arabic and Latin numerals to Latin digits-only output."""
    return re.sub(r"\D", "", text.translate(ARABIC_TO_LATIN_DIGITS))


def validate_birth_date(text: str, min_year: int, max_year: int) -> Tuple[bool, Dict[str, str]]:
    """Validate birth date candidate with conservative parsing."""
    digits = normalize_digits(text)
    if len(digits) < 8:
        return False, {"digits_only": digits, "formatted": "", "reason": "insufficient_digits"}
    dd, mm, yyyy = int(digits[:2]), int(digits[2:4]), int(digits[4:8])
    try:
        dt = datetime(yyyy, mm, dd)
    except ValueError:
        return False, {"digits_only": digits, "formatted": "", "reason": "invalid_date"}
    if yyyy < min_year or yyyy > max_year:
        return False, {"digits_only": digits, "formatted": "", "reason": "year_out_of_range"}
    return True, {"digits_only": digits[:8], "formatted": dt.strftime("%Y-%m-%d"), "reason": ""}


def validate_id_number(text: str, valid_lengths: list[int]) -> Tuple[bool, Dict[str, str]]:
    """Validate ID number with conservative length and non-triviality checks."""
    digits = normalize_digits(text)
    if len(digits) not in valid_lengths:
        return False, {"digits_only": digits, "reason": "bad_length"}
    if len(set(digits)) <= 2:
        return False, {"digits_only": digits, "reason": "low_entropy_digits"}
    return True, {"digits_only": digits, "reason": ""}
