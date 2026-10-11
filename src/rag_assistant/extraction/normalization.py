from __future__ import annotations

import re
import unicodedata

NORMALIZATION_VERSION = "persian-retrieval-v1"

_CHARACTER_TRANSLATION = str.maketrans(
    {
        "\u064a": "\u06cc",  # Arabic yeh
        "\u0649": "\u06cc",  # alef maksura
        "\u0643": "\u06a9",  # Arabic kaf
        "\u0660": "0",
        "\u0661": "1",
        "\u0662": "2",
        "\u0663": "3",
        "\u0664": "4",
        "\u0665": "5",
        "\u0666": "6",
        "\u0667": "7",
        "\u0668": "8",
        "\u0669": "9",
        "\u06f0": "0",
        "\u06f1": "1",
        "\u06f2": "2",
        "\u06f3": "3",
        "\u06f4": "4",
        "\u06f5": "5",
        "\u06f6": "6",
        "\u06f7": "7",
        "\u06f8": "8",
        "\u06f9": "9",
    }
)
_REMOVED_FORMAT_CHARACTERS = frozenset(
    {
        "\u00ad",  # soft hyphen
        "\u061c",  # Arabic letter mark
        "\u0640",  # tatweel
        "\u200b",  # zero-width space
        "\u200d",  # zero-width joiner
        "\u200e",
        "\u200f",
        "\u202a",
        "\u202b",
        "\u202c",
        "\u202d",
        "\u202e",
        "\u2066",
        "\u2067",
        "\u2068",
        "\u2069",
    }
)
_WHITESPACE = re.compile(r"\s+")
_ZWNJ = "\u200c"


def _is_arabic_mark(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x0610 <= codepoint <= 0x061A
        or 0x064B <= codepoint <= 0x065F
        or codepoint == 0x0670
        or 0x06D6 <= codepoint <= 0x06ED
    )


def _is_arabic_script_letter(character: str) -> bool:
    codepoint = ord(character)
    in_arabic_block = (
        0x0600 <= codepoint <= 0x06FF
        or 0x0750 <= codepoint <= 0x077F
        or 0x08A0 <= codepoint <= 0x08FF
    )
    return in_arabic_block and unicodedata.category(character).startswith("L")


class PersianTextNormalizer:
    """Create the deterministic retrieval representation of a source block."""

    version = NORMALIZATION_VERSION

    def normalize(self, text: str) -> str:
        value = unicodedata.normalize("NFKC", text).translate(_CHARACTER_TRANSLATION)
        filtered = "".join(
            character
            for character in value
            if character not in _REMOVED_FORMAT_CHARACTERS and not _is_arabic_mark(character)
        )
        filtered = self._normalize_zwnj(filtered)
        return _WHITESPACE.sub(" ", filtered).strip()

    @staticmethod
    def _normalize_zwnj(text: str) -> str:
        output: list[str] = []
        index = 0
        while index < len(text):
            character = text[index]
            if character != _ZWNJ:
                output.append(character)
                index += 1
                continue

            while index + 1 < len(text) and text[index + 1] == _ZWNJ:
                index += 1
            previous = output[-1] if output else ""
            following = text[index + 1] if index + 1 < len(text) else ""
            if _is_arabic_script_letter(previous) and _is_arabic_script_letter(following):
                output.append(_ZWNJ)
            index += 1
        return "".join(output)
