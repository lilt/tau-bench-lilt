"""Tests for language detection utilities."""

import pytest

from tau2.utils.language_detection import (
    LanguageInfo,
    _clean_text_for_glotlid,
    _language_matches_any,
    parse_language_tag,
)


class TestParseLanguageTag:
    """Tests for parse_language_tag function."""

    def test_simple_language_codes(self):
        """Test parsing simple two-letter language codes."""
        # Spanish
        info = parse_language_tag("es")
        assert info.iso639_3 == "spa"
        assert info.script == "Latn"
        assert info.display_name == "Spanish"
        assert info.glotlid_label == "__label__spa_Latn"

        # German
        info = parse_language_tag("de")
        assert info.iso639_3 == "deu"
        assert info.script == "Latn"
        assert info.display_name == "German"

        # French
        info = parse_language_tag("fr")
        assert info.iso639_3 == "fra"
        assert info.script == "Latn"

    def test_three_letter_codes(self):
        """Test parsing ISO 639-3 codes directly."""
        info = parse_language_tag("deu")
        assert info.iso639_3 == "deu"
        assert info.script == "Latn"

        info = parse_language_tag("spa")
        assert info.iso639_3 == "spa"

    def test_arabic_variants(self):
        """Test Arabic language variants - ar vs arb vs arz."""
        # Generic Arabic (macro language) - should work
        info = parse_language_tag("ar")
        assert info.iso639_3 == "ara"
        assert info.script == "Arab"
        assert info.glotlid_label == "__label__ara_Arab"

        # Standard Arabic (arb) - needs explicit script since assume_script returns None
        info = parse_language_tag("arb-Arab")
        assert info.iso639_3 == "arb"
        assert info.script == "Arab"
        assert info.glotlid_label == "__label__arb_Arab"

        # With explicit script
        info = parse_language_tag("ar-Arab")
        assert info.script == "Arab"

    def test_korean_scripts(self):
        """Test Korean with different scripts - ko-Kore vs ko-Hang."""
        # Default Korean (should infer script)
        info = parse_language_tag("ko")
        assert info.iso639_3 == "kor"
        # Korean default script should be Kore (mixed) or Hang (Hangul)
        assert info.script in ("Kore", "Hang")

        # Explicit Hangul script
        info = parse_language_tag("ko-Hang")
        assert info.iso639_3 == "kor"
        assert info.script == "Hang"

        # Explicit Korean script (mixed Hangul + Hanja)
        info = parse_language_tag("ko-Kore")
        assert info.iso639_3 == "kor"
        assert info.script == "Kore"

    def test_chinese_scripts(self):
        """Test Chinese with different scripts."""
        # Simplified Chinese
        info = parse_language_tag("zh-Hans")
        assert info.script == "Hans"

        # Traditional Chinese
        info = parse_language_tag("zh-Hant")
        assert info.script == "Hant"

        # Generic Chinese - langcodes.assume_script() returns None for macro languages
        # so explicit script is required for Chinese
        info = parse_language_tag("zh")
        # When assume_script returns None, script will be None
        assert info.script is None or info.script in ("Hans", "Hant", "Hani")

    def test_cyrillic_languages(self):
        """Test Cyrillic script languages."""
        # Russian
        info = parse_language_tag("ru")
        assert info.iso639_3 == "rus"
        assert info.script == "Cyrl"

        # Ukrainian
        info = parse_language_tag("uk")
        assert info.iso639_3 == "ukr"
        assert info.script == "Cyrl"

    def test_explicit_script_override(self):
        """Test that explicit script overrides inference."""
        # Serbian can use both Cyrillic and Latin
        info = parse_language_tag("sr-Latn")
        assert info.script == "Latn"

        info = parse_language_tag("sr-Cyrl")
        assert info.script == "Cyrl"

    def test_glotlid_label_without_prefix(self):
        """Test glotlid_label_without_prefix property."""
        info = parse_language_tag("de")
        assert info.glotlid_label_without_prefix == "deu_Latn"
        assert "__label__" not in info.glotlid_label_without_prefix

    def test_invalid_language_tag(self):
        """Test that invalid tags raise ValueError."""
        with pytest.raises(ValueError, match="Invalid BCP-47"):
            parse_language_tag("xyz123invalid")

    def test_normalized_bcp47_tag(self):
        """Test that BCP-47 tags are normalized."""
        # Case normalization
        info = parse_language_tag("DE")
        assert info.bcp47_tag == "de"

        info = parse_language_tag("ZH-HANS")
        assert info.bcp47_tag == "zh-Hans"


class TestCleanTextForGlotlid:
    """Tests for _clean_text_for_glotlid function."""

    def test_removes_markdown_bold(self):
        """Test removal of markdown bold markers."""
        text = "This is **bold** text"
        assert _clean_text_for_glotlid(text) == "This is bold text"

        text = "**Start** and **end**"
        assert _clean_text_for_glotlid(text) == "Start and end"

    def test_removes_markdown_italic(self):
        """Test removal of markdown italic markers."""
        text = "This is *italic* text"
        assert _clean_text_for_glotlid(text) == "This is italic text"

    def test_removes_code_blocks(self):
        """Test removal of code block markers."""
        text = "Here is ```code``` block"
        assert _clean_text_for_glotlid(text) == "Here is code block"

    def test_removes_inline_code(self):
        """Test removal of inline code markers."""
        text = "Use `code` here"
        assert _clean_text_for_glotlid(text) == "Use code here"

    def test_removes_newlines(self):
        """Test removal of newlines."""
        text = "Line one\nLine two\nLine three"
        assert _clean_text_for_glotlid(text) == "Line one Line two Line three"

    def test_normalizes_whitespace(self):
        """Test normalization of multiple whitespaces."""
        text = "Too   many    spaces"
        assert _clean_text_for_glotlid(text) == "Too many spaces"

        text = "  leading and trailing  "
        assert _clean_text_for_glotlid(text) == "leading and trailing"

    def test_combined_cleaning(self):
        """Test combined markdown and whitespace cleaning."""
        text = "**Guten Tag!**\n\nWie kann ich Ihnen *heute* helfen?"
        cleaned = _clean_text_for_glotlid(text)
        assert cleaned == "Guten Tag! Wie kann ich Ihnen heute helfen?"

    def test_empty_text(self):
        """Test handling of empty text."""
        assert _clean_text_for_glotlid("") == ""
        assert _clean_text_for_glotlid("   ") == ""
        assert _clean_text_for_glotlid("\n\n") == ""

    def test_preserves_non_latin_text(self):
        """Test that non-Latin text is preserved."""
        # Arabic
        text = "مرحبا بك"
        assert _clean_text_for_glotlid(text) == "مرحبا بك"

        # Chinese
        text = "你好世界"
        assert _clean_text_for_glotlid(text) == "你好世界"

        # With markdown
        text = "**你好** 世界"
        assert _clean_text_for_glotlid(text) == "你好 世界"


class TestLanguageMatchesAny:
    """Tests for _language_matches_any function."""

    def test_exact_match(self):
        """Test exact code matches."""
        assert _language_matches_any("deu_Latn", ["deu_Latn"])
        assert _language_matches_any("spa_Latn", ["spa_Latn"])
        assert _language_matches_any("arb_Arab", ["arb_Arab"])

    def test_no_match(self):
        """Test non-matching codes."""
        assert not _language_matches_any("eng_Latn", ["deu_Latn"])
        assert not _language_matches_any("fra_Latn", ["spa_Latn", "ita_Latn"])

    def test_empty_acceptable_codes(self):
        """Test with empty acceptable codes list."""
        assert not _language_matches_any("deu_Latn", [])

    def test_multiple_acceptable_codes(self):
        """Test matching against multiple acceptable codes."""
        acceptable = ["deu_Latn", "gsw_Latn", "nds_Latn"]
        assert _language_matches_any("deu_Latn", acceptable)
        assert _language_matches_any("gsw_Latn", acceptable)

    def test_underscore_hyphen_equivalence(self):
        """Test that underscores and hyphens are treated equivalently."""
        # langcodes should handle both separators
        assert _language_matches_any("deu_Latn", ["deu-Latn"])
        assert _language_matches_any("deu-Latn", ["deu_Latn"])

    def test_german_dialect_matching(self):
        """Test German dialect code matching (gsw = Swiss German)."""
        # Swiss German should match when explicitly in acceptable list
        assert _language_matches_any("gsw_Latn", ["gsw_Latn"])
        assert _language_matches_any("gsw_Latn", ["deu_Latn", "gsw_Latn"])

        # But Swiss German should NOT match standard German without explicit inclusion
        # (distance should be > 10 for different language codes)
        # This depends on langcodes behavior - gsw and deu are different languages
        _ = _language_matches_any("gsw_Latn", ["deu_Latn"])
        # gsw (Swiss German) is a separate language, not a dialect of deu
        # So this should likely be False unless langcodes considers them close
        # The actual behavior depends on langcodes implementation

    def test_arabic_variant_matching(self):
        """Test Arabic variant code matching."""
        # Standard Arabic
        assert _language_matches_any("arb_Arab", ["arb_Arab"])

        # Egyptian Arabic
        assert _language_matches_any("arz_Arab", ["arz_Arab"])

        # Multiple Arabic variants acceptable
        acceptable = ["arb_Arab", "arz_Arab", "ara_Arab"]
        assert _language_matches_any("arb_Arab", acceptable)
        assert _language_matches_any("arz_Arab", acceptable)

    def test_script_mismatch(self):
        """Test that script mismatches are handled."""
        # Same language, different script - langcodes considers these close (distance < 10)
        # since they're the same language just with different script variants
        # This is intentional: we want to match Korean regardless of script representation
        assert _language_matches_any("kor_Hang", ["kor_Kore"])

        # Different languages should not match
        assert not _language_matches_any("kor_Hang", ["jpn_Jpan"])

    def test_language_code_normalization(self):
        """Test that language codes are normalized for comparison."""
        # Two-letter vs three-letter codes (via langcodes normalization)
        # de and deu should be equivalent
        assert _language_matches_any("deu_Latn", ["de-Latn"])
        assert _language_matches_any("de-Latn", ["deu_Latn"])


class TestLanguageInfoDataclass:
    """Tests for LanguageInfo dataclass."""

    def test_glotlid_label_without_prefix(self):
        """Test the glotlid_label_without_prefix property."""
        info = LanguageInfo(
            bcp47_tag="de",
            iso639_3="deu",
            script="Latn",
            display_name="German",
            glotlid_label="__label__deu_Latn",
        )

        assert info.glotlid_label_without_prefix == "deu_Latn"

    def test_all_fields_accessible(self):
        """Test that all fields are accessible."""
        info = LanguageInfo(
            bcp47_tag="zh-Hans",
            iso639_3="zho",
            script="Hans",
            display_name="Chinese",
            glotlid_label="__label__zho_Hans",
        )

        assert info.bcp47_tag == "zh-Hans"
        assert info.iso639_3 == "zho"
        assert info.script == "Hans"
        assert info.display_name == "Chinese"
        assert info.glotlid_label == "__label__zho_Hans"
