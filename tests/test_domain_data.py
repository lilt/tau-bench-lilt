"""Tests for localized data resolution.

A non-English language without any installed localized data fails before the
run starts instead of silently running the English data. A locale that lacks
one domain still falls back to the English data for it, by design.
"""

from argparse import Namespace

import pytest

import tau2.domain_data
from tau2.cli import resolve_language_args
from tau2.domain_data import (
    LocalizedDataNotFoundError,
    get_domain_data_dir,
    require_localized_data,
)


@pytest.fixture(autouse=True)
def localized_data_dir(tmp_path, monkeypatch):
    """de-DE and en-US are installed, with the airline domain only."""
    for locale in ("de-DE", "en-US"):
        (tmp_path / "tau2" / "localized" / locale / "domains" / "airline").mkdir(
            parents=True
        )
    monkeypatch.setattr(tau2.domain_data, "DATA_DIR", tmp_path)
    monkeypatch.setattr(tau2.domain_data, "_data_lang", None)


def make_args(language=None, data_lang=None, domain="airline"):
    return Namespace(
        language=language,
        data_lang=data_lang,
        domain=domain,
        glotlid_threshold=None,
        glotlid_acceptable_codes=None,
        first_agent_message=None,
    )


class TestRequireLocalizedData:
    def test_no_language_passes(self):
        require_localized_data(None)

    def test_english_language_without_locale_passes(self):
        require_localized_data(None, "en")

    def test_installed_locale_passes(self):
        require_localized_data("de-DE", "de")

    def test_language_without_locale_fails(self):
        with pytest.raises(LocalizedDataNotFoundError, match="Hugging Face"):
            require_localized_data(None, "ko")

    def test_explicit_locale_not_installed_fails(self):
        with pytest.raises(LocalizedDataNotFoundError, match="ko-KR"):
            require_localized_data("ko-KR", "ko")


class TestResolveLanguageArgs:
    def test_localized_language_resolves(self):
        data_lang, language_info = resolve_language_args(make_args(language="de"))
        assert data_lang == "de-DE"
        assert language_info.bcp47_tag == "de"

    def test_language_without_localized_data_exits(self):
        with pytest.raises(SystemExit, match="No localized data"):
            resolve_language_args(make_args(language="ko"))

    def test_explicit_data_lang_not_installed_exits(self):
        with pytest.raises(SystemExit, match="ko-KR"):
            resolve_language_args(make_args(language="de", data_lang="ko-KR"))

    def test_locale_without_domain_falls_back_to_english(self, tmp_path):
        data_lang, _ = resolve_language_args(make_args(language="de", domain="telecom"))
        assert data_lang == "de-DE"
        assert (
            get_domain_data_dir("telecom") == tmp_path / "tau2" / "domains" / "telecom"
        )

    def test_no_language_unchanged(self):
        data_lang, language_info = resolve_language_args(make_args())
        assert data_lang is None
        assert language_info is None
