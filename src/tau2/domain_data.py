"""Central resolver for per-domain data directories.

At import time, domain `utils.py` modules used to freeze path constants to
`DATA_DIR / "tau2" / "domains" / <domain>`. That happens before the CLI can
know which locale to use, so localized runs had no way to redirect reads.

This module decouples the lookup: call `set_data_lang(locale)` once (after
arg parsing) and then every `get_domain_data_dir(domain)` call returns the
localized path `DATA_DIR / "tau2" / "localized" / <locale> / "domains" /
<domain>` when that directory exists, otherwise falls back to the
non-localized path with a warning.

A non-English language must match an installed locale, though: see
`require_localized_data`.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from tau2.utils.utils import DATA_DIR, is_english_language

_data_lang: str | None = None


class LocalizedDataNotFoundError(RuntimeError):
    """A non-English language or locale has no localized data installed."""


def set_data_lang(lang: str | None) -> None:
    global _data_lang
    _data_lang = lang


def get_data_lang() -> str | None:
    return _data_lang


def get_domain_data_dir(domain: str) -> Path:
    """Return the data directory for `domain`, honoring the active data_lang."""
    base = DATA_DIR / "tau2"
    if _data_lang:
        localized = base / "localized" / _data_lang / "domains" / domain
        if localized.exists():
            return localized
        logger.warning(
            f"No localized data for domain={domain!r} at {localized}; "
            f"falling back to default English data"
        )
    return base / "domains" / domain


def require_localized_data(locale: str | None, language: str | None = None) -> None:
    """Fail fast when a non-English run has no localized data at all.

    Raises `LocalizedDataNotFoundError` when `language` is non-English but no
    locale was resolved for it (the localized dataset is not installed), or
    when a non-English `locale` has no directory under data/tau2/localized/.
    A locale that lacks a particular domain still falls back to the English
    data in `get_domain_data_dir`, by design.
    """
    root = DATA_DIR / "tau2" / "localized"
    hint = (
        "The localized task data is distributed separately as a gated Hugging "
        "Face dataset; download it into data/tau2/ (see README.md)."
    )
    if locale is None:
        if language and not is_english_language(language):
            raise LocalizedDataNotFoundError(
                f"No localized data under {root} matches language {language!r}. {hint}"
            )
    elif not is_english_language(locale) and not (root / locale).is_dir():
        raise LocalizedDataNotFoundError(
            f"No localized data for locale {locale!r} under {root}. {hint}"
        )


def resolve_data_lang(
    language: str | None, explicit_data_lang: str | None
) -> str | None:
    """Resolve the effective data_lang from CLI flags.

    Precedence:
      1. --data-lang passed explicitly (may be a full locale like "ko-KR"
         or a short prefix like "ko").
      2. --language, auto-matched against available localized dirs.

    Returns the locale directory name (e.g. "ko-KR") or None if no
    localized data should be used.
    """
    if explicit_data_lang:
        return match_locale_dir(explicit_data_lang) or explicit_data_lang
    if language:
        return match_locale_dir(language)
    return None


def match_locale_dir(candidate: str) -> str | None:
    """Match `candidate` against locale directories under data/tau2/localized/.

    Uses ``langcodes.closest_match`` for soft BCP-47 equivalence so that e.g.
    ``ko`` / ``kor`` / ``ko-KR`` all resolve to the ``ko-KR`` localized dir,
    and ``zh`` / ``cmn-Hans`` both resolve to ``zh-CN`` while ``zh-Hant`` does
    not. Returns ``None`` when no directory matches within langcodes' default
    distance threshold.
    """
    root = DATA_DIR / "tau2" / "localized"
    if not root.exists():
        return None

    dirs = [p.name for p in root.iterdir() if p.is_dir()]
    if not dirs:
        return None

    import langcodes

    try:
        match, _distance = langcodes.closest_match(candidate, dirs)
    except Exception:
        return None
    if match == "und":
        return None
    return match
