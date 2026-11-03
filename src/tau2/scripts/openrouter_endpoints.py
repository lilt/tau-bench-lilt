"""Print the providers currently serving an OpenRouter model.

An operator pastes the output into ``--agent-openrouter-providers``. This is a
**config-writing aid only**: a run never calls it, because a pool resolved live
would differ between two cells of the same eval and could not be reproduced
from the recorded results. The pool that matters is the one written down in the
command line and stored in ``results.json`` under ``info.openrouter_provider_pin``.

    python -m tau2.scripts.openrouter_endpoints anthropic/claude-opus-5
    python -m tau2.scripts.openrouter_endpoints openrouter/openai/gpt-5.6-sol:exacto

No API key is needed. Endpoints come and go, so re-check before a run and
re-record the pool with it.
"""

import argparse
import json
import sys
import urllib.error
import urllib.request

ENDPOINTS_URL = "https://openrouter.ai/api/v1/models/{author}/{slug}/endpoints"


def split_model_slug(model: str) -> tuple[str, str]:
    """``openrouter/anthropic/claude-opus-5:exacto`` -> ``("anthropic", "claude-opus-5")``.

    The ``openrouter/`` prefix and any ``:variant`` suffix are tau2/litellm
    spellings; the metadata API wants the bare ``author/slug``.
    """
    name = model.removeprefix("openrouter/")
    author, _, slug = name.partition("/")
    if not author or not slug:
        raise ValueError(
            f"Expected an 'author/slug' model name, got {model!r} "
            "(e.g. anthropic/claude-opus-5)"
        )
    return author, slug.split(":", 1)[0]


def fetch_endpoints(model: str, timeout: float = 20.0) -> list[dict]:
    author, slug = split_model_slug(model)
    url = ENDPOINTS_URL.format(author=author, slug=slug)
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
        payload = json.load(response)
    return payload.get("data", {}).get("endpoints", []) or []


def provider_tag(endpoint: dict) -> str:
    """The token to put in ``provider.order`` for one endpoint."""
    return endpoint.get("tag") or endpoint.get("provider_name") or "?"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", help="e.g. anthropic/claude-opus-5")
    parser.add_argument(
        "--json", action="store_true", help="Dump the raw endpoint objects."
    )
    args = parser.parse_args(argv)

    try:
        endpoints = fetch_endpoints(args.model)
    except (urllib.error.URLError, ValueError, json.JSONDecodeError) as exc:
        print(f"Could not list endpoints for {args.model!r}: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(endpoints, indent=2))
        return 0

    if not endpoints:
        print(f"No endpoints reported for {args.model!r}.", file=sys.stderr)
        return 1

    tags = []
    for endpoint in endpoints:
        tag = provider_tag(endpoint)
        tags.append(tag)
        context = endpoint.get("context_length")
        print(f"{tag:<24} {endpoint.get('provider_name', '')} (context {context})")
    print()
    print("--agent-openrouter-providers " + ",".join(sorted(set(tags))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
