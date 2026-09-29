"""Live smoke test of the three wrapped APIs (uses the network; NOT part of pytest).

Runs the real MASH queries through the same tool functions the MCP servers use, saves every raw
HTTP response under ``tests/fixtures/recorded/`` (so they can replace hand-written fixtures), and
reports PASS/FAIL per check. Exit code is 1 if any check fails.

Usage (responses go to .cache/live_smoke; add --record to overwrite the test fixtures):
    NCBI_EMAIL=you@example.com uv run python scripts/live_smoke.py [--out DIR]

Optional: NCBI_API_KEY, OPENFDA_API_KEY.
"""

import argparse
import asyncio
import re
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path

import httpx

from mash_agent.mcp_servers import clinicaltrials, openfda, pubmed
from mash_agent.mcp_servers.client import ApiClient
from mash_agent.ratelimit import RateLimiter

ROOT = Path(__file__).resolve().parent.parent
RECORDED_DIR = ROOT / "tests" / "fixtures" / "recorded"
DEFAULT_OUT = ROOT / ".cache" / "live_smoke"  # untracked; use --record to refresh fixtures


class RecordingTransport(httpx.AsyncBaseTransport):
    """Wraps a transport and saves each response body to ``out_dir``."""

    def __init__(
        self, service: str, out_dir: Path, inner: httpx.AsyncBaseTransport | None = None
    ) -> None:
        self._service = service
        self._out_dir = out_dir
        self._inner = inner or httpx.AsyncHTTPTransport()
        self._count = 0
        self.saved: list[Path] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        resp = await self._inner.handle_async_request(request)
        await resp.aread()
        body = resp.content  # already decompressed
        name = re.sub(r"\W+", "_", request.url.path.strip("/")) or "root"
        suffix = "xml" if body.lstrip().startswith(b"<") else "json"
        self._count += 1
        self._out_dir.mkdir(parents=True, exist_ok=True)
        path = (
            self._out_dir / f"{self._service}_{self._count:02d}_{name}_{resp.status_code}.{suffix}"
        )
        path.write_bytes(body)
        self.saved.append(path)
        headers = {
            k: v
            for k, v in resp.headers.items()
            if k.lower() not in {"content-encoding", "content-length", "transfer-encoding"}
        }
        return httpx.Response(resp.status_code, headers=headers, content=body, request=request)


class Report:
    def __init__(self) -> None:
        self.failures = 0

    def check(self, label: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" - {detail}" if detail else ""))
        self.failures += 0 if ok else 1


async def _run(
    name: str,
    report: Report,
    out_dir: Path,
    base_url: str,
    rate: float,
    body: Callable[[ApiClient], Awaitable[None]],
) -> None:
    print(f"{name}:")
    transport = RecordingTransport(name, out_dir)
    client = ApiClient(base_url, RateLimiter(rate), cache=None, transport=transport, max_retries=2)
    try:
        await body(client)
    except Exception as exc:  # smoke test: report any failure, keep going
        report.check("call completed", False, repr(exc))
    finally:
        await client.aclose()
    for p in transport.saved:
        print(f"  saved {p}")


async def main_async(out_dir: Path) -> int:
    report = Report()

    async def pubmed_body(client: ApiClient) -> None:
        res = await pubmed.search_pubmed(client, "resmetirom AND (MASH OR NASH)", 5)
        report.check("total_count > 0", res.total_count > 0, str(res.total_count))
        report.check("articles returned", len(res.articles) > 0, str(len(res.articles)))
        report.check("PMIDs numeric", all(a.pmid.isdigit() for a in res.articles))
        report.check("titles non-empty", all(a.title for a in res.articles))
        with_abs = sum(1 for a in res.articles if a.abstract)
        report.check(
            "at least one abstract parsed", with_abs > 0, f"{with_abs}/{len(res.articles)}"
        )
        report.check("years parsed", any(a.year for a in res.articles))

    async def trials_body(client: ApiClient) -> None:
        res = await clinicaltrials.search_trials(client, "MASH", max_results=20)
        report.check("trials returned", len(res.trials) > 0, str(len(res.trials)))
        report.check(
            "NCT IDs well-formed", all(re.fullmatch(r"NCT\d{8}", t.nct_id) for t in res.trials)
        )
        report.check("all Phase 2/3", all(set(t.phases) & {"PHASE2", "PHASE3"} for t in res.trials))
        report.check("sponsors present", any(t.sponsor for t in res.trials))
        report.check("primary endpoints present", any(t.primary_endpoints for t in res.trials))

        broad = 'MASH OR NASH OR "metabolic dysfunction-associated steatohepatitis"'
        res2 = await clinicaltrials.search_trials(client, broad, max_results=20)
        report.check(
            "OR condition query returns trials", len(res2.trials) > 0, str(len(res2.trials))
        )
        liver = re.compile(r"MASH|NASH|steato|fatty liver|MASLD|NAFLD", re.I)
        report.check("conditions parsed", all(t.conditions for t in res2.trials))
        offtopic = [
            f"{t.nct_id}: {t.title} {t.conditions}"
            for t in res2.trials
            if not liver.search(" ".join([t.title, *t.conditions]))
        ]
        # Informational: the OR query can still return unrelated trials, which the trials
        # specialist must filter by conditions, so this is reported but not a failure.
        print(f"  [INFO] {len(offtopic)}/{len(res2.trials)} not liver-related: {offtopic}")

    async def fda_body(client: ApiClient) -> None:
        for drug in ("Rezdiffra", "resmetirom"):
            res = await openfda.get_drug_labels(client, drug)
            report.check(f"{drug}: label found", len(res.labels) > 0, str(len(res.labels)))
            sections = {s.section for label in res.labels for s in label.sections}
            report.check(
                f"{drug}: warnings section",
                bool(sections & {"warnings_and_cautions", "warnings"}),
                str(sorted(sections)),
            )
            report.check(f"{drug}: adverse_reactions section", "adverse_reactions" in sections)
            report.check(f"{drug}: label_id set", all(label.label_id for label in res.labels))

    await _run("pubmed", report, out_dir, pubmed.BASE_URL, 2.5, pubmed_body)
    await _run("clinicaltrials", report, out_dir, clinicaltrials.BASE_URL, 2.0, trials_body)
    await _run("openfda", report, out_dir, openfda.BASE_URL, 0.5, fda_body)

    print(
        f"\n{'ALL CHECKS PASSED' if report.failures == 0 else f'{report.failures} CHECK(S) FAILED'}"
    )
    return 1 if report.failures else 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="directory for raw responses")
    parser.add_argument(
        "--record", action="store_true", help=f"write into {RECORDED_DIR} (overwrites fixtures)"
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(main_async(RECORDED_DIR if args.record else args.out)))


if __name__ == "__main__":
    main()
