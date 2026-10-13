"""Kaynak edinimi. Her indirme `data/raw/` altında önbelleğe alınır; ikinci kez indirilmez (R12).

Görsel indirilmez (R4): OpenStax deposundaki `resources/` (figürler) klonlandıktan sonra silinir.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)

UBERON_URL = "http://purl.obolibrary.org/obo/uberon.obo"
OPENSTAX_REPO = "https://github.com/philschatz/anatomy-book"
# Gray's 1918 (20. baskı, W. H. Lewis) — Internet Archive taraması,
# "NOT_IN_COPYRIGHT" (public domain).
GRAYS_TXT_URL = (
    "https://archive.org/download/anatomyofhumanbo00grayrich/anatomyofhumanbo00grayrich_djvu.txt"
)
USER_AGENT = "anatomy-knowledge-service/0.1 (educational; contact: project owner)"
_MIN_INTERVAL_S = 1.0
_last_request = 0.0


def _fetch(url: str, dest: Path) -> Path:
    global _last_request
    if dest.exists() and dest.stat().st_size > 0:
        log.info("cache hit", extra={"extra_data": {"path": str(dest)}})
        return dest
    wait = _MIN_INTERVAL_S - (time.monotonic() - _last_request)
    if wait > 0:
        time.sleep(wait)
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req, timeout=120) as r, tmp.open("wb") as f:  # noqa: S310
        shutil.copyfileobj(r, f)
    _last_request = time.monotonic()
    tmp.rename(dest)
    return dest


def ensure_uberon(raw_dir: Path) -> Path:
    return _fetch(UBERON_URL, raw_dir / "uberon.obo")


def ensure_openstax(raw_dir: Path) -> Path:
    dest = raw_dir / "anatomy-book"
    if not (dest / "SUMMARY.md").exists():
        subprocess.run(["git", "clone", "--depth", "1", OPENSTAX_REPO, str(dest)], check=True)
    # R4: görseller atılır
    for junk in ("resources", ".git", "cover.png"):
        p = dest / junk
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()
    lic = (dest / "LICENSE.txt").read_text(encoding="utf-8")
    if "Creative Commons Attribution" not in lic or "NonCommercial" in lic:
        raise RuntimeError("OpenStax LICENSE.txt CC BY değil — dur ve raporla (Bölüm 3).")
    return dest


def ensure_grays(raw_dir: Path) -> Path:
    return _fetch(GRAYS_TXT_URL, raw_dir / "grays" / "grays_1918_djvu.txt")
