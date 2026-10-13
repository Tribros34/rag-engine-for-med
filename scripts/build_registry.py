"""Uberon OBO → data/registry/structures.json (Faz 1)."""

from __future__ import annotations

import json
import sys

from anatomy_ks.config import get_settings
from anatomy_ks.ingestion.download import ensure_uberon
from anatomy_ks.ontology import build_registry
from anatomy_ks.registry import Registry


def main() -> int:
    st = get_settings()
    obo = ensure_uberon(st.raw_dir)
    structures, report = build_registry(
        obo, st.config_dir / "region_roots.yaml", st.config_dir / "name_tr.yaml"
    )
    Registry.save(structures, st.registry_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
