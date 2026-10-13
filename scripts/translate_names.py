"""Registry'deki Türkçe adı olmayan yapılar için LLM ile toplu çeviri (Faz 1).

Çıktı config/name_tr.yaml'a EKLENİR (mevcut girdilere dokunulmaz); ID içermez (R2).
Sonra `python scripts/build_registry.py` ile registry yeniden üretilir. Tüm çeviriler
name_tr_reviewed: false olarak kalır (uzman onayı gerekir).

    python scripts/translate_names.py --limit 500 [--system muscular]
"""

from __future__ import annotations

import argparse
import json
import re
import sys

import yaml

from anatomy_ks.config import get_settings
from anatomy_ks.llm.base import LLMUnavailable, get_llm
from anatomy_ks.registry import Registry

SYSTEM = (
    "Sen bir tıbbi terminoloji çevirmenisin. Verilen İngilizce anatomik terimleri, Türk tıp "
    "eğitiminde kullanılan Türkçe karşılıklarına çevir. Emin olmadığın terimde Latince/İngilizce "
    'terimi Türkçe okunuşla bırak. YALNIZCA JSON döndür: {"<ingilizce>": "<türkçe>", ...}'
)
BATCH = 40


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--system")
    args = ap.parse_args()
    st = get_settings()
    if not st.llm_configured:
        print("LLM yapılandırılmadı.")
        return 2
    llm = get_llm(st)
    reg = Registry.load(st.registry_path)
    path = st.config_dir / "name_tr.yaml"
    existing = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    have = {str(k).casefold() for k in existing}
    todo = [
        s.name_en
        for s in reg.all()
        if s.name_tr == s.name_en
        and s.name_en.casefold() not in have
        and (not args.system or args.system in s.system)
    ]
    todo.sort(key=lambda n: -reg.descendant_count(next(x.id for x in reg.all() if x.name_en == n)))
    todo = todo[: args.limit]
    added: dict[str, str] = {}
    for i in range(0, len(todo), BATCH):
        batch = todo[i : i + BATCH]
        for _ in range(2):
            try:
                raw = llm.complete(SYSTEM, json.dumps(batch, ensure_ascii=False), 2000)
                m = re.search(r"\{.*\}", raw, re.S)
                data = json.loads(m.group(0)) if m else {}
                added.update({k: str(v) for k, v in data.items() if k in batch and v})
                break
            except (ValueError, json.JSONDecodeError):
                continue
            except LLMUnavailable as e:
                print(f"LLM hatası: {e}")
                return 3
        print(f"{min(i + BATCH, len(todo))}/{len(todo)}")
    with path.open("a", encoding="utf-8") as f:
        f.write("\n# ---------- LLM toplu çeviri (uzman onayı bekliyor) ----------\n")
        for k, v in added.items():
            key, val = json.dumps(k, ensure_ascii=False), json.dumps(v, ensure_ascii=False)
            f.write(f"{key}: {{tr: {val}}}\n")
    print(f"eklendi: {len(added)} → {path}. Şimdi: python scripts/build_registry.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
