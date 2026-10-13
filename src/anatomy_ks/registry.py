"""Bellek içi yapı kayıt defteri: çözümleme, arama, ID genişletme (Bölüm 8)."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable
from pathlib import Path

from rapidfuzz import fuzz, process

from anatomy_ks.models import Structure

_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")
_ID_RE = re.compile(r"^([A-Za-z]+)[:_](\d+)$")


class StructureNotFound(KeyError):
    pass


def normalize_id(raw: str) -> str:
    """`FMA_<n>` / `FMA%3A<n>` / `fma:<n>` → `FMA:<n>`."""
    raw = raw.strip().replace("%3A", ":").replace("%3a", ":")
    m = _ID_RE.match(raw)
    if not m:
        return raw
    return f"{m.group(1).upper()}:{m.group(2)}"


def normalize_name(text: str) -> str:
    """casefold + aksan/noktalama temizliği (Türkçe ı/İ dahil)."""
    t = text.replace("İ", "i").replace("I", "ı").casefold()
    t = t.replace("ı", "i")
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = _PUNCT_RE.sub(" ", t)
    return _WS_RE.sub(" ", t).strip()


class Registry:
    def __init__(self, structures: Iterable[Structure], group_min_descendants: int = 100) -> None:
        self._by_id: dict[str, Structure] = {s.id: s for s in structures}
        self._names: dict[str, list[str]] = {}  # normalize ad -> [id]
        self._name_keys: list[str] = []
        for s in self._by_id.values():
            for n in self.names_of(s):
                key = normalize_name(n)
                if key:
                    self._names.setdefault(key, [])
                    if s.id not in self._names[key]:
                        self._names[key].append(s.id)
        self._name_keys = list(self._names)
        # alt yapı sayısı (bölge/sistem tespiti için)
        self._desc_count: dict[str, int] = {}
        for s in self._by_id.values():
            for a in s.ancestors:
                if a != s.id:
                    self._desc_count[a] = self._desc_count.get(a, 0) + 1
        self.group_min_descendants = group_min_descendants

    # ---------- yükleme ----------
    @classmethod
    def load(cls, path: Path, group_min_descendants: int = 100) -> Registry:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls((Structure.model_validate(x) for x in data), group_min_descendants)

    @staticmethod
    def save(structures: Iterable[Structure], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [s.model_dump() for s in structures]
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---------- erişim ----------
    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, sid: object) -> bool:
        return isinstance(sid, str) and normalize_id(sid) in self._by_id

    def all(self) -> list[Structure]:
        return list(self._by_id.values())

    def resolve(self, structure_id: str) -> Structure:
        sid = normalize_id(structure_id)
        s = self._by_id.get(sid)
        if s is None:
            raise StructureNotFound(sid)
        return s

    def get(self, structure_id: str) -> Structure | None:
        return self._by_id.get(normalize_id(structure_id))

    @staticmethod
    def names_of(s: Structure) -> list[str]:
        return [s.name_en, s.name_la, s.name_tr, *s.synonyms]

    def display_name(self, s: Structure) -> str:
        if s.name_tr and s.name_tr.casefold() != s.name_la.casefold():
            return f"{s.name_tr} ({s.name_la})"
        return s.name_la

    # ---------- ontoloji ----------
    def descendant_count(self, structure_id: str) -> int:
        return self._desc_count.get(normalize_id(structure_id), 0)

    def is_group(self, s: Structure) -> bool:
        """Bölge/sistem düğümü mü? (çok sayıda alt yapısı olan düğüm)"""
        return self.descendant_count(s.id) >= self.group_min_descendants

    def search_ids(self, structure_id: str, max_neighbors: int = 40, depth: int = 1) -> list[str]:
        """Retrieval filtresine girecek ID listesi: yapı + genel yapı + komşular."""
        s = self.resolve(structure_id)
        base_id = s.generic_id or s.id
        base = self._by_id.get(base_id, s)
        ids: list[str] = [s.id]
        if base.id != s.id:
            ids.append(base.id)
        frontier = [base]
        for _ in range(max(depth, 0)):
            nxt: list[Structure] = []
            for f in frontier:
                for nb in f.neighbors:
                    if len(ids) >= max_neighbors + 2:
                        break
                    if nb not in ids and nb in self._by_id:
                        ids.append(nb)
                        nxt.append(self._by_id[nb])
            frontier = nxt
        return ids

    def filter_field(self, s: Structure) -> str:
        return "ancestors" if self.is_group(s) else "related_ids"

    # ---------- arama ----------
    def search(
        self,
        q: str = "",
        system: str | None = None,
        region: str | None = None,
        limit: int = 20,
    ) -> list[Structure]:
        def ok(s: Structure) -> bool:
            if s.deprecated:
                return False
            if system and system not in s.system:
                return False
            return not (region and region not in s.region)

        key = normalize_name(q) if q else ""
        if not key:
            res = [s for s in self._by_id.values() if ok(s)]
            res.sort(key=lambda s: (-self.descendant_count(s.id), s.name_en))
            return res[:limit]

        scored: dict[str, float] = {}

        def add(sid: str, score: float) -> None:
            if scored.get(sid, -1.0) < score:
                scored[sid] = score

        # 1) birebir
        for sid in self._names.get(key, []):
            st = self._by_id[sid]
            if key == normalize_name(st.name_en):
                add(sid, 320.0)
            elif key in (normalize_name(st.name_tr), normalize_name(st.name_la)):
                add(sid, 310.0)
            else:
                add(sid, 300.0)
        # 2) önek / kelime öneki
        for name_key in self._name_keys:
            if name_key.startswith(key):
                for sid in self._names[name_key]:
                    add(sid, 200.0 - min(len(name_key) - len(key), 99) * 0.5)
            elif f" {key}" in f" {name_key}":
                for sid in self._names[name_key]:
                    add(sid, 150.0 - min(len(name_key) - len(key), 99) * 0.5)
        # 3) fuzzy
        if len(scored) < limit:
            for name_key, score, _ in process.extract(
                key, self._name_keys, scorer=fuzz.WRatio, limit=limit * 3, score_cutoff=80
            ):
                for sid in self._names[name_key]:
                    add(sid, float(score))

        items = [(sc, self._by_id[sid]) for sid, sc in scored.items() if ok(self._by_id[sid])]
        # eşit skorda: taraflı olmayan, daha genel (çok alt yapılı) ve kısa ad önce
        items.sort(
            key=lambda t: (
                -t[0],
                t[1].generic_id is not None,
                -min(self.descendant_count(t[1].id), 50),
                len(t[1].name_en),
            )
        )
        return [s for _, s in items[:limit]]
