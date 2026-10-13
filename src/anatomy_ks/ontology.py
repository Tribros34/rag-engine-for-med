"""Uberon OBO → FMA tabanlı yapı kayıtları (registry) üretimi (Faz 1).

Kurallar:
- Yalnızca FMA çapraz referansı olan Uberon terimleri alınır (R2: ID'ler yalnızca dosyadan gelir).
- Sistem/bölge kökleri `config/region_roots.yaml` içinde terim ADI olarak verilir.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict, deque
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import networkx as nx
import obonet
import yaml

from anatomy_ks.models import Structure

log = logging.getLogger(__name__)

# Ata zinciri ve sistem/bölge türetimi için izlenen kenarlar (çocuk → ebeveyn).
HIERARCHY_RELS = ("is_a", "part_of")
# Derinlik-1 komşuluk için kullanılan anatomik ilişki kenarları (iki yönlü kullanılır).
# Öncelik sırası: önce işlevsel/anatomik ilişkiler, sonra parça-bütün.
RELATIONAL_RELS = (
    "has_muscle_origin",
    "has_muscle_insertion",
    "innervated_by",
    "innervates",
    "vessel_supplies_blood_to",
    "vessel_drains_blood_from",
    "connected_to",
    "connects",
    "proximally_connected_to",
    "distally_connected_to",
    "attaches_to",
    "adjacent_to",
    "continuous_with",
    "tributary_of",
    "branching_part_of",
    "connecting_branch_of",
    "surrounds",
)
PART_RELS = ("part_of", "has_part")

_SYN_RE = re.compile(
    r'^"(?P<text>(?:[^"\\]|\\.)*)"\s+(?P<scope>EXACT|RELATED|NARROW|BROAD)(?P<rest>.*)$'
)
LATIN_TAG = "OMO:0003011"
_SIDE_RE = re.compile(r"^(left|right)\s+(.+)$")
STORED_NEIGHBORS = 200
LIMB_REGIONS = {"upper_limb", "lower_limb"}


def parse_synonym(raw: str) -> tuple[str, str, bool, bool] | None:
    """OBO synonym satırı → (metin, kapsam, latin_mi, homoloji_mi)."""
    m = _SYN_RE.match(raw.strip())
    if not m:
        return None
    rest = m.group("rest")
    return m.group("text"), m.group("scope"), LATIN_TAG in rest, "HOMOLOGY" in rest


def load_graph(obo_path: Path) -> nx.MultiDiGraph:
    g: nx.MultiDiGraph = obonet.read_obo(str(obo_path))
    return g


def load_roots(path: Path) -> dict[str, dict[str, list[str]]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {"system": data.get("system", {}), "region": data.get("region", {})}


def load_name_overrides(path: Path) -> dict[str, dict[str, str]]:
    """İngilizce ad → {"tr": Türkçe ad, "la": tercih edilen Latince ad}.

    Dosya ID içermez (R2); anahtar Uberon'daki İngilizce terim adıdır.
    Değer düz metinse yalnızca Türkçe ad kabul edilir.
    """
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out: dict[str, dict[str, str]] = {}
    for k, v in data.items():
        if isinstance(v, str):
            out[str(k).casefold()] = {"tr": v}
        elif isinstance(v, dict):
            out[str(k).casefold()] = {kk: str(vv) for kk, vv in v.items() if vv}
    return out


def _fma_of(attrs: dict[str, Any]) -> list[str]:
    return [x.split()[0] for x in attrs.get("xref", []) if x.startswith("FMA:")]


def _hier_ancestors(g: nx.MultiDiGraph, node: str) -> set[str]:
    seen: set[str] = {node}
    q: deque[str] = deque([node])
    while q:
        cur = q.popleft()
        for _, parent, key in g.out_edges(cur, keys=True):
            if key in HIERARCHY_RELS and parent not in seen:
                seen.add(parent)
                q.append(parent)
    return seen


def _ordered_unique(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for i in items:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def build_registry(
    obo_path: Path,
    roots_path: Path,
    tr_names_path: Path | None = None,
) -> tuple[list[Structure], dict[str, Any]]:
    """Registry kayıtlarını ve bir özet raporu döndürür."""
    g = load_graph(obo_path)
    roots_cfg = load_roots(roots_path)
    overrides = load_name_overrides(tr_names_path) if tr_names_path else {}

    by_name: dict[str, str] = {}
    for n, d in g.nodes(data=True):
        name = d.get("name")
        if name and n.startswith("UBERON:"):
            by_name.setdefault(name, n)

    # Uberon → FMA (ilk FMA xref'i; bir FMA birden fazla Uberon'a düşerse ilk gelen kazanır)
    u2f: dict[str, str] = {}
    f2u: dict[str, str] = {}
    for n, d in g.nodes(data=True):
        if not n.startswith("UBERON:") or d.get("is_obsolete"):
            continue
        fmas = _fma_of(d)
        if not fmas:
            continue
        fma = fmas[0]
        if fma in f2u:
            continue
        u2f[n] = fma
        f2u[fma] = n

    # kökleri çöz
    missing_roots: list[str] = []
    root_map: dict[str, list[tuple[str, str]]] = defaultdict(list)  # uberon_id -> [(kind, value)]
    for kind in ("system", "region"):
        for value, names in roots_cfg[kind].items():
            for nm in names:
                uid = by_name.get(nm)
                if uid is None:
                    missing_roots.append(nm)
                    continue
                root_map[uid].append((kind, value))

    # taraf: "left X"/"right X" → generic X
    name_of = {u: str(g.nodes[u].get("name", "")) for u in u2f}
    sided: dict[str, tuple[str, str]] = {}  # uberon -> (side, generic_uberon)
    for u, nm in name_of.items():
        m = _SIDE_RE.match(nm)
        if m:
            gen = by_name.get(m.group(2))
            if gen and gen in u2f:
                sided[u] = (m.group(1), gen)
    has_sides = {gen for _, gen in sided.values()}

    # komşuluk (iki yönlü), FMA'lı düğümlerle sınırlı
    rel_nb: dict[str, list[str]] = defaultdict(list)
    part_nb: dict[str, list[str]] = defaultdict(list)
    for u, v, key in g.edges(keys=True):
        if u not in u2f or v not in u2f or u == v:
            continue
        if key in RELATIONAL_RELS:
            rel_nb[u].append(v)
            rel_nb[v].append(u)
        elif key in PART_RELS:
            part_nb[u].append(v)
            part_nb[v].append(u)

    structures: list[Structure] = []
    for u, fma in u2f.items():
        d = g.nodes[u]
        name_en = str(d.get("name", ""))
        anc_u = _hier_ancestors(g, u)
        ancestors = _ordered_unique([fma] + sorted(u2f[a] for a in anc_u if a in u2f and a != u))
        parents = sorted(
            {u2f[p] for _, p, k in g.out_edges(u, keys=True) if k in HIERARCHY_RELS and p in u2f}
        )

        systems: set[str] = set()
        regions: set[str] = set()
        for a in anc_u:
            for kind, value in root_map.get(a, []):
                (systems if kind == "system" else regions).add(value)

        synonyms: list[str] = []
        latin: list[str] = []
        for raw in d.get("synonym", []):
            parsed = parse_synonym(raw)
            if not parsed:
                continue
            text, scope, is_latin, is_homology = parsed
            if is_homology:
                continue
            if is_latin:
                latin.append(text)
                synonyms.append(text)
            elif scope == "EXACT":
                synonyms.append(text)

        laterality = "none"
        generic_id = None
        if u in sided:
            laterality = sided[u][0]
            generic_id = u2f[sided[u][1]]
        elif (
            u in has_sides
            or any(k == "in_lateral_side_of" for _, _, k in g.out_edges(u, keys=True))
            or regions & LIMB_REGIONS
        ):
            # Uberon çoğu çift yapı için sol/sağ terim içermez; ekstremite yapıları çifttir.
            laterality = "bilateral"

        ov = overrides.get(name_en.casefold(), {})
        name_tr = ov.get("tr", "")
        if name_tr:
            synonyms.append(name_tr)
        name_la = ov.get("la") or (latin[0] if latin else name_en[:1].upper() + name_en[1:])
        if ov.get("la"):
            synonyms.append(ov["la"])

        neighbors = _ordered_unique(
            [u2f[x] for x in rel_nb.get(u, [])] + [u2f[x] for x in part_nb.get(u, [])]
        )
        neighbors = [x for x in neighbors if x != fma][:STORED_NEIGHBORS]

        structures.append(
            Structure(
                id=fma,
                uberon_id=u,
                name_la=name_la,
                name_en=name_en,
                name_tr=name_tr or name_en,
                name_tr_reviewed=False,
                synonyms=_ordered_unique(s for s in synonyms if s.casefold() != name_en.casefold()),
                system=sorted(systems),
                region=sorted(regions),
                laterality=laterality,  # type: ignore[arg-type]
                generic_id=generic_id,
                parent_ids=parents,
                ancestors=ancestors,
                neighbors=neighbors,
                deprecated=False,
            )
        )

    structures.sort(key=lambda s: s.id)
    report = {
        "uberon_terms": sum(1 for n in g.nodes if str(n).startswith("UBERON:")),
        "structures": len(structures),
        "with_tr_name": sum(1 for s in structures if s.name_tr != s.name_en),
        "with_system": sum(1 for s in structures if s.system),
        "with_region": sum(1 for s in structures if s.region),
        "sided": len(sided),
        "missing_roots": missing_roots,
        "data_version": str(g.graph.get("data-version", "")),
    }
    return structures, report
