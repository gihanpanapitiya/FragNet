"""
Contribution-Guided Fragment Optimizer.

Workflow:
  1. Analyze seed molecule with FragNet → fragment contributions
  2. Identify the worst-contributing fragment(s)
  3. Enumerate swaps: replace each with BRICS-compatible library alternatives
  4. Score all candidate molecules with FragNet (single batch)
  5. Rank by improvement and return top-k
"""

import os
import logging
from pathlib import Path

import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import BRICS, rdDepictor
from rdkit.Chem.BRICS import BRICSDecompose, BRICSBuild, BreakBRICSBonds
from rdkit.Chem.Draw import rdMolDraw2D
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

from fragnet.dataset.data import collate_fn as dataset_collate_fn
from fragnet.dataset.dataset import FinetuneData
from fragnet.dataset.utils import extract_data


# ---------------------------------------------------------------------------
# Reference molecules used to seed the fragment library
# ---------------------------------------------------------------------------
_REFERENCE_SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",              # aspirin
    "CN1CCC[C@H]1c2cccnc2",               # nicotine
    "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",   # ibuprofen
    "CC(=O)Nc1ccc(O)cc1",                 # paracetamol
    "c1ccncc1",                           # pyridine
    "C1CCNCC1",                           # piperidine
    "C1COCCN1",                           # morpholine
    "c1ccc2ccccc2c1",                     # naphthalene
    "Cc1ccccc1",                          # toluene
    "c1ccc(cc1)O",                        # phenol
    "c1ccc(cc1)N",                        # aniline
    "c1ccc(cc1)F",                        # fluorobenzene
    "c1ccc(cc1)Cl",                       # chlorobenzene
    "c1ccco1",                            # furan
    "c1ccsc1",                            # thiophene
    "C1CCCC1",                            # cyclopentane
    "C1CCCCC1",                           # cyclohexane
    "CC(=O)c1ccccc1",                     # acetophenone
    "N#Cc1ccccc1",                        # benzonitrile
    "CC(=O)N",                            # acetamide
    "CC(=O)O",                            # acetic acid
    "c1cnc2ccccc2n1",                     # benzimidazole
    "Cc1ccc(cc1)S(=O)(=O)N",             # sulfonamide core
    "O=C1CCCC(=O)N1",                     # glutarimide
    "c1ccc(cc1)C(=O)O",                   # benzoic acid
    "CCN(CC)CC",                          # triethylamine
    "CC(C)(C)c1ccccc1",                   # tert-butylbenzene
    "CC(C)O",                             # isopropanol
    "OCC",                                # ethanol
    "NCC",                                # ethylamine
    "c1ccc(cc1)Br",                       # bromobenzene
    "c1ccc2c(c1)ccc(c2)O",               # naphthol
    "CN(C)C",                             # trimethylamine
    "C1CC1",                              # cyclopropane
    "CC(C)N",                             # isopropylamine
    "c1cnccn1",                           # pyrimidine
    "c1ccnc2ccccc12",                     # quinoline
    "c1ccc2[nH]cccc2c1",                  # indole
    "O=C(O)c1ccco1",                      # furan-2-carboxylic acid
    "CC(=O)Nc1cccs1",                     # thiophene acetamide
]


# ---------------------------------------------------------------------------
# Fragment library: built once per source, keyed by attachment-type tuple
# ---------------------------------------------------------------------------
# Selectable library sources:
#   "auto"      — use the pre-built ChEMBL+FDA pickle if present, else reference
#   "chembl"    — force the pre-built pickle (error surfaces if unavailable)
#   "reference" — force the small built-in BRICS set (no pickle)
LIBRARY_SOURCES = ("auto", "chembl", "reference")
_DEFAULT_SOURCE = "auto"

# One built dict cached per resolved source.
_LIBRARY_CACHE: dict[str, dict] = {}

# FragmentLibrary objects cached per source (only populated when the pkl is used).
_FRAG_LIB_OBJ_CACHE: dict[str, object] = {}

# Pre-built ChEMBL + FDA fragment library (see build_chembl_library.py).
# Override the location with the FRAGNET_LIBRARY_PKL env var.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LIBRARY_PKL_NAME = "chembl_library.pkl"


def _find_library_pkl() -> Path | None:
    """Locate the pre-built fragment library pickle, if present."""
    env_path = os.environ.get("FRAGNET_LIBRARY_PKL")
    candidates = [Path(env_path)] if env_path else []
    candidates += [Path.cwd() / _LIBRARY_PKL_NAME, _PROJECT_ROOT / _LIBRARY_PKL_NAME]
    for path in candidates:
        if path.is_file():
            return path
    return None


def _load_pkl_library(path: Path, source: str) -> dict:
    """Load a FragmentLibrary pickle, cache the object, return the flat dict."""
    from fragnet.vizualize.fragment_library import FragmentLibrary

    lib = FragmentLibrary.load(str(path))
    _FRAG_LIB_OBJ_CACHE[source] = lib          # cache for contribution-prior access

    library: dict = {}
    for entry in lib.entries:
        key = tuple(entry["attachment_types"])
        smi = entry["smiles"]
        bucket = library.setdefault(key, [])
        if smi not in bucket:
            bucket.append(smi)
    return library


def _get_fragment_library_obj(source: str = _DEFAULT_SOURCE):
    """Return the cached FragmentLibrary object, or None for the reference set."""
    if source not in _FRAG_LIB_OBJ_CACHE:
        _get_library(source)   # triggers load and caching
    return _FRAG_LIB_OBJ_CACHE.get(source)


def _contribution_score(entry: dict, prop_type: str, direction: str,
                        min_n: int = 2, eps: float = 0.1) -> float:
    """Signal-to-noise score for a library fragment's mean contribution.

    score = mean / (std + eps), sign-flipped for 'minimize'.
    Returns 0.0 (neutral) when stats are absent or n < min_n, so unknown
    fragments sort after known-good ones rather than being excluded.
    """
    stat = entry["contributions"].get(prop_type)
    if stat is None or stat["n"] < min_n:
        return 0.0
    score = stat["mean"] / (stat["std"] + eps)
    return score if direction == "maximize" else -score


def _build_library() -> dict:
    """Decompose reference molecules and collect BRICS fragments by attachment type."""
    library: dict = {}
    for smi in _REFERENCE_SMILES:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            frag_smiles_set = BRICSDecompose(mol)
        except Exception:
            continue
        for fsmi in frag_smiles_set:
            fmol = Chem.MolFromSmiles(fsmi)
            if fmol is None:
                continue
            key = _attachment_types(fmol)
            if key not in library:
                library[key] = []
            if fsmi not in library[key]:
                library[key].append(fsmi)
    return library


def _get_library(source: str = _DEFAULT_SOURCE) -> dict:
    """Return the fragment library for the requested source (built once, cached).

    source:
        "auto"      — pre-built ChEMBL+FDA pickle if present, else reference set
        "chembl"    — force the pre-built pickle; raises if it cannot be loaded
        "reference" — force the small built-in BRICS reference set
    """
    if source not in LIBRARY_SOURCES:
        raise ValueError(
            f"Unknown library source {source!r}; expected one of {LIBRARY_SOURCES}"
        )
    if source in _LIBRARY_CACHE:
        return _LIBRARY_CACHE[source]

    if source == "reference":
        library = _build_library()
        logger.info("Built fragment library from reference set (%d types)", len(library))
        _LIBRARY_CACHE[source] = library
        return library

    pkl_path = _find_library_pkl()
    if pkl_path is None:
        if source == "chembl":
            raise FileNotFoundError(
                f"No fragment library pickle found (looked for {_LIBRARY_PKL_NAME} "
                "via $FRAGNET_LIBRARY_PKL, CWD, and project root)"
            )
        # auto → fall back to reference set
        library = _build_library()
        logger.info("No pickle found; built fragment library from reference set "
                    "(%d types)", len(library))
        _LIBRARY_CACHE[source] = library
        return library

    try:
        library = _load_pkl_library(pkl_path, source)
    except Exception as exc:
        if source == "chembl":
            raise
        logger.warning("Failed to load %s (%s); falling back to reference set",
                       pkl_path, exc)
        library = _build_library()
        _LIBRARY_CACHE[source] = library
        return library

    n_frags = sum(len(v) for v in library.values())
    logger.info("Loaded fragment library from %s (%d fragments, %d types)",
                pkl_path, n_frags, len(library))
    _LIBRARY_CACHE[source] = library
    return library


def _attachment_types(frag_mol) -> tuple:
    """Return sorted tuple of BRICS dummy-atom isotope numbers (attachment types)."""
    return tuple(sorted(
        a.GetIsotope()
        for a in frag_mol.GetAtoms()
        if a.GetAtomicNum() == 0
    ))


# ---------------------------------------------------------------------------
# Core / scaffold locking
# ---------------------------------------------------------------------------

def get_fragment_atom_map(smiles: str) -> list[list[int]]:
    """
    Return a list of heavy-atom index lists, one per BRICS fragment.
    Ordering matches FragNetVizApp fragment ordering (sorted by lowest atom index).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    try:
        broken = BreakBRICSBonds(mol)
        frags = Chem.GetMolFrags(broken)          # tuples of original atom indices
        # strip dummy-atom placeholders: keep only atoms that exist in the original mol
        n_atoms = mol.GetNumAtoms()
        return [sorted(a for a in frag if a < n_atoms) for frag in frags]
    except Exception:
        return []


def get_core_protected_indices(smiles: str, core_smiles: str) -> set[int]:
    """
    Return the set of fragment indices whose atoms overlap with the core substructure.
    These fragments will be excluded from swapping.

    core_smiles can be a SMILES string or a SMARTS pattern.
    Returns an empty set if the core is not found in the molecule.
    """
    if not core_smiles or not core_smiles.strip():
        return set()

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return set()

    # Try SMARTS first (more flexible), fall back to SMILES
    core_query = Chem.MolFromSmarts(core_smiles)
    if core_query is None:
        core_mol = Chem.MolFromSmiles(core_smiles)
        if core_mol is None:
            return set()
        core_query = core_mol

    matches = mol.GetSubstructMatches(core_query)
    if not matches:
        return set()

    # Union of all match atom sets (handles symmetry / multiple matches)
    core_atoms: set[int] = set()
    for match in matches:
        core_atoms.update(match)

    # Map core atoms → fragment indices
    frag_atom_map = get_fragment_atom_map(smiles)
    protected: set[int] = set()
    for frag_idx, frag_atoms in enumerate(frag_atom_map):
        if set(frag_atoms) & core_atoms:
            protected.add(frag_idx)

    return protected


# ---------------------------------------------------------------------------
# Scaffold-preserving swap enumeration
# ---------------------------------------------------------------------------

def build_scaffold_with_dummies(mol, frag_atom_indices: list):
    """
    Build a scaffold mol = the entire molecule minus the target fragment, with
    BRICS-labeled dummy atoms at the cut points.

    Strategy: break ONLY the BRICS bonds that cross the fragment boundary.
    Passing (scaffold_mol, replacement_mol) to BRICSBuild with onlyCompleteMols=True
    then guarantees BOTH are used, so the scaffold is always preserved.

    Returns (scaffold_mol, n_cut_bonds) or (None, 0) on failure.
    """
    import copy
    from rdkit.Chem.BRICS import FindBRICSBonds

    frag_set = set(frag_atom_indices)

    # Find BRICS bonds that cross the fragment boundary
    crossing = []
    for (a1, a2), (t1, t2) in FindBRICSBonds(mol):
        if (a1 in frag_set) != (a2 in frag_set):
            bond_type = mol.GetBondBetweenAtoms(a1, a2).GetBondType()
            if a1 in frag_set:
                crossing.append((a1, a2, t1, t2, bond_type))
            else:
                crossing.append((a2, a1, t2, t1, bond_type))

    if not crossing:
        return None, 0

    rwmol = Chem.RWMol(copy.deepcopy(mol))

    # Remove crossing bonds before removing atoms
    for frag_atom, scaffold_atom, *_ in crossing:
        rwmol.RemoveBond(frag_atom, scaffold_atom)

    # Remove fragment atoms in reverse order to keep lower atom indices stable
    for idx in sorted(frag_set, reverse=True):
        rwmol.RemoveAtom(idx)

    # Compute new indices for scaffold atoms after fragment removal
    def shifted(old, removed_set):
        return old - sum(1 for r in removed_set if r < old)

    # Add BRICS-labeled dummy atoms at each former attachment point.
    # The dummy gets the env type that was on the SCAFFOLD side of the cut
    # (= scaffold_env), so BRICSBuild can match it against library fragments
    # that carry the FRAGMENT side type (= frag_env) — BRICS pairs (t1, t2)
    # are recorded symmetrically, so either ordering works for BRICSBuild.
    for frag_atom, scaffold_atom, frag_env, scaffold_env, bond_type in crossing:
        new_sc = shifted(scaffold_atom, frag_set)
        d = rwmol.AddAtom(Chem.Atom(0))
        rwmol.GetAtomWithIdx(d).SetIsotope(int(scaffold_env))
        rwmol.AddBond(new_sc, d, bond_type)

    try:
        Chem.SanitizeMol(rwmol)
        return rwmol.GetMol(), len(crossing)
    except Exception:
        return None, 0


def enumerate_fragment_swaps(smiles: str, worst_frag_indices: list, max_per_frag: int = 40,
                             library_source: str = _DEFAULT_SOURCE,
                             use_contribution_prior: bool = False,
                             prop_type: str = "Solubility",
                             direction: str = "maximize") -> list:
    """
    Replace target fragments while guaranteeing the rest of the molecule is preserved.

    Approach per target fragment:
      1. Build a scaffold mol = the molecule minus that fragment, with BRICS
         dummy atoms at the cut points (via build_scaffold_with_dummies).
      2. Call BRICSBuild([scaffold, lib_frag], maxDepth=1, onlyCompleteMols=True).
         With only 2 fragments both must be consumed, so the scaffold is always kept.

    Args:
        smiles:                  Seed SMILES string
        worst_frag_indices:      Fragment indices to swap (matching FragNet ordering)
        max_per_frag:            Max library fragments to try per target fragment
        library_source:          Which fragment library to draw from
                                 ("auto" | "chembl" | "reference"; see _get_library)
        use_contribution_prior:  If True, sort each attachment-type bucket by
                                 mean/std contribution score before iterating, so
                                 the max_per_frag budget is spent on the most
                                 promising candidates first. No-op when the library
                                 has no contribution stats (degrades to flat order).
        prop_type:               Property name for contribution lookup (e.g. "Solubility")
        direction:               "maximize" or "minimize" — determines score sign

    Returns:
        List of unique valid SMILES strings that each preserve the scaffold.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    frag_atom_map = get_fragment_atom_map(smiles)
    if not frag_atom_map or len(frag_atom_map) <= 1:
        return []

    library = _get_library(library_source)
    seed_canonical = Chem.MolToSmiles(mol)
    candidates: set = set()

    # When using the contribution prior, build a smiles→entry lookup from the
    # FragmentLibrary object so we can score each candidate fragment.
    contrib_map: dict[str, dict] = {}
    if use_contribution_prior:
        lib_obj = _get_fragment_library_obj(library_source)
        if lib_obj is not None:
            contrib_map = {e["smiles"]: e for e in lib_obj.entries}

    def _sorted_bucket(smis: list[str]) -> list[str]:
        """Sort a bucket by contribution score (best first), unknown last."""
        if not contrib_map:
            return smis
        return sorted(
            smis,
            key=lambda s: _contribution_score(
                contrib_map.get(s, {"contributions": {}}),
                prop_type, direction,
            ),
            reverse=True,
        )

    for worst_idx in worst_frag_indices:
        if worst_idx >= len(frag_atom_map):
            continue

        frag_atoms = frag_atom_map[worst_idx]
        if not frag_atoms:
            continue

        scaffold_mol, n_cuts = build_scaffold_with_dummies(mol, frag_atoms)
        if scaffold_mol is None:
            continue

        scaffold_dummy_types = tuple(sorted(
            a.GetIsotope() for a in scaffold_mol.GetAtoms() if a.GetAtomicNum() == 0
        ))

        # Build a per-fragment-position candidate list, sorted by contribution
        # prior within each attachment-type bucket, then flattened. This ensures
        # the max_per_frag budget is spent on the most confident candidates first.
        ordered_smis: list[str] = []
        for att_key, smis in library.items():
            if n_cuts == 1 and len(att_key) != 1:
                continue
            if n_cuts > 1 and not (set(scaffold_dummy_types) & set(att_key)):
                continue
            ordered_smis.extend(_sorted_bucket(smis))

        candidates_tried = 0
        for lib_smi in ordered_smis:
            if candidates_tried >= max_per_frag:
                break
            lib_mol = Chem.MolFromSmiles(lib_smi)
            if lib_mol is None:
                continue
            candidates_tried += 1
            try:
                # Only 2 fragments → BRICSBuild must use both → scaffold preserved
                new_mols = list(BRICSBuild([scaffold_mol, lib_mol], maxDepth=1))
                for new_mol in new_mols[:6]:
                    try:
                        Chem.SanitizeMol(new_mol)
                        new_smi = Chem.MolToSmiles(Chem.RemoveHs(new_mol))
                        if new_smi and new_smi != seed_canonical:
                            candidates.add(new_smi)
                    except Exception:
                        pass
            except Exception:
                pass

    return list(candidates)


# ---------------------------------------------------------------------------
# Batch scoring
# ---------------------------------------------------------------------------

def batch_score(viz_app, smiles_list: list) -> dict:
    """
    Score multiple SMILES with the already-loaded FragNetVizApp model.

    Uses a single batched DataLoader call for efficiency.

    Returns:
        dict mapping SMILES → predicted value (float), skipping failures
    """
    valid: list = []
    for smi in smiles_list:
        try:
            if Chem.MolFromSmiles(smi) is not None:
                valid.append(smi)
        except Exception:
            pass

    if not valid:
        return {}

    df = pd.DataFrame({"smiles": valid, "log_sol": [0.0] * len(valid)})
    ds = viz_app.dataset.get_ft_dataset(df)
    ds = extract_data(ds)

    loader = DataLoader(ds, collate_fn=dataset_collate_fn,
                        batch_size=len(ds), shuffle=False, drop_last=False)
    batch = next(iter(loader))

    with torch.no_grad():
        viz_app.model.eval()
        result = viz_app.model(batch)
        preds = result[0] if isinstance(result, tuple) else result

    preds = preds.numpy().ravel()
    return {smi: float(preds[i]) for i, smi in enumerate(valid)}


# ---------------------------------------------------------------------------
# Full optimization pipeline
# ---------------------------------------------------------------------------

def optimize_molecule(
    smiles: str,
    viz_app,
    prop_type: str = "Solubility",
    direction: str = "maximize",
    n_worst: int = 1,
    max_candidates: int = 60,
    top_k: int = 10,
    frag_contributions: list | None = None,
    seed_prediction: float | None = None,
    protected_fragment_indices: set | None = None,
    library_source: str = _DEFAULT_SOURCE,
    use_contribution_prior: bool = False,
) -> dict:
    """
    Contribution-guided fragment optimization with optional core locking.

    Args:
        smiles:                    Seed molecule SMILES
        viz_app:                   Loaded FragNetVizApp instance
        prop_type:                 'Solubility' or 'Lipophilicity'
        direction:                 'maximize' or 'minimize'
        n_worst:                   Number of worst-contributing fragments to swap
        max_candidates:            Cap on candidates scored by FragNet
        top_k:                     Top results to return
        frag_contributions:        Pre-computed from get_attr_image
        seed_prediction:           Pre-computed seed prediction value
        protected_fragment_indices: Fragment indices that must not be swapped
                                    (e.g. core/scaffold atoms). Pass the result
                                    of get_core_protected_indices() here.
        library_source:            Fragment library to draw replacements from:
                                    "auto" (ChEMBL pickle if present, else
                                    reference), "chembl" (force pickle), or
                                    "reference" (built-in 47-fragment set).
        use_contribution_prior:    If True, sort replacement candidates by mean/std
                                    contribution score before spending the
                                    max_candidates budget. Requires the library pkl
                                    to have been built with --contributions. No-op
                                    when stats are absent (degrades gracefully).

    Returns:
        dict with keys: seed_smiles, seed_prediction, worst_fragment_indices,
        protected_fragment_indices, frag_contributions, candidates,
        n_candidates_evaluated, n_eligible_fragments
    """
    protected = set(protected_fragment_indices or [])

    # Step 1: Fragment contributions (reuse if provided)
    if seed_prediction is None:
        score_map = batch_score(viz_app, [smiles])
        seed_prediction = score_map.get(smiles, 0.0)

    if frag_contributions is None:
        n_frags = max(1, len(get_fragment_atom_map(smiles)))
        frag_contributions = [
            {"fragment_index": i, "atoms": [], "contribution": 0.0}
            for i in range(n_frags)
        ]

    # Step 2: Rank fragments, exclude protected, pick worst eligible
    df_c = pd.DataFrame(frag_contributions)
    ascending = (direction == "maximize")   # ascending → most-negative first
    df_sorted = df_c.sort_values("contribution", ascending=ascending)

    eligible = df_sorted[~df_sorted["fragment_index"].isin(protected)]
    n_eligible = len(eligible)
    worst_indices = eligible.head(n_worst)["fragment_index"].tolist()

    # Step 3: Enumerate swap candidates (scaffold-preserving by construction)
    candidates_smiles = enumerate_fragment_swaps(
        smiles, worst_indices, max_per_frag=max_candidates,
        library_source=library_source,
        use_contribution_prior=use_contribution_prior,
        prop_type=prop_type,
        direction=direction,
    )

    # Post-filter: if locked fragments exist, verify each candidate still contains
    # every locked fragment as a substructure (belt-and-suspenders guard)
    if protected and candidates_smiles:
        protected_queries = []
        frag_atom_map = get_fragment_atom_map(smiles)
        mol_noh = Chem.MolFromSmiles(smiles)
        for fid in protected:
            if fid < len(frag_atom_map):
                frag_smi = Chem.MolFragmentToSmiles(mol_noh, frag_atom_map[fid])
                q = Chem.MolFromSmiles(frag_smi)
                if q is not None:
                    protected_queries.append(q)
        if protected_queries:
            def preserves_scaffold(smi):
                m = Chem.MolFromSmiles(smi)
                return m is not None and all(m.HasSubstructMatch(q) for q in protected_queries)
            candidates_smiles = [s for s in candidates_smiles if preserves_scaffold(s)]

    candidates_smiles = candidates_smiles[:max_candidates]

    if not candidates_smiles:
        return {
            "seed_smiles": smiles,
            "seed_prediction": seed_prediction,
            "worst_fragment_indices": worst_indices,
            "protected_fragment_indices": protected,
            "frag_contributions": frag_contributions,
            "candidates": [],
            "n_candidates_evaluated": 0,
            "n_eligible_fragments": n_eligible,
        }

    # Step 4: Score all candidates in one batch
    scores = batch_score(viz_app, candidates_smiles)

    # Step 5: Rank by improvement
    results = []
    for smi, score in scores.items():
        delta = score - seed_prediction
        improvement = delta if direction == "maximize" else -delta
        results.append({
            "smiles": smi,
            "prediction": round(score, 4),
            "delta": round(delta, 4),
            "improvement": round(improvement, 4),
        })

    results.sort(key=lambda x: x["improvement"], reverse=True)

    return {
        "seed_smiles": smiles,
        "seed_prediction": seed_prediction,
        "worst_fragment_indices": worst_indices,
        "protected_fragment_indices": protected,
        "frag_contributions": frag_contributions,
        "candidates": results[:top_k],
        "n_candidates_evaluated": len(scores),
        "n_eligible_fragments": n_eligible,
    }


# ---------------------------------------------------------------------------
# Visualization helper
# ---------------------------------------------------------------------------

def mol_to_image(smiles: str, width: int = 300, height: int = 200):
    """Render a SMILES string to a PIL Image using RDKit Cairo renderer."""
    import io
    from PIL import Image

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    png_bytes = drawer.GetDrawingText()
    return Image.open(io.BytesIO(png_bytes))
