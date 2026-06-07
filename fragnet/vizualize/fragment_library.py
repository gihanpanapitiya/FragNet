"""
FragNet-Annotated Fragment Library.

Builds and manages an expanded BRICS fragment library with quality filtering
and FragNet-specific annotations for RL-based molecule design.

Pipeline:
  1. Decompose source molecules (FDA drugs + optional external set) with BRICS
  2. Filter by size, Ro3, PAINS, SA score; deduplicate
  3. (Optional) Compute FragNet embeddings via standalone-molecule method
  4. (Optional) Compute mean contribution statistics from source molecules
  5. Save/load; provide typed, contribution-ranked, and similarity lookup

Usage:
    lib = FragmentLibrary.build(viz_app=viz_app, prop_type="Solubility")
    lib.save("fragment_library.pkl")

    lib = FragmentLibrary.load("fragment_library.pkl")
    candidates = lib.get_compatible((3, 7))
    top = lib.get_by_contribution((3, 7), "Solubility", "maximize", top_k=20)
"""

import re
import pickle
import logging
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem.BRICS import BRICSDecompose
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from torch.utils.data import DataLoader

from fragnet.dataset.data import collate_fn as dataset_collate_fn
from fragnet.dataset.dataset import FinetuneData
from fragnet.dataset.utils import extract_data
from torch_scatter import scatter_add

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional SA score
# ---------------------------------------------------------------------------
try:
    from rdkit.Contrib.SA_Score import sascorer as _sascorer
    _HAS_SA = True
except Exception:
    _HAS_SA = False

# ---------------------------------------------------------------------------
# PAINS catalog (lazy singleton)
# ---------------------------------------------------------------------------
_PAINS_CATALOG = None


def _get_pains_catalog() -> FilterCatalog:
    global _PAINS_CATALOG
    if _PAINS_CATALOG is None:
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        _PAINS_CATALOG = FilterCatalog(params)
    return _PAINS_CATALOG


# ---------------------------------------------------------------------------
# FDA-approved drug SMILES — diverse reference set for fragment generation
# ---------------------------------------------------------------------------
FDA_DRUG_SMILES: list[str] = [
    # Analgesics / anti-inflammatory
    "CC(=O)Oc1ccccc1C(=O)O",              # aspirin
    "CC(=O)Nc1ccc(O)cc1",                 # paracetamol
    "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",   # ibuprofen
    "COc1ccc2cc(ccc2c1)[C@@H](C)C(=O)O",  # naproxen
    "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1",  # celecoxib
    "OC(=O)c1ccccc1O",                    # salicylic acid
    "CC(=O)Nc1ccc(OC)cc1",               # methacetin
    # CNS
    "CN1CCC[C@H]1c2cccnc2",              # nicotine
    "CNCCC(Oc1ccc(F)cc1)c1ccccc1",       # fluoxetine
    "CN1C(=O)CN=C(c2ccccc2)c2ccccc21",   # diazepam
    "O=C1CN=C(c2ccccc2Cl)c2ccccc2N1",    # chlordiazepoxide analog
    "CN(C)CCCN1c2ccccc2Sc2ccccc21",       # promazine
    "CNCC(O)c1ccc(O)c(O)c1",             # adrenaline-like
    # Cardiovascular
    "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1",     # salbutamol
    "CCCC1(CC)C(=O)N(C(=O)NC1=O)c1ccccc1",  # phenobarbital
    # Statins
    "CC(C)c1nc(C(C)C)c(-c2ccc(F)cc2)c1CCc1ccc(O)cc1",  # atorvastatin core
    # Antibiotics
    "CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O",  # penicillin G
    "CC1(C)SC2C(NC(=O)C(N)c3ccccc3)C(=O)N2C1C(=O)O",  # ampicillin
    # Antivirals
    "Nc1ncnc2c1ncn2[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O",  # adenosine
    # Antihistamines
    "CN(C)CCOC(c1ccccc1)c1ccccc1",        # diphenhydramine
    # Proton pump inhibitors
    "Cc1cnc(CS(=O)c2nc3ccccc3[nH]2)nc1OC",  # omeprazole
    # Antifungals
    "Clc1ccc(C(c2ccc(Cl)cc2)n2ccnc2)cc1",  # clotrimazole
    # Antidiabetics
    "CN(C)C(=N)NC(=N)N",                  # metformin
    # Anticancer
    "NC(=O)c1ccncc1",                     # nicotinamide
    # Core scaffolds
    "c1ccncc1",                           # pyridine
    "C1CCNCC1",                           # piperidine
    "C1COCCN1",                           # morpholine
    "c1ccc2ccccc2c1",                     # naphthalene
    "c1ccsc1",                            # thiophene
    "c1ccoc1",                            # furan
    "C1CCCCC1",                           # cyclohexane
    "c1cnc2ccccc2n1",                     # benzimidazole
    "c1ccnc2ccccc12",                     # quinoline
    "c1ccc2[nH]cccc2c1",                  # indole
    "c1cnccn1",                           # pyrimidine
    "c1ccc2ncccc2c1",                     # isoquinoline
    "C1CCCC1",                            # cyclopentane
    "C1CC1",                              # cyclopropane
    "C1CCC1",                             # cyclobutane
    "c1ccc2sccc2c1",                      # benzothiophene
    "c1ccc2occc2c1",                      # benzofuran
    "C1CCSC1",                            # tetrahydrothiophene
    "C1CCOC1",                            # tetrahydrofuran
    "N1CCCC1",                            # pyrrolidine
    "C1CNCCN1",                           # piperazine
    # Substituted benzenes
    "Cc1ccccc1",                          # toluene
    "c1ccc(O)cc1",                        # phenol
    "c1ccc(N)cc1",                        # aniline
    "c1ccc(F)cc1",                        # fluorobenzene
    "c1ccc(Cl)cc1",                       # chlorobenzene
    "c1ccc(Br)cc1",                       # bromobenzene
    "c1ccc(cc1)C(=O)O",                   # benzoic acid
    "c1ccc(cc1)C(=O)N",                   # benzamide
    "c1ccc(cc1)S(=O)(=O)N",              # benzenesulfonamide
    "Cc1ccc(cc1)S(=O)(=O)N",             # p-toluenesulfonamide
    "CC(=O)c1ccccc1",                     # acetophenone
    "N#Cc1ccccc1",                        # benzonitrile
    "O=Cc1ccccc1",                        # benzaldehyde
    "c1ccc(cc1)-c1ccccc1",               # biphenyl
    "COc1ccccc1",                         # anisole
    # Heteroaromatic acids
    "OC(=O)c1ccco1",                      # furan-2-carboxylic acid
    "OC(=O)c1cccs1",                      # thiophene-2-carboxylic acid
    "OC(=O)c1cccnc1",                     # nicotinic acid
    "OC(=O)c1ccc(O)cc1",                  # 4-hydroxybenzoic acid
    "OC(=O)c1ccc(N)cc1",                  # 4-aminobenzoic acid
    "OC(=O)c1ccc(Cl)cc1",                 # 4-chlorobenzoic acid
    "OC(=O)c1ccc(F)cc1",                  # 4-fluorobenzoic acid
    "OC(=O)c1ccccc1O",                    # salicylic acid dup (fine — dedup handles it)
    # Lactams / cyclic amides
    "O=C1CCCC(=O)N1",                     # glutarimide
    "O=C1CCCN1",                          # pyrrolidinone
    "O=C1CCCCN1",                         # caprolactam
    "O=C1CCCO1",                          # gamma-butyrolactone
    # Amino acids
    "CC(N)C(=O)O",                        # alanine
    "NCC(=O)O",                           # glycine
    "NC(Cc1ccccc1)C(=O)O",               # phenylalanine
    # Aliphatic chains
    "CCC(=O)O",                           # propionic acid
    "CC(=O)O",                            # acetic acid
    "CC(C)O",                             # isopropanol
    "CC(C)N",                             # isopropylamine
    "CCN(CC)CC",                          # triethylamine
    # Piperidine / pyrrolidine variants
    "c1ccc(cc1)N1CCCCC1",                # N-phenylpiperidine
    "c1ccc(cc1)N1CCCC1",                  # N-phenylpyrrolidine
    # Sulfa / urea fragments
    "Nc1ccc(cc1)S(=O)(=O)N",             # sulfanilamide
    "O=C(N)N",                            # urea
    "O=C(Nc1ccccc1)Nc1ccccc1",           # diphenylurea
    # Additional drug scaffolds
    "Cc1nc2ccccc2s1",                     # 2-methylbenzothiazole
    "Cc1nc2ccccc2[nH]1",                  # 2-methylbenzimidazole
    "CC(=O)Nc1nc2ccccc2s1",              # benzothiazole acetamide
    "O=C1c2ccccc2C(=O)c2ccccc21",        # anthraquinone
    "O=c1[nH]c2ccccc2[nH]1",            # benzimidazolone
    "O=c1ccnc(=O)[nH]1",                 # uracil
    "Nc1ncnc2[nH]cnc12",                  # adenine
    "Nc1nc(=O)c2[nH]cnc2[nH]1",          # guanine
    "c1ccc2[nH]nnc2c1",                  # benzotriazole
    "c1cnc2ccccn12",                      # imidazo[1,2-a]pyridine
    "c1ccc2c(c1)ccn2",                   # indolizine
    "Cc1ccc(-c2ccccn2)nc1",              # 2-methylpyridyl pyridine
    "OC(=O)CCC(=O)O",                    # succinic acid
    "OC(=O)CC(=O)O",                     # malonic acid
    "CC(O)C(=O)O",                        # lactic acid
    "OCC(O)CO",                           # glycerol
    "CC(C)(CO)CO",                        # neopentyl glycol
    "O=C(O)c1ccc(cc1)C(=O)O",           # terephthalic acid
]


# ---------------------------------------------------------------------------
# Fragment utilities
# ---------------------------------------------------------------------------

def _attachment_types(frag_mol) -> tuple:
    """Return sorted tuple of BRICS dummy-atom isotope numbers."""
    return tuple(sorted(
        a.GetIsotope()
        for a in frag_mol.GetAtoms()
        if a.GetAtomicNum() == 0
    ))


def frag_to_clean_smiles(frag_smiles: str) -> str | None:
    """
    Strip BRICS dummy atoms ([n*]) from a fragment SMILES and return a
    valid, sanitized SMILES suitable for standalone FragNet scoring.

    Attachment points are capped with implicit H by removing the dummy atom
    and its bond, restoring the original valence.
    """
    cleaned = re.sub(r'\[\d+\*\]', '[H]', frag_smiles)
    mol = Chem.MolFromSmiles(cleaned)
    if mol is None:
        return None
    mol = Chem.RemoveHs(mol)
    try:
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def _heavy_atom_count(mol) -> int:
    return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() != 1)


def _passes_filters(
    mol,
    min_atoms: int = 3,
    max_atoms: int = 25,
    max_mw: float = 300.0,
    max_sa: float = 4.0,
    check_pains: bool = True,
) -> bool:
    """Return True if the fragment passes all quality filters."""
    n_ha = _heavy_atom_count(mol)
    if n_ha < min_atoms or n_ha > max_atoms:
        return False
    mw = Descriptors.MolWt(mol)
    if mw > max_mw:
        return False
    if _HAS_SA and max_sa < 10.0:
        try:
            sa = _sascorer.calculateScore(mol)
            if sa > max_sa:
                return False
        except Exception:
            pass
    if check_pains and _get_pains_catalog().HasMatch(mol):
        return False
    return True


# ---------------------------------------------------------------------------
# Library building
# ---------------------------------------------------------------------------

def _decompose_smiles(smiles_list: list[str]) -> dict[str, int]:
    """
    Decompose all molecules with BRICS and return a dict of
    canonical fragment SMILES → occurrence count.
    """
    counts: dict[str, int] = defaultdict(int)
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            frags = BRICSDecompose(mol)
        except Exception:
            continue
        for fsmi in frags:
            fmol = Chem.MolFromSmiles(fsmi)
            if fmol is None:
                continue
            canon = Chem.MolToSmiles(fmol)
            counts[canon] += 1
    return counts


# ---------------------------------------------------------------------------
# FragNet embedding
# ---------------------------------------------------------------------------

def _embed_smiles_batch(clean_smiles_list: list[str], viz_app) -> dict[str, np.ndarray]:
    """
    Compute FragNet molecular embeddings for a list of clean (no dummy atom)
    SMILES by running them through viz_app.model.pretrain.

    Returns a dict mapping SMILES → 128-dim numpy array (mean-pooled x_frags).
    Molecules that fail featurization are silently skipped.
    """
    valid = [s for s in clean_smiles_list if Chem.MolFromSmiles(s) is not None]
    if not valid:
        return {}

    df = pd.DataFrame({"smiles": valid, "log_sol": [0.0] * len(valid)})

    dataset_obj = FinetuneData(
        target_name="log_sol", data_type="exp1s", frag_type="brics"
    )
    ds = dataset_obj.get_ft_dataset(df)
    ds = extract_data(ds)

    loader = DataLoader(
        ds,
        collate_fn=dataset_collate_fn,
        batch_size=len(ds),
        shuffle=False,
        drop_last=False,
    )
    batch = next(iter(loader))

    with torch.no_grad():
        viz_app.model.eval()
        # pretrain returns (x_atoms, x_frags, x_edge, x_fedge, *attn_weights)
        out = viz_app.model.pretrain(batch)
        x_frags = out[1]  # [total_frags, emb_dim]
        frag_batch = batch["frag_batch"]  # [total_frags]
        # mean-pool fragments per molecule
        n_mols = frag_batch.max().item() + 1
        emb = torch.zeros(n_mols, x_frags.shape[1])
        counts = torch.zeros(n_mols, 1)
        for mol_id in range(n_mols):
            mask = frag_batch == mol_id
            emb[mol_id] = x_frags[mask].mean(0)

    emb_np = emb.numpy()
    return {smi: emb_np[i] for i, smi in enumerate(valid)}


# ---------------------------------------------------------------------------
# Contribution computation from source molecules
# ---------------------------------------------------------------------------

def _compute_source_contributions(
    source_smiles: list[str],
    viz_app,
    prop_type: str = "Solubility",
) -> dict[str, list[float]]:
    """
    For each source molecule, decompose with BRICS, run fragment occlusion,
    and record each fragment's contribution.

    Returns a dict: canonical_fragment_smiles → list of contribution values
    across all source molecules where that fragment appears.
    """
    from fragnet.vizualize.optimizer import batch_score, get_fragment_atom_map

    contrib_map: dict[str, list[float]] = defaultdict(list)

    for smi in source_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        frag_atom_map = get_fragment_atom_map(smi)
        if not frag_atom_map:
            continue

        # Score the full molecule
        score_full = batch_score(viz_app, [smi]).get(smi)
        if score_full is None:
            continue

        # For each fragment, build masked molecule (fragment replaced by nothing)
        # We can't literally mask here — instead score the molecule with each
        # fragment's atoms zeroed out, following the occlusion approach.
        # Simpler: use the contribution computation from model_attr.get_attr_image,
        # but that's slow. Here we approximate by scoring without that fragment
        # using BRICSDecompose → fragment SMILES.
        try:
            frag_smiles_set = BRICSDecompose(mol)
        except Exception:
            continue

        for fsmi in frag_smiles_set:
            fmol = Chem.MolFromSmiles(fsmi)
            if fmol is None:
                continue
            canon = Chem.MolToSmiles(fmol)
            # Contribution proxy: score difference when fragment is removed.
            # Build molecule without this fragment using BRICS rebuild with
            # the remaining fragments.
            remaining = frag_smiles_set - {fsmi}
            if not remaining:
                continue
            remaining_mols = [Chem.MolFromSmiles(r) for r in remaining]
            remaining_mols = [m for m in remaining_mols if m is not None]
            if not remaining_mols:
                continue
            from rdkit.Chem.BRICS import BRICSBuild
            try:
                rebuilt = list(BRICSBuild(remaining_mols, maxDepth=4))
            except Exception:
                continue
            if not rebuilt:
                continue
            best_smi = None
            for rb in rebuilt[:5]:
                try:
                    Chem.SanitizeMol(rb)
                    best_smi = Chem.MolToSmiles(Chem.RemoveHs(rb))
                    break
                except Exception:
                    pass
            if best_smi is None:
                continue
            score_masked = batch_score(viz_app, [best_smi]).get(best_smi)
            if score_masked is None:
                continue
            contrib = float(score_full - score_masked)
            contrib_map[canon].append(contrib)

    return dict(contrib_map)


# ---------------------------------------------------------------------------
# FragmentLibrary
# ---------------------------------------------------------------------------

class FragmentLibrary:
    """
    Expanded, annotated BRICS fragment library for RL-based molecule design.

    Each entry is a dict:
        smiles          — BRICS fragment SMILES (may contain dummy atoms)
        clean_smiles    — valid SMILES with dummy atoms capped by H
        attachment_types — sorted tuple of BRICS dummy isotope numbers
        heavy_atoms     — number of heavy atoms
        mw              — molecular weight
        n_sources       — number of source molecules it appeared in
        embedding       — numpy array [emb_dim] or None
        contributions   — dict {prop_type: {"mean": float, "std": float, "n": int}}
    """

    def __init__(self):
        self.entries: list[dict] = []
        self._type_index: dict[tuple, list[int]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        extra_smiles: list[str] | None = None,
        viz_app=None,
        prop_type: str = "Solubility",
        min_freq: int = 1,
        min_atoms: int = 3,
        max_atoms: int = 25,
        max_mw: float = 300.0,
        max_sa: float = 4.0,
        check_pains: bool = True,
        compute_embeddings: bool = True,
        compute_contributions: bool = False,
    ) -> "FragmentLibrary":
        """
        Build the library from FDA drugs plus any extra_smiles.

        Args:
            extra_smiles:          Additional source SMILES (e.g. from a CSV).
            viz_app:               Loaded FragNetVizApp instance (needed for
                                   embeddings and contributions).
            prop_type:             Property name for contribution statistics.
            min_freq:              Minimum source-molecule occurrences to keep.
            min_atoms / max_atoms: Heavy-atom count range.
            max_mw:                Maximum molecular weight (Ro3-style).
            max_sa:                Maximum SA score (ignored if rdkit SA unavailable).
            check_pains:           Filter PAINS compounds.
            compute_embeddings:    Compute FragNet embeddings (requires viz_app).
            compute_contributions: Compute mean contribution stats (slow, requires
                                   viz_app; uses the source SMILES list).
        """
        source_smiles = list(FDA_DRUG_SMILES)
        if extra_smiles:
            source_smiles += list(extra_smiles)

        logger.info("Decomposing %d source molecules with BRICS …", len(source_smiles))
        counts = _decompose_smiles(source_smiles)
        logger.info("Raw fragments before filtering: %d", len(counts))

        lib = cls()
        clean_to_brics: dict[str, str] = {}  # clean_smiles → brics_smiles

        for brics_smi, freq in counts.items():
            if freq < min_freq:
                continue
            fmol = Chem.MolFromSmiles(brics_smi)
            if fmol is None:
                continue
            clean_smi = frag_to_clean_smiles(brics_smi)
            if clean_smi is None:
                continue
            clean_mol = Chem.MolFromSmiles(clean_smi)
            if clean_mol is None:
                continue
            if not _passes_filters(
                clean_mol, min_atoms, max_atoms, max_mw, max_sa, check_pains
            ):
                continue
            att = _attachment_types(fmol)
            if not att:  # no attachment points → can't be used in swap
                continue
            entry = {
                "smiles": brics_smi,
                "clean_smiles": clean_smi,
                "attachment_types": att,
                "heavy_atoms": _heavy_atom_count(clean_mol),
                "mw": Descriptors.MolWt(clean_mol),
                "n_sources": freq,
                "embedding": None,
                "contributions": {},
            }
            idx = len(lib.entries)
            lib.entries.append(entry)
            lib._type_index[att].append(idx)
            clean_to_brics[clean_smi] = brics_smi

        logger.info("Filtered library size: %d fragments", len(lib.entries))

        if viz_app is not None and compute_embeddings:
            lib.compute_fragnet_embeddings(viz_app)

        if viz_app is not None and compute_contributions:
            lib.compute_mean_contributions(source_smiles, viz_app, prop_type)

        return lib

    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        smiles_col: str = "smiles",
        viz_app=None,
        **build_kwargs,
    ) -> "FragmentLibrary":
        """Build from a CSV file with a SMILES column (appended to FDA drugs)."""
        df = pd.read_csv(csv_path)
        extra = df[smiles_col].dropna().tolist()
        return cls.build(extra_smiles=extra, viz_app=viz_app, **build_kwargs)

    # ------------------------------------------------------------------
    # Annotation (can be called after initial build)
    # ------------------------------------------------------------------

    def compute_fragnet_embeddings(self, viz_app) -> None:
        """Compute and store FragNet embeddings for all entries."""
        need = [e for e in self.entries if e["embedding"] is None]
        if not need:
            return
        clean_smiles = list({e["clean_smiles"] for e in need})
        logger.info("Computing FragNet embeddings for %d unique fragments …", len(clean_smiles))
        emb_map = _embed_smiles_batch(clean_smiles, viz_app)
        for entry in need:
            emb = emb_map.get(entry["clean_smiles"])
            if emb is not None:
                entry["embedding"] = emb
        n_done = sum(1 for e in self.entries if e["embedding"] is not None)
        logger.info("Embeddings computed: %d / %d", n_done, len(self.entries))

    def compute_mean_contributions(
        self,
        source_smiles: list[str],
        viz_app,
        prop_type: str = "Solubility",
    ) -> None:
        """
        Compute mean fragment contribution statistics from the source molecules
        and store them per entry under entry["contributions"][prop_type].
        """
        logger.info(
            "Computing fragment contributions from %d source molecules …", len(source_smiles)
        )
        raw = _compute_source_contributions(source_smiles, viz_app, prop_type)

        clean_to_entries: dict[str, list[int]] = defaultdict(list)
        for i, entry in enumerate(self.entries):
            clean_to_entries[entry["clean_smiles"]].append(i)

        for brics_smi, contrib_list in raw.items():
            clean_smi = frag_to_clean_smiles(brics_smi)
            if clean_smi is None:
                continue
            arr = np.array(contrib_list)
            stat = {"mean": float(arr.mean()), "std": float(arr.std()), "n": len(arr)}
            for idx in clean_to_entries.get(clean_smi, []):
                self.entries[idx]["contributions"][prop_type] = stat

        n_annotated = sum(1 for e in self.entries if prop_type in e["contributions"])
        logger.info(
            "Contribution stats added for %d / %d entries (%s)",
            n_annotated, len(self.entries), prop_type,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Library saved to %s (%d entries)", path, len(self.entries))

    @classmethod
    def load(cls, path: str) -> "FragmentLibrary":
        with open(path, "rb") as f:
            lib = pickle.load(f)
        # Rebuild index in case it wasn't serialized cleanly
        lib._rebuild_index()
        logger.info("Library loaded from %s (%d entries)", path, len(lib.entries))
        return lib

    def _rebuild_index(self) -> None:
        self._type_index = defaultdict(list)
        for i, entry in enumerate(self.entries):
            self._type_index[entry["attachment_types"]].append(i)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_compatible(self, attachment_types: tuple) -> list[dict]:
        """
        Return all entries whose attachment types exactly match the given tuple.
        For single-cut scaffolds (len == 1), strict match.
        For multi-cut, also return entries that cover a subset of required types.
        """
        exact = [self.entries[i] for i in self._type_index.get(attachment_types, [])]
        if len(attachment_types) <= 1 or exact:
            return exact
        # Partial match: any entry whose types intersect with the required set
        req = set(attachment_types)
        partial = [
            e for e in self.entries
            if set(e["attachment_types"]) & req
        ]
        return partial

    def get_by_contribution(
        self,
        attachment_types: tuple,
        prop_type: str,
        direction: str = "maximize",
        top_k: int = 20,
    ) -> list[dict]:
        """
        Return compatible fragments sorted by mean contribution (best first).

        If no contribution statistics are available for a fragment, it is
        placed last with a contribution of 0.
        """
        candidates = self.get_compatible(attachment_types)
        ascending = direction != "maximize"

        def sort_key(e):
            stat = e["contributions"].get(prop_type)
            val = stat["mean"] if stat else 0.0
            return val if not ascending else -val

        return sorted(candidates, key=sort_key, reverse=True)[:top_k]

    def get_by_similarity(
        self,
        query_embedding: np.ndarray,
        attachment_types: tuple,
        top_k: int = 20,
    ) -> list[dict]:
        """
        Return compatible fragments sorted by cosine similarity to a query
        FragNet embedding.  Requires embeddings to have been computed.
        """
        candidates = self.get_compatible(attachment_types)
        with_emb = [(e, e["embedding"]) for e in candidates if e["embedding"] is not None]
        if not with_emb:
            return candidates[:top_k]

        q = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
        scored = []
        for entry, emb in with_emb:
            e_norm = emb / (np.linalg.norm(emb) + 1e-9)
            sim = float(np.dot(q, e_norm))
            scored.append((sim, entry))
        scored.sort(reverse=True, key=lambda x: x[0])
        return [e for _, e in scored[:top_k]]

    # ------------------------------------------------------------------
    # Stats / introspection
    # ------------------------------------------------------------------

    def summary(self) -> str:
        n = len(self.entries)
        n_emb = sum(1 for e in self.entries if e["embedding"] is not None)
        type_counts = sorted(
            ((k, len(v)) for k, v in self._type_index.items()),
            key=lambda x: -x[1],
        )[:5]
        lines = [
            f"FragmentLibrary: {n} entries",
            f"  With embeddings: {n_emb}",
            f"  Top attachment-type groups: {type_counts}",
        ]
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.entries)

    def __repr__(self) -> str:
        return f"FragmentLibrary(n={len(self.entries)})"


# ---------------------------------------------------------------------------
# Convenience: build and return a default library (no viz_app required)
# ---------------------------------------------------------------------------

_DEFAULT_LIBRARY: FragmentLibrary | None = None


def get_default_library() -> FragmentLibrary:
    """Return a cached default library built from FDA drugs (no embeddings)."""
    global _DEFAULT_LIBRARY
    if _DEFAULT_LIBRARY is None:
        _DEFAULT_LIBRARY = FragmentLibrary.build(compute_embeddings=False)
    return _DEFAULT_LIBRARY
