"""
FragNet Molecule Design RL Environment.

Wraps BRICS fragment-swap molecule optimization as a Gym-style episode:

    State  : molecule + FragNet's 4-level interpretability signals
    Action : (fragment_index, replacement_brics_smiles)
    Reward : Δ(property) + shaped Δ(contribution signal)
    Done   : max_steps reached | no valid actions | property threshold hit

Quick start:
    from fragnet.vizualize.rl_env import MoleculeDesignEnv
    from fragnet.vizualize.fragment_library import FragmentLibrary

    lib = FragmentLibrary.load("chembl_library.pkl")
    env = MoleculeDesignEnv(viz_app, lib, prop_type="Solubility", direction="maximize")

    state = env.reset("CC(=O)Oc1ccccc1C(=O)O")
    actions = env.get_valid_actions()
    frag_idx, replacement = actions[0]           # pick first valid action
    state, reward, done, info = env.step(frag_idx, replacement)
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem.BRICS import BRICSDecompose
from torch.utils.data import DataLoader
from torch_scatter import scatter_add

from fragnet.dataset.data import collate_fn as dataset_collate_fn
from fragnet.dataset.dataset import FinetuneData
from fragnet.dataset.utils import extract_data
from fragnet.vizualize.fragment_library import FragmentLibrary
from fragnet.vizualize.optimizer import (
    batch_score,
    build_scaffold_with_dummies,
    enumerate_fragment_swaps,
    get_core_protected_indices,
    get_fragment_atom_map,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State dataclass
# ---------------------------------------------------------------------------

@dataclass
class MolState:
    """
    All information about the molecule at a given step.

    Signals at each level:
      atom_attention    [N_atoms]         — GAT attention from last layer
      bond_attention    [N_bonds]         — GAT attention, bond graph
      frag_attention    [N_frags]         — GAT attention, fragment graph
      conn_attention    [N_connections]   — GAT attention, connection graph
      frag_contributions [N_frags]        — occlusion-based causal contribution
      frag_embeddings   [N_frags, emb_dim]— fragment hidden states (for similarity lookup)

    All per-node arrays are numpy float32, normalized to [0, 1] within the molecule.
    """
    smiles: str
    prediction: float
    n_frags: int

    # 4-level attention signals (fast — one forward pass)
    atom_attention: np.ndarray = field(default_factory=lambda: np.array([]))
    bond_attention: np.ndarray = field(default_factory=lambda: np.array([]))
    frag_attention: np.ndarray = field(default_factory=lambda: np.array([]))
    conn_attention: np.ndarray = field(default_factory=lambda: np.array([]))

    # Fragment-level contribution (slow — O(N_frags) passes, optional)
    frag_contributions: np.ndarray = field(default_factory=lambda: np.array([]))

    # Fragment embeddings from FragNet pretrain backbone
    frag_embeddings: np.ndarray = field(default_factory=lambda: np.array([]))

    # Fragment-to-atom map for protection / action masking
    frag_atom_map: list[list[int]] = field(default_factory=list)

    def as_policy_input(self) -> dict[str, np.ndarray]:
        """
        Per-fragment feature matrix for the policy network.

        Each fragment gets a feature vector:
          [contribution, frag_attention, mean_atom_attention,
           max_atom_attention, n_atoms_in_frag]
        Shape: [N_frags, 5]
        """
        n = self.n_frags
        contribs = self.frag_contributions if len(self.frag_contributions) == n else np.zeros(n)
        fattn = self.frag_attention if len(self.frag_attention) == n else np.zeros(n)

        mean_atom_attn = np.zeros(n)
        max_atom_attn = np.zeros(n)
        n_atoms_in_frag = np.zeros(n)
        if len(self.atom_attention) > 0 and self.frag_atom_map:
            for i, atoms in enumerate(self.frag_atom_map):
                if atoms and max(atoms) < len(self.atom_attention):
                    vals = self.atom_attention[atoms]
                    mean_atom_attn[i] = vals.mean()
                    max_atom_attn[i] = vals.max()
                    n_atoms_in_frag[i] = len(atoms)

        return {
            "frag_features": np.column_stack([
                contribs, fattn, mean_atom_attn, max_atom_attn, n_atoms_in_frag
            ]).astype(np.float32),
            "frag_embeddings": self.frag_embeddings,
            "prediction": np.array([self.prediction], dtype=np.float32),
        }


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class MoleculeDesignEnv:
    """
    Gym-style RL environment for FragNet-guided molecule optimization.

    Args:
        viz_app:                  Loaded FragNetVizApp instance.
        library:                  FragmentLibrary with BRICS fragments.
        prop_type:                'Solubility' or 'Lipophilicity'.
        direction:                'maximize' or 'minimize'.
        max_steps:                Maximum fragment swaps per episode.
        protected_smarts:         Optional SMARTS/SMILES to lock as scaffold.
        reward_weights:           Dict with keys 'property', 'contribution'.
                                  Defaults to {'property': 0.8, 'contribution': 0.2}.
        compute_contributions:    Whether to compute occlusion contributions at
                                  each step. Slow (O(N_frags) passes) but gives
                                  the full 4-level state. Default True.
        property_threshold:       Episode terminates early if prediction crosses
                                  this value (optional).
        max_library_per_frag:     Max replacement candidates to try per fragment.
    """

    def __init__(
        self,
        viz_app,
        library: FragmentLibrary,
        prop_type: str = "Solubility",
        direction: str = "maximize",
        max_steps: int = 5,
        protected_smarts: str | None = None,
        reward_weights: dict | None = None,
        compute_contributions: bool = True,
        property_threshold: float | None = None,
        max_library_per_frag: int = 50,
    ):
        self.viz_app = viz_app
        self.library = library
        self.prop_type = prop_type
        self.direction = direction
        self.max_steps = max_steps
        self.protected_smarts = protected_smarts
        self.reward_weights = reward_weights or {"property": 0.8, "contribution": 0.2}
        self.compute_contributions = compute_contributions
        self.property_threshold = property_threshold
        self.max_library_per_frag = max_library_per_frag

        # Episode state
        self._state: MolState | None = None
        self._step_count: int = 0
        self._seed_smiles: str | None = None
        self._seed_prediction: float | None = None
        self._episode_history: list[dict] = []

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed_smiles: str,
        frag_contributions: list[dict] | None = None,
    ) -> MolState:
        """
        Start a new episode.

        Args:
            seed_smiles:         Starting molecule SMILES.
            frag_contributions:  Pre-computed contribution dicts from get_attr_image
                                 (list of {'fragment_index': i, 'contribution': v}).
                                 If None, computed via batch_score approximation.

        Returns:
            Initial MolState.
        """
        self._step_count = 0
        self._seed_smiles = seed_smiles
        self._episode_history = []

        self._state = self._build_state(seed_smiles, frag_contributions)
        self._seed_prediction = self._state.prediction
        return self._state

    def step(
        self,
        frag_idx: int,
        replacement_smiles: str,
    ) -> tuple[MolState, float, bool, dict]:
        """
        Apply one fragment swap.

        Args:
            frag_idx:            Index of the fragment to replace (0-indexed,
                                 matching FragNet's fragment ordering).
            replacement_smiles:  BRICS SMILES from the library (with dummy atoms).

        Returns:
            (new_state, reward, done, info)
        """
        if self._state is None:
            raise RuntimeError("Call reset() before step().")

        old_state = self._state
        info: dict[str, Any] = {
            "step": self._step_count,
            "old_smiles": old_state.smiles,
            "frag_idx": frag_idx,
            "replacement": replacement_smiles,
        }

        # --- Apply the swap ---
        new_smiles = self._apply_swap(old_state.smiles, frag_idx, replacement_smiles)

        if new_smiles is None:
            # Invalid action — penalise and do not advance
            info["error"] = "invalid_swap"
            return old_state, -0.1, False, info

        # --- Build new state ---
        new_state = self._build_state(new_smiles)
        self._state = new_state
        self._step_count += 1

        # --- Reward ---
        reward = self._compute_reward(old_state, new_state)

        # --- Done? ---
        done = self._is_done(new_state)

        info.update({
            "new_smiles": new_smiles,
            "old_prediction": old_state.prediction,
            "new_prediction": new_state.prediction,
            "delta": new_state.prediction - old_state.prediction,
            "total_improvement": new_state.prediction - self._seed_prediction,
            "reward": reward,
        })
        self._episode_history.append(info)

        logger.debug(
            "Step %d: %s → %s  Δ=%.4f  r=%.4f",
            self._step_count, old_state.smiles[:20], new_smiles[:20],
            info["delta"], reward,
        )

        return new_state, reward, done, info

    def get_valid_actions(self) -> list[tuple[int, str]]:
        """
        Return all valid (frag_idx, replacement_brics_smiles) pairs for the
        current state, sorted by descending contribution-weighted priority.

        The fragment selector pre-filters using:
          - Protection mask (protected_smarts)
          - Atom attention: skip fragments with max atom attention > threshold
          - Contribution: prioritise most-negative contributions

        The replacement selector uses library.get_compatible() masked by BRICS type.
        """
        if self._state is None:
            raise RuntimeError("Call reset() before get_valid_actions().")

        state = self._state
        mol = Chem.MolFromSmiles(state.smiles)
        if mol is None:
            return []

        # Protected fragment indices
        protected = get_core_protected_indices(state.smiles, self.protected_smarts or "")

        # Atom attention protection threshold (top-quartile atoms are load-bearing)
        attn_threshold = (
            np.percentile(state.atom_attention, 75)
            if len(state.atom_attention) > 0
            else 1.0
        )

        # Rank fragments: most negative contribution first (maximize) or
        # most positive first (minimize)
        frag_scores: list[tuple[float, int]] = []
        for i in range(state.n_frags):
            if i in protected:
                continue
            # Skip load-bearing fragments
            if state.frag_atom_map and i < len(state.frag_atom_map):
                atoms = state.frag_atom_map[i]
                if atoms and len(state.atom_attention) > 0:
                    max_attn = state.atom_attention[atoms].max() if atoms else 0.0
                    if max_attn > attn_threshold:
                        continue
            contrib = (
                float(state.frag_contributions[i])
                if len(state.frag_contributions) > i
                else 0.0
            )
            priority = -contrib if self.direction == "maximize" else contrib
            frag_scores.append((priority, i))

        frag_scores.sort(reverse=True)

        # Build action list
        actions: list[tuple[int, str]] = []
        from fragnet.vizualize.optimizer import _attachment_types as _att
        from rdkit.Chem.BRICS import FindBRICSBonds, BreakBRICSBonds

        for _, frag_idx in frag_scores:
            # Get attachment types for this fragment's scaffold side
            if not state.frag_atom_map or frag_idx >= len(state.frag_atom_map):
                continue
            frag_atoms = state.frag_atom_map[frag_idx]
            scaffold_mol, n_cuts = build_scaffold_with_dummies(mol, frag_atoms)
            if scaffold_mol is None:
                continue
            scaffold_att = tuple(sorted(
                a.GetIsotope()
                for a in scaffold_mol.GetAtoms()
                if a.GetAtomicNum() == 0
            ))
            replacements = self.library.get_compatible(scaffold_att)
            for entry in replacements[: self.max_library_per_frag]:
                actions.append((frag_idx, entry["smiles"]))

        return actions

    def episode_summary(self) -> dict:
        """Return a summary of the current episode trajectory."""
        if not self._episode_history:
            return {}
        return {
            "seed_smiles": self._seed_smiles,
            "final_smiles": self._state.smiles if self._state else None,
            "seed_prediction": self._seed_prediction,
            "final_prediction": self._state.prediction if self._state else None,
            "total_improvement": (
                (self._state.prediction - self._seed_prediction)
                if self._state else 0.0
            ),
            "n_steps": self._step_count,
            "trajectory": [
                {"step": h["step"], "smiles": h["new_smiles"],
                 "prediction": h["new_prediction"], "reward": h["reward"]}
                for h in self._episode_history
            ],
        }

    # ------------------------------------------------------------------
    # State construction
    # ------------------------------------------------------------------

    def _build_state(
        self,
        smiles: str,
        prefilled_contributions: list[dict] | None = None,
    ) -> MolState:
        """Build a full MolState from a SMILES string."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        frag_atom_map = get_fragment_atom_map(smiles)
        n_frags = len(frag_atom_map)

        # --- Fast pass: prediction + all 4-level attention weights ---
        prediction = self.viz_app.calc_weights(smiles)

        atom_attn = self._pool_attention(self.viz_app.summed_attn_weights_atoms)
        bond_attn = self._pool_attention(self.viz_app.summed_attn_weights_bonds)
        frag_attn = self._pool_attention(self.viz_app.summed_attn_weights_frags)
        conn_attn = self._pool_attention(self.viz_app.summed_attn_weights_fbonds)

        # --- Fragment embeddings from pretrain backbone ---
        frag_emb = self._get_frag_embeddings(smiles)

        # --- Fragment contributions ---
        if prefilled_contributions is not None:
            contribs = np.array(
                [c["contribution"] for c in sorted(
                    prefilled_contributions, key=lambda x: x["fragment_index"]
                )],
                dtype=np.float32,
            )
        elif self.compute_contributions:
            contribs = self._compute_frag_contributions(smiles, n_frags)
        else:
            contribs = np.zeros(n_frags, dtype=np.float32)

        return MolState(
            smiles=smiles,
            prediction=float(prediction),
            n_frags=n_frags,
            atom_attention=atom_attn,
            bond_attention=bond_attn,
            frag_attention=frag_attn,
            conn_attention=conn_attn,
            frag_contributions=contribs,
            frag_embeddings=frag_emb,
            frag_atom_map=frag_atom_map,
        )

    def _pool_attention(self, attn_tensor: torch.Tensor) -> np.ndarray:
        """Mean-pool multi-head attention weights → normalized 1-D array."""
        if attn_tensor is None or attn_tensor.numel() == 0:
            return np.array([], dtype=np.float32)
        arr = attn_tensor.mean(dim=-1).detach().numpy().astype(np.float32)
        rng = arr.max() - arr.min()
        if rng > 1e-9:
            arr = (arr - arr.min()) / rng
        return arr

    def _get_frag_embeddings(self, smiles: str) -> np.ndarray:
        """Extract per-fragment hidden states from the pretrain backbone."""
        df = pd.DataFrame({"smiles": [smiles], "log_sol": [0.0]})
        dataset_obj = FinetuneData(target_name="log_sol", data_type="exp1s", frag_type="brics")
        ds = dataset_obj.get_ft_dataset(df)
        ds = extract_data(ds)
        loader = DataLoader(ds, collate_fn=dataset_collate_fn, batch_size=1,
                            shuffle=False, drop_last=False)
        batch = next(iter(loader))
        with torch.no_grad():
            self.viz_app.model.eval()
            out = self.viz_app.model.pretrain(batch)
            x_frags = out[1].detach().numpy()  # [N_frags, emb_dim]
        return x_frags.astype(np.float32)

    def _compute_frag_contributions(self, smiles: str, n_frags: int) -> np.ndarray:
        """
        Approximate fragment contributions via batch_score occlusion:
          contribution_i = score(full) - score(scaffold_without_frag_i)

        O(N_frags) scoring calls. Falls back to zeros on any failure.
        """
        full_score = batch_score(self.viz_app, [smiles]).get(smiles)
        if full_score is None:
            return np.zeros(n_frags, dtype=np.float32)

        mol = Chem.MolFromSmiles(smiles)
        frag_atom_map = get_fragment_atom_map(smiles)
        contribs = np.zeros(n_frags, dtype=np.float32)

        for i, frag_atoms in enumerate(frag_atom_map):
            scaffold_mol, _ = build_scaffold_with_dummies(mol, frag_atoms)
            if scaffold_mol is None:
                continue
            # Cap dummy atoms with H to get a scoreable SMILES
            from rdkit.Chem import RWMol
            rw = RWMol(scaffold_mol)
            dummies = [a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 0]
            for d_idx in sorted(dummies, reverse=True):
                rw.GetAtomWithIdx(d_idx).SetAtomicNum(1)
            try:
                Chem.SanitizeMol(rw)
                sc_smi = Chem.MolToSmiles(Chem.RemoveHs(rw.GetMol()))
            except Exception:
                continue
            sc_score = batch_score(self.viz_app, [sc_smi]).get(sc_smi)
            if sc_score is not None:
                contribs[i] = float(full_score - sc_score)

        return contribs

    # ------------------------------------------------------------------
    # Swap execution
    # ------------------------------------------------------------------

    def _apply_swap(
        self, smiles: str, frag_idx: int, replacement_brics_smiles: str
    ) -> str | None:
        """
        Replace fragment frag_idx with replacement_brics_smiles using
        BRICSBuild (scaffold-preserving). Returns canonical SMILES or None.
        """
        from rdkit.Chem.BRICS import BRICSBuild

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        frag_atom_map = get_fragment_atom_map(smiles)
        if frag_idx >= len(frag_atom_map):
            return None

        frag_atoms = frag_atom_map[frag_idx]
        scaffold_mol, n_cuts = build_scaffold_with_dummies(mol, frag_atoms)
        if scaffold_mol is None:
            return None

        rep_mol = Chem.MolFromSmiles(replacement_brics_smiles)
        if rep_mol is None:
            return None

        try:
            new_mols = list(BRICSBuild([scaffold_mol, rep_mol], maxDepth=1))
        except Exception:
            return None

        seed_canon = Chem.MolToSmiles(mol)
        for new_mol in new_mols[:8]:
            try:
                Chem.SanitizeMol(new_mol)
                new_smi = Chem.MolToSmiles(Chem.RemoveHs(new_mol))
                if new_smi and new_smi != seed_canon:
                    return new_smi
            except Exception:
                continue
        return None

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self, old: MolState, new: MolState) -> float:
        """
        Weighted combination of:
          - property improvement (primary)
          - improvement in total positive fragment contribution (shaped)
        """
        w_prop = self.reward_weights.get("property", 0.8)
        w_contrib = self.reward_weights.get("contribution", 0.2)

        delta = new.prediction - old.prediction
        property_reward = delta if self.direction == "maximize" else -delta

        # Shaped: change in sum of positive contributions
        shaped_reward = 0.0
        if w_contrib > 0 and len(old.frag_contributions) > 0 and len(new.frag_contributions) > 0:
            old_pos = float(np.sum(np.maximum(0, old.frag_contributions)))
            new_pos = float(np.sum(np.maximum(0, new.frag_contributions)))
            shaped_reward = new_pos - old_pos

        return float(w_prop * property_reward + w_contrib * shaped_reward)

    # ------------------------------------------------------------------
    # Done condition
    # ------------------------------------------------------------------

    def _is_done(self, state: MolState) -> bool:
        if self._step_count >= self.max_steps:
            return True
        if self.property_threshold is not None:
            if self.direction == "maximize" and state.prediction >= self.property_threshold:
                return True
            if self.direction == "minimize" and state.prediction <= self.property_threshold:
                return True
        return False
