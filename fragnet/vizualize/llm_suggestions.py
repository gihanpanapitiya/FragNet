"""
LLM-guided fragment replacement suggestions via Claude.

Given a seed molecule and FragNet fragment contributions, Claude acts as a
medicinal chemist and proposes complete modified molecules — replacing the
worst-contributing fragment(s) with chemically reasoned alternatives.

The caller is responsible for scoring the returned SMILES with FragNet.
"""
import json
import os
import re

from rdkit import Chem


# Property-specific chemistry hints injected into the prompt
_PROP_HINTS = {
    "Solubility": (
        "logS (aqueous solubility). Higher = more soluble. "
        "To increase: add H-bond donors/acceptors (OH, NH, COOH, SO3H), "
        "reduce hydrophobic/aromatic surface area, break up flat ring systems, "
        "add ionisable groups, lower MW."
    ),
    "Lipophilicity": (
        "logP (lipophilicity). "
        "To increase: add aromatic rings, halogens (F, Cl, Br), alkyl chains, "
        "remove polar groups. "
        "To decrease: add polar/ionisable groups (OH, NH2, COOH)."
    ),
}


def _fragment_smiles(smiles: str, frag_atom_map: list) -> list[str]:
    """Return the SMILES of each fragment as it appears in the parent molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ["?"] * len(frag_atom_map)
    result = []
    for atoms in frag_atom_map:
        try:
            fsmi = Chem.MolFragmentToSmiles(mol, atomsToUse=atoms, isomericSmiles=False)
            result.append(fsmi or "?")
        except Exception:
            result.append("?")
    return result


def suggest_replacements(
    smiles: str,
    frag_contributions: list,
    frag_atom_map: list,
    prop_type: str,
    direction: str,
    seed_prediction: float,
    protected_indices: set,
    n_worst: int = 1,
    n_suggestions: int = 8,
    model: str = "claude-sonnet-4-6",
) -> list[dict]:
    """
    Ask Claude to suggest modified molecules that should improve ``prop_type``.

    Args:
        smiles:             Seed molecule SMILES.
        frag_contributions: List of {"fragment_index": int, "contribution": float, ...}.
        frag_atom_map:      List of atom-index lists, one per fragment.
        prop_type:          "Solubility" or "Lipophilicity".
        direction:          "maximize" or "minimize".
        seed_prediction:    FragNet's predicted value for the seed.
        protected_indices:  Fragment indices that must not be changed.
        n_worst:            How many worst-contributing fragments to target.
        n_suggestions:      How many molecule suggestions to request from Claude.
        model:              Anthropic model ID.

    Returns:
        List of {"smiles": str, "rationale": str} — only valid, unique SMILES
        that differ from the seed molecule.

    Raises:
        EnvironmentError: if ANTHROPIC_API_KEY is not set.
        RuntimeError:     if Claude's response cannot be parsed as JSON.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set. "
            "Export it before starting the app:\n"
            "  export ANTHROPIC_API_KEY=sk-ant-..."
        )

    import anthropic  # imported here so the module loads without the SDK installed

    frag_smiles_list = _fragment_smiles(smiles, frag_atom_map)

    seed_mol = Chem.MolFromSmiles(smiles)
    seed_canonical = Chem.MolToSmiles(seed_mol) if seed_mol else smiles

    # Identify worst-contributing eligible fragments
    ascending = (direction == "maximize")   # ascending → most-negative first
    sorted_c = sorted(frag_contributions, key=lambda x: x["contribution"],
                      reverse=not ascending)
    eligible = [c for c in sorted_c if c["fragment_index"] not in protected_indices]
    worst = eligible[:n_worst]

    # Build fragment table
    frag_lines = []
    for c in frag_contributions:
        fid = c["fragment_index"]
        fsmi = frag_smiles_list[fid] if fid < len(frag_smiles_list) else "?"
        tag = " [LOCKED — do not modify]" if fid in protected_indices else ""
        frag_lines.append(
            f"  Fragment {fid}: {fsmi}  contribution={c['contribution']:+.4f}{tag}"
        )

    worst_desc = "\n".join(
        f"  Fragment {c['fragment_index']}: "
        f"{frag_smiles_list[c['fragment_index']] if c['fragment_index'] < len(frag_smiles_list) else '?'}  "
        f"contribution={c['contribution']:+.4f}"
        for c in worst
    )

    hint = _PROP_HINTS.get(prop_type, prop_type)
    action = "increase" if direction == "maximize" else "decrease"

    prompt = f"""You are an expert medicinal chemist helping to optimise a molecule.

Seed SMILES: {smiles}
Property: {prop_type} — {hint}
Current predicted value: {seed_prediction:.4f}
Goal: {action} {prop_type}

Fragment decomposition (BRICS):
{chr(10).join(frag_lines)}

Target fragment(s) to replace (worst-contributing to the optimisation goal):
{worst_desc}

Instructions:
1. Replace ONLY the target fragment(s). Every LOCKED fragment must remain unchanged.
2. Suggest {n_suggestions} diverse modified complete molecules.
3. Each must be a valid, synthetically reasonable SMILES string.
4. Provide one concise sentence explaining the chemical rationale per suggestion.

Respond with ONLY a valid JSON array and nothing else:
[
  {{"smiles": "COMPLETE_MOLECULE_SMILES", "rationale": "one-sentence explanation"}},
  ...
]"""

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=1800,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = message.content[0].text.strip()

    # Extract JSON array from response (Claude sometimes adds a preamble like
    # "here are suggestions [as requested]:" before the actual JSON).
    # Match [ followed by { or whitespace+{ to avoid false hits on prose brackets.
    m = re.search(r'\[\s*\{', raw)
    start = m.start() if m else raw.find("[")
    end = raw.rfind("]") + 1
    if start == -1 or end == 0:
        raise RuntimeError(
            f"Claude did not return a JSON array.\nResponse preview:\n{raw[:400]}"
        )
    suggestions = json.loads(raw[start:end])

    # Validate and deduplicate
    seen: set[str] = {seed_canonical}
    valid: list[dict] = []
    for item in suggestions:
        raw_smi = item.get("smiles", "").strip()
        mol = Chem.MolFromSmiles(raw_smi)
        if mol is None:
            continue
        canonical = Chem.MolToSmiles(mol)
        if canonical in seen:
            continue
        seen.add(canonical)
        valid.append({"smiles": canonical, "rationale": item.get("rationale", "")})

    return valid
