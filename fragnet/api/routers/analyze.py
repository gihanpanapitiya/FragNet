import io
import json
import base64
import numpy as np
from fastapi import APIRouter, HTTPException

from fragnet.api.models import AnalyzeRequest, AnalyzeResponse, FragCentroid
from fragnet.api.dependencies import get_viz_app, MODEL_CONFIGS
from fragnet.vizualize.model_attr import get_attr_image
from fragnet.vizualize.optimizer import get_fragment_atom_map
from fragnet.vizualize.mol_interactive import _compute_flat_2d, _FRAG_PALETTE

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

router = APIRouter()

SVG_W, SVG_H = 600, 400


def _to_records(df) -> list[dict]:
    """DataFrame → plain Python list-of-dicts (numpy scalars converted)."""
    if df is None or (hasattr(df, "empty") and df.empty):
        return []
    return json.loads(df.to_json(orient="records"))


def _sanitise(obj):
    """Recursively convert numpy scalars/arrays to Python native types."""
    if isinstance(obj, dict):
        return {k: _sanitise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitise(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _pil_to_data_uri(img) -> str:
    """Convert PIL Image, raw PNG bytes, or SVG string → data URI."""
    if img is None:
        return ""
    # SVG string (returned by get_attr_image)
    if isinstance(img, str):
        return "data:image/svg+xml;base64," + base64.b64encode(img.encode()).decode()
    # Raw PNG/SVG bytes
    if isinstance(img, bytes):
        # Detect SVG bytes
        if img.lstrip()[:5] in (b"<?xml", b"<svg "):
            return "data:image/svg+xml;base64," + base64.b64encode(img).decode()
        return "data:image/png;base64," + base64.b64encode(img).decode()
    # PIL Image
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _build_interactive_svg(smiles: str, frag_contributions: list) -> tuple[str, list[FragCentroid]]:
    """
    Render molecule as a fragment-coloured SVG and compute fragment centroid
    pixel positions using GetDrawCoords() — same coordinate space as the SVG.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "", []

    _compute_flat_2d(mol)

    frag_atom_map = get_fragment_atom_map(smiles)
    contrib_map = {c["fragment_index"]: c["contribution"] for c in frag_contributions}

    highlight_atoms: list[int] = []
    atom_colors: dict[int, tuple] = {}
    for fid, atoms in enumerate(frag_atom_map):
        rgb = _FRAG_PALETTE[fid % len(_FRAG_PALETTE)]
        for a in atoms:
            if a < mol.GetNumAtoms():
                highlight_atoms.append(a)
                atom_colors[a] = rgb

    drawer = rdMolDraw2D.MolDraw2DSVG(SVG_W, SVG_H)
    drawer.drawOptions().padding = 0.15
    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=atom_colors,
        highlightBonds=[],
        highlightBondColors={},
    )
    drawer.FinishDrawing()

    # Pixel positions in SVG coordinate space (call AFTER FinishDrawing)
    draw_coords: dict[int, tuple[float, float]] = {}
    for i in range(mol.GetNumAtoms()):
        pt = drawer.GetDrawCoords(i)
        draw_coords[i] = (pt.x, pt.y)

    centroids: list[FragCentroid] = []
    for fid, atoms in enumerate(frag_atom_map):
        valid = [a for a in atoms if a in draw_coords]
        if not valid:
            continue
        cx = sum(draw_coords[a][0] for a in valid) / len(valid)
        cy = sum(draw_coords[a][1] for a in valid) / len(valid)
        centroids.append(FragCentroid(
            fragment_index=fid,
            cx=round(cx, 2),
            cy=round(cy, 2),
            contribution=contrib_map.get(fid, 0.0),
            atom_indices=valid,
        ))

    svg_str = drawer.GetDrawingText()
    svg_b64 = base64.b64encode(svg_str.encode()).decode()
    return svg_b64, centroids


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    if req.prop_type not in MODEL_CONFIGS:
        raise HTTPException(400, f"Unknown prop_type '{req.prop_type}'")
    if Chem.MolFromSmiles(req.smiles) is None:
        raise HTTPException(400, "Invalid SMILES string")

    viz = get_viz_app(req.prop_type)
    cfg = MODEL_CONFIGS[req.prop_type]

    try:
        prediction = viz.calc_weights(req.smiles)

        png_frag_attn, png_frag_highlight, frag_w, connection_w, atoms_in_frags = viz.frag_weight_highlight()
        png_atoms, _ = viz.vizualize_atom_weights(True, False)
        png_bonds, _ = viz.vizualize_atom_weights(False, True)
        df_atom, df_bond, df_fbond = viz.get_all_contributions(req.prop_type)
        png_attr, _, frag_contributions = get_attr_image(req.smiles, cfg["config"], cfg["chkpt"], req.prop_type)

        mol_svg_b64, centroids = _build_interactive_svg(req.smiles, frag_contributions)

        return AnalyzeResponse(
            smiles=req.smiles,
            prop_type=req.prop_type,
            prediction=float(prediction),
            unit="logS" if req.prop_type == "Solubility" else "logP",
            mol_svg_b64=mol_svg_b64,
            mol_svg_width=SVG_W,
            mol_svg_height=SVG_H,
            fragment_centroids=centroids,
            frag_atom_map=get_fragment_atom_map(req.smiles),
            atom_contribs=_to_records(df_atom),
            bond_contribs=_to_records(df_bond),
            fbond_contribs=_to_records(df_fbond),
            frag_contribs=_sanitise(frag_contributions),
            frag_weights=_to_records(frag_w) if hasattr(frag_w, "to_dict") else [],
            connection_weights=_to_records(connection_w) if hasattr(connection_w, "to_dict") else [],
            atoms_in_frags=_sanitise({str(k): list(v) for k, v in atoms_in_frags.items()}),
            img_atom_attn=_pil_to_data_uri(png_atoms),
            img_bond_attn=_pil_to_data_uri(png_bonds),
            img_frag_attn=_pil_to_data_uri(png_frag_attn),
            img_frag_highlight=_pil_to_data_uri(png_frag_highlight),
            img_frag_attr=_pil_to_data_uri(png_attr),
        )
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc
