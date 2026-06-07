"""
Interactive RDKit molecule visualization for Dash via Plotly.

Renders the molecule as a fragment-coloured PNG background image and overlays
clickable scatter markers at each fragment's centroid.  Clicking a marker
toggles the lock state; the caller is responsible for storing that state.
"""
import io
import base64

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Geometry import rdGeometry
import plotly.graph_objects as go

# Pastel palette — one colour per fragment (cycles if >8 frags)
_FRAG_PALETTE = [
    (0.55, 0.78, 1.00),  # blue
    (1.00, 0.78, 0.55),  # orange
    (0.55, 1.00, 0.65),  # green
    (1.00, 0.60, 0.75),  # pink
    (0.80, 0.60, 1.00),  # purple
    (1.00, 1.00, 0.55),  # yellow
    (0.55, 1.00, 1.00),  # cyan
    (1.00, 0.75, 0.75),  # salmon
]


def _compute_flat_2d(mol) -> None:
    """
    Compute 2-D coordinates and rotate them so the molecule's principal axis
    is horizontal, giving a flat landscape orientation.

    Uses CoordGen (Schrodinger algorithm) if compiled into RDKit, which
    produces better layouts for drug-like molecules; falls back to the
    built-in rdDepictor otherwise.

    Modifies mol's conformer in-place.
    """
    # Prefer CoordGen — significantly better layouts for fused/complex rings
    try:
        from rdkit.Chem import rdCoordGen
        rdCoordGen.AddCoords(mol)
    except (ImportError, AttributeError, Exception):
        rdDepictor.Compute2DCoords(mol, canonOrient=True)

    if mol.GetNumConformers() == 0 or mol.GetNumAtoms() < 3:
        return

    conf = mol.GetConformer()
    n = mol.GetNumAtoms()
    pts = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y]
                    for i in range(n)])

    # PCA: find the direction of maximum variance and rotate it to the x-axis
    centre = pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts - centre, full_matrices=False)
    principal = Vt[0]                          # unit vector of max spread
    angle = -np.arctan2(principal[1], principal[0])

    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    pts_rot = (pts - centre) @ R.T + centre

    # Write back into the conformer
    for i in range(n):
        conf.SetAtomPosition(i, rdGeometry.Point3D(
            float(pts_rot[i, 0]), float(pts_rot[i, 1]), 0.0
        ))


def _render_highlighted_png(mol, frag_atom_map: list, locked_frags: set,
                             width: int, height: int) -> bytes:
    """
    Render molecule as PNG using the 2-D coords already on ``mol``.
    Does NOT recompute coordinates — call _compute_flat_2d first.
    """
    highlight_atoms: list[int] = []
    atom_colors: dict[int, tuple] = {}
    for fid, atoms in enumerate(frag_atom_map):
        rgb = _FRAG_PALETTE[fid % len(_FRAG_PALETTE)]
        if fid in locked_frags:
            rgb = tuple(0.4 * c + 0.6 * 0.65 for c in rgb)
        for a in atoms:
            if a < mol.GetNumAtoms():
                highlight_atoms.append(a)
                atom_colors[a] = rgb
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    drawer.drawOptions().padding = 0.15
    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=atom_colors,
        highlightBonds=[],
        highlightBondColors={},
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def mol_to_interactive_figure(
    smiles: str,
    frag_atom_map: list,
    locked_frags: set | None = None,
    contributions: list | None = None,
    width: int = 440,
    height: int = 320,
) -> go.Figure:
    """
    Return a Plotly Figure with:
    - Fragment-coloured molecule as background (flat / horizontal orientation).
    - Clickable scatter markers at each fragment centroid.
    - Red   = hurts the property  (contribution < -0.05)
    - Green = helps the property  (contribution > +0.05)
    - Blue  = neutral
    - Grey  = locked (will not be swapped)

    Each marker's ``customdata`` is the integer fragment index so the
    Dash ``clickData`` callback can read it directly.
    """
    locked_frags = locked_frags or set()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return go.Figure()

    # ── Single coordinate computation, shared by PNG and Plotly overlay ──
    _compute_flat_2d(mol)

    # Render PNG from the normalised conformer
    png_bytes = _render_highlighted_png(mol, frag_atom_map, locked_frags, width, height)
    img_b64 = base64.b64encode(png_bytes).decode()

    # Extract the same coordinates for the Plotly overlay
    if mol.GetNumConformers() == 0:
        return go.Figure()
    conf = mol.GetConformer()
    atom_coords = {i: (conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y)
                   for i in range(mol.GetNumAtoms())}

    xs = [v[0] for v in atom_coords.values()]
    ys = [v[1] for v in atom_coords.values()]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    pad_x = max(0.5, (x_max - x_min) * 0.18)
    pad_y = max(0.5, (y_max - y_min) * 0.18)

    contrib_map: dict[int, float] = {}
    if contributions:
        for c in contributions:
            contrib_map[c["fragment_index"]] = c["contribution"]

    fig = go.Figure()

    # Background: molecule PNG mapped exactly onto the RDKit coordinate space
    fig.add_layout_image(dict(
        source=f"data:image/png;base64,{img_b64}",
        xref="x", yref="y",
        x=x_min - pad_x,
        y=y_max + pad_y,
        sizex=(x_max - x_min) + 2 * pad_x,
        sizey=(y_max - y_min) + 2 * pad_y,
        sizing="stretch",
        layer="below",
    ))

    # One clickable scatter marker per fragment
    for fid, atoms in enumerate(frag_atom_map):
        coords = [(atom_coords[a][0], atom_coords[a][1]) for a in atoms if a in atom_coords]
        if not coords:
            continue
        cx = sum(x for x, _ in coords) / len(coords)
        cy = sum(y for _, y in coords) / len(coords)

        contrib = contrib_map.get(fid, 0.0)
        is_locked = fid in locked_frags

        if is_locked:
            color, symbol, label = "#7f8c8d", "square", "🔒"
        elif contrib < -0.05:
            color, symbol, label = "#e74c3c", "circle", f"F{fid}"
        elif contrib > 0.05:
            color, symbol, label = "#27ae60", "circle", f"F{fid}"
        else:
            color, symbol, label = "#2980b9", "circle", f"F{fid}"

        action = "Click to UNLOCK" if is_locked else "Click to LOCK"
        fig.add_trace(go.Scatter(
            x=[cx], y=[cy],
            mode="markers+text",
            marker=dict(size=26, color=color, opacity=0.72,
                        symbol=symbol, line=dict(width=2, color="rgba(0,0,0,0.5)")),
            text=[label],
            textposition="middle center",
            textfont=dict(size=9, color="white"),
            name=f"Frag {fid}",
            customdata=[fid],
            hovertemplate=(
                f"<b>Fragment {fid}</b><br>"
                f"Contribution: {contrib:+.4f}<br>"
                f"{action}<extra></extra>"
            ),
        ))

    fig.update_layout(
        xaxis=dict(range=[x_min - pad_x, x_max + pad_x], visible=False, constrain="domain"),
        yaxis=dict(range=[y_min - pad_y, y_max + pad_y], visible=False, scaleanchor="x"),
        margin=dict(l=0, r=0, t=0, b=0),
        height=height,
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        dragmode=False,
    )
    return fig
