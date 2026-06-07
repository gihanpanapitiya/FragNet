"""
FragNet Dash Application.

Interactive fragment-based molecular property analysis with clickable
fragment locking for the optimizer.

Run with:
    python -m fragnet.vizualize.dash_app
or:
    python fragnet/vizualize/dash_app.py
"""
import io
import base64
import traceback

import dash
from dash import dcc, html, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from fragnet.vizualize.viz import FragNetVizApp
from fragnet.vizualize.model_attr import get_attr_image
from fragnet.vizualize.optimizer import (
    optimize_molecule, mol_to_image, batch_score,
    get_fragment_atom_map,
)
from fragnet.vizualize.mol_interactive import mol_to_interactive_figure
from fragnet.vizualize.llm_suggestions import suggest_replacements


# ─────────────────────────────────────────────────────────────────────────────
# Model configs & lazy loading
# ─────────────────────────────────────────────────────────────────────────────

MODEL_CONFIGS = {
    "Solubility": {
        "config": "./fragnet/exps/ft/pnnl_full/fragnet_hpdl_exp1s_h4pt4_10/config_exp100.yaml",
        "chkpt":  "./fragnet/exps/ft/pnnl_full/fragnet_hpdl_exp1s_h4pt4_10/ft_100.pt",
    },
    "Lipophilicity": {
        "config": "./fragnet/exps/ft/lipo/fragnet_hpdl_exp1s_pt4_30/config_exp100.yaml",
        "chkpt":  "./fragnet/exps/ft/lipo/fragnet_hpdl_exp1s_pt4_30/ft_100.pt",
    },
}

_viz_cache: dict[str, FragNetVizApp] = {}


def get_viz(prop_type: str) -> FragNetVizApp | None:
    if prop_type not in MODEL_CONFIGS:
        return None
    if prop_type not in _viz_cache:
        cfg = MODEL_CONFIGS[prop_type]
        _viz_cache[prop_type] = FragNetVizApp(cfg["config"], cfg["chkpt"])
    return _viz_cache[prop_type]


# ─────────────────────────────────────────────────────────────────────────────
# Image helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_b64(img) -> str | None:
    """Convert PIL Image, raw PNG bytes, or numpy array to base64 PNG string."""
    if img is None:
        return None
    if isinstance(img, bytes):
        return base64.b64encode(img).decode()
    if isinstance(img, str):
        return img  # already encoded
    try:
        from PIL import Image as PILImage
        if isinstance(img, np.ndarray):
            pil = PILImage.fromarray(img.astype("uint8"))
        else:
            pil = img  # assume PIL Image
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def b64_src(b64: str) -> str:
    return f"data:image/png;base64,{b64}"


def _df_safe(obj) -> list:
    """Convert DataFrame or list-of-dicts to list of dicts for dcc.Store."""
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict("records")
    return obj or []


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    assets_folder="assets",
    title="FragNet",
    suppress_callback_exceptions=True,
)

# ── Layout ────────────────────────────────────────────────────────────────────

navbar = dbc.Navbar(
    dbc.Container([
        # Brand
        html.Span("🧬 FragNet", className="navbar-brand fw-bold fs-5 me-4"),

        # SMILES input
        dbc.Col(
            dbc.Input(
                id="smiles-input",
                value="CC1(C)CC(O)CC(C)(C)N1[O]",
                debounce=True,
                placeholder="Enter SMILES…",
                className="smiles-input",
                size="sm",
            ),
            width="auto",
            className="me-3",
        ),

        # Property radio
        dbc.Col(
            dcc.RadioItems(
                id="prop-type",
                options=["Solubility", "Lipophilicity"],
                value="Solubility",
                inline=True,
                className="prop-radio",
            ),
            width="auto",
            className="me-3",
        ),

        # Prediction badge (updated by callback)
        dbc.Col(html.Div(id="prediction-display"), width="auto"),
    ], fluid=True),
    color="primary",
    dark=True,
    sticky="top",
    className="fragnet-navbar py-2",
)

mol_panel = html.Div([
    html.P("Click a fragment to lock / unlock it for the Optimizer",
           className="mol-title"),
    dcc.Loading(
        dcc.Graph(
            id="mol-figure",
            config={"displayModeBar": False, "scrollZoom": False},
            style={"height": "460px"},
        ),
        type="circle",
    ),
    html.Div(id="lock-legend"),
], id="mol-panel")

tab_panel = html.Div([
    dcc.Tabs(
        id="main-tabs",
        value="tab-frags",
        className="custom-tabs",
        children=[
            dcc.Tab(label="⚛️ Atoms",       value="tab-atoms"),
            dcc.Tab(label="🔗 Bonds",       value="tab-bonds"),
            dcc.Tab(label="🧩 Fragments",   value="tab-frags"),
            dcc.Tab(label="🔀 Connections", value="tab-fconn"),
            dcc.Tab(label="🔬 Optimizer",   value="tab-opt"),
        ],
    ),
    dcc.Loading(html.Div(id="tab-content", className="p-3"), type="default"),
], id="tab-panel")

app.layout = html.Div([
    dcc.Store(id="viz-store"),
    dcc.Store(id="locked-frags-store", data=[]),

    navbar,

    dbc.Row([
        dbc.Col(mol_panel, width=5, className="p-0"),
        dbc.Col(tab_panel, width=7, className="p-0"),
    ], className="g-0"),
])


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("viz-store", "data"),
    Output("prediction-display", "children"),
    Input("smiles-input", "value"),
    Input("prop-type", "value"),
)
def run_model(smiles, prop_type):
    """Run FragNet and persist all serialisable results in dcc.Store."""
    if not smiles or not smiles.strip():
        return no_update, dbc.Alert("Enter a SMILES string.", color="secondary", className="small p-2")

    try:
        viz = get_viz(prop_type)
        if viz is None:
            return no_update, dbc.Alert(f"No model for {prop_type}.", color="danger", className="small p-2")

        cfg = MODEL_CONFIGS[prop_type]
        pred = viz.calc_weights(smiles)

        png_frag_attn, png_frag_highlight, frag_w, connection_w, atoms_in_frags = viz.frag_weight_highlight()
        png_atoms, atom_weights = viz.vizualize_atom_weights(True, False)
        png_bonds, bond_atom_weights = viz.vizualize_atom_weights(False, True)
        df_atom, df_bond, df_fbond = viz.get_all_contributions(prop_type)
        png_attr, _, frag_contributions = get_attr_image(smiles, cfg["config"], cfg["chkpt"], prop_type)

        # atom_weights may be a tensor or numpy array
        def _to_list(x):
            if hasattr(x, "detach"):
                return x.detach().numpy().tolist()
            if hasattr(x, "tolist"):
                return x.tolist()
            return list(x)

        store = {
            "smiles": smiles,
            "prop_type": prop_type,
            "prop_prediction": float(pred),
            "model_config": cfg["config"],
            "chkpt_path": cfg["chkpt"],
            # images as base64
            "png_frag_attn":      _to_b64(png_frag_attn),
            "png_frag_highlight": _to_b64(png_frag_highlight),
            "png_atoms":          _to_b64(png_atoms),
            "png_bonds":          _to_b64(png_bonds),
            "png_attr":           _to_b64(png_attr),
            # DataFrames as records
            "frag_w":      _df_safe(frag_w),
            "connection_w": _df_safe(connection_w),
            "df_atom":  _df_safe(df_atom),
            "df_bond":  _df_safe(df_bond),
            "df_fbond": _df_safe(df_fbond),
            # misc
            "atom_weights":      _to_list(atom_weights),
            "bond_atom_weights": _to_list(bond_atom_weights),
            "atoms_in_frags":    {str(k): list(v) for k, v in atoms_in_frags.items()},
            "frag_contributions": frag_contributions,
            "frag_atom_map":     get_fragment_atom_map(smiles),
        }

        unit = "logS" if prop_type == "Solubility" else "logP"
        badge = html.Div([
            html.Span(f"Predicted {prop_type}", className="pred-label"),
            html.Span(f"{pred:.4f} {unit}", className="pred-badge"),
        ])
        return store, badge

    except Exception as exc:
        traceback.print_exc()
        return no_update, dbc.Alert(f"Error: {exc}", color="danger", className="small p-2")


@app.callback(
    Output("mol-figure", "figure"),
    Output("lock-legend", "children"),
    Input("viz-store", "data"),
    Input("locked-frags-store", "data"),
)
def update_mol_figure(store, locked_frags):
    """Redraw the interactive molecule whenever the analysis or lock state changes."""
    if store is None:
        return go.Figure(), None

    locked_set = set(locked_frags or [])
    fig = mol_to_interactive_figure(
        store["smiles"],
        store["frag_atom_map"],
        locked_frags=locked_set,
        contributions=store.get("frag_contributions"),
        width=520,
        height=440,
    )

    n_locked = len(locked_set)
    n_total  = len(store["frag_atom_map"])
    dots = [
        html.Span("● hurts",   className="legend-dot", style={"color": "#e74c3c"}),
        html.Span("● helps",   className="legend-dot", style={"color": "#27ae60"}),
        html.Span("● neutral", className="legend-dot", style={"color": "#2980b9"}),
        html.Span("■ locked",  className="legend-dot", style={"color": "#7f8c8d"}),
    ]
    lock_note = (
        dbc.Badge(f"{n_locked} locked", color="secondary", className="ms-2")
        if n_locked else None
    )
    legend = html.Div([
        html.Div(dots, className="lock-legend-bar"),
        html.Div(
            [html.Small(f"{n_total} fragments detected", className="text-muted"), lock_note],
            className="text-center mt-1",
        ),
    ])
    return fig, legend


@app.callback(
    Output("locked-frags-store", "data"),
    Input("mol-figure", "clickData"),
    State("locked-frags-store", "data"),
    prevent_initial_call=True,
)
def toggle_lock(click_data, locked_frags):
    """Toggle a fragment's lock state when its marker is clicked."""
    if click_data is None:
        return locked_frags or []
    locked_set = set(locked_frags or [])
    try:
        fid = int(click_data["points"][0]["customdata"])
        if fid in locked_set:
            locked_set.discard(fid)
        else:
            locked_set.add(fid)
    except (KeyError, IndexError, TypeError, ValueError):
        pass
    return list(locked_set)


@app.callback(
    Output("tab-content", "children"),
    Input("main-tabs", "value"),
    Input("viz-store", "data"),
    Input("locked-frags-store", "data"),
)
def render_tab(tab, store, locked_frags):
    if store is None:
        return dbc.Alert("Enter a SMILES string to begin.", color="info")
    locked_set = set(locked_frags or [])

    if tab == "tab-atoms":
        return _tab_atoms(store)
    if tab == "tab-bonds":
        return _tab_bonds(store)
    if tab == "tab-frags":
        return _tab_frags(store)
    if tab == "tab-fconn":
        return _tab_fconn(store)
    if tab == "tab-opt":
        return _tab_opt(store, locked_set)
    return html.P("Unknown tab.")


# ─────────────────────────────────────────────────────────────────────────────
# Tab renderers (pure layout functions, no callbacks)
# ─────────────────────────────────────────────────────────────────────────────

def _card(children, className=""):
    return dbc.Card(children, className=f"analysis-card {className}", body=True)


def _section(title, children):
    return html.Div([
        html.P(title, className="section-title"),
        children,
    ])


def _contrib_table(records: list, contrib_col: str, cols: list, labels: list):
    df = pd.DataFrame(records)
    if df.empty or contrib_col not in df.columns:
        return dbc.Alert("No data.", color="secondary", className="small")
    df = df[cols].copy()
    df["_abs"] = df[contrib_col].abs()
    df = df.sort_values("_abs", ascending=False).drop("_abs", axis=1)
    df.columns = labels
    return dbc.Table.from_dataframe(
        df.head(20), striped=True, bordered=False,
        hover=True, size="sm", responsive=True,
        className="contrib-table",
    )


def _tab_atoms(store):
    return dbc.Row([
        dbc.Col(_card(_section("Atom Attention Weights",
            html.Img(src=b64_src(store["png_atoms"]), className="mol-img"),
        )), width=6),
        dbc.Col(_card(_section("Atom Contributions (masking)",
            _contrib_table(store["df_atom"], "attr",
                           ["atom_index", "atom_type", "attr"],
                           ["Idx", "Symbol", "Contribution"]),
        )), width=6),
    ])


def _tab_bonds(store):
    return dbc.Row([
        dbc.Col(_card(_section("Bond Attention Weights",
            html.Img(src=b64_src(store["png_bonds"]), className="mol-img"),
        )), width=6),
        dbc.Col(_card(_section("Bond Contributions (masking)",
            _contrib_table(store["df_bond"], "attr",
                           ["bond_index", "begin_atom", "end_atom", "attr"],
                           ["Idx", "Begin", "End", "Contribution"]),
        )), width=6),
    ])


def _tab_frags(store):
    df_frag = pd.DataFrame(store["frag_contributions"])
    if "atoms" in df_frag.columns:
        df_frag["atoms"] = df_frag["atoms"].apply(lambda x: ", ".join(str(a) for a in x))

    atoms_in_frags = store["atoms_in_frags"]
    df_map = pd.DataFrame(
        {k: pd.Series(v) for k, v in atoms_in_frags.items()}
    ).T
    df_map.index.rename("Fragment", inplace=True)

    return html.Div([
        dbc.Row([
            dbc.Col(_card(html.Div([
                _section("Fragment Decomposition",
                    html.Img(src=b64_src(store["png_frag_highlight"]), className="mol-img")),
                html.Div(style={"height": "8px"}),
                _section("Fragment Attention Weights",
                    html.Img(src=b64_src(store["png_frag_attn"]), className="mol-img")),
            ])), width=6),
            dbc.Col(_card(_section("Fragment Atom Mapping",
                dbc.Table.from_dataframe(df_map.reset_index(), striped=True, size="sm",
                                         responsive=True, hover=True, className="contrib-table"),
            )), width=6),
        ]),
        dbc.Row([
            dbc.Col(_card(_section("Fragment Attribution (visual)",
                html.Img(src=b64_src(store["png_attr"]), className="mol-img"),
            )), width=6),
            dbc.Col(_card(_section("Fragment Contributions (masking)",
                _contrib_table(df_frag.to_dict("records"), "contribution",
                               ["fragment_index", "atoms", "contribution"],
                               ["Frag #", "Atoms", "Contribution"]),
            )), width=6),
        ]),
    ])


def _tab_fconn(store):
    conn_df = pd.DataFrame(store["connection_w"])
    return dbc.Row([
        dbc.Col(_card(html.Div([
            _section("Fragment Decomposition",
                html.Img(src=b64_src(store["png_frag_highlight"]), className="mol-img")),
            html.Div(style={"height": "8px"}),
            _section("Connection Weights",
                dbc.Table.from_dataframe(conn_df, striped=True, size="sm",
                                         responsive=True, className="contrib-table")
                if not conn_df.empty
                else dbc.Alert("No connection data.", color="secondary", className="small"),
            ),
        ])), width=6),
        dbc.Col(_card(_section("Fragment Connection Contributions",
            _contrib_table(store["df_fbond"], "attr",
                           ["fragbond_node_index", "begin_index", "end_index", "attr"],
                           ["Conn #", "Frag A", "Frag B", "Contribution"])
            if store["df_fbond"]
            else dbc.Alert("Single fragment — no inter-fragment connections.",
                           color="secondary", className="small"),
        )), width=6),
    ])


def _tab_opt(store, locked_set: set):
    frag_contributions = store["frag_contributions"]
    frag_atom_map      = store["frag_atom_map"]
    all_fids           = sorted(c["fragment_index"] for c in frag_contributions)
    n_available        = sum(1 for fid in all_fids if fid not in locked_set)

    # Fragment status rows
    status_rows = []
    for frag in frag_contributions:
        fid   = frag["fragment_index"]
        atoms = frag_atom_map[fid] if fid < len(frag_atom_map) else []
        contrib = frag["contribution"]
        status_rows.append({
            "Frag":        fid,
            "Atoms":       str(atoms),
            "Contribution": round(contrib, 4),
            "Status":      "🔒 Locked" if fid in locked_set else "🔓 Available",
        })

    # Settings card
    settings_card = html.Div([
        dbc.Row([
            dbc.Col([
                html.Label("Direction", className="small fw-bold"),
                dcc.Dropdown(
                    id="opt-direction",
                    options=[{"label": "Maximize ↑", "value": "maximize"},
                             {"label": "Minimize ↓", "value": "minimize"}],
                    value="maximize", clearable=False,
                ),
            ], width=4),
            dbc.Col([
                html.Label("Fragments to target", className="small fw-bold"),
                dcc.Slider(
                    id="opt-n-worst",
                    min=1, max=min(3, max(1, n_available)), value=1, step=1,
                    marks={i: str(i) for i in range(1, min(4, n_available + 1))},
                    className="mt-1",
                ),
            ], width=4),
            dbc.Col([
                html.Label("Max candidates", className="small fw-bold"),
                dcc.Slider(
                    id="opt-max-cands",
                    min=10, max=100, value=50, step=10,
                    marks={i: str(i) for i in range(10, 101, 20)},
                    className="mt-1",
                ),
            ], width=4),
        ]),
    ], className="opt-settings-card")

    avail_badge = dbc.Badge(
        f"{n_available} / {len(all_fids)} fragments available",
        color="primary" if n_available > 0 else "secondary",
        className="me-2",
    )
    lock_badge = (
        dbc.Badge(f"{len(locked_set)} locked", color="secondary")
        if locked_set else None
    )

    return html.Div([
        # ── Header ────────────────────────────────────────────────────────
        dbc.Row([
            dbc.Col([
                html.H5("🔬 Fragment Optimizer", className="mb-1"),
                html.P(
                    "Lock fragments to preserve in the molecule panel, "
                    "then run the BRICS optimizer or ask Claude for suggestions.",
                    className="text-muted small mb-0",
                ),
            ]),
        ], className="mb-3"),

        # ── Fragment status ────────────────────────────────────────────────
        _card(html.Div([
            html.Div([avail_badge, lock_badge], className="mb-2"),
            dbc.Table.from_dataframe(
                pd.DataFrame(status_rows), striped=True, size="sm",
                hover=True, responsive=True, className="contrib-table",
            ),
        ])),

        # ── Settings ──────────────────────────────────────────────────────
        _card(html.Div([
            html.P("⚙️ Settings", className="section-title mb-2"),
            settings_card,
        ])),

        # ── BRICS optimizer ────────────────────────────────────────────────
        dbc.Button(
            "🚀 Run BRICS Optimizer", id="run-opt-btn", color="primary",
            disabled=(n_available == 0), className="me-2",
        ),
        dcc.Loading(html.Div(id="opt-results", className="mt-3"), type="default"),

        html.Hr(className="my-4"),

        # ── LLM suggestions ────────────────────────────────────────────────
        _card(html.Div([
            html.P("🤖 LLM-Guided Suggestions", className="section-title mb-1"),
            html.P(
                "Claude reasons about the chemistry and proposes targeted modifications. "
                "Each suggestion is then scored by FragNet.",
                className="text-muted small mb-2",
            ),
            dbc.Alert(
                [html.I(className="me-1"), "Requires ", html.Code("ANTHROPIC_API_KEY"),
                 " in the environment."],
                color="light", className="small py-1 mb-2",
            ),
            dbc.Button(
                "✨ Get LLM Suggestions", id="llm-suggest-btn", color="secondary",
                disabled=(n_available == 0),
            ),
        ])),
        dcc.Loading(html.Div(id="llm-results", className="mt-3"), type="default"),
    ])


@app.callback(
    Output("opt-results", "children"),
    Input("run-opt-btn", "n_clicks"),
    State("viz-store", "data"),
    State("locked-frags-store", "data"),
    State("opt-direction", "value"),
    State("opt-n-worst", "value"),
    State("opt-max-cands", "value"),
    prevent_initial_call=True,
)
def run_optimizer(_, store, locked_frags, direction, n_worst, max_cands):
    if store is None:
        return dbc.Alert("Run the model analysis first.", color="warning", className="small")

    protected = set(locked_frags or [])
    try:
        viz    = get_viz(store["prop_type"])
        result = optimize_molecule(
            smiles=store["smiles"],
            viz_app=viz,
            prop_type=store["prop_type"],
            direction=direction or "maximize",
            n_worst=n_worst or 1,
            max_candidates=max_cands or 50,
            top_k=10,
            frag_contributions=store["frag_contributions"],
            seed_prediction=store["prop_prediction"],
            protected_fragment_indices=protected,
        )
    except Exception as exc:
        traceback.print_exc()
        return dbc.Alert(f"Optimization failed: {exc}", color="danger", className="small")

    candidates = result["candidates"]
    n_eval     = result["n_candidates_evaluated"]
    n_lock     = len(protected)
    seed_val   = result["seed_prediction"]

    banner = dbc.Alert(
        f"Evaluated {n_eval} candidates · "
        f"{n_lock} fragment(s) locked · "
        f"Top {len(candidates)} shown.",
        color="success", className="small py-2",
    )

    if not candidates:
        return html.Div([
            banner,
            dbc.Alert(
                "No valid swap candidates found. This happens for single-fragment molecules, "
                "fully locked scaffolds, or when no BRICS-compatible library alternatives exist.",
                color="warning", className="small",
            ),
        ])

    best = candidates[0]
    n_improved = sum(1 for c in candidates if c["improvement"] > 0)

    def _metric_card(label, value, sub=None):
        return dbc.Card(dbc.CardBody([
            html.P(label, className="metric-label"),
            html.Div(value, className="metric-value"),
            html.Small(sub, className=f"text-{'success' if '+' in str(sub) else 'danger'}")
            if sub else None,
        ]), className="metric-card")

    metrics_row = dbc.Row([
        dbc.Col(_metric_card("Seed", f"{seed_val:.4f}"), width=4),
        dbc.Col(_metric_card(
            "Best Candidate",
            f"{best['prediction']:.4f}",
            sub=f"Δ {best['delta']:+.4f}",
        ), width=4),
        dbc.Col(_metric_card("Improved", str(n_improved)), width=4),
    ], className="mb-3")

    df_res = pd.DataFrame(candidates)
    df_res.index += 1
    df_res.columns = ["SMILES", "Prediction", "Δ vs Seed", "Improvement"]
    results_table = _card(_section("Ranked Candidates",
        dbc.Table.from_dataframe(df_res, striped=True, size="sm",
                                  hover=True, responsive=True, className="contrib-table"),
    ))

    # Top 6 structures
    top_n = min(6, len(candidates))
    grid_rows = []
    for row_start in range(0, top_n, 3):
        cols = []
        for i in range(row_start, min(row_start + 3, top_n)):
            cand = candidates[i]
            img  = mol_to_image(cand["smiles"], width=260, height=180)
            b64  = _to_b64(img)
            sign = "+" if cand["delta"] >= 0 else ""
            improved = cand["improvement"] > 0
            cols.append(dbc.Col(dbc.Card([
                html.Div(
                    f"#{i+1}  pred={cand['prediction']:.3f}  Δ={sign}{cand['delta']:.3f}",
                    className="candidate-header-positive" if improved else "candidate-header-neutral",
                ),
                dbc.CardBody([
                    html.Img(src=b64_src(b64), style={"width": "100%"}) if b64 else html.P("N/A"),
                    html.Span(cand["smiles"], className="candidate-smiles"),
                ], className="p-2"),
            ], className="candidate-card"), width=4, className="mb-3"))
        grid_rows.append(dbc.Row(cols))

    return html.Div([
        banner,
        metrics_row,
        results_table,
        html.P("Top Structures", className="section-title mt-3"),
        *grid_rows,
    ])


@app.callback(
    Output("llm-results", "children"),
    Input("llm-suggest-btn", "n_clicks"),
    State("viz-store", "data"),
    State("locked-frags-store", "data"),
    State("opt-direction", "value"),
    State("opt-n-worst", "value"),
    prevent_initial_call=True,
)
def run_llm_suggestions(_, store, locked_frags, direction, n_worst):
    if store is None:
        return dbc.Alert("Run the model analysis first.", color="warning", className="small")

    protected = set(locked_frags or [])
    direction  = direction or "maximize"
    n_worst    = n_worst or 1

    try:
        suggestions = suggest_replacements(
            smiles=store["smiles"],
            frag_contributions=store["frag_contributions"],
            frag_atom_map=store["frag_atom_map"],
            prop_type=store["prop_type"],
            direction=direction,
            seed_prediction=store["prop_prediction"],
            protected_indices=protected,
            n_worst=n_worst,
            n_suggestions=8,
        )
    except EnvironmentError as exc:
        return dbc.Alert(str(exc), color="warning", className="small")
    except Exception as exc:
        traceback.print_exc()
        return dbc.Alert(f"LLM suggestion failed: {exc}", color="danger", className="small")

    if not suggestions:
        return dbc.Alert("Claude returned no valid SMILES.", color="secondary", className="small")

    # Score with FragNet
    viz = get_viz(store["prop_type"])
    seed_val = store["prop_prediction"]
    try:
        scores = batch_score(viz, [s["smiles"] for s in suggestions])
    except Exception:
        scores = {}

    # Merge scores and rank
    results = []
    for s in suggestions:
        smi  = s["smiles"]
        pred = scores.get(smi)
        if pred is not None:
            delta       = pred - seed_val
            improvement = delta if direction == "maximize" else -delta
        else:
            delta = improvement = None
        results.append({
            "smiles":      smi,
            "rationale":   s["rationale"],
            "prediction":  pred,
            "delta":       delta,
            "improvement": improvement,
        })
    results.sort(key=lambda x: x["improvement"] if x["improvement"] is not None else -999,
                 reverse=True)

    n_scored   = sum(1 for r in results if r["prediction"] is not None)
    n_improved = sum(1 for r in results if (r["improvement"] or 0) > 0)

    banner = dbc.Alert(
        f"Claude proposed {len(results)} molecules · {n_scored} scored by FragNet "
        f"· {n_improved} improved over seed ({seed_val:.4f})",
        color="info", className="small py-2",
    )

    cards = []
    for i, r in enumerate(results):
        img = mol_to_image(r["smiles"], width=260, height=180)
        b64 = _to_b64(img)
        improved = (r["improvement"] or 0) > 0

        if r["prediction"] is not None:
            sign      = "+" if r["delta"] >= 0 else ""
            score_txt = f"pred={r['prediction']:.3f}  Δ={sign}{r['delta']:.3f}"
        else:
            score_txt = "score unavailable"

        cards.append(dbc.Col(dbc.Card([
            html.Div(f"#{i+1}  {score_txt}",
                     className="candidate-header-positive" if improved
                     else "candidate-header-neutral"),
            dbc.CardBody([
                html.Img(src=b64_src(b64), style={"width": "100%"}) if b64 else None,
                html.Small(r["rationale"], className="text-muted d-block mt-1 fst-italic"),
                html.Span(r["smiles"], className="candidate-smiles"),
            ], className="p-2"),
        ], className="candidate-card"), width=4, className="mb-3"))

    grid = [dbc.Row(cards[i: i + 3]) for i in range(0, len(cards), 3)]
    return html.Div([banner, *grid])


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
