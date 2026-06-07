import streamlit as st
import pandas as pd
from pathlib import Path
import base64
from fragnet.vizualize.viz import FragNetVizApp
from fragnet.vizualize.model import FragNetPreTrainViz
from streamlit_ketcher import st_ketcher
from fragnet.vizualize.model_attr import get_attr_image
from fragnet.vizualize.optimizer import optimize_molecule, mol_to_image
# Initial page config

st.set_page_config(
     page_title='FragNet',
     layout="wide",
     initial_sidebar_state="expanded",
     page_icon="🧪"
)

# Thanks to streamlitopedia for the following code snippet

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

# sidebar
def input_callback():
    st.session_state.input = st.session_state.my_input
# def cs_sidebar():

def predict_cdrp(smiles, cell_line, cell_line_df):
    gene_expr = cell_line_df.loc[cell_line,:].values
    viz.calc_weights_cdrp(smiles, gene_expr)
    prop_prediction = -1   
    return viz, prop_prediction     



def resolve_prop_model(prop_type):

    if prop_type == 'Solubility':
        model_config = './fragnet/exps/ft/pnnl_full/fragnet_hpdl_exp1s_h4pt4_10/config_exp100.yaml'
        chkpt_path = './fragnet/exps/ft/pnnl_full/fragnet_hpdl_exp1s_h4pt4_10/ft_100.pt'
        # model_config = './fragnet/exps/ft/pnnl_set2/fragnet_hpdl_exp1s_h4pt4_10/config_exp100.yaml'
        # chkpt_path = '../fragnet/exps/ft/pnnl_set2/fragnet_hpdl_exp1s_h4pt4_10/ft_100.pt'
        
        viz = FragNetVizApp(model_config, chkpt_path)

        prop_prediction = viz.calc_weights(selected)


    elif prop_type == 'Lipophilicity':
        model_config =  './fragnet/exps/ft/lipo/fragnet_hpdl_exp1s_pt4_30/config_exp100.yaml'
        chkpt_path = './fragnet/exps/ft/lipo/fragnet_hpdl_exp1s_pt4_30/ft_100.pt'
        viz = FragNetVizApp(model_config, chkpt_path)

        prop_prediction = viz.calc_weights(selected)  

    elif prop_type == 'Energy':
        model_config = '../fragnet/fragnet/exps/pt/unimol_exp1s4/config.yaml'
        chkpt_path = '../fragnet/fragnet/exps/pt/unimol_exp1s4/pt.pt'
        viz = FragNetVizApp(model_config, chkpt_path, 'energy')
        prop_prediction = viz.calc_weights(selected)

    return viz, prop_prediction, model_config, chkpt_path

def resolve_DRP(smiles, cell_line, cell_line_df):

    model_config = '../fragnet/fragnet/exps/ft/gdsc/fragnet_hpdl_exp1s_pt4_30/config_exp100.yaml'
    chkpt_path = '../fragnet/fragnet/exps/ft/gdsc/fragnet_hpdl_exp1s_pt4_30/ft_100.pt'
    viz = FragNetVizApp(model_config, chkpt_path,'cdrp')

    # viz, prop_prediction = predict_cdrp(smiles=selected, cell_line=cell_line, cell_line_df=cell_line_df)
    gene_expr = cell_line_df.loc[cell_line,:].values
    viz.calc_weights_cdrp(smiles, gene_expr)
    prop_prediction = -1   

    return viz, prop_prediction, model_config, chkpt_path


# st.sidebar.markdown('''[<img src='data:image/png;base64,{}' class='img-fluid' width=32 height=32>](https://streamlit.io/)'''.format(img_to_bytes("logomark_website.png")), unsafe_allow_html=True)
st.sidebar.title('🧬 FragNet')
st.sidebar.markdown("**Interpretable graph neural network predictions with fragment-based analysis**")
st.sidebar.markdown('---')

st.sidebar.subheader("⚙️ Configuration")
prop_type = st.sidebar.radio(
    "Property Type",
    ["Solubility", "Lipophilicity"],
    captions = ["In logS units", "Lipophilicity coefficient"],
    help="Select the molecular property to predict and visualize"
)

st.sidebar.markdown('---')

        # def input_callback():
        #     st.session_state.input = st.session_state.my_input
        # selected = st.text_input("Input Your Own SMILES :", key="my_input",on_change=input_callback,args=None)

st.sidebar.subheader("📝 Molecule Input")
selected = st.sidebar.text_input(
    "SMILES String", 
    key="my_input",
    on_change=input_callback,
    args=None,
    value="CC1(C)CC(O)CC(C)(C)N1[O]",
    help="Enter a valid SMILES string for the molecule"
)
selected = st_ketcher(selected)

st.sidebar.markdown('---')

# if prop_type=="DRP":

#     cell_line = st.sidebar.selectbox(
#     'Select the cell line identifier',
#     ['DATA.906826',
#     'DATA.687983',
#     'DATA.910927',
#     'DATA.1240138',
#     'DATA.1240139',
#     'DATA.906792',
#     'DATA.910688',
#     'DATA.1240135',
#     'DATA.1290812',
#     'DATA.907045',
#     'DATA.906861',
#     'DATA.906830',
#     'DATA.909750',
#     'DATA.1240137',
#     'DATA.753552',
#     'DATA.907065',
#     'DATA.925338',
#     'DATA.1290809',
#     'DATA.949158',
#     'DATA.924110'])
# cell_line='DATA.924110'
# cell_line_df = pd.read_csv('../fragnet/fragnet/assets/cell_line_data.csv', index_col=0)

#     st.sidebar.write(f'selected cell line: {cell_line}')

if prop_type in ["Solubility", "Lipophilicity", "Energy"]:
    viz, prop_prediction, model_config, chkpt_path = resolve_prop_model(prop_type)


# Display prediction in a prominent metric card
with st.sidebar:
    if prop_type == "Solubility":
        st.metric(label="📊 Predicted Solubility (logS)", value=f"{prop_prediction:.4f}")
    elif prop_type == "Lipophilicity":
        st.metric(label="📊 Predicted Lipophilicity", value=f"{prop_prediction:.4f}")
    elif prop_type == "Energy":
        st.metric(label="📊 Predicted Energy", value=f"{prop_prediction:.4f}")


col1, col2, col3 = st.columns(3)


# hide_bond_weights = st.sidebar.checkbox("Hide bond weights", help="Toggle bond weight visualization")
# hide_atom_weights = st.sidebar.checkbox("Hide atom weights", help="Toggle atom weight visualization")

hide_bond_weights=False
png_frag_attn, png_frag_highlight, frag_w, connection_w, atoms_in_frags = viz.frag_weight_highlight()
png_attr, attr_atom_weights, frag_contributions = get_attr_image(selected, model_config, chkpt_path, prop_type)

def highlight_contribution(val):
    if isinstance(val, (int, float)):
        color = 'lightcoral' if val < 0 else 'lightblue'
        return f'background-color: {color}'
    return ''

def show_contrib_table(df, contrib_col, display_cols, label_cols):
    df_display = df[display_cols].copy()
    df_display['abs_attr'] = df_display[contrib_col].abs()
    df_display = df_display.sort_values('abs_attr', ascending=False).drop('abs_attr', axis=1)
    df_display.columns = label_cols
    st.dataframe(
        df_display.style.applymap(highlight_contribution, subset=['Contribution']),
        hide_index=True,
        use_container_width=True,
        height=350
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("Mean", f"{df[contrib_col].mean():.4f}")
    m2.metric("Max", f"{df[contrib_col].max():.4f}")
    m3.metric("Min", f"{df[contrib_col].min():.4f}")

try:
    with st.spinner("🔄 Calculating contributions..."):
        df_atom_contrib, df_bond_contrib, df_fbond_contrib = viz.get_all_contributions(prop_type)

    tab_atoms, tab_bonds, tab_frags, tab_fconn, tab_opt = st.tabs([
        "⚛️ Atoms", "🔗 Bonds", "🧩 Fragments", "🔀 Fragment Connections", "🔬 Optimizer"
    ])

    with tab_atoms:
        v_col, t_col = st.columns(2)
        with v_col:
            st.subheader("Atom Weights")
            png_atoms, atom_weights = viz.vizualize_atom_weights(True, False)
            st.image(png_atoms, use_column_width=True)
            with st.expander("📊 View Atom Weight Values", expanded=False):
                attn_atoms = pd.DataFrame(atom_weights)
                attn_atoms.index.rename('Atom Index', inplace=True)
                attn_atoms.columns = ['Weight']
                st.dataframe(attn_atoms, use_container_width=True)
        with t_col:
            st.subheader("Atom Contributions")
            st.markdown("Impact of each **atom** on the prediction (masking).")
            show_contrib_table(df_atom_contrib, 'attr',
                               ['atom_index', 'atom_type', 'attr'],
                               ['Atom Index', 'Symbol', 'Contribution'])

    with tab_bonds:
        v_col, t_col = st.columns(2)
        with v_col:
            st.subheader("Bond Weights")
            png_bonds, bond_atom_weights = viz.vizualize_atom_weights(False, True)
            st.image(png_bonds, use_column_width=True)
            with st.expander("📊 View Bond Weight Values", expanded=False):
                bond_atoms = pd.DataFrame(bond_atom_weights)
                bond_atoms.index.rename('Atom Index', inplace=True)
                bond_atoms.columns = ['Weight']
                st.dataframe(bond_atoms, use_container_width=True)
        with t_col:
            st.subheader("Bond Contributions")
            st.markdown("Impact of each **bond** on the prediction (masking).")
            if not df_bond_contrib.empty:
                show_contrib_table(df_bond_contrib, 'attr',
                                   ['bond_index', 'begin_atom', 'end_atom', 'attr'],
                                   ['Bond Index', 'Begin Atom', 'End Atom', 'Contribution'])
            else:
                st.info("No bonds to analyze.")

    with tab_frags:
        # Row 1: Fragment decomposition + atom mapping
        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            st.subheader("Fragment Decomposition")
            st.image(png_frag_highlight, use_column_width=True)
            st.image(png_frag_attn, use_column_width=True)
            st.caption("Attention-based fragment weights")
            with st.expander("📊 View Fragment Weight Values", expanded=False):
                st.dataframe(frag_w, use_container_width=True)
        with row1_col2:
            st.subheader("Fragment Atom Mapping")
            with st.expander("📋 View Fragment Atom Mapping", expanded=True):
                df_atoms_in_frags = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in atoms_in_frags.items()])).T
                df_atoms_in_frags.index.rename('Fragment', inplace=True)
                st.dataframe(df_atoms_in_frags, use_container_width=True)

        st.markdown('---')

        # Row 2: Fragment attribution image + contribution table
        row2_col1, row2_col2 = st.columns(2)
        with row2_col1:
            st.subheader("Fragment Contributions (Visual)")
            st.image(png_attr, use_column_width=True)
            st.caption("Fragment contributions via masking-based attribution")
        with row2_col2:
            st.subheader("Fragment Contributions (Table)")
            st.markdown("Impact of each **fragment** on the prediction (masking).")
            df_frag_contrib = pd.DataFrame(frag_contributions)
            df_frag_contrib['atoms'] = df_frag_contrib['atoms'].apply(lambda x: ', '.join(str(a) for a in x))
            show_contrib_table(df_frag_contrib, 'contribution',
                               ['fragment_index', 'atoms', 'contribution'],
                               ['Fragment Index', 'Atoms', 'Contribution'])

    with tab_fconn:
        # Row 1: Fragment decomposition + atom mapping
        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            st.subheader("Fragment Decomposition")
            st.image(png_frag_highlight, use_column_width=True)
        with row1_col2:
            st.subheader("Fragment Atom Mapping")
            with st.expander("📋 View Fragment Atom Mapping", expanded=True):
                df_atoms_in_frags2 = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in atoms_in_frags.items()])).T
                df_atoms_in_frags2.index.rename('Fragment', inplace=True)
                st.dataframe(df_atoms_in_frags2, use_container_width=True)

        st.markdown('---')

        # Row 2: Connection weights + contribution table
        v_col, t_col = st.columns(2)
        with v_col:
            st.subheader("Connection Weight Values")
            with st.expander("🔗 View Connection Weight Values", expanded=True):
                st.dataframe(connection_w, use_container_width=True)
        with t_col:
            st.subheader("Fragment Connection Contributions")
            st.markdown("Impact of **inter-fragment connections** on the prediction (masking).")
            if not df_fbond_contrib.empty:
                show_contrib_table(df_fbond_contrib, 'attr',
                                   ['fragbond_node_index', 'begin_index', 'end_index', 'attr'],
                                   ['Connection Index', 'Begin Fragment', 'End Fragment', 'Contribution'])
            else:
                st.info("Single fragment molecule — no inter-fragment connections.")

    # -------------------------------------------------------------------------
    # Optimizer tab
    # -------------------------------------------------------------------------
    with tab_opt:
        st.subheader("Contribution-Guided Fragment Optimizer")
        st.markdown(
            "FragNet identifies which fragments hurt the target property most. "
            "The optimizer replaces them with BRICS-compatible alternatives and "
            "re-scores every candidate in a single batch."
        )
        st.markdown("---")

        from fragnet.vizualize.optimizer import get_core_protected_indices, get_fragment_atom_map

        # Pre-computed contributions carried from the Fragments tab
        frag_contrib_df = pd.DataFrame(frag_contributions)
        all_frag_indices = sorted(frag_contrib_df["fragment_index"].tolist())
        frag_atom_map = get_fragment_atom_map(selected)

        # ── Core / scaffold locking ──────────────────────────────────────────
        st.subheader("🔒 Core / Scaffold Lock")
        st.markdown(
            "Fragments overlapping the core will be **protected** from swapping. "
            "Only peripheral fragments will be optimized."
        )
        lock_col1, lock_col2 = st.columns([2, 1])
        with lock_col1:
            core_smiles_input = st.text_input(
                "Core SMILES or SMARTS",
                value="",
                placeholder="e.g. c1ccccc1  or  leave blank to lock by fragment index",
                help="Substructure that must be preserved. Accepts SMILES or SMARTS.",
            )
        with lock_col2:
            manual_lock = st.multiselect(
                "Or lock specific fragment indices",
                options=all_frag_indices,
                default=[],
                help="Fragments selected here will not be swapped regardless of their contribution.",
            )

        # Resolve protected set: union of core-matched + manually locked
        core_protected: set = set()
        core_match_valid = False
        if core_smiles_input.strip():
            core_protected = get_core_protected_indices(selected, core_smiles_input.strip())
            core_match_valid = len(core_protected) > 0
            if not core_match_valid:
                st.warning("Core substructure not found in the molecule — check your SMILES/SMARTS.")

        protected_indices: set = core_protected | set(manual_lock)

        # Fragment status table
        status_rows = []
        for frag in frag_contributions:
            fid = frag["fragment_index"]
            contrib = frag["contribution"]
            atoms = frag_atom_map[fid] if fid < len(frag_atom_map) else []
            locked = fid in protected_indices
            status_rows.append({
                "Frag #": fid,
                "Atom Indices": str(atoms),
                "Contribution": round(contrib, 4),
                "Status": "🔒 Locked" if locked else "🔓 Available",
            })
        status_df = pd.DataFrame(status_rows)

        n_available = len([r for r in status_rows if r["Status"] == "🔓 Available"])
        st.markdown(
            f"**{n_available} / {len(all_frag_indices)} fragments available for swapping**"
            + (f" · {len(protected_indices)} locked" if protected_indices else "")
        )
        st.dataframe(
            status_df.style
            .applymap(highlight_contribution, subset=["Contribution"])
            .applymap(lambda v: "background-color: #ffe0b2" if "🔒" in str(v) else "", subset=["Status"]),
            hide_index=True,
            use_container_width=True,
            height=min(200, 40 + 35 * len(status_rows)),
        )

        # ── Optimization settings ────────────────────────────────────────────
        st.markdown("---")
        st.subheader("⚙️ Settings")
        cfg_col1, cfg_col2, cfg_col3 = st.columns(3)
        with cfg_col1:
            opt_direction = st.selectbox(
                "Optimization direction",
                ["maximize", "minimize"],
                help="maximize → higher value is better (e.g. solubility); minimize → lower is better",
            )
        with cfg_col2:
            n_worst = st.slider(
                "Fragments to target", min_value=1,
                max_value=min(3, max(1, n_available)),
                value=1,
                help="Number of worst-contributing eligible fragments to swap",
            )
        with cfg_col3:
            max_candidates = st.slider(
                "Max candidates to score", min_value=10, max_value=100, value=50, step=10,
                help="More candidates = more coverage but slower",
            )

        # Preview which fragments will actually be targeted
        ascending = (opt_direction == "maximize")
        eligible_df = frag_contrib_df[~frag_contrib_df["fragment_index"].isin(protected_indices)]
        worst_preview = eligible_df.sort_values("contribution", ascending=ascending).head(n_worst)

        st.markdown("**Fragments targeted for replacement:**")
        preview_df = worst_preview[["fragment_index", "atoms", "contribution"]].copy()
        preview_df["atoms"] = preview_df["atoms"].apply(
            lambda x: str(list(x)) if hasattr(x, "__iter__") else str(x)
        )
        preview_df.columns = ["Fragment Index", "Atom Indices", "Contribution"]
        if preview_df.empty:
            st.warning("No eligible fragments to target — all are locked. Unlock some fragments to proceed.")
        else:
            st.dataframe(
                preview_df.style.applymap(highlight_contribution, subset=["Contribution"]),
                hide_index=True,
                use_container_width=True,
            )

        st.markdown("---")
        run_opt = st.button(
            "Run Optimizer",
            type="primary",
            disabled=preview_df.empty,
        )

        if run_opt:
            with st.spinner("Enumerating and scoring candidates… this may take 30–90 s"):
                try:
                    opt_result = optimize_molecule(
                        smiles=selected,
                        viz_app=viz,
                        prop_type=prop_type,
                        direction=opt_direction,
                        n_worst=n_worst,
                        max_candidates=max_candidates,
                        top_k=10,
                        frag_contributions=frag_contributions,
                        seed_prediction=prop_prediction,
                        protected_fragment_indices=protected_indices,
                    )
                    st.session_state["opt_result"] = opt_result
                except Exception as e:
                    st.error(f"Optimization failed: {e}")
                    st.exception(e)

        # Display results (persist across reruns via session_state)
        if "opt_result" in st.session_state:
            res = st.session_state["opt_result"]

            # Sanity check: invalidate stale results if molecule changed
            if res.get("seed_smiles") != selected:
                st.info("Molecule has changed — run the optimizer again.")
            else:
                n_eval = res["n_candidates_evaluated"]
                candidates = res["candidates"]
                seed_val = res["seed_prediction"]

                n_protected = len(res.get("protected_fragment_indices", set()))
                lock_note = f" · {n_protected} fragment(s) locked" if n_protected else ""
                st.success(
                    f"Evaluated **{n_eval}** candidates "
                    f"({res.get('n_eligible_fragments', '?')} eligible fragments{lock_note}). "
                    f"Top {len(candidates)} shown below."
                )

                if not candidates:
                    st.warning(
                        "No valid swap candidates were found. This can happen for single-fragment "
                        "molecules, fully locked scaffolds, or when no BRICS-compatible library "
                        "alternatives exist for the eligible fragments."
                    )
                else:
                    # Summary metrics
                    best = candidates[0]
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Seed prediction", f"{seed_val:.4f}")
                    m2.metric(
                        "Best candidate",
                        f"{best['prediction']:.4f}",
                        delta=f"{best['delta']:+.4f}",
                    )
                    m3.metric("Candidates improved", str(sum(1 for c in candidates if c["improvement"] > 0)))

                    st.markdown("---")
                    st.subheader("Ranked Candidates")

                    # Build results table
                    df_results = pd.DataFrame(candidates)
                    df_results.index += 1
                    df_results.columns = ["SMILES", "Prediction", "Δ vs Seed", "Improvement"]

                    st.dataframe(
                        df_results.style.applymap(
                            highlight_contribution, subset=["Δ vs Seed"]
                        ).background_gradient(subset=["Improvement"], cmap="RdYlGn"),
                        use_container_width=True,
                    )

                    st.markdown("---")
                    st.subheader("Top Candidate Structures")

                    # Draw top candidates in a grid (up to 6)
                    top_n = min(6, len(candidates))
                    cols_per_row = 3
                    for row_start in range(0, top_n, cols_per_row):
                        row_cols = st.columns(cols_per_row)
                        for col_idx, cand_idx in enumerate(range(row_start, min(row_start + cols_per_row, top_n))):
                            cand = candidates[cand_idx]
                            img = mol_to_image(cand["smiles"], width=260, height=180)
                            with row_cols[col_idx]:
                                delta_sign = "+" if cand["delta"] >= 0 else ""
                                st.markdown(
                                    f"**#{cand_idx + 1}** &nbsp; "
                                    f"pred={cand['prediction']:.3f} &nbsp; "
                                    f"Δ={delta_sign}{cand['delta']:.3f}"
                                )
                                if img:
                                    st.image(img, use_column_width=True)
                                st.code(cand["smiles"], language=None)

except Exception as e:
    st.error(f"❌ Error calculating contributions: {str(e)}")
    st.exception(e)

# Footer section
st.markdown('---')
st.sidebar.markdown('---')
st.sidebar.info(f"**Current Molecule:** `{selected}`")