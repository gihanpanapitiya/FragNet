# FragNet

FragNet is a Graph Neural Network designed for molecular property prediction, that can offer insights into how different substructures influence the predictions. More details of FragNet can be found in our paper,

[FragNet: A Graph Neural Network for Molecular Property Prediction with Four Levels of Interpretability](https://pubs.acs.org/doi/10.1021/jacs.5c22620)

[Arxiv version: https://arxiv.org/abs/2410.12156](https://arxiv.org/abs/2410.12156)

<img src="fragnet/assets/fragnet.png" alt="drawing" width="500"/>

Figure 1: FragNet’s architecture and data representation. (a) Atom and Fragment graphs’
edge features are learned from Bond and Fragment connection graphs respectively. b) Initial
fragment features for the fragment graph are the summation of the updated atom features
that compose the fragment. (c) Illustration of FragNet’s message passing taking place be-
tween two non-covalently bonded substructures. Fragment-Fragment connections are also
present between adjacent fragments in each non-covalently bonded structure of the com-
pound.

<img src="fragnet/assets/weights_main.png" alt="drawing" width="500"/>

Figure 2: Different types of attention weights and contribution values available in FragNet visualized for CC[NH+](CCCl)CCOc1cccc2ccccc12.[Cl-] with atom, bond, and fragment at-
tention weights shown in (a),(b), and (c) and fragment contribution values shown in (d).
The top table provides the atom to fragment mapping and the bottom table provides the
fragment connection attention weights. Atom and bond attention weights are scaled to val-
ues between 0 and 1. The fragment and fragment connection weights are not scaled. The
numbers in blue boxes in (d) correspond to Fragment IDs in ‘Atoms in Fragments’ table.
## Quick Start

**New to FragNet?** The fastest way to get started:

1. **Install FragNet** (see Installation section below)
2. **Explore visualizations**: See how atoms, bonds, and fragments contribute to predictions

For the web application or advanced usage, continue reading below.
# Usage

### Installation

The installation has been tested with python 3.11 and cuda 12.1

#### For CPU

1. Create a python 3.11 virtual environment and install the required packages using the command `pip install -r requirements.txt`
2. Install torch-scatter using `pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cpu.html`
3. Next install FragNet. In the directory where `setup.py` is, run the command `pip install .`

Alternatively and more conveniently, you can run `bash install_cpu.sh` which will install FragNet and create pretraining and finetuning data for ESOL dataset.

#### For GPU

1. Create a python 3.11 virtual environment and install the required packages using the command `pip instal -r requirements.txt`
2. Install torch-scatter using `pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu121.html`
3. Next install FragNet. In the directory where `setup.py` is, run the command `pip install .`

Alternatively do `bash install_gpu.sh`.

-------

### Creating pretraining data

FragNet was pretrained using part of the data used by [UniMol](https://github.com/deepmodeling/Uni-Mol/tree/main/unimol).

Here, we use ESOL dataset to demonstrate the data creation. The following commands should be run at the `FragNet/fragnet` directory.

First, create a directory to save data.

```mkdir -p finetune_data/moleculenet/esol/raw/```

Next, download ESOL dataset.

```
wget -O finetune_data/moleculenet/esol/raw/delaney-processed.csv https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv
```

Next, run the following command to create pretraining data.

```
python data_create/create_pretrain_datasets.py --save_path pretrain_data/esol --data_type exp1s --maxiters 500 --raw_data_path finetune_data/moleculenet/esol/raw/delaney-processed.csv
```


- save_path: where the datasets should be saved
- data_type: use exp1s for all the calculations 
- maxiters: maximum number of iterations for 3D coordinate generation
- raw_data_path: location of the smiles dataset

------

### Creating finetuning data

Creating data for finetuning for MoleculeNet datasets can be done as follows,


`python data_create/create_finetune_datasets.py --dataset_name moleculenet --dataset_subset esol --use_molebert True --output_dir finetune_data/moleculenet_exp1s --data_dir finetune_data/moleculenet --data_type exp1s`


- dataset_name: dataset type
- dataset_subset: dataset sub-type
- use_molebert: whether to use the dataset splitting method used by MoleBert model

------

### Pretrain

To pretrain run the following command. All the input parameters have to be given in a config file.

```
python train/pretrain/pretrain_gat2.py --config exps/pt/unimol_exp1s4/config.yaml
```

------

### Finetune
```
python train/finetune/finetune_gat2.py --config exps/ft/esol/e1pt4.yaml
```


------

## Interactive Applications

FragNet provides multiple user-friendly ways to interact with the model and visualize molecular property predictions:

### Option 1: Interactive Jupyter Notebooks (Recommended for Beginners)

We provide two ready-to-use Jupyter notebooks that require no additional setup beyond installation:

- **[fragnet/notebooks/FragNet.ipynb](fragnet/notebooks/FragNet.ipynb)** - A streamlined notebook for quick interpretability analysis

**How to use:**
```bash
# Navigate to the FragNet directory
cd /path/to/FragNet

# Launch Jupyter
jupyter notebook

# Open either FragNet_Interactive_Demo.ipynb or fragnet/notebooks/interprete.ipynb
# Run all cells (Cell → Run All) and follow the inline instructions
```

These notebooks provide:
- Step-by-step guidance with clear explanations
- Pre-configured examples you can run immediately
- Visualizations of atom weights, bond contributions, and fragment attributions
- Easy modification of SMILES strings to analyze your own molecules

### Option 2: Interactive Web Application

For a browser-based GUI experience, you can launch the Streamlit application:

**Prerequisites:**
- Ensure FragNet is installed (see Installation section above)
- Install Streamlit: `pip install streamlit streamlit-ketcher`

**To launch the application:**
```bash
# From the root FragNet directory
streamlit run fragnet/vizualize/app.py
```

The application will open in your browser at `http://localhost:8501`

**Features:**
- Draw molecules directly in the browser using the Ketcher molecular editor
- Input SMILES strings for prediction
- Select different property types (Solubility, Lipophilicity, Energy, Drug Response)
- Interactive visualizations of molecular interpretability

<img src="fragnet/assets/app.png" alt="drawing" width="500"/>

**Troubleshooting:**
- If the app doesn't open automatically, manually navigate to `http://localhost:8501`
- Ensure no other applications are using port 8501
- Check that all dependencies are installed: `pip install -r requirements.txt`

------

## Optional
### Hyperparameter tuning
```
python  hp/hpoptuna.py --config exps/ft/esol/e1pt4.yaml --n_trials 10 \
--chkpt hpruns/pt.pt --seed 10 --ft_epochs 10 --prune 1
```

- config: initial parameters
- n_trials: number of hp optimization trails
- chkpt: this is where the checkoint during hp optimization will be saved. Note that you will have to create an output directory for this (in this case hpruns). Otherwise the output directory is assumed to be the current working directory.
- seed: random seed
- ft_epochs: number of training epochs
- prune: For Optuna runs. Whether to prune an optimization.



## Citation
If you use our work, please cite it as,

```
@article{doi:10.1021/jacs.5c22620,
author = {Panapitiya, Gihan and Gao, Peiyuan and Maupin, C. Mark and Saldanha, Emily G.},
title = {FragNet: A Graph Neural Network for Molecular Property Prediction with Four Levels of Interpretability},
journal = {Journal of the American Chemical Society},
volume = {148},
number = {9},
pages = {9930-9950},
year = {2026},
doi = {10.1021/jacs.5c22620},note ={PMID: 41738545},
URL = { https://doi.org/10.1021/jacs.5c22620},
eprint = { https://doi.org/10.1021/jacs.5c22620}
}
```

## Docker

To run the FragNet Streamlit app in a Docker container:

1. Build the Docker image:
      ```sh
      docker build -t fragnet-app .
      ```
2. Run the container:
      ```sh
      docker run -p 8501:8501 fragnet-app
      ```

This will start the app at [http://localhost:8501](http://localhost:8501).

**Note:** The Dockerfile installs required system libraries (e.g., `libxrender1`, `libxext6`) and Python build tools for compatibility with scientific packages. If you encounter missing library errors, install the relevant system package in the Dockerfile.

## Running the Visualizer with Optimizer

The web application pairs a React frontend with a FastAPI backend. Both must run
simultaneously. All commands are run from the **project root** (`FragNet/`).

### Prerequisites

**Python environment** (same environment used to install FragNet):
```bash
pip install fastapi uvicorn python-dotenv
```

**Node.js** (v18+ recommended):
```bash
cd frontend
npm install       # installs React, Vite, Mantine, etc.
cd ..
```

**Model checkpoints** — the API expects fine-tuned weights at:
```
fragnet/exps/ft/pnnl_full/fragnet_hpdl_exp1s_h4pt4_10/config_exp100.yaml  (Solubility)
fragnet/exps/ft/pnnl_full/fragnet_hpdl_exp1s_h4pt4_10/ft_100.pt
fragnet/exps/ft/lipo/fragnet_hpdl_exp1s_pt4_30/config_exp100.yaml          (Lipophilicity)
fragnet/exps/ft/lipo/fragnet_hpdl_exp1s_pt4_30/ft_100.pt
```
These paths are configured in `fragnet/api/dependencies.py`.

**Environment file** — copy the example and fill in your values:
```bash
cp .env.example .env
```
`.env` is listed in `.gitignore` and must never be committed. The file contains:

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes (for LLM tab) | Anthropic API key — get one at console.anthropic.com |
| `FRAGNET_LIBRARY_PKL` | No | Path to the fragment library pickle (default: `chembl_library.pkl`) |
| `FRAGNET_LIBRARY_SOURCE` | No | `auto` (default) \| `chembl` \| `reference` — see Fragment library section |
| `FRAGNET_CONTRIBUTION_PRIOR` | No | `false` (default) \| `true` — enable contribution-ranked replacement ordering |

The backend reads `.env` from the project root on startup via `python-dotenv`.
See `.env.example` for descriptions of all options.

---

### Starting the app

Open two terminals, both from the project root:

**Terminal 1 — API backend** (port 8000):
```bash
bash start_api.sh
```
This activates the Python environment and starts Uvicorn with hot-reload enabled.
You should see `Uvicorn running on http://0.0.0.0:8000`. The first request to each
property type will take a few seconds while the model checkpoint loads; subsequent
requests are fast (model is cached per property).

**Terminal 2 — Frontend dev server** (port 5173):
```bash
cd frontend
npm run dev
```
Vite proxies all `/api/*` requests to the backend automatically, so no CORS
configuration is needed during development.

Open **http://localhost:5173** in your browser.

---

### Using the app

1. **Enter a SMILES string** in the input box, select a property
   (*Solubility* or *Lipophilicity*), and click **Analyze**.

2. The **Analysis panel** shows:
   - Predicted property value (logS or logP)
   - Interactive molecule SVG colour-coded by fragment
   - Per-fragment contribution scores — click any fragment to lock/unlock it
   - Tabbed views for atom, bond, fragment, and connection attributions

3. Switch to the **Optimizer tab** to run fragment swaps:
   - Set the optimization direction (*maximize* / *minimize*)
   - Locked fragments (clicked in step 2) are preserved as a protected core
   - Click **Optimize** — the backend enumerates BRICS-compatible replacements
     for the worst-contributing fragment and re-scores every candidate with FragNet
   - Candidates are returned ranked by Δ (improvement over the seed)

4. The **LLM Suggestions** button sends the fragment attribution context to Claude,
   which proposes modifications from a medicinal chemistry perspective.
   Requires `ANTHROPIC_API_KEY` to be set.

---

### Fragment library

The optimizer draws replacements from `chembl_library.pkl` (1,313 drug-like BRICS
fragments from ChEMBL + FDA drugs). If the file is missing it falls back to a
small built-in reference set. To regenerate the library from scratch:

```bash
python fetch_chembl.py              # → chembl_druglike.csv  (~3000 molecules)
python build_chembl_library.py      # → chembl_library.pkl   (1,313 fragments)
```

---

### Production build

To serve the app from a single process (API serves the compiled frontend):
```bash
cd frontend && npm run build && cd ..
bash start_api.sh        # serves React from /  and API from /api/*
```
The compiled assets are written to `frontend/dist/` and served as static files
by FastAPI.



