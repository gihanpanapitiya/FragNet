"""
Lazy model registry — loads each FragNetVizApp once per (prop_type).
Plain functions (not FastAPI Depends) keep things simple for a single-user tool.
"""
from functools import lru_cache
from fragnet.vizualize.viz import FragNetVizApp

MODEL_CONFIGS: dict[str, dict] = {
    "Solubility": {
        "config": "./fragnet/exps/ft/pnnl_full/fragnet_hpdl_exp1s_h4pt4_10/config_exp100.yaml",
        "chkpt":  "./fragnet/exps/ft/pnnl_full/fragnet_hpdl_exp1s_h4pt4_10/ft_100.pt",
    },
    "Lipophilicity": {
        "config": "./fragnet/exps/ft/lipo/fragnet_hpdl_exp1s_pt4_30/config_exp100.yaml",
        "chkpt":  "./fragnet/exps/ft/lipo/fragnet_hpdl_exp1s_pt4_30/ft_100.pt",
    },
}


@lru_cache(maxsize=4)
def get_viz_app(prop_type: str) -> FragNetVizApp:
    cfg = MODEL_CONFIGS[prop_type]
    return FragNetVizApp(cfg["config"], cfg["chkpt"])
