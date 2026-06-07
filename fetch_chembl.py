"""
Fetch drug-like molecules from ChEMBL and save as chembl_druglike.csv.

Filters applied:
  - MW 150–500 (small-molecule range, generous for fragment diversity)
  - Heavy atom count 10–50
  - AlogP -2 to 6 (solubility/lipophilicity range)
  - Passes RO5 violations <= 1
  - Molecule type: Small molecule
  - Max ~3000 molecules (enough for ~1500 usable BRICS fragments)

Usage:
    python fetch_chembl.py              # saves chembl_druglike.csv
    python fetch_chembl.py --n 5000     # fetch more
"""

import argparse
import pandas as pd
from chembl_webresource_client.new_client import new_client
from rdkit import Chem

def fetch(n: int = 3000, out: str = "chembl_druglike.csv") -> None:
    molecule = new_client.molecule

    print(f"Querying ChEMBL for drug-like small molecules (target: {n}) …")
    results = molecule.filter(
        molecule_properties__mw_freebase__gte=150,
        molecule_properties__mw_freebase__lte=500,
        molecule_properties__heavy_atoms__gte=10,
        molecule_properties__heavy_atoms__lte=50,
        molecule_properties__alogp__gte=-2,
        molecule_properties__alogp__lte=6,
        molecule_properties__num_ro5_violations__lte=1,
        molecule_type="Small molecule",
    ).only(["molecule_chembl_id", "molecule_structures"])

    smiles_list = []
    ids = []
    for r in results:
        if len(smiles_list) >= n:
            break
        structs = r.get("molecule_structures") or {}
        smi = structs.get("canonical_smiles")
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        smiles_list.append(smi)
        ids.append(r["molecule_chembl_id"])
        if len(smiles_list) % 500 == 0:
            print(f"  … fetched {len(smiles_list)}")

    df = pd.DataFrame({"chembl_id": ids, "smiles": smiles_list})
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} molecules to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3000)
    parser.add_argument("--out", default="chembl_druglike.csv")
    args = parser.parse_args()
    fetch(args.n, args.out)
