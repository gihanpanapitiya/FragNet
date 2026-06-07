"""
Build the FragNet fragment library (chembl_library.pkl) from a CSV of source
molecules (e.g. chembl_druglike.csv produced by fetch_chembl.py).

Pipeline (see fragnet/vizualize/fragment_library.py):
  1. Source molecules = FDA drugs + the SMILES in the CSV
  2. BRICS-decompose, count fragment frequencies
  3. Filter by heavy-atom range, MW, SA score, PAINS; require attachment points
  4. (Optional) Compute FragNet embeddings / mean contributions (needs a model)
  5. Save to a pickle

Usage:
    python build_chembl_library.py                      # structural library
    python build_chembl_library.py --csv chembl_druglike.csv --out chembl_library.pkl
    python build_chembl_library.py --embeddings --prop-type Solubility
"""

import argparse
import logging

from fragnet.vizualize.fragment_library import FragmentLibrary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="chembl_druglike.csv",
                        help="CSV of source molecules (appended to FDA drugs)")
    parser.add_argument("--smiles-col", default="smiles",
                        help="Name of the SMILES column in the CSV")
    parser.add_argument("--out", default="chembl_library.pkl",
                        help="Output pickle path")
    parser.add_argument("--min-atoms", type=int, default=3)
    parser.add_argument("--max-atoms", type=int, default=25)
    parser.add_argument("--max-mw", type=float, default=300.0)
    parser.add_argument("--max-sa", type=float, default=4.0)
    parser.add_argument("--min-freq", type=int, default=1,
                        help="Minimum source-molecule occurrences to keep a fragment")
    parser.add_argument("--no-pains", action="store_true",
                        help="Disable PAINS filtering")
    parser.add_argument("--embeddings", action="store_true",
                        help="Compute FragNet embeddings (requires a loaded model)")
    parser.add_argument("--contributions", action="store_true",
                        help="Compute mean contribution stats (slow; requires a model)")
    parser.add_argument("--prop-type", default="Solubility",
                        help="Property model to use for embeddings/contributions")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # A model is only needed when annotating with embeddings/contributions.
    viz_app = None
    if args.embeddings or args.contributions:
        from fragnet.api.dependencies import get_viz_app
        viz_app = get_viz_app(args.prop_type)

    lib = FragmentLibrary.from_csv(
        args.csv,
        smiles_col=args.smiles_col,
        viz_app=viz_app,
        prop_type=args.prop_type,
        min_freq=args.min_freq,
        min_atoms=args.min_atoms,
        max_atoms=args.max_atoms,
        max_mw=args.max_mw,
        max_sa=args.max_sa,
        check_pains=not args.no_pains,
        compute_embeddings=args.embeddings,
        compute_contributions=args.contributions,
    )

    lib.save(args.out)
    print(f"Built library with {len(lib.entries)} fragments "
          f"across {len(lib._type_index)} attachment-type buckets → {args.out}")


if __name__ == "__main__":
    main()
