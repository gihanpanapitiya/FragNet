It's a two-step pipeline, and the file artifacts confirm it (CSV saved May 30 17:21, pkl 3 minutes later at 17:24):

Step 1 — Fetch source molecules (fetch_chembl.py)
python fetch_chembl.py   # → chembl_druglike.csv
Queries the ChEMBL web API for ~3000 drug-like small molecules with filters: MW 150–500, heavy atoms 10–50, AlogP −2 to 6, ≤1 Ro5 violation, type "Small molecule". Each SMILES is validated through RDKit and written to chembl_druglike.csv (the file has 3000 molecules — 3001 lines incl. header).

Step 2 — Build & save the library (fragnet/vizualize/fragment_library.py)
Via FragmentLibrary.from_csv("chembl_druglike.csv") → .build(...) → .save("chembl_library.pkl"). The build pipeline:
1. Source set = FDA_DRUG_SMILES + the 3000 ChEMBL SMILES from the CSV.
2. BRICS-decompose all of them and count fragment frequencies.
3. Filter each fragment: convert BRICS → clean SMILES, then keep only those passing heavy-atom range (3–25), MW ≤ 300, SA score ≤ 4, PAINS filter, and that have ≥1 attachment point.
4. Index surviving fragments by attachment-type tuple.

Result in the actual pkl:
- 1313 fragments, grouped into 99 attachment-type buckets
- heavy-atom range 3–22; n_sources (how many source molecules each fragment came from) ranges 1–356
- 0 embeddings, 0 contributions → it was built with viz_app=None (the structural-only path; FragNet embeddings/contribution stats were not computed, since those branches only run when a viz_app is passed)

So it's a purely structural BRICS fragment library: ChEMBL drug-like set + FDA drugs, decomposed and quality-filtered down to 1313 fragments. It's loaded by fragnet/vizualize/rl_env.py (FragmentLibrary.load("chembl_library.pkl")) for RL-based design — and notably is not the library the running API uses (the API uses the optimizer's lightweight 47-fragment pool).

Note: the build step is now captured in a committed script, `build_chembl_library.py` (see below). Previously it had been run interactively (REPL/notebook) with no recorded kwargs, but the pkl's stats matched the default build parameters — which the script reproduces exactly.

---

Done. Created build_chembl_library.py and verified it exactly reproduces the committed library: 1313 fragments, 99 attachment-type buckets, identical fragment set.

build_chembl_library.py

It's the missing build step that pairs with fetch_chembl.py, matching the repo's script style (module docstring + argparse + __main__). It calls FragmentLibrary.from_csv(...).save(...) with the defaults that produced the original pkl.

Default run reproduces chembl_library.pkl:
python build_chembl_library.py
# FDA drugs + chembl_druglike.csv → BRICS → filter → 1313 fragments → chembl_library.pkl

Useful flags:
- --csv / --out / --smiles-col — input CSV and output path
- --min-atoms / --max-atoms / --max-mw / --max-sa / --min-freq / --no-pains — filter tuning
- --embeddings / --contributions — opt into the FragNet-annotated build (these lazily load a model via get_viz_app; the original committed pkl has neither, so they're off by default and reproduce it exactly)

The Can't kekulize lines are harmless RDKit warnings from BRICS on a few fragments — those get dropped by the filter, which is why the count still matches.

A note on faithfulness: I verified by writing to /tmp/chembl_library_test.pkl, not by overwriting your existing chembl_library.pkl — so the committed artifact is untouched.

---

## Reproducing chembl_library.pkl (canonical commands)

Full two-step pipeline from scratch:

```bash
# 1. Fetch ~3000 drug-like small molecules from ChEMBL → chembl_druglike.csv
python fetch_chembl.py

# 2. Build the structural BRICS fragment library → chembl_library.pkl
#    (FDA drugs + chembl_druglike.csv, BRICS-decomposed and quality-filtered)
python build_chembl_library.py
```

The default `build_chembl_library.py` run reproduces the committed pkl exactly:
**1313 fragments across 99 attachment-type buckets** (verified by rebuilding to a
temp file and comparing the clean-SMILES set — identical).

Flags:
- `--csv / --out / --smiles-col` — input CSV and output path
- `--min-atoms / --max-atoms / --max-mw / --max-sa / --min-freq / --no-pains` — filter tuning
  (defaults: min_atoms 3, max_atoms 25, max_mw 300, max_sa 4, min_freq 1, PAINS on)
- `--embeddings / --contributions` — opt into the FragNet-annotated build; these lazily
  load a model via `get_viz_app(--prop-type)`. The committed pkl has neither (built with
  `viz_app=None`), so they are **off by default** to reproduce it.

---

## How the optimizer uses the fragment library

The fragment library is the **pool of replacement parts** the optimizer draws on
when it tries to improve a molecule. The optimizer (`fragnet/vizualize/optimizer.py`,
driven by the `/optimize` API endpoint) never invents new chemistry from scratch — it
swaps a poorly-contributing fragment of the seed molecule for a chemically-compatible
alternative pulled from this library.

### Library representation in the optimizer

`_get_library()` returns a dict keyed by **attachment-type tuple** → **list of BRICS
fragment SMILES**:

```python
{ (1,): ["[1*]c1ccccc1O", ...],        # one-attachment fragments
  (1, 5): ["[1*]C(=O)[5*]", ...],      # two-attachment fragments
  ... }
```

The key is the sorted tuple of BRICS dummy-atom isotopes (the "attachment types"),
computed by `_attachment_types()`. This is exactly the join-point compatibility code
that BRICS uses, so the key tells the optimizer *where and how* a fragment can be
reconnected.

Source of that dict (see `_get_library()`):
1. **Preferred:** load the pre-built pickle via `_find_library_pkl()`
   (`$FRAGNET_LIBRARY_PKL`, then CWD, then repo root) and convert each
   `FragmentLibrary` entry into the `{attachment_types: [smiles]}` map
   (`_load_pkl_library`). → **1313 ChEMBL+FDA fragments.**
2. **Fallback:** if the pickle is missing/unreadable, decompose the built-in
   `_REFERENCE_SMILES` set with BRICS (`_build_library`). → **47 fragments.**

The library is built once and cached in the module-level `_LIBRARY`.

### Where it's consumed — `enumerate_fragment_swaps()`

This is the only consumer. Given the seed and the index of the fragment to replace:

1. **Cut out the target fragment.** `build_scaffold_with_dummies()` produces a
   *scaffold* = the whole molecule minus the target fragment, leaving BRICS dummy
   atoms at the cut points.
2. **Materialize the library once.** All library SMILES are converted to RDKit mols
   (`all_lib_mols`) for the run.
3. **Pre-filter by attachment compatibility.** For a single-cut scaffold, only
   library fragments with exactly one attachment point are tried; for a multi-cut
   scaffold, only fragments whose attachment types overlap the scaffold's dummy types.
   This is a cheap gate that shrinks the search space before the expensive step.
4. **Reassemble.** `BRICSBuild([scaffold, lib_frag], maxDepth=1, onlyComplete...)` is
   called. With exactly two pieces, both must be consumed, so the scaffold is always
   preserved and only the swapped fragment changes. Each produced molecule is
   sanitized, canonicalized, and collected (deduped, seed excluded).
5. **Bounded cost.** `max_per_frag` caps how many library fragments are actually fed
   to `BRICSBuild` per target fragment, so a 1313-fragment library does **not** mean
   1313 build attempts.

### How it fits the full pipeline — `optimize_molecule()`

1. Get per-fragment **FragNet contributions** for the seed.
2. Rank fragments, drop any **protected/locked** (core) indices, pick the `n_worst`
   worst-contributing eligible fragments.
3. **`enumerate_fragment_swaps()`** generates scaffold-preserving candidates by
   pulling compatible alternatives from the library (the step described above).
4. (If fragments are locked) substructure-verify each candidate still contains every
   locked fragment.
5. **Batch-score** all candidates with FragNet in one pass, rank by improvement
   (Δ vs. the seed prediction in the requested direction), return top-k.

So the library only feeds **step 3**: it determines *what fragments are available to
swap in*. Its quality and breadth directly set the diversity of candidates the
optimizer can propose — which is why moving from the 47-fragment reference pool to the
1313-fragment ChEMBL+FDA library yields more, and more drug-like, candidates. The
attachment-type keying guarantees every proposed swap is BRICS-valid by construction.

---

## Selecting the library source

`_get_library(source)` and `optimize_molecule(..., library_source=...)` accept three values:

| source | behaviour |
|---|---|
| `"auto"` *(default)* | ChEMBL+FDA pickle if found (`$FRAGNET_LIBRARY_PKL`, CWD, repo root), else reference set |
| `"chembl"` | Force the pickle; raises `FileNotFoundError` if absent (fail-loud for misconfigured deploys) |
| `"reference"` | Force the built-in 47-fragment reference set |

The library is built once and cached per source in `_LIBRARY_CACHE`. The full
`FragmentLibrary` object is also cached in `_FRAG_LIB_OBJ_CACHE` (needed for
contribution stats — see below). Set `FRAGNET_LIBRARY_PKL=/path/to/lib.pkl` to
override the pickle location without code changes.

Exposed in the API as `OptimizeRequest.library_source` (default `"auto"`).

---

## Contribution prior for replacement selection (optional)

### The problem

Fragment contributions in FragNet are **not** a property of the fragment alone — they
are a property of `(fragment, its chemical environment)`. A mean contribution averaged
over a fragment's appearances across ChEMBL/FDA molecules is an out-of-context
extrapolation when applied to a new scaffold. Using it as a hard filter would be
chemically unsound: the final `batch_score` re-scores every candidate in its actual
context anyway, so only the budget-allocation step can benefit from a prior.

### The approach

When `use_contribution_prior=True`, `enumerate_fragment_swaps()` **sorts each
attachment-type bucket** by a signal-to-noise score before iterating, so the
`max_per_frag` budget is spent on the most statistically confident candidates first.
Fragments that are not scored (or score poorly) are tried last, not excluded — the
downstream `batch_score` is still the arbiter.

**Score formula** (`_contribution_score()` in optimizer.py):

```
score = mean / (std + ε)         if n ≥ min_n
      = 0.0                       otherwise
```

- `ε = 0.1`, `min_n = 2` by default.
- Sign-flipped for `direction="minimize"`.
- **Low std amplifies the mean** (fragment behaves consistently across contexts →
  more transferable signal). **High std dampens it** (context-sensitive → mean is
  unreliable). Fragments with `n < 2` or missing stats score 0.0 and sort last.

**Why per-bucket sorting?** Sorting within each attachment-type group compares only
fragments competing for the same structural role (same BRICS join points). This is
the most honest use of the cross-context averaged stats — we're not comparing a
one-attachment aromatic ring to a two-attachment linker.

### When it is and isn't useful

| Property | Usefulness |
|---|---|
| Lipophilicity (logP) | **Higher** — logP is near-additive (Crippen/π-constants), so mean contributions transfer reasonably well |
| Solubility (logS) | **Lower** — crystal packing, ionisation, context coupling make the mean much noisier; rely more on `batch_score` |

The feature is **completely inert** until the pkl is rebuilt with contribution stats.
With the current structural-only pkl (no stats), all scores are 0.0 and order is
unchanged — it degrades gracefully.

### Enabling it

**Step 1 — rebuild the library with contributions:**
```bash
python build_chembl_library.py --contributions --prop-type Solubility \
    --out chembl_library_with_contribs.pkl
```
This requires a loaded `viz_app` model (slow — runs FragNet over all source molecules).

**Step 2 — use it in the API:**
```json
{
  "library_source": "chembl",
  "use_contribution_prior": true,
  ...
}
```

Or directly in Python:
```python
optimize_molecule(smiles, viz_app, use_contribution_prior=True, prop_type="Lipophilicity")
```

Exposed in the API as `OptimizeRequest.use_contribution_prior` (default `false`).

---

