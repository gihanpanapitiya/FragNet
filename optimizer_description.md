# Contribution-Guided Fragment Optimizer

## Overview

We present a fragment-level molecular optimizer that leverages the hierarchical
interpretability of FragNet to propose structurally valid, property-improving
modifications to a seed molecule. Unlike generative approaches that operate in a
continuous latent space, the optimizer works entirely in discrete molecular space
using the Breaking Retrosynthetically Interesting Chemical Substructures (BRICS)
formalism [Degen et al., 2008], ensuring every proposed molecule is synthetically
accessible by construction. The method requires no additional training: it reuses
the already-fine-tuned FragNet model both to identify which fragment of the seed
molecule to modify and to evaluate all proposed modifications.

---

## 2.1 Fragment-Level Attribution

FragNet decomposes molecules into fragments using BRICS and encodes the resulting
fragment graph hierarchically. Following fine-tuning on a target property
(aqueous solubility, logS; or lipophilicity, logP), fragment-level contributions
are computed by a masking-based attribution scheme: the model is run twice for each
fragment — once on the intact molecule and once with a fragment's features masked
(zeroed) at the graph level — and the contribution of fragment *i* is defined as the
signed difference in predicted property:

$$c_i = \hat{y}(\mathcal{G}) - \hat{y}(\mathcal{G} \setminus i)$$

where $\mathcal{G}$ is the full molecular graph and $\mathcal{G} \setminus i$
denotes the graph with fragment *i* masked. A positive value indicates the fragment
raises the predicted property; a negative value indicates it lowers it. The same
masking scheme is applied at the atom and bond levels to provide finer-grained
attribution, and at the fragment-bond level to quantify the contribution of
inter-fragment connections.

---

## 2.2 Fragment Library

The optimizer draws replacement fragments from a pre-built library derived from
3,000 drug-like molecules retrieved from ChEMBL (MW 150–500 Da, heavy atoms 10–50,
AlogP −2 to 6, Lipinski violations ≤ 1) together with a curated set of
FDA-approved drugs spanning major therapeutic areas and scaffold classes. Each
source molecule is decomposed with BRICS, yielding a pool of candidate fragments
that are then filtered by the following criteria:

- **Size:** 3–25 heavy atoms
- **Molecular weight:** ≤ 300 Da (Rule-of-Three compliant)
- **Synthetic accessibility:** SA score ≤ 4.0 [Ertl and Schuffenhauer, 2009]
- **Pan-assay interference:** PAINS filter applied [Baell and Holloway, 2010]
- **Connectability:** at least one BRICS attachment point retained

After filtering and deduplication, the library contains **1,313 unique fragments**
indexed into **99 attachment-type buckets**, where each bucket groups fragments by
their sorted tuple of BRICS dummy-atom isotope numbers. This indexing directly
encodes BRICS join-point compatibility, so library lookup is O(1) per attachment
type.

Each library entry records the BRICS SMILES, a clean (H-capped) SMILES for
standalone scoring, the attachment-type tuple, heavy-atom count, molecular weight,
and the number of source molecules the fragment appeared in.

---

## 2.3 Optimization Algorithm

The optimizer implements a five-step pipeline (Figure X):

**Step 1 — Analyze.** Given a seed SMILES and a target property, the fine-tuned
FragNet model computes fragment contributions $\{c_i\}$ for all fragments of the
seed molecule using the masking-based attribution described above.

**Step 2 — Select target.** Fragments are ranked by contribution in the direction
opposing the optimization goal (ascending for maximization, descending for
minimization). Any user-specified *protected* fragments — substructures designated as
a pharmacophoric core that must be preserved — are excluded from consideration.
The $n$ worst-contributing eligible fragments are selected as swap targets
(default $n = 1$).

**Step 3 — Enumerate candidates.** For each target fragment, the optimizer:

1. Constructs a *scaffold* by removing the target fragment from the seed molecule
   and installing BRICS dummy atoms at each cut point (via `build_scaffold_with_dummies`).
2. Retrieves all library fragments whose attachment types are compatible with the
   scaffold's dummy atoms.
3. Calls `BRICSBuild([scaffold, lib\_frag], maxDepth=1, onlyCompleteMols=True)` for
   each retrieved fragment.

Because exactly two fragments are supplied to `BRICSBuild`, both must be consumed to
form a complete molecule. This guarantees **by construction** that the scaffold —
and therefore every protected fragment within it — is preserved in all generated
candidates. The number of library fragments attempted per target is bounded by
`max_per_frag` (default 50).

**Step 4 — Score.** All unique valid candidates are collected, sanitized, and
submitted to FragNet in a **single batched forward pass**, avoiding repeated model
loading overhead. This produces a predicted property value $\hat{y}$ for each
candidate.

**Step 5 — Rank and return.** Candidates are ranked by improvement over the seed:

$$\Delta_i = \hat{y}(m_i) - \hat{y}(m_0)$$

where $m_0$ is the seed molecule and $m_i$ is the $i$-th candidate. The top-$k$
candidates are returned together with their SMILES, predicted property, and $\Delta$.

---

## 2.4 Scaffold Locking

Users may designate any substructure of the seed molecule as a protected core. The
optimizer identifies which BRICS fragments overlap the specified substructure
(matched by SMARTS) and marks those fragment indices as protected. In addition to
the structural guarantee from Step 3 above, each generated candidate undergoes a
substructure verification check to confirm it still contains every locked fragment,
providing a belt-and-suspenders guard against edge cases in `BRICSBuild`.

---

## 2.5 Optional Contribution-Prior Ordering

When the fragment library has been annotated with per-property contribution
statistics (computed by running FragNet over all source molecules and averaging the
masked-attribution contribution across each fragment's observed contexts), an
optional contribution-prior mode prioritizes which library fragments are attempted
within the `max_per_frag` budget.

Fragments are ordered within each attachment-type bucket by a signal-to-noise score:

$$s = \frac{\bar{c}}{\sigma_c + \varepsilon}, \quad \varepsilon = 0.1$$

where $\bar{c}$ and $\sigma_c$ are the mean and standard deviation of the fragment's
contribution across all source molecules in which it appeared. Fragments with
$n < 2$ observations or no recorded statistics receive $s = 0$ and are tried after
fragments with reliable estimates. The score is sign-flipped for minimization.

The weighting by $1/(\sigma_c + \varepsilon)$ is deliberate: it selects for
context-insensitive fragments — those whose property effect is consistent regardless
of the host scaffold — as these are the fragments whose averaged contribution is most
likely to transfer to a new chemical environment. High-$\sigma$ fragments, whose
contribution is strongly context-dependent, are deprioritized because their mean
contribution is an unreliable guide in a novel scaffold context.

Crucially, this ordering affects only the allocation of the enumeration budget, not
the final ranking: all enumerated candidates are still scored in full context by
FragNet (Step 4), which remains the sole arbiter of the final ranking. The prior
degrades gracefully to insertion order when no contribution statistics are available.

---

## 2.6 Properties Supported

The optimizer currently supports two target properties, each backed by a separate
fine-tuned FragNet checkpoint:

| Property | Unit | Direction |
|---|---|---|
| Aqueous solubility | log S (mol/L) | maximize or minimize |
| Lipophilicity | log P | maximize or minimize |

Both models share the same optimization interface; the property type is specified
at the API call level.

---

## 2.7 Discussion

The approach has three design properties worth noting. First, **chemical validity
is guaranteed without a validity filter**: every proposed molecule is assembled from
real BRICS fragments through a BRICS-valid join, so sanitization failures are rare
and result from RDKit edge cases, not invalid chemistry. Second, **interpretability
and optimization are tightly coupled**: the same model representations that explain
the seed molecule drive the choice of edit site, and the same model re-evaluates
every candidate. The optimizer does not rely on a surrogate or an auxiliary scoring
function. Third, the method is **inherently local**: it changes one fragment at a
time, which makes individual steps chemically interpretable but limits exploration
to single-fragment neighborhoods of the seed. Multi-step application (iterative
optimization) or targeting multiple fragments simultaneously ($n > 1$) can extend
coverage at the cost of a larger candidate space.

A known limitation is that fragment contributions are measured in the context of
the seed molecule and do not directly predict the contribution of a replacement
fragment in the same position: the GNN aggregation over the new molecular graph will
in general differ from a simple substitution of attribution values. The full
FragNet re-score in Step 4 accounts for this, but the selection of *which*
fragment to remove (Step 2) is based on the seed's attribution and therefore
carries this approximation. For properties with near-additive fragment contributions
(lipophilicity) this approximation is reasonable; for properties with strong
cooperativity (solubility) the re-score correction is particularly important.
