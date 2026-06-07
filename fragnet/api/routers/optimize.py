import io
import base64
from fastapi import APIRouter, HTTPException

from fragnet.api.models import OptimizeRequest, OptimizeResponse, CandidateResult
from fragnet.api.dependencies import get_viz_app
from fragnet.vizualize.optimizer import optimize_molecule, mol_to_image

router = APIRouter()


def _img_b64(img) -> str:
    if img is None:
        return ""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@router.post("/optimize", response_model=OptimizeResponse)
def optimize(req: OptimizeRequest):
    viz = get_viz_app(req.prop_type)
    try:
        result = optimize_molecule(
            smiles=req.smiles,
            viz_app=viz,
            prop_type=req.prop_type,
            direction=req.direction,
            n_worst=req.n_worst,
            max_candidates=req.max_candidates,
            top_k=req.top_k,
            frag_contributions=req.frag_contribs,
            seed_prediction=req.seed_prediction,
            protected_fragment_indices=set(req.locked_fragment_indices),
            library_source=req.library_source,
            use_contribution_prior=req.use_contribution_prior,
        )
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc

    candidates = []
    for c in result["candidates"]:
        img = mol_to_image(c["smiles"], width=260, height=180)
        candidates.append(CandidateResult(
            smiles=c["smiles"],
            prediction=c["prediction"],
            delta=c["delta"],
            improvement=c["improvement"],
            mol_img_b64=_img_b64(img),
        ))

    return OptimizeResponse(
        seed_smiles=result["seed_smiles"],
        seed_prediction=result["seed_prediction"],
        n_candidates_evaluated=result["n_candidates_evaluated"],
        n_eligible_fragments=result["n_eligible_fragments"],
        worst_fragment_indices=result["worst_fragment_indices"],
        candidates=candidates,
    )
