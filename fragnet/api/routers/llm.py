import io
import base64
from fastapi import APIRouter, HTTPException

from fragnet.api.models import LLMSuggestRequest, LLMSuggestResponse, LLMSuggestion
from fragnet.api.dependencies import get_viz_app
from fragnet.vizualize.llm_suggestions import suggest_replacements
from fragnet.vizualize.optimizer import batch_score, mol_to_image

router = APIRouter()


def _img_b64(img) -> str:
    if img is None:
        return ""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@router.post("/llm-suggest", response_model=LLMSuggestResponse)
def llm_suggest(req: LLMSuggestRequest):
    try:
        raw = suggest_replacements(
            smiles=req.smiles,
            frag_contributions=req.frag_contribs,
            frag_atom_map=req.frag_atom_map,
            prop_type=req.prop_type,
            direction=req.direction,
            seed_prediction=req.seed_prediction,
            protected_indices=set(req.locked_fragment_indices),
            n_worst=req.n_worst,
            n_suggestions=req.n_suggestions,
        )
    except EnvironmentError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc

    if not raw:
        return LLMSuggestResponse(suggestions=[], n_scored=0, n_improved=0)

    viz = get_viz_app(req.prop_type)
    try:
        scores = batch_score(viz, [s["smiles"] for s in raw])
    except Exception:
        scores = {}

    suggestions: list[LLMSuggestion] = []
    for s in raw:
        smi = s["smiles"]
        pred = scores.get(smi)
        delta = float(pred - req.seed_prediction) if pred is not None else None
        improvement = (delta if req.direction == "maximize" else -delta) if delta is not None else None
        img = mol_to_image(smi, width=260, height=180)
        suggestions.append(LLMSuggestion(
            smiles=smi,
            rationale=s["rationale"],
            prediction=round(float(pred), 4) if pred is not None else None,
            delta=round(delta, 4) if delta is not None else None,
            improvement=round(improvement, 4) if improvement is not None else None,
            mol_img_b64=_img_b64(img),
        ))

    suggestions.sort(key=lambda x: x.improvement if x.improvement is not None else -999, reverse=True)
    n_scored = sum(1 for s in suggestions if s.prediction is not None)
    n_improved = sum(1 for s in suggestions if (s.improvement or 0) > 0)

    return LLMSuggestResponse(suggestions=suggestions, n_scored=n_scored, n_improved=n_improved)
