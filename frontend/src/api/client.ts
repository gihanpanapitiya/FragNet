import type {
  AnalyzeResponse, OptimizeResponse, LLMSuggestResponse, PropType,
} from '../types';

const BASE = '/api';

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
    throw new Error(err.detail ?? `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  analyze: (smiles: string, propType: PropType) =>
    post<AnalyzeResponse>('/analyze', { smiles, prop_type: propType }),

  optimize: (body: {
    smiles: string;
    prop_type: PropType;
    direction: string;
    n_worst: number;
    max_candidates: number;
    top_k: number;
    frag_contribs: unknown[];
    seed_prediction: number;
    locked_fragment_indices: number[];
  }) => post<OptimizeResponse>('/optimize', body),

  llmSuggest: (body: {
    smiles: string;
    prop_type: PropType;
    direction: string;
    n_worst: number;
    n_suggestions: number;
    frag_contribs: unknown[];
    frag_atom_map: number[][];
    seed_prediction: number;
    locked_fragment_indices: number[];
  }) => post<LLMSuggestResponse>('/llm-suggest', body),
};
