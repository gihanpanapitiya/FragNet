export type PropType = 'Solubility' | 'Lipophilicity';

export interface FragCentroid {
  fragment_index: number;
  cx: number;
  cy: number;
  contribution: number;
  atom_indices: number[];
}

export interface AnalyzeResponse {
  smiles: string;
  prop_type: PropType;
  prediction: number;
  unit: string;
  mol_svg_b64: string;
  mol_svg_width: number;
  mol_svg_height: number;
  fragment_centroids: FragCentroid[];
  frag_atom_map: number[][];
  atom_contribs: Record<string, unknown>[];
  bond_contribs: Record<string, unknown>[];
  fbond_contribs: Record<string, unknown>[];
  frag_contribs: Record<string, unknown>[];
  frag_weights: Record<string, unknown>[];
  connection_weights: Record<string, unknown>[];
  atoms_in_frags: Record<string, number[]>;
  img_atom_attn: string;
  img_bond_attn: string;
  img_frag_attn: string;
  img_frag_highlight: string;
  img_frag_attr: string;
}

export interface CandidateResult {
  smiles: string;
  prediction: number;
  delta: number;
  improvement: number;
  mol_img_b64: string;
}

export interface OptimizeResponse {
  seed_smiles: string;
  seed_prediction: number;
  n_candidates_evaluated: number;
  n_eligible_fragments: number;
  worst_fragment_indices: number[];
  candidates: CandidateResult[];
}

export interface LLMSuggestion {
  smiles: string;
  rationale: string;
  prediction: number | null;
  delta: number | null;
  improvement: number | null;
  mol_img_b64: string;
}

export interface LLMSuggestResponse {
  suggestions: LLMSuggestion[];
  n_scored: number;
  n_improved: number;
}
