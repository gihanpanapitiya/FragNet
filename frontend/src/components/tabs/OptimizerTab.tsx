import { useState } from 'react';
import { Group as PanelGroup, Panel, Separator as PanelResizeHandle } from 'react-resizable-panels';
import {
  Stack, Group, Title, Text, Badge, Select, Slider, Button,
  NumberInput, Alert, SimpleGrid, Card, Divider, Loader, Paper,
} from '@mantine/core';
import { IconRocket, IconSparkles, IconAlertCircle } from '@tabler/icons-react';
import ContribTable from '../ContribTable';
import CandidateGrid from '../CandidateGrid';
import MoleculeViewer from '../MoleculeViewer';
import { api } from '../../api/client';
import type { AnalyzeResponse, OptimizeResponse, LLMSuggestResponse } from '../../types';

interface Props {
  data: AnalyzeResponse;
  lockedFrags: Set<number>;
  onToggleLock: (fid: number) => void;
}

export default function OptimizerTab({ data, lockedFrags, onToggleLock }: Props) {
  const [direction, setDirection] = useState('maximize');
  const [nWorst, setNWorst] = useState(1);
  const [maxCands, setMaxCands] = useState(50);

  const [optResult, setOptResult] = useState<OptimizeResponse | null>(null);
  const [llmResult, setLlmResult] = useState<LLMSuggestResponse | null>(null);
  const [optLoading, setOptLoading] = useState(false);
  const [llmLoading, setLlmLoading] = useState(false);
  const [optError, setOptError] = useState<string | null>(null);
  const [llmError, setLlmError] = useState<string | null>(null);

  const allFids = data.frag_contribs.map((f) => f.fragment_index as number);
  const nAvailable = allFids.filter((f) => !lockedFrags.has(f)).length;
  const maxNWorst = Math.min(3, Math.max(1, nAvailable));

  const statusRows = data.frag_contribs.map((f) => ({
    'Frag #': f.fragment_index,
    Atoms: Array.isArray(f.atoms) ? (f.atoms as number[]).join(', ') : String(f.atoms),
    Contribution: f.contribution,
    Status: lockedFrags.has(f.fragment_index as number) ? '🔒 Locked' : '🔓 Available',
  }));

  async function runOptimizer() {
    setOptError(null);
    setOptLoading(true);
    try {
      const result = await api.optimize({
        smiles: data.smiles,
        prop_type: data.prop_type,
        direction,
        n_worst: nWorst,
        max_candidates: maxCands,
        top_k: 10,
        frag_contribs: data.frag_contribs,
        seed_prediction: data.prediction,
        locked_fragment_indices: [...lockedFrags],
      });
      setOptResult(result);
    } catch (e) {
      setOptError((e as Error).message);
    } finally {
      setOptLoading(false);
    }
  }

  async function runLLM() {
    setLlmError(null);
    setLlmLoading(true);
    try {
      const result = await api.llmSuggest({
        smiles: data.smiles,
        prop_type: data.prop_type,
        direction,
        n_worst: nWorst,
        n_suggestions: 8,
        frag_contribs: data.frag_contribs,
        frag_atom_map: data.frag_atom_map,
        seed_prediction: data.prediction,
        locked_fragment_indices: [...lockedFrags],
      });
      setLlmResult(result);
    } catch (e) {
      setLlmError((e as Error).message);
    } finally {
      setLlmLoading(false);
    }
  }

  return (
    <Stack gap="lg">
      {/* ── Molecule + fragment status (resizable) ── */}
      <PanelGroup orientation="horizontal" style={{ minHeight: 420, display: 'flex' }}>

        {/* Molecule panel */}
        <Panel defaultSize={60} minSize={30}>
          <Paper withBorder radius="md" p="sm" h="100%">
            <Text size="xs" c="dimmed" ta="center" mb={6}>
              Click a fragment to <strong>lock</strong> / <strong>unlock</strong> it
            </Text>
            <Group gap={10} justify="center" mb={8} wrap="wrap">
              {[
                { color: '#e74c3c', label: 'hurts' },
                { color: '#27ae60', label: 'helps' },
                { color: '#2980b9', label: 'neutral' },
                { color: '#7f8c8d', label: 'locked' },
              ].map(({ color, label }) => (
                <Group key={label} gap={4}>
                  <div style={{ width: 9, height: 9, borderRadius: '50%', background: color }} />
                  <Text size="xs" c="dimmed">{label}</Text>
                </Group>
              ))}
            </Group>
            <MoleculeViewer
              molSvgB64={data.mol_svg_b64}
              molSvgWidth={data.mol_svg_width}
              molSvgHeight={data.mol_svg_height}
              fragmentCentroids={data.fragment_centroids}
              lockedFrags={lockedFrags}
              onToggleLock={onToggleLock}
            />
            {lockedFrags.size > 0 && (
              <Text size="xs" c="dimmed" ta="center" mt={6}>
                {lockedFrags.size} fragment(s) locked
              </Text>
            )}
          </Paper>
        </Panel>

        {/* Drag handle */}
        <PanelResizeHandle style={{
          width: 6,
          background: 'transparent',
          cursor: 'col-resize',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}>
          <div style={{
            width: 3,
            height: 40,
            borderRadius: 2,
            background: 'var(--mantine-color-gray-4)',
            transition: 'background 0.15s',
          }} />
        </PanelResizeHandle>

        {/* Fragment status panel */}
        <Panel defaultSize={40} minSize={15}>
          <Paper withBorder radius="md" p="sm" h="100%" style={{ overflow: 'auto' }}>
            <Stack gap={6} mb="xs">
              <Title order={6}>Fragments</Title>
              <Badge color={nAvailable > 0 ? 'blue' : 'gray'} size="sm">
                {nAvailable}/{allFids.length} free
              </Badge>
              {lockedFrags.size > 0 && (
                <Badge color="gray" size="sm">{lockedFrags.size} locked</Badge>
              )}
            </Stack>
            <ContribTable
              rows={statusRows}
              columns={['Frag #', 'Contribution', 'Status']}
              labels={['#', 'Contrib', 'Status']}
              contribKey="Contribution"
              maxRows={20}
            />
          </Paper>
        </Panel>

      </PanelGroup>

      <Divider />

      {/* Settings */}
      <div>
        <Title order={5} mb="sm">⚙️ Settings</Title>
        <SimpleGrid cols={3} spacing="md">
          <Select
            label="Direction"
            value={direction}
            onChange={(v) => setDirection(v ?? 'maximize')}
            data={[
              { value: 'maximize', label: 'Maximize ↑' },
              { value: 'minimize', label: 'Minimize ↓' },
            ]}
          />
          <div>
            <Text size="sm" fw={500} mb={4}>Fragments to target: {nWorst}</Text>
            <Slider
              value={nWorst}
              onChange={setNWorst}
              min={1} max={maxNWorst} step={1}
              marks={Array.from({ length: maxNWorst }, (_, i) => ({ value: i + 1, label: String(i + 1) }))}
            />
          </div>
          <NumberInput
            label="Max candidates"
            value={maxCands}
            onChange={(v) => setMaxCands(Number(v) || 50)}
            min={10} max={200} step={10}
          />
        </SimpleGrid>
      </div>

      <Divider />

      {/* BRICS Optimizer */}
      <div>
        <Group mb="sm">
          <Button
            leftSection={optLoading ? <Loader size="xs" color="white" /> : <IconRocket size={16} />}
            onClick={runOptimizer}
            disabled={nAvailable === 0 || optLoading}
          >
            Run BRICS Optimizer
          </Button>
        </Group>

        {optError && (
          <Alert icon={<IconAlertCircle />} color="red" mb="md">{optError}</Alert>
        )}

        {optResult && (
          <Stack gap="sm">
            <SimpleGrid cols={3}>
              {[
                { label: 'Seed', value: optResult.seed_prediction.toFixed(4) },
                {
                  label: 'Best candidate',
                  value: optResult.candidates[0]?.prediction.toFixed(4) ?? '—',
                  delta: optResult.candidates[0]?.delta,
                },
                {
                  label: 'Improved',
                  value: String(optResult.candidates.filter((c) => c.improvement > 0).length),
                },
              ].map((m) => (
                <Card key={m.label} withBorder radius="md" padding="sm">
                  <Text size="xs" c="dimmed" tt="uppercase" fw={600}>{m.label}</Text>
                  <Text size="xl" fw={700}>{m.value}</Text>
                  {m.delta !== undefined && (
                    <Text size="xs" c={m.delta > 0 ? 'green' : 'red'}>
                      Δ {m.delta > 0 ? '+' : ''}{m.delta.toFixed(4)}
                    </Text>
                  )}
                </Card>
              ))}
            </SimpleGrid>
            <Text size="xs" c="dimmed">
              {optResult.n_candidates_evaluated} candidates evaluated · {optResult.n_eligible_fragments} eligible fragments
            </Text>
            <CandidateGrid items={optResult.candidates} seedPrediction={optResult.seed_prediction} />
          </Stack>
        )}
      </div>

      <Divider />

      {/* LLM Suggestions */}
      <div>
        <Title order={5} mb={4}>🤖 LLM-Guided Suggestions</Title>
        <Text size="xs" c="dimmed" mb="sm">
          Claude reasons about the chemistry and proposes targeted modifications,
          each scored by FragNet. Requires <code>ANTHROPIC_API_KEY</code> in the environment.
        </Text>
        <Group mb="sm">
          <Button
            variant="outline"
            leftSection={llmLoading ? <Loader size="xs" /> : <IconSparkles size={16} />}
            onClick={runLLM}
            disabled={nAvailable === 0 || llmLoading}
          >
            Get LLM Suggestions
          </Button>
        </Group>

        {llmError && (
          <Alert icon={<IconAlertCircle />} color="orange" mb="md">{llmError}</Alert>
        )}

        {llmResult && (
          <Stack gap="sm">
            <Text size="xs" c="dimmed">
              {llmResult.suggestions.length} suggestions · {llmResult.n_scored} scored · {llmResult.n_improved} improved
            </Text>
            <CandidateGrid items={llmResult.suggestions} seedPrediction={data.prediction} />
          </Stack>
        )}
      </div>
    </Stack>
  );
}
