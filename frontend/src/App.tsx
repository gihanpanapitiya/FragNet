import { useState, useCallback } from 'react';
import {
  AppShell, Group, TextInput, SegmentedControl, Badge, Text,
  Loader, Alert, Stack, Title, Paper,
} from '@mantine/core';
import { IconAlertCircle, IconFlask } from '@tabler/icons-react';
import { useMutation } from '@tanstack/react-query';
import AnalysisPanel from './components/AnalysisPanel';
import { api } from './api/client';
import type { AnalyzeResponse, PropType } from './types';

const DEFAULT_SMILES = 'CC1(C)CC(O)CC(C)(C)N1[O]';

export default function App() {
  const [smiles, setSmiles] = useState(DEFAULT_SMILES);
  const [propType, setPropType] = useState<PropType>('Solubility');
  const [lockedFrags, setLockedFrags] = useState<Set<number>>(new Set());

  const { mutate, data, isPending, error } = useMutation<AnalyzeResponse, Error, { smiles: string; propType: PropType }>({
    mutationFn: ({ smiles: s, propType: p }) => api.analyze(s, p),
    onSuccess: () => setLockedFrags(new Set()),
  });

  const handleAnalyze = useCallback(() => {
    if (smiles.trim()) mutate({ smiles: smiles.trim(), propType });
  }, [smiles, propType, mutate]);

  const toggleLock = useCallback((fid: number) => {
    setLockedFrags((prev) => {
      const next = new Set(prev);
      next.has(fid) ? next.delete(fid) : next.add(fid);
      return next;
    });
  }, []);

  return (
    <AppShell header={{ height: 58 }} padding={0}>
      {/* ── Header ──────────────────────────────────────────────────── */}
      <AppShell.Header
        style={{ background: 'var(--mantine-color-blue-7)', borderBottom: 'none' }}
      >
        <Group h="100%" px="md" gap="md">
          <Group gap={6}>
            <IconFlask size={22} color="white" />
            <Title order={4} c="white">FragNet</Title>
          </Group>

          <TextInput
            value={smiles}
            onChange={(e) => setSmiles(e.currentTarget.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleAnalyze()}
            placeholder="Enter SMILES and press Enter…"
            style={{ flex: 1, maxWidth: 520 }}
            styles={{
              input: {
                fontFamily: 'monospace',
                fontSize: 13,
                background: 'rgba(255,255,255,0.15)',
                color: 'white',
                border: '1px solid rgba(255,255,255,0.3)',
              },
            }}
          />

          <SegmentedControl
            value={propType}
            onChange={(v) => setPropType(v as PropType)}
            data={['Solubility', 'Lipophilicity']}
            color="blue"
            styles={{ root: { background: 'rgba(255,255,255,0.15)' }, label: { color: 'white' } }}
          />

          {isPending && <Loader size="sm" color="white" />}

          {data && (
            <Badge size="lg" variant="white" color="blue" style={{ fontWeight: 700 }}>
              {data.prediction.toFixed(4)} {data.unit}
            </Badge>
          )}
        </Group>
      </AppShell.Header>

      {/* ── Main content ────────────────────────────────────────────── */}
      <AppShell.Main style={{ background: '#f0f2f5', overflow: 'auto' }}>
        <div style={{ padding: '16px', minHeight: '100%' }}>
          {!data && !isPending && !error && (
            <Paper p="xl" radius="md" withBorder ta="center" maw={480} mx="auto" mt={80}>
              <IconFlask size={40} color="var(--mantine-color-blue-6)" />
              <Title order={3} mt="sm">FragNet</Title>
              <Text c="dimmed" mt="xs">
                Enter a SMILES string in the top bar and press <kbd>Enter</kbd> to analyse.
              </Text>
            </Paper>
          )}
          {isPending && (
            <Stack align="center" mt={80} gap="md">
              <Loader size="xl" />
              <Text c="dimmed" size="sm">Running FragNet analysis…</Text>
            </Stack>
          )}
          {error && (
            <Alert icon={<IconAlertCircle />} color="red" title="Analysis failed" maw={600} mx="auto" mt={40}>
              {error.message}
            </Alert>
          )}
          {data && !isPending && (
            <AnalysisPanel data={data} lockedFrags={lockedFrags} onToggleLock={toggleLock} />
          )}
        </div>
      </AppShell.Main>
    </AppShell>
  );
}
