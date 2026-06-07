import { Tabs } from '@mantine/core';
import AtomsTab from './tabs/AtomsTab';
import BondsTab from './tabs/BondsTab';
import FragmentsTab from './tabs/FragmentsTab';
import ConnectionsTab from './tabs/ConnectionsTab';
import OptimizerTab from './tabs/OptimizerTab';
import type { AnalyzeResponse } from '../types';

interface Props {
  data: AnalyzeResponse;
  lockedFrags: Set<number>;
  onToggleLock: (fid: number) => void;
}

export default function AnalysisPanel({ data, lockedFrags, onToggleLock }: Props) {
  return (
    <Tabs defaultValue="fragments" keepMounted={false}>
      <Tabs.List>
        <Tabs.Tab value="atoms">⚛️ Atoms</Tabs.Tab>
        <Tabs.Tab value="bonds">🔗 Bonds</Tabs.Tab>
        <Tabs.Tab value="fragments">🧩 Fragments</Tabs.Tab>
        <Tabs.Tab value="connections">🔀 Connections</Tabs.Tab>
        <Tabs.Tab value="optimizer">🔬 Optimizer</Tabs.Tab>
      </Tabs.List>

      <Tabs.Panel value="atoms"       pt="md"><AtomsTab data={data} /></Tabs.Panel>
      <Tabs.Panel value="bonds"       pt="md"><BondsTab data={data} /></Tabs.Panel>
      <Tabs.Panel value="fragments"   pt="md"><FragmentsTab data={data} /></Tabs.Panel>
      <Tabs.Panel value="connections" pt="md"><ConnectionsTab data={data} /></Tabs.Panel>
      <Tabs.Panel value="optimizer"   pt="md">
        <OptimizerTab data={data} lockedFrags={lockedFrags} onToggleLock={onToggleLock} />
      </Tabs.Panel>
    </Tabs>
  );
}
