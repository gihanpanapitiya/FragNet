import { SimpleGrid, Image, Stack, Title } from '@mantine/core';
import ContribTable from '../ContribTable';
import type { AnalyzeResponse } from '../../types';

export default function AtomsTab({ data }: { data: AnalyzeResponse }) {
  return (
    <SimpleGrid cols={2} spacing="md">
      <Stack gap="xs">
        <Title order={5}>Atom Attention Weights</Title>
        <Image src={data.img_atom_attn} radius="md" fit="contain" style={{ background: '#fff', border: '1px solid #e9ecef' }} />
      </Stack>
      <Stack gap="xs">
        <Title order={5}>Atom Contributions (masking)</Title>
        <ContribTable
          rows={data.atom_contribs}
          columns={['atom_index', 'atom_type', 'attr']}
          labels={['Idx', 'Symbol', 'Contribution']}
          contribKey="attr"
        />
      </Stack>
    </SimpleGrid>
  );
}
