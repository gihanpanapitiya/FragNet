import { SimpleGrid, Image, Stack, Title } from '@mantine/core';
import ContribTable from '../ContribTable';
import type { AnalyzeResponse } from '../../types';

export default function BondsTab({ data }: { data: AnalyzeResponse }) {
  return (
    <SimpleGrid cols={2} spacing="md">
      <Stack gap="xs">
        <Title order={5}>Bond Attention Weights</Title>
        <Image src={data.img_bond_attn} radius="md" fit="contain" style={{ background: '#fff', border: '1px solid #e9ecef' }} />
      </Stack>
      <Stack gap="xs">
        <Title order={5}>Bond Contributions (masking)</Title>
        <ContribTable
          rows={data.bond_contribs}
          columns={['bond_index', 'begin_atom', 'end_atom', 'attr']}
          labels={['Idx', 'Begin', 'End', 'Contribution']}
          contribKey="attr"
        />
      </Stack>
    </SimpleGrid>
  );
}
