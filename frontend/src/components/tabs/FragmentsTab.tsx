import { SimpleGrid, Image, Stack, Title, Text } from '@mantine/core';
import ContribTable from '../ContribTable';
import type { AnalyzeResponse } from '../../types';

export default function FragmentsTab({ data }: { data: AnalyzeResponse }) {
  const fragContribs = data.frag_contribs.map((f) => ({
    ...f,
    atoms: Array.isArray(f.atoms) ? (f.atoms as number[]).join(', ') : f.atoms,
  }));

  return (
    <Stack gap="lg">
      <SimpleGrid cols={2} spacing="md">
        <Stack gap="xs">
          <Title order={5}>Fragment Decomposition</Title>
          <Image src={data.img_frag_highlight} radius="md" fit="contain" style={{ background: '#fff', border: '1px solid #e9ecef' }} />
          <Text size="xs" c="dimmed" mt={4}>Fragment attention weights</Text>
          <Image src={data.img_frag_attn} radius="md" fit="contain" style={{ background: '#fff', border: '1px solid #e9ecef' }} />
        </Stack>
        <Stack gap="xs">
          <Title order={5}>Fragment Attribution (masking)</Title>
          <Image src={data.img_frag_attr} radius="md" fit="contain" style={{ background: '#fff', border: '1px solid #e9ecef' }} />
          <Title order={5} mt="sm">Fragment Contributions</Title>
          <ContribTable
            rows={fragContribs}
            columns={['fragment_index', 'atoms', 'contribution']}
            labels={['Frag #', 'Atoms', 'Contribution']}
            contribKey="contribution"
          />
        </Stack>
      </SimpleGrid>
    </Stack>
  );
}
