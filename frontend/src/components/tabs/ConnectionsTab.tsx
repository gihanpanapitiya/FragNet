import { SimpleGrid, Image, Stack, Title, Text } from '@mantine/core';
import ContribTable from '../ContribTable';
import type { AnalyzeResponse } from '../../types';

export default function ConnectionsTab({ data }: { data: AnalyzeResponse }) {
  return (
    <SimpleGrid cols={2} spacing="md">
      <Stack gap="xs">
        <Title order={5}>Fragment Decomposition</Title>
        <Image src={data.img_frag_highlight} radius="md" fit="contain" style={{ background: '#fff', border: '1px solid #e9ecef' }} />
        <Title order={5} mt="sm">Connection Weights</Title>
        {data.connection_weights.length > 0 ? (
          <ContribTable
            rows={data.connection_weights}
            columns={['connection', 'weight']}
            labels={['Connection', 'Weight']}
            contribKey="weight"
          />
        ) : (
          <Text c="dimmed" size="sm">No connection data.</Text>
        )}
      </Stack>
      <Stack gap="xs">
        <Title order={5}>Fragment Connection Contributions</Title>
        {data.fbond_contribs.length > 0 ? (
          <ContribTable
            rows={data.fbond_contribs}
            columns={['fragbond_node_index', 'begin_index', 'end_index', 'attr']}
            labels={['Conn #', 'Frag A', 'Frag B', 'Contribution']}
            contribKey="attr"
          />
        ) : (
          <Text c="dimmed" size="sm">Single fragment — no inter-fragment connections.</Text>
        )}
      </Stack>
    </SimpleGrid>
  );
}
