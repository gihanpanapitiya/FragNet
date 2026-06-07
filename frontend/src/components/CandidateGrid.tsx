import { SimpleGrid, Card, Text, Badge, Code, Stack, Image } from '@mantine/core';
import type { CandidateResult, LLMSuggestion } from '../types';

type Item = (CandidateResult & { rationale?: string }) | LLMSuggestion;

interface Props {
  items: Item[];
  seedPrediction?: number;
}

function DeltaBadge({ delta }: { delta: number | null }) {
  if (delta === null) return <Badge color="gray" variant="light">unscored</Badge>;
  return (
    <Badge color={delta > 0 ? 'green' : 'red'} variant="light">
      {delta > 0 ? '+' : ''}{delta.toFixed(3)}
    </Badge>
  );
}

export default function CandidateGrid({ items }: Props) {
  if (!items.length) return <Text c="dimmed" size="sm">No candidates.</Text>;

  return (
    <SimpleGrid cols={{ base: 1, sm: 2, md: 3 }} spacing="md">
      {items.map((item, i) => {
        const improved = (item.improvement ?? 0) > 0;
        return (
          <Card
            key={i}
            withBorder
            radius="md"
            padding="sm"
            style={{ borderColor: improved ? '#40c057' : '#dee2e6', borderWidth: improved ? 2 : 1 }}
          >
            <Stack gap={6}>
              <Text fw={700} size="sm" c={improved ? 'green' : 'dimmed'}>
                #{i + 1}
                {item.prediction !== null && (
                  <Text span c="dark" fw={400} ml={6}>
                    pred = {item.prediction?.toFixed(3)}
                  </Text>
                )}
                <DeltaBadge delta={item.delta} />
              </Text>

              {item.mol_img_b64 && (
                <Image
                  src={`data:image/png;base64,${item.mol_img_b64}`}
                  radius="sm"
                  fit="contain"
                  h={160}
                  style={{ background: '#f8f9fa' }}
                />
              )}

              {'rationale' in item && item.rationale && (
                <Text size="xs" c="dimmed" fs="italic">{item.rationale}</Text>
              )}

              <Code fz={9} style={{ wordBreak: 'break-all', whiteSpace: 'pre-wrap' }}>
                {item.smiles}
              </Code>
            </Stack>
          </Card>
        );
      })}
    </SimpleGrid>
  );
}
