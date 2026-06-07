import { Table, ScrollArea, Text } from '@mantine/core';

interface Props {
  rows: Record<string, unknown>[];
  columns: string[];   // keys to show
  labels: string[];    // header labels
  contribKey?: string; // key to colour-code
  maxRows?: number;
}

function cellStyle(val: unknown, isContrib: boolean): React.CSSProperties {
  if (!isContrib || typeof val !== 'number') return {};
  return { color: val < 0 ? '#e74c3c' : '#27ae60', fontWeight: 600 };
}

export default function ContribTable({ rows, columns, labels, contribKey, maxRows = 20 }: Props) {
  if (!rows || rows.length === 0) {
    return <Text c="dimmed" size="sm">No data.</Text>;
  }

  const sorted = contribKey
    ? [...rows].sort((a, b) => Math.abs(b[contribKey] as number) - Math.abs(a[contribKey] as number))
    : rows;

  const visible = sorted.slice(0, maxRows);

  return (
    <ScrollArea>
      <Table striped highlightOnHover withTableBorder withColumnBorders fz="xs">
        <Table.Thead>
          <Table.Tr>
            {labels.map((l) => <Table.Th key={l}>{l}</Table.Th>)}
          </Table.Tr>
        </Table.Thead>
        <Table.Tbody>
          {visible.map((row, i) => (
            <Table.Tr key={i}>
              {columns.map((col) => (
                <Table.Td key={col} style={cellStyle(row[col], col === contribKey)}>
                  {typeof row[col] === 'number'
                    ? (row[col] as number).toFixed(4)
                    : String(row[col] ?? '')}
                </Table.Td>
              ))}
            </Table.Tr>
          ))}
        </Table.Tbody>
      </Table>
    </ScrollArea>
  );
}
