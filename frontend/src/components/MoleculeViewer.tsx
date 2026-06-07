import { Text } from '@mantine/core';
import type { FragCentroid } from '../types';

interface Props {
  molSvgB64: string;
  molSvgWidth: number;
  molSvgHeight: number;
  fragmentCentroids: FragCentroid[];
  lockedFrags: Set<number>;
  onToggleLock: (fragIdx: number) => void;
}

function fragColor(contribution: number, isLocked: boolean): string {
  if (isLocked) return '#7f8c8d';
  if (contribution < -0.05) return '#e74c3c';
  if (contribution > 0.05) return '#27ae60';
  return '#2980b9';
}

export default function MoleculeViewer({
  molSvgB64, molSvgWidth, molSvgHeight,
  fragmentCentroids, lockedFrags, onToggleLock,
}: Props) {
  if (!molSvgB64) {
    return (
      <Text c="dimmed" size="sm" ta="center" pt="xl">
        Enter a SMILES string to visualise the molecule.
      </Text>
    );
  }

  return (
    <div style={{ position: 'relative', width: '100%' }}>
      {/* Molecule SVG — scaled to fill container width, aspect ratio locked */}
      <img
        src={`data:image/svg+xml;base64,${molSvgB64}`}
        width={molSvgWidth}
        height={molSvgHeight}
        style={{ width: '100%', height: 'auto', display: 'block' }}
        alt="molecule"
      />

      {/* Overlay — exact same viewBox as the molecule SVG */}
      <svg
        viewBox={`0 0 ${molSvgWidth} ${molSvgHeight}`}
        style={{
          position: 'absolute',
          top: 0, left: 0,
          width: '100%', height: '100%',
        }}
      >
        {fragmentCentroids.map((c) => {
          const locked = lockedFrags.has(c.fragment_index);
          const color = fragColor(c.contribution, locked);
          return (
            <g
              key={c.fragment_index}
              onClick={() => onToggleLock(c.fragment_index)}
              style={{ cursor: 'pointer' }}
            >
              <circle cx={c.cx} cy={c.cy} r={16} fill={color} opacity={0.78} />
              <text
                x={c.cx} y={c.cy + 0.5}
                dominantBaseline="middle"
                textAnchor="middle"
                fill="white"
                fontSize={locked ? 13 : 10}
                fontWeight="700"
                style={{ pointerEvents: 'none', userSelect: 'none' }}
              >
                {locked ? '🔒' : `F${c.fragment_index}`}
              </text>
              <title>
                {`Fragment ${c.fragment_index}\nContribution: ${c.contribution.toFixed(4)}\n${locked ? 'Click to unlock' : 'Click to lock'}`}
              </title>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
