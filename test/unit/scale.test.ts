import assert from 'node:assert/strict'
import { describe, test } from 'node:test'
import { ml } from '../../src/index.ts'

// biome-ignore lint/suspicious/noControlCharactersInRegex: ANSI escapes
const plain = (text: string) => text.replace(/\u001b\[\d+m/g, '')

const close = (actual: number | undefined, expected: number) =>
  assert.ok(
    Math.abs((actual as number) - expected) < 1e-12,
    `${actual} is not ${expected}`
  )

const mean = (xs: readonly number[]) =>
  xs.reduce((a, b) => a + b, 0) / xs.length
const std = (xs: readonly number[]) => {
  const m = mean(xs)
  return Math.sqrt(mean(xs.map(x => (x - m) ** 2)))
}

describe('scale', () => {
  const schema = ml.features({
    id: ml.number(),
    x: ml.number().optional(),
    level: ml.orderedCategory(['low', 'mid', 'high']),
    flat: ml.number()
  })
  const data = ml.dataset(schema, [
    { id: 0, x: 1, level: 'low', flat: 5 },
    { id: 1, x: 2, level: 'high', flat: 5 },
    { id: 2, x: 6, level: 'mid', flat: 5 },
    { id: 3, level: 'high', flat: 5 },
    { id: 4, x: 100, level: 'low', flat: 5 }
  ])
  const { train, test: held } = data.split({ test: 0.2, seed: 1 })
  const pipeline = ml.pipeline(schema).scale(['x', 'level', 'flat'])
  const fitted = pipeline.fit(train)

  test('standardizes with the mean and population std of training', () => {
    const xs = train.rows.map(row => row.x).filter(x => x !== undefined)
    for (const [i, row] of fitted.transform(held).rows.entries()) {
      const original = held.rows[i]
      if (original?.x === undefined) assert.equal(row.x, undefined)
      else close(row.x, (original.x - mean(xs)) / std(xs))
    }
  })

  test('scales an ordered category by its position', () => {
    const positions = train.rows.map(row =>
      ['low', 'mid', 'high'].indexOf(row.level)
    )
    fitted.transform(train).rows.forEach((row, i) => {
      close(
        row.level,
        ((positions[i] as number) - mean(positions)) / std(positions)
      )
    })
  })

  test('only centers a field with no spread, and says so', () => {
    for (const row of fitted.transform(held).rows) assert.equal(row.flat, 0)
    assert.match(
      plain(String(fitted)),
      /flat: mean 5, std 0 \(no spread in training: only centered\)/
    )
  })

  test('a constant after scale is scaled with the field', () => {
    const filled = ml
      .pipeline(schema)
      .scale(['x'])
      .missing({ x: { constant: 0 } })
      .fit(train)
    const xs = train.rows.map(row => row.x).filter(x => x !== undefined)
    const row = train.rows.findIndex(r => r.x === undefined)
    assert.notEqual(row, -1, 'the row without x must be in training')
    close(filled.transform(train).rows[row]?.x, (0 - mean(xs)) / std(xs))
  })

  test('the matrix uses the scaled values', () => {
    const complete = ml
      .pipeline(schema)
      .scale(['level'])
      .missing({ x: 'median' })
      .fit(train)
    const prepared = complete.transform(train)
    const matrix = prepared.matrix()
    const column = matrix.columns.indexOf('level')
    prepared.rows.forEach((row, i) => {
      assert.equal(matrix.values[i]?.[column], row.level)
    })
  })

  test('prints the configuration', () => {
    assert.equal(
      plain(String(pipeline)),
      "pipeline\n  scale    x, level, flat\npending: 'x' need a missing-value policy"
    )
  })
})
