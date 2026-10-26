import assert from 'node:assert/strict'
import { describe, test } from 'node:test'
import { ml } from '../../src/index.ts'

// biome-ignore lint/suspicious/noControlCharactersInRegex: ANSI escapes
const plain = (text: string) => text.replace(/\u001b\[\d+m/g, '')

// Two clusters: small birds on Dream, big birds on Biscoe. Row 9 is the only
// test row (checked below), so the neighbors always come from rows 0-8.
const birdSchema = ml.features({
  id: ml.number(),
  island: ml.category(['Biscoe', 'Dream']),
  length: ml.number(),
  mass: ml.number().optional(),
  size: ml.category(['small', 'big']).optional()
})
const rows = [
  { id: 0, island: 'Dream', length: 30, mass: 3000, size: 'small' },
  { id: 1, island: 'Dream', length: 31, mass: 3100, size: 'small' },
  { id: 2, island: 'Dream', length: 32, mass: 3200, size: 'small' },
  { id: 3, island: 'Biscoe', length: 50, mass: 5000, size: 'big' },
  { id: 4, island: 'Biscoe', length: 51, mass: 5100, size: 'big' },
  { id: 5, island: 'Biscoe', length: 52, mass: 5200, size: 'big' },
  { id: 6, island: 'Dream', length: 31 },
  { id: 7, island: 'Biscoe', length: 51 },
  { id: 8, island: 'Biscoe', length: 49, mass: 4900, size: 'big' },
  { id: 9, island: 'Dream', length: 30.5 }
]

const split = (seed: number) =>
  ml.dataset(birdSchema, rows).split({ test: 0.1, seed })
const seed = [1, 2, 3, 4, 5, 6, 7, 8].find(
  s => split(s).test.rows[0]?.id === 9
) as number
const { train, test: held } = split(seed)

const pipeline = ml
  .pipeline(birdSchema)
  .scale(['length'])
  .missing({
    mass: { knn: 3, using: ['length', 'island'] },
    size: { knn: 3, using: ['length', 'island'] }
  })

describe('knn imputation', () => {
  test('fills from the nearest training rows', () => {
    const fitted = pipeline.fit(train)
    const byId = new Map(fitted.transform(train).rows.map(r => [r.id, r]))
    assert.equal(byId.get(6)?.mass, 3100)
    assert.equal(byId.get(6)?.size, 'small')
    assert.equal(byId.get(7)?.mass, 5100)
    assert.equal(byId.get(7)?.size, 'big')
  })

  test('test rows take neighbors from training only', () => {
    const fitted = pipeline.fit(train)
    const [row] = fitted.transform(held).rows
    assert.equal(row?.mass, 3100)
    assert.equal(row?.size, 'small')
  })

  test('prints what it keeps', () => {
    assert.match(
      plain(String(pipeline.fit(train))),
      /mass: knn 3 over length, island \(7 training rows with a value\)/
    )
  })

  test('needs at least knn training rows with a value', () => {
    const greedy = ml
      .pipeline(birdSchema)
      .scale(['length'])
      .missing({ mass: { knn: 8, using: ['length'] }, size: 'mode' })
    assert.throws(
      () => greedy.fit(train),
      /'mass' has 7 training rows with a value, fewer than knn: 8/
    )
  })

  test('knn must be a positive integer', () => {
    assert.throws(
      () =>
        ml
          .pipeline(birdSchema)
          .scale(['length'])
          .missing({ mass: { knn: 0, using: ['length'] } }),
      /knn for 'mass' must be a positive integer, received 0/
    )
  })

  test('using needs at least one field', () => {
    assert.throws(
      () => ml.pipeline(birdSchema).missing({ mass: { knn: 2, using: [] } }),
      /knn for 'mass' needs at least one field in using/
    )
  })
})
