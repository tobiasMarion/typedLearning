import assert from 'node:assert/strict'
import { describe, test } from 'node:test'
import { ml } from '../../src/index.ts'

// biome-ignore lint/suspicious/noControlCharactersInRegex: ANSI escapes
const plain = (text: string) => text.replace(/\u001b\[\d+m/g, '')

const pointSchema = ml
  .features({
    x: ml.number(),
    label: ml.category(['a', 'b'])
  })
  .target('label')

const points = ml.dataset(
  pointSchema,
  Array.from({ length: 10 }, (_, x) => ({ x, label: x % 2 ? 'b' : 'a' }))
)
const { train, test: testBlock } = points.split({ test: 0.3, seed: 1 })

describe('ml.pipeline', () => {
  test('fit returns a new value and leaves the pipeline as it was', () => {
    const pipeline = ml.pipeline(pointSchema)
    const first = pipeline.fit(train)
    const second = pipeline.fit(train)
    assert.notEqual(first, second)
    assert.ok(Object.isFrozen(pipeline))
    assert.ok(Object.isFrozen(first))
  })

  test('transform keeps the rows of a block in order', () => {
    const fitted = ml.pipeline(pointSchema).fit(train)
    assert.deepEqual(fitted.transform(testBlock).rows, testBlock.rows)
    assert.ok(Object.isFrozen(fitted.transform(testBlock).rows))
  })

  test('a prepared block prints its role and the positions of its rows', () => {
    const prepared = ml.pipeline(pointSchema).fit(train).transform(testBlock)
    const table = plain(String(prepared)).split('\n')
    assert.equal(table[0], 'prepared test: 3 rows')
    assert.deepEqual(
      table.slice(2).map(line => line.trim().split(/\s+/)[0]),
      testBlock.rows.map(row => String(points.rows.indexOf(row)))
    )
  })

  test('transform rejects a value that is not a block', () => {
    const fitted = ml.pipeline(pointSchema).fit(train)
    assert.throws(
      () => fitted.transform({ rows: [] } as never),
      /expects a block from split\(\) or folds\(\)/
    )
  })

  test('a fitted pipeline without steps prints the encoding', () => {
    const fitted = ml.pipeline(pointSchema).fit(train)
    assert.equal(plain(String(fitted)), 'fitted pipeline')
  })
})

describe('matrix', () => {
  const penguinSchema = ml
    .features({
      species: ml.category(['Adelie', 'Gentoo']),
      island: ml.category(['Biscoe', 'Dream', 'Torgersen']),
      billLengthMm: ml.number(),
      bodyCondition: ml.orderedCategory(['poor', 'fair', 'good']),
      isAdult: ml.boolean(),
      observedAt: ml.timestamp()
    })
    .target('species')
  const penguins = ml.dataset(penguinSchema, [
    {
      species: 'Adelie',
      island: 'Dream',
      billLengthMm: 39.1,
      bodyCondition: 'good',
      isAdult: true,
      observedAt: 0
    },
    {
      species: 'Gentoo',
      island: 'Biscoe',
      billLengthMm: 46.5,
      bodyCondition: 'poor',
      isAdult: false,
      observedAt: 0
    },
    {
      species: 'Adelie',
      island: 'Biscoe',
      billLengthMm: 40,
      bodyCondition: 'fair',
      isAdult: true,
      observedAt: 0
    }
  ])
  const { train } = penguins.split({ test: 0.3, seed: 1 })
  const fitted = ml.pipeline(penguinSchema).fit(train)
  const matrix = fitted.transform(train).matrix()

  test('encodes fields from the schema, with the target apart', () => {
    assert.deepEqual(matrix.columns, [
      'island=Biscoe',
      'island=Dream',
      'island=Torgersen',
      'billLengthMm',
      'bodyCondition',
      'isAdult'
    ])
    const expected = new Map([
      [39.1, [0, 1, 0, 39.1, 2, 1]],
      [46.5, [1, 0, 0, 46.5, 0, 0]],
      [40, [1, 0, 0, 40, 1, 1]]
    ])
    assert.equal(matrix.values.length, 2)
    train.rows.forEach((row, i) => {
      assert.deepEqual(matrix.values[i], expected.get(row.billLengthMm))
    })
    assert.deepEqual(
      matrix.y,
      train.rows.map(row => row.species)
    )
  })

  test('reports declared categories that training never had', () => {
    assert.match(
      plain(String(fitted)),
      /island: one-hot \(Biscoe, Dream, Torgersen\), not in training: .*'Torgersen'/
    )
    assert.match(
      plain(String(fitted)),
      /bodyCondition: position \(poor < fair < good\)/
    )
    assert.match(plain(String(fitted)), /observedAt: left out \(timestamp\)/)
  })

  test('names the one-hot fields with every category', () => {
    assert.deepEqual(matrix.fullOneHot, ['island'])
    const referenced = ml
      .pipeline(penguinSchema)
      .encode({ island: { reference: 'Dream' } })
      .fit(train)
      .transform(train)
      .matrix()
    assert.deepEqual(referenced.fullOneHot, [])
  })

  test('prints as a table numbered by position in the data', () => {
    const table = plain(String(matrix)).split('\n')
    assert.equal(table[0], "train matrix: 2 rows, 6 columns, target 'species'")
    assert.match(table[1] ?? '', /^#\s+island=Biscoe\s+island=Dream/)
  })

  test('throws while a field still needs a missing-value policy', () => {
    const schema = ml.features({ x: ml.number().optional() })
    const { train } = ml
      .dataset(schema, [{ x: 1 }, { x: 2 }])
      .split({ test: 0.5, seed: 1 })
    const prepared = ml.pipeline(schema).fit(train).transform(train)
    assert.deepEqual(prepared.pending, ['x'])
    assert.throws(
      () => (prepared as { matrix(): unknown }).matrix(),
      /'x' still need a missing-value policy/
    )
  })
})

describe('encode', () => {
  const reviewSchema = ml.features({
    condition: ml.orderedCategory(['poor', 'fair', 'good']),
    rating: ml.number()
  })
  const { train } = ml
    .dataset(reviewSchema, [
      { condition: 'fair', rating: 3 },
      { condition: 'good', rating: 5 },
      { condition: 'poor', rating: 1 }
    ])
    .split({ test: 0.3, seed: 1 })

  test('turns an ordered category into one-hot columns', () => {
    const pipeline = ml.pipeline(reviewSchema).encode({ condition: 'one-hot' })
    const matrix = pipeline.fit(train).transform(train).matrix()
    assert.deepEqual(matrix.columns, [
      'condition=poor',
      'condition=fair',
      'condition=good',
      'rating'
    ])
    train.rows.forEach((row, i) => {
      const position = ['poor', 'fair', 'good'].indexOf(row.condition)
      assert.deepEqual(
        matrix.values[i]?.slice(0, 3),
        [0, 1, 2].map(c => (c === position ? 1 : 0))
      )
    })
    assert.equal(
      plain(String(pipeline)),
      'pipeline\n  encode   condition: one-hot'
    )
    assert.match(
      plain(String(pipeline.fit(train))),
      /condition: one-hot \(poor, fair, good\)/
    )
  })

  test('leaves the column of a reference out', () => {
    const pipeline = ml
      .pipeline(reviewSchema)
      .encode({ condition: { reference: 'fair' } })
    const matrix = pipeline.fit(train).transform(train).matrix()
    assert.deepEqual(matrix.columns, [
      'condition=poor',
      'condition=good',
      'rating'
    ])
    train.rows.forEach((row, i) => {
      assert.deepEqual(matrix.values[i]?.slice(0, 2), [
        row.condition === 'poor' ? 1 : 0,
        row.condition === 'good' ? 1 : 0
      ])
    })
    assert.equal(
      plain(String(pipeline)),
      "pipeline\n  encode   condition: reference 'fair'"
    )
    assert.match(
      plain(String(pipeline.fit(train))),
      /condition: one-hot \(poor, good; reference fair\)/
    )
  })

  test('rejects a reference the schema does not declare', () => {
    assert.throws(
      () =>
        ml
          .pipeline(reviewSchema)
          .encode({ condition: { reference: 'great' } } as never),
      /the reference of 'condition' must be one of 'poor', 'fair', 'good', received 'great'/
    )
  })
})
