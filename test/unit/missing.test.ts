import assert from 'node:assert/strict'
import { describe, test } from 'node:test'
import { ml } from '../../src/index.ts'

// biome-ignore lint/suspicious/noControlCharactersInRegex: ANSI escapes
const plain = (text: string) => text.replace(/\u001b\[\d+m/g, '')

const surveySchema = ml.features({
  id: ml.number(),
  age: ml.number().optional(),
  smoker: ml.boolean().optional(),
  city: ml.category(['Lisbon', 'Porto', 'Faro']).optional(),
  score: ml.orderedCategory(['low', 'mid', 'high']).optional()
})

// The checks read rows by id, since the split shuffles them.
const rows = [
  { id: 0, age: 20, smoker: true, city: 'Porto', score: 'low' },
  { id: 1, age: 30, smoker: false, city: 'Porto', score: 'high' },
  { id: 2, age: 60, smoker: false, city: 'Lisbon', score: 'high' },
  { id: 3, age: null, smoker: null, city: null, score: null },
  { id: 4, age: 40, smoker: true, city: 'Lisbon', score: 'low' }
]
const { train } = ml
  .dataset(surveySchema, [...rows, { id: 5, age: 50 }])
  .split({ test: 1 / 6, seed: 1 })
assert.ok(
  train.rows.some(row => row.id === 3),
  'row 3 must be in training'
)
const byId = <R extends { id: number }>(prepared: {
  readonly rows: readonly R[]
}) => new Map(prepared.rows.map(row => [row.id, row]))

describe('missing', () => {
  test('fills with statistics of the training block', () => {
    const fitted = ml
      .pipeline(surveySchema)
      .missing({ age: 'mean', smoker: 'mode', city: 'mode', score: 'mode' })
      .fit(train)
    const filled = byId(fitted.transform(train)).get(3)
    const present = train.rows.filter(row => row.age !== undefined)
    const ages = present.map(row => row.age as number)
    assert.equal(filled?.age, ages.reduce((a, b) => a + b) / ages.length)
    assert.equal(typeof filled?.smoker, 'boolean')
    assert.notEqual(filled?.city, undefined)
    assert.notEqual(filled?.score, undefined)
  })

  test('median takes the middle of an even count as the mean of both', () => {
    const schema = ml.features({ x: ml.number().optional() })
    const { train } = ml
      .dataset(schema, [{ x: 1 }, { x: 2 }, { x: 10 }, { x: 20 }, {}, {}])
      .split({ test: 1 / 6, seed: 3 })
    const fitted = ml.pipeline(schema).missing({ x: 'median' }).fit(train)
    const xs = train.rows
      .map(row => row.x)
      .filter(x => x !== undefined)
      .sort((a, b) => a - b)
    const middle = xs.length / 2
    const expected =
      xs.length % 2 === 1
        ? xs[Math.floor(middle)]
        : ((xs[middle - 1] as number) + (xs[middle] as number)) / 2
    assert.match(plain(String(fitted)), new RegExp(`x: median ${expected}`))
  })

  test('breaks ties in mode by declared order', () => {
    const schema = ml.features({
      c: ml.category(['b', 'a']).optional(),
      flag: ml.boolean().optional()
    })
    const { train } = ml
      .dataset(schema, [
        { c: 'a', flag: true },
        { c: 'b', flag: false },
        {},
        {}
      ])
      .split({ test: 0.25, seed: 1 })
    // Whichever row went to test, one 'a' and one 'b' can't both remain
    // unless it was a missing one; check against the counts that did.
    const fitted = ml
      .pipeline(schema)
      .missing({ c: 'mode', flag: 'mode' })
      .fit(train)
    const cs = train.rows.map(row => row.c).filter(c => c !== undefined)
    const expected = cs.length === 2 ? 'b' : cs[0]
    assert.match(plain(String(fitted)), new RegExp(`c: mode '${expected}'`))
  })

  test('fills with a constant', () => {
    const fitted = ml
      .pipeline(surveySchema)
      .missing({
        age: { constant: 0 },
        smoker: { constant: false },
        city: { constant: 'Faro' },
        score: { constant: 'mid' }
      })
      .fit(train)
    assert.deepEqual(byId(fitted.transform(train)).get(3), {
      id: 3,
      age: 0,
      smoker: false,
      city: 'Faro',
      score: 'mid'
    })
  })

  test("'error' throws at the row, by its position", () => {
    const pipeline = ml.pipeline(surveySchema).missing({
      age: 'error',
      smoker: 'mode',
      city: 'mode',
      score: 'mode'
    })
    assert.throws(
      () => pipeline.fit(train),
      /'age' is missing in row 3, and its missing-value policy is 'error'/
    )
  })

  test('a field with no training values has no mean', () => {
    const schema = ml.features({ x: ml.number().optional(), id: ml.number() })
    const { train } = ml
      .dataset(schema, [{ id: 1 }, { id: 2 }, { id: 3 }])
      .split({ test: 0.3, seed: 1 })
    assert.throws(
      () => ml.pipeline(schema).missing({ x: 'mean' }).fit(train),
      /'x' has no values in the training block, so it has no mean/
    )
  })

  test('test rows are filled with what training learned', () => {
    const schema = ml.features({ x: ml.number().optional(), id: ml.number() })
    const { train, test } = ml
      .dataset(schema, [
        { id: 0, x: 1 },
        { id: 1, x: 3 },
        { id: 2, x: 1000 },
        { id: 3 }
      ])
      .split({ test: 0.5, seed: 2 })
    const fitted = ml.pipeline(schema).missing({ x: 'mean' }).fit(train)
    const xs = train.rows.map(row => row.x).filter(x => x !== undefined)
    const mean = xs.reduce((a, b) => a + b, 0) / xs.length
    for (const row of fitted.transform(test).rows) {
      const original = test.rows.find(r => r.id === row.id)
      assert.equal(row.x, original?.x ?? mean)
    }
  })

  test('prints the policies and what each one learned', () => {
    const pipeline = ml
      .pipeline(surveySchema)
      .missing({ age: 'median', smoker: 'mode' })
    assert.equal(
      plain(String(pipeline)),
      [
        'pipeline',
        '  missing  age: median, smoker: mode',
        "pending: 'city', 'score' need a missing-value policy"
      ].join('\n')
    )
  })
})
