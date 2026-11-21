// The errors of docs/methodology.md that the library catches at runtime or
// prevents by design, by ID. The casts stand for JavaScript callers, where the
// types don't reach.
import assert from 'node:assert/strict'
import { describe, test } from 'node:test'
import { FeatureValidationError, ml } from '../../src/index.ts'

const priceSchema = ml.features({
  date: ml.timestamp(),
  ticker: ml.category(['AAPL', 'MSFT']),
  close: ml.number()
})

const date = (day: number) => `2025-01-${String(day).padStart(2, '0')}`
const days = Array.from({ length: 10 }, (_, i) => i + 1)
const rows = days.map(day => ({ date: date(day), ticker: 'AAPL', close: day }))

type Options = Parameters<
  ReturnType<typeof ml.series<typeof priceSchema.fields>>['split']
>[0]

const issueCodes = (run: () => unknown) => {
  try {
    run()
  } catch (error) {
    assert.ok(error instanceof FeatureValidationError)
    return error.issues.map(issue => issue.code)
  }
  assert.fail('expected a FeatureValidationError')
}

describe('time series', () => {
  test('T1: a random split of a series throws', () => {
    const series = ml.series(priceSchema, rows, { time: 'date' })
    assert.throws(
      () => series.split({ test: 0.2, seed: 42 } as unknown as Options),
      TypeError
    )
  })

  test('T2: the split follows time, not the order the rows arrived in', () => {
    const shuffled = [...rows].reverse()
    const { train, test } = ml
      .series(priceSchema, shuffled, { time: 'date' })
      .split({ testFrom: date(8), gap: { steps: 0 } })
    assert.deepEqual(
      train.rows.map(row => row.close),
      [1, 2, 3, 4, 5, 6, 7]
    )
    assert.deepEqual(
      test.rows.map(row => row.close),
      [8, 9, 10]
    )
  })

  test('T3: series stacked without a group repeat times and throw', () => {
    const stacked = [...rows, ...rows.map(row => ({ ...row, ticker: 'MSFT' }))]
    assert.deepEqual(
      issueCodes(() => ml.series(priceSchema, stacked, { time: 'date' })),
      ['duplicate_time']
    )
  })

  test('T4: a split without a gap throws', () => {
    const series = ml.series(priceSchema, rows, { time: 'date' })
    assert.throws(
      () => series.split({ testFrom: date(8) } as unknown as Options),
      TypeError
    )
  })

  test('T6: every group is cut at the same time', () => {
    // MSFT has no row on Jan 6 and Jan 7, so counting steps per group would
    // drop Jan 4 and 5 from MSFT and leave it a different cut than AAPL
    const panel = [
      ...rows,
      ...rows
        .filter(row => row.close !== 6 && row.close !== 7)
        .map(row => ({ ...row, ticker: 'MSFT' }))
    ]
    const { train, test } = ml
      .series(priceSchema, panel, { time: 'date', group: 'ticker' })
      .split({ testFrom: date(8), gap: { steps: 2 } })
    const firstTest = Math.min(...test.rows.map(row => row.date))
    assert.equal(firstTest, Date.parse(date(8)))
    for (const ticker of ['AAPL', 'MSFT']) {
      const dates = train.rows
        .filter(row => row.ticker === ticker)
        .map(row => row.date)
      assert.equal(Math.max(...dates), Date.parse(date(5)), ticker)
    }
  })
})

describe('train and test', () => {
  test('S3: comparing models on the test block throws at the second one', () => {
    const massSchema = ml
      .features({ flipperLengthMm: ml.number(), bodyMassG: ml.number() })
      .target('bodyMassG')
    const masses = ml.dataset(
      massSchema,
      Array.from({ length: 8 }, (_, i) => ({
        flipperLengthMm: 180 + i,
        bodyMassG: 3000 + 40 * i + (i % 2) * 15
      }))
    )
    const { train, test } = masses.split({ test: 0.25, seed: 42 })
    const candidates = [
      ml.linearRegression({ solver: 'normal-equation' }),
      ml.linearRegression({
        solver: 'gradient-descent',
        learningRate: 0.1,
        epochs: 100
      })
    ]
    const pipeline = ml.pipeline(massSchema).scale(['flipperLengthMm'])
    assert.throws(
      () =>
        candidates.map(candidate =>
          pipeline.estimator(candidate).fit(train).evaluate(test, [ml.mse])
        ),
      /already evaluated as test data/
    )
  })
})

describe('evaluation', () => {
  test('E1: a random split without a seed throws', () => {
    const penguinSchema = ml.features({ billLengthMm: ml.number() })
    const penguins = ml.dataset(penguinSchema, [
      { billLengthMm: 40 },
      { billLengthMm: 41 }
    ])
    assert.throws(
      () =>
        penguins.split({ test: 0.5 } as unknown as Parameters<
          typeof penguins.split
        >[0]),
      RangeError
    )
  })
})

describe('data', () => {
  test('D1: unknown categories and impossible dates throw', () => {
    assert.deepEqual(
      issueCodes(() =>
        ml.series(
          priceSchema,
          [
            { date: '2025-02-30', ticker: 'AAPL', close: 1 },
            { date: date(1), ticker: 'GOOG', close: 1 }
          ],
          { time: 'date' }
        )
      ),
      ['invalid_timestamp', 'unknown_category']
    )
  })

  test('D2: a missing value in a required field throws', () => {
    assert.deepEqual(
      issueCodes(() =>
        ml.series(priceSchema, [{ date: date(1), ticker: 'AAPL' }], {
          time: 'date'
        })
      ),
      ['missing']
    )
  })
})
