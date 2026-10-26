// Time series example with the classic Box & Jenkins airline data: monthly
// international airline passengers, in thousands, from 1949 to 1960.
// Run with: npm run example:air-passengers

import { ml } from '../../src/index.ts'
import raw from './data.json' with { type: 'json' }

// Rows as they would come from a CSV parser: untrusted until parsed.
const rows: unknown[] = raw

// ── M1: Feature DSL ──────────────────────────────────────────────────────────

const airPassengerSchema = ml.features({
  month: ml.timestamp(), // the first day of each month, in UTC
  passengers: ml.number()
})

// ── M2: Series, split and cross-validation ───────────────────────────────────

// The entry point for training rows: validated and sorted by time, once, here.
const airPassengers = ml.series(airPassengerSchema, rows, { time: 'month' })
console.log(`${airPassengers.rows.length} months of passengers`)

// The last year is the test block, kept out of every choice below. A target
// that looks one month ahead needs a gap of at least one month.
const { train, test } = airPassengers.split({
  testFrom: '1960-01-01',
  gap: { months: 1 }
})
console.log(
  `${train.rows.length} months for training, ${test.rows.length} for testing\n`
)

// Train on 3 years, validate on the next one, and move a year at a time.
// Printing the folds draws them, so the cut can be checked before anything
// trains on it.
const rolling = ml.walkForward({
  validationFrom: '1955-01-01',
  train: { years: 3 },
  validation: { years: 1 },
  gap: { months: 1 }
})
console.log(rolling.folds(train))

// Or keep all the history and validate every 6 months
const expanding = ml.walkForward({
  validationFrom: '1957-01-01',
  validation: { months: 6 },
  train: 'expanding',
  gap: { months: 1 }
})
console.log(`\n${expanding.folds(train)}`)

// A fold lists the months it trains and validates on
console.log(`\n${rolling.folds(train)[0]}`)

// ── M3: Preprocessing ────────────────────────────────────────────────────────
// ── M4: Linear regression ────────────────────────────────────────────────────
// ── M5: Pipeline, backtest and metrics ───────────────────────────────────────

// const model = ml
//   .pipeline(airPassengerSchema)
//   .target('passengers', { transform: 'log-return', horizon: 1 })
//   .lag('passengers', [1, 12]) // last month and the same month last year
//   .missing({ warmup: 'drop' })
//   .estimator(ml.linearRegression({ solver: 'normal-equation' }))

// const report = model.backtest(train, rolling)
// console.log(report)
