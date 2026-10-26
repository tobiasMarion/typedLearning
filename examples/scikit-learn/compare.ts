// Fits the body mass regression on all 344 Palmer Penguins with TL and writes
// what scikit-learn needs to repeat it: the rows of each block, the weights,
// the test predictions and how long each step took. compare.py reads it.
// Run both with: npm run compare:scikit-learn

import { writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { ml } from '../../src/index.ts'
import all from '../penguins/all.json' with { type: 'json' }

const REPEATS = 50

const massSchema = ml
  .features({
    island: ml.category(['Biscoe', 'Dream', 'Torgersen']),
    billLengthMm: ml.number().optional(),
    billDepthMm: ml.number().optional(),
    flipperLengthMm: ml.number().optional(),
    sex: ml.category(['male', 'female']).optional(),
    bodyMassG: ml.number()
  })
  .target('bodyMassG')

// The target has to be present: two penguins were never weighed
const weighed = all.filter(row => row.bodyMassG !== null)
const penguins = ml.dataset(massSchema, weighed)
const { train, test } = penguins.split({
  test: 0.25,
  seed: 42,
  stratify: 'island'
})

const pipeline = ml
  .pipeline(massSchema)
  .missing({
    billLengthMm: 'median',
    billDepthMm: 'median',
    flipperLengthMm: 'median',
    sex: 'mode'
  })
  .encode({ island: { reference: 'Biscoe' }, sex: { reference: 'female' } })
  .scale(['billLengthMm', 'billDepthMm', 'flipperLengthMm'])

/** The median time of `REPEATS` runs, in milliseconds, after one warm-up run. */
const time = <T>(run: () => T): { result: T; ms: number } => {
  let result = run()
  const times: number[] = []
  for (let i = 0; i < REPEATS; i++) {
    const start = performance.now()
    result = run()
    times.push(performance.now() - start)
  }
  times.sort((a, b) => a - b)
  return { result, ms: times[Math.floor(REPEATS / 2)] as number }
}

const prepare = time(() => {
  const prepared = pipeline.fit(train)
  return {
    trainMatrix: prepared.transform(train).matrix(),
    testMatrix: prepared.transform(test).matrix()
  }
})
const { trainMatrix, testMatrix } = prepare.result

const exact = time(() =>
  ml.linearRegression({ solver: 'normal-equation' }).fit(trainMatrix)
)
const stepped = time(() =>
  ml
    .linearRegression({
      solver: 'gradient-descent',
      learningRate: 0.1,
      epochs: 2000
    })
    .fit(trainMatrix)
)

const out = join(tmpdir(), 'tl-penguin-mass.json')
writeFileSync(
  out,
  JSON.stringify({
    // Positions in the weighed rows, which compare.py loads the same way
    train: train.rows.map(row => penguins.rows.indexOf(row)),
    test: test.rows.map(row => penguins.rows.indexOf(row)),
    columns: trainMatrix.columns,
    normalEquation: {
      intercept: exact.result.intercept,
      weights: exact.result.weights,
      predictions: exact.result.predict(testMatrix)
    },
    gradientDescent: {
      intercept: stepped.result.intercept,
      weights: stepped.result.weights
    },
    ms: {
      preprocessing: prepare.ms,
      normalEquation: exact.ms,
      gradientDescent: stepped.ms
    },
    repeats: REPEATS
  })
)
console.log(exact.result)
console.log(`TL results written to ${out}`)
