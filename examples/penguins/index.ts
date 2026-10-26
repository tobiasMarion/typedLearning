// End-to-end example with the Palmer Penguins dataset.
// It starts fully commented out: uncomment each block as its feature lands.
// Run with: npm run example (or npm run example:bad for a messy dataset)

import { ml } from '../../src/index.ts'
import badData from './bad-data.json' with { type: 'json' }
import goodData from './data.json' with { type: 'json' }

// Rows as they would come from a CSV parser: untrusted until parsed.
const rows: unknown[] = process.argv.includes('--bad') ? badData : goodData

// ── M1: Feature DSL ──────────────────────────────────────────────────────────

// The target is the field a model predicts: a category, so classification.
const penguinSchema = ml
  .features({
    species: ml.category(['Adelie', 'Chinstrap', 'Gentoo']),
    island: ml.category(['Biscoe', 'Dream', 'Torgersen']),
    billLengthMm: ml.number().optional(),
    billDepthMm: ml.number().optional(),
    flipperLengthMm: ml.number().optional(),
    bodyMassG: ml.number().optional(),
    sex: ml.category(['male', 'female']).optional(),
    // Not in the original dataset: a made-up field-assessment score, from lowest
    // to highest, to show ordered categories.
    bodyCondition: ml.orderedCategory(['poor', 'fair', 'good'])
  })
  .target('species')

type Penguin = typeof penguinSchema.infer

// Validating a single record on its own, for example a form entry. Training
// rows go through ml.dataset below; predict trusts typed input like this one.
const entry: Penguin = penguinSchema.parse({
  species: 'Gentoo',
  island: 'Biscoe',
  billLengthMm: 47.5,
  billDepthMm: null,
  flipperLengthMm: 218,
  bodyMassG: 4950,
  sex: 'male',
  bodyCondition: 'good'
})
console.log(entry)

// ── M2: Dataset, split and cross-validation ──────────────────────────────────

// The entry point for training rows: validated once, here.
const penguins = ml.dataset(penguinSchema, rows)
console.log(`${penguins.rows.length} valid penguins`)

const { train, validation, test } = penguins.split({
  validation: 0.15,
  test: 0.15,
  seed: 42,
  stratify: 'species'
})
console.log(
  `${train.rows.length} for training, ${validation.rows.length} for validation, ${test.rows.length} for testing`
)

// Cross-validation resamples only the training block: each fold holds out a
// different part of it, and the test block stays out of every fold.
// Printing the folds draws them, with the species each one holds out; printing
// a fold lists its rows, numbered by their position in the data.
const cv = ml.kFold({ k: 3, seed: 42, stratify: 'species' })
const folds = cv.folds(train)
console.log(folds)
console.log(folds[0])

// ── M3: Preprocessing ────────────────────────────────────────────────────────

// Built from the schema: nothing here has seen data yet. Every optional field
// needs a missing-value policy, and KNN measures distances, so the numbers it
// compares are scaled first.
const pipeline = ml
  .pipeline(penguinSchema)
  .missing({
    billLengthMm: 'median',
    billDepthMm: 'median',
    flipperLengthMm: 'median',
    sex: 'mode'
  })
  .scale(['billLengthMm', 'billDepthMm', 'flipperLengthMm', 'bodyMassG'])
  .missing({
    bodyMassG: { knn: 3, using: ['billLengthMm', 'flipperLengthMm', 'island'] }
  })
console.log(pipeline)

// fit learns every statistic from the training block only; transform applies
// them to any block, which keeps its role. The matrix is what an estimator
// will get: categories one-hot, the ordered category by position, the target
// apart.
const fitted = pipeline.fit(train)
console.log(fitted)
console.log(fitted.transform(test).matrix())

// ── M4: Linear regression ────────────────────────────────────────────────────

// Predicting the body mass, a number, so regression. The target has to be
// present: rows without it can't train or score, so they leave here, at the
// boundary, before the split.
const massSchema = ml
  .features({
    island: ml.category(['Biscoe', 'Dream', 'Torgersen']),
    flipperLengthMm: ml.number().optional(),
    sex: ml.category(['male', 'female']).optional(),
    bodyMassG: ml.number()
  })
  .target('bodyMassG')

const weighed = (rows as { bodyMassG?: unknown }[]).filter(
  row => row.bodyMassG != null
)
const masses = ml.dataset(massSchema, weighed)
const massSplit = masses.split({ test: 0.25, seed: 42, stratify: 'island' })

// Next to the intercept, a column for every island would add up to it, and
// the weights would have no single answer: each one-hot field reads against a
// reference category instead.
const massPipeline = ml
  .pipeline(massSchema)
  .missing({ flipperLengthMm: 'median', sex: 'mode' })
  .encode({ island: { reference: 'Biscoe' }, sex: { reference: 'female' } })
  .scale(['flipperLengthMm'])
const preparedMasses = massPipeline.fit(massSplit.train)
const trainMatrix = preparedMasses.transform(massSplit.train).matrix()
console.log(trainMatrix)

// The closed-form solution, and gradient descent from zero: with enough
// epochs they find the same weights.
const exact = ml
  .linearRegression({ solver: 'normal-equation' })
  .fit(trainMatrix)
console.log(exact)
const stepped = ml
  .linearRegression({
    solver: 'gradient-descent',
    learningRate: 0.3,
    epochs: 5000
  })
  .fit(trainMatrix)
console.log(stepped)

// Predictions for the test block, in row order. Scoring them is evaluate's
// job, in M5.
console.log(
  exact.predict(preparedMasses.transform(massSplit.test).matrix()),
  massSplit.test.rows.map(row => row.bodyMassG)
)

// ── M5: Pipeline and metrics ─────────────────────────────────────────────────
// ── M6: Second algorithm ─────────────────────────────────────────────────────

// const model = pipeline.estimator(ml.knn({ k: 5, distance: 'euclidean' }))

// const fitted = model.fit(train)
// console.log(fitted.evaluate(test, [ml.accuracy, ml.confusionMatrix]))

// console.log(
//   fitted.predict({
//     island: 'Biscoe',
//     billLengthMm: 47.5,
//     billDepthMm: 15.0,
//     flipperLengthMm: 218,
//     bodyMassG: 4950,
//     sex: 'male'
//   })
// )
