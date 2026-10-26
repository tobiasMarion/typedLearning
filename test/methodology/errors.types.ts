// The errors of docs/methodology.md that don't compile, by ID. Runtime and
// by-design statuses are in errors.test.ts.
import { ml, type Train } from '../../src/index.ts'

const penguinSchema = ml.features({
  species: ml.category(['Adelie', 'Chinstrap', 'Gentoo']),
  billLengthMm: ml.number()
})
type P = typeof penguinSchema.fields
const penguins = ml.dataset(penguinSchema, [])
const { test } = penguins.split({ test: 0.2, seed: 42 })

const priceSchema = ml.features({ date: ml.timestamp(), close: ml.number() })
const series = ml.series(priceSchema, [], { time: 'date' })

// S1: training on the test block
// @ts-expect-error — a Test isn't a Train
const _s1: Train<P> = test

// S4: fitting preprocessing on data that isn't the training block
// @ts-expect-error — a pipeline learns only from a Train
ml.pipeline(penguinSchema).fit(test)
// @ts-expect-error — and is built from a schema, not from data
ml.pipeline(penguins)

// S1: and fitting an estimator on the matrix of the test block
const massSchema = ml
  .features({ flipperLengthMm: ml.number(), bodyMassG: ml.number() })
  .target('bodyMassG')
const masses = ml.dataset(massSchema, []).split({ test: 0.2, seed: 42 })
const preparedMasses = ml.pipeline(massSchema).fit(masses.train)
ml.linearRegression({ solver: 'normal-equation' }).fit(
  // @ts-expect-error — the matrix keeps the role of its block
  preparedMasses.transform(masses.test).matrix()
)

// S2: splitting the test block again
// @ts-expect-error — only the original data has split
test.split({ test: 0.5, seed: 42 })

// T1: random split of a time series
// @ts-expect-error — a series only splits by time
series.split({ test: 0.2, seed: 42 })

// T1: shuffling the training block of a time series into folds
const seriesTrain = series.split({
  testFrom: '2025-01-01',
  gap: { steps: 1 }
}).train
// @ts-expect-error — k-fold only accepts a training block cut from a dataset
ml.kFold({ k: 5, seed: 42 }).folds(seriesTrain)

// T4: no gap between train and test
// @ts-expect-error — gap is required, even { steps: 0 }
series.split({ testFrom: '2025-01-01' })
// @ts-expect-error — walk-forward requires it too
ml.walkForward({
  validationFrom: 0,
  validation: { months: 1 },
  train: 'expanding'
})

// E1: a split that can't be repeated
// @ts-expect-error — seed is required
penguins.split({ test: 0.2 })

// D2: missing values filled by a default nobody chose
const birdSchema = ml.features({ bodyMassG: ml.number().optional() })
const { train: birds } = ml
  .dataset(birdSchema, [])
  .split({ test: 0.2, seed: 42 })
// @ts-expect-error — 'bodyMassG' needs a missing-value policy first
ml.pipeline(birdSchema).fit(birds).transform(birds).matrix()
