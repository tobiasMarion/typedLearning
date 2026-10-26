import { ml, type Prepared } from '../../src/index.ts'
import type { Equal, Expect } from './helpers.ts'

const penguinSchema = ml
  .features({
    species: ml.category(['Adelie', 'Chinstrap', 'Gentoo']),
    island: ml.category(['Biscoe', 'Dream']),
    billLengthMm: ml.number(),
    bodyCondition: ml.orderedCategory(['poor', 'good']),
    isAdult: ml.boolean(),
    observedAt: ml.timestamp(),
    bodyMassG: ml.number().optional()
  })
  .target('species')
type S = typeof penguinSchema.fields

const penguins = ml.dataset(penguinSchema, [])
const { train, validation, test } = penguins.split({
  validation: 0.2,
  test: 0.2,
  seed: 42
})

const fitted = ml.pipeline(penguinSchema).fit(train)

// ── roles ────────────────────────────────────────────────────────────────────

// A prepared block keeps the role of the block it came from
const preparedTrain = fitted.transform(train)
const preparedValidation = fitted.transform(validation)
const preparedTest = fitted.transform(test)

type _train = Expect<
  Equal<
    typeof preparedTrain,
    Prepared<S, 'species', 'bodyMassG', never, never, 'train'>
  >
>
type _validation = Expect<
  Equal<
    typeof preparedValidation,
    Prepared<S, 'species', 'bodyMassG', never, never, 'validation'>
  >
>
type _test = Expect<
  Equal<
    typeof preparedTest,
    Prepared<S, 'species', 'bodyMassG', never, never, 'test'>
  >
>

// @ts-expect-error — a prepared test block isn't a prepared training block
export const testAsTrain: Prepared<
  S,
  'species',
  'bodyMassG',
  never,
  never,
  'train'
> = preparedTest

// @ts-expect-error — preprocessing learns only from training data
ml.pipeline(penguinSchema).fit(test)

// @ts-expect-error — preprocessing learns only from training data
ml.pipeline(penguinSchema).fit(validation)

// @ts-expect-error — the pipeline starts from a schema, not from data
ml.pipeline(penguins)

// @ts-expect-error — the training block must come from the same schema
ml.pipeline(ml.features({ x: ml.number() })).fit(train)

// ── rows ─────────────────────────────────────────────────────────────────────

type _rows = Expect<
  Equal<
    (typeof preparedTest.rows)[number],
    {
      species: 'Adelie' | 'Chinstrap' | 'Gentoo'
      island: 'Biscoe' | 'Dream'
      billLengthMm: number
      bodyCondition: 'poor' | 'good'
      isAdult: boolean
      observedAt: number
      bodyMassG?: number | undefined
    }
  >
>

// ── matrix ───────────────────────────────────────────────────────────────────

// @ts-expect-error — 'bodyMassG' still needs a missing-value policy
preparedTest.matrix()

const complete = ml
  .features({
    species: ml.category(['Adelie', 'Chinstrap', 'Gentoo']),
    island: ml.category(['Biscoe', 'Dream']),
    billLengthMm: ml.number(),
    bodyCondition: ml.orderedCategory(['poor', 'good']),
    isAdult: ml.boolean(),
    observedAt: ml.timestamp()
  })
  .target('species')
const completeTrain = ml
  .dataset(complete, [])
  .split({ test: 0.5, seed: 1 }).train
const matrix = ml
  .pipeline(complete)
  .fit(completeTrain)
  .transform(completeTrain)
  .matrix()

// One-hot columns come from the declared categories; the target and
// timestamps aren't columns
type _columns = Expect<
  Equal<
    (typeof matrix.columns)[number],
    | 'island=Biscoe'
    | 'island=Dream'
    | 'billLengthMm'
    | 'bodyCondition'
    | 'isAdult'
  >
>
type _y = Expect<
  Equal<typeof matrix.y, readonly ('Adelie' | 'Chinstrap' | 'Gentoo')[]>
>

// Without a target, there is no y
const pointSchema = ml.features({ x: ml.number() })
const points = ml.dataset(pointSchema, []).split({ test: 0.5, seed: 1 }).train
const unlabeled = ml
  .pipeline(pointSchema)
  .fit(points)
  .transform(points)
  .matrix()
type _noY = Expect<Equal<typeof unlabeled.y, undefined>>

// ── missing ──────────────────────────────────────────────────────────────────

const surveySchema = ml
  .features({
    label: ml.number(),
    age: ml.number().optional(),
    smoker: ml.boolean().optional(),
    city: ml.category(['Lisbon', 'Porto']).optional(),
    score: ml.orderedCategory(['low', 'high']).optional()
  })
  .target('label')

const filled = ml
  .pipeline(surveySchema)
  .missing({ age: 'median', smoker: 'mode' })
  .missing({ city: { constant: 'Porto' }, score: 'error' })
const surveyTrain = ml
  .dataset(surveySchema, [])
  .split({ test: 0.5, seed: 1 }).train
const surveyMatrix = filled.fit(surveyTrain).transform(surveyTrain).matrix()
type _filledRows = Expect<
  Equal<
    ReturnType<
      ReturnType<typeof filled.fit>['transform']
    >['rows'][number]['age'],
    number
  >
>
export const _surveyMatrix = surveyMatrix

// @ts-expect-error — 'label' is the target, and required
ml.pipeline(surveySchema).missing({ label: 'mean' })

// @ts-expect-error — 'mode' is for categories and booleans
ml.pipeline(surveySchema).missing({ age: 'mode' })

// @ts-expect-error — 'mean' is for numbers
ml.pipeline(surveySchema).missing({ city: 'mean' })

// @ts-expect-error — a constant must be a declared category
ml.pipeline(surveySchema).missing({ city: { constant: 'Faro' } })

// @ts-expect-error — 'drop' isn't a policy: filter rows before ml.dataset
ml.pipeline(surveySchema).missing({ age: 'drop' })

// @ts-expect-error — 'age' already has a policy
filled.missing({ age: 'mean' })

// ── encode ───────────────────────────────────────────────────────────────────

const reviewSchema = ml
  .features({
    rating: ml.number(),
    condition: ml.orderedCategory(['poor', 'fair', 'good']),
    size: ml.orderedCategory(['small', 'large']),
    color: ml.category(['red', 'blue'])
  })
  .target('rating')
const reviewTrain = ml
  .dataset(reviewSchema, [])
  .split({ test: 0.5, seed: 1 }).train
const encoded = ml.pipeline(reviewSchema).encode({ condition: 'one-hot' })
const encodedMatrix = encoded.fit(reviewTrain).transform(reviewTrain).matrix()
type _encodedColumns = Expect<
  Equal<
    (typeof encodedMatrix.columns)[number],
    | 'condition=poor'
    | 'condition=fair'
    | 'condition=good'
    | 'size'
    | 'color=red'
    | 'color=blue'
  >
>

// @ts-expect-error — a category is always one-hot
ml.pipeline(reviewSchema).encode({ color: 'one-hot' })

// @ts-expect-error — 'position' is the default; only 'one-hot' overrides it
ml.pipeline(reviewSchema).encode({ size: 'position' })

// @ts-expect-error — 'condition' is already one-hot
encoded.encode({ condition: 'one-hot' })

// @ts-expect-error — no ordered category is left to encode
encoded.encode({ size: 'one-hot' }).encode({ size: 'one-hot' })

// ── scale ────────────────────────────────────────────────────────────────────

const scaled = ml
  .pipeline(surveySchema)
  .scale(['age', 'score'])
  .missing({ age: 'mean', smoker: 'mode', city: 'mode', score: 'mode' })
type ScaledRow = ReturnType<
  ReturnType<typeof scaled.fit>['transform']
>['rows'][number]

// A scaled ordered category is its scaled position
type _scaledScore = Expect<Equal<ScaledRow['score'], number>>
type _unscaledCity = Expect<Equal<ScaledRow['city'], 'Lisbon' | 'Porto'>>

// @ts-expect-error — 'age' is already scaled
scaled.scale(['age'])

// @ts-expect-error — a category is one-hot, not scaled
ml.pipeline(surveySchema).scale(['city'])

// @ts-expect-error — 'label' is the target
ml.pipeline(surveySchema).scale(['label'])

// @ts-expect-error — a scaled ordered category keeps its position encoding
ml.pipeline(reviewSchema).scale(['condition']).encode({ condition: 'one-hot' })

// @ts-expect-error — a one-hot ordered category isn't scaled
encoded.scale(['condition'])

// @ts-expect-error — scale needs at least one field
ml.pipeline(surveySchema).scale([])

// ── knn ──────────────────────────────────────────────────────────────────────

const birdSchema = ml
  .features({
    species: ml.category(['Adelie', 'Gentoo']),
    island: ml.category(['Biscoe', 'Dream']),
    billLengthMm: ml.number(),
    condition: ml.orderedCategory(['poor', 'good']),
    isAdult: ml.boolean(),
    seenAt: ml.timestamp(),
    sex: ml.category(['male', 'female']).optional(),
    bodyMassG: ml.number().optional()
  })
  .target('species')

ml.pipeline(birdSchema)
  .scale(['billLengthMm', 'condition'])
  .missing({ sex: 'mode' })
  .missing({
    bodyMassG: {
      knn: 5,
      using: ['billLengthMm', 'condition', 'island', 'isAdult', 'sex']
    }
  })

// One-hot categories are comparable without scale
ml.pipeline(birdSchema)
  .encode({ condition: 'one-hot' })
  .missing({ bodyMassG: { knn: 3, using: ['condition', 'island'] } })

ml.pipeline(birdSchema).missing({
  // @ts-expect-error — a number has to be scaled to measure a distance
  bodyMassG: { knn: 5, using: ['billLengthMm'] }
})

ml.pipeline(birdSchema).missing({
  // @ts-expect-error — an ordered position has to be scaled too
  bodyMassG: { knn: 5, using: ['condition'] }
})

ml.pipeline(birdSchema).missing({
  // @ts-expect-error — 'sex' is still incomplete
  bodyMassG: { knn: 5, using: ['island', 'sex'] }
})

ml.pipeline(birdSchema).missing({
  // @ts-expect-error — the target can't define a neighbor
  bodyMassG: { knn: 5, using: ['species'] }
})

ml.pipeline(birdSchema).missing({
  // @ts-expect-error — a timestamp has no encoding
  bodyMassG: { knn: 5, using: ['seenAt'] }
})

ml.pipeline(birdSchema).missing({
  // @ts-expect-error — using has no default
  bodyMassG: { knn: 5 }
})
