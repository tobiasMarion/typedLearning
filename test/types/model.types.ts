import {
  type FittedLinearRegression,
  type FittedModel,
  ml,
  type Unlabeled
} from '../../src/index.ts'
import type { Equal, Expect } from './helpers.ts'

const massSchema = ml
  .features({
    island: ml.category(['Biscoe', 'Dream', 'Torgersen']),
    flipperLengthMm: ml.number().optional(),
    sex: ml.category(['male', 'female']),
    bodyMassG: ml.number()
  })
  .target('bodyMassG')
const { train, test } = ml
  .dataset(massSchema, [])
  .split({ validation: 0.2, test: 0.2, seed: 1 })

const regression = ml.linearRegression({ solver: 'normal-equation' })
const pipeline = ml
  .pipeline(massSchema)
  .missing({ flipperLengthMm: 'median' })
  .encode({ island: { reference: 'Biscoe' }, sex: { reference: 'female' } })
  .scale(['flipperLengthMm'])

const fitted = pipeline.estimator(regression).fit(train)

type C = 'island=Dream' | 'island=Torgersen' | 'flipperLengthMm' | 'sex=male'
type _estimator = Expect<
  Equal<typeof fitted.estimator, FittedLinearRegression<C>>
>
type _fitted = Expect<
  Equal<
    typeof fitted,
    FittedModel<
      (typeof massSchema)['fields'],
      'bodyMassG',
      'flipperLengthMm',
      never,
      'island=Biscoe' | 'sex=female'
    >
  >
>

// Records to predict have every field but the target
type _unlabeled = Expect<
  Equal<
    Unlabeled<(typeof massSchema)['fields'], 'bodyMassG'>,
    {
      island: 'Biscoe' | 'Dream' | 'Torgersen'
      flipperLengthMm?: number | undefined
      sex: 'male' | 'female'
    }
  >
>
const predictions = fitted.predict([
  { island: 'Dream', sex: 'male', flipperLengthMm: 200 },
  { island: 'Biscoe', sex: 'female' }
])
type _predictions = Expect<Equal<typeof predictions, readonly number[]>>

// @ts-expect-error — the target isn't an input
fitted.predict([{ island: 'Dream', sex: 'male', bodyMassG: 4000 }])

// @ts-expect-error — 'Gentoo' isn't an island
fitted.predict([{ island: 'Gentoo', sex: 'male' }])

const model = pipeline.estimator(regression)
// @ts-expect-error — only a training block fits
model.fit(test)

const unfilled = ml
  .pipeline(massSchema)
  .encode({ island: { reference: 'Biscoe' }, sex: { reference: 'female' } })
// @ts-expect-error — flipperLengthMm still needs a missing-value policy
unfilled.estimator(regression)

const noReference = ml
  .pipeline(massSchema)
  .missing({ flipperLengthMm: 'median' })
// @ts-expect-error — 'island' and 'sex' need a reference category
noReference.estimator(regression)

const speciesSchema = ml
  .features({ species: ml.category(['Adelie', 'Gentoo']), x: ml.number() })
  .target('species')
// @ts-expect-error — linear regression predicts a number; 'species' is a category
ml.pipeline(speciesSchema).estimator(regression)

// @ts-expect-error — without a target there is nothing to learn
ml.pipeline(ml.features({ x: ml.number() })).estimator(regression)

// evaluate takes a held-out block and returns one number per metric
const { validation } = ml
  .dataset(massSchema, [])
  .split({ validation: 0.2, test: 0.2, seed: 2 })
const scores = fitted.evaluate(validation, [ml.mse, ml.r2])
type _scores = Expect<
  Equal<typeof scores, { readonly mse: number; readonly r2: number }>
>
fitted.evaluate(test, [ml.mae])

// @ts-expect-error — the training block isn't held out
fitted.evaluate(train, [ml.mse])

// @ts-expect-error — at least one metric
fitted.evaluate(validation, [])
