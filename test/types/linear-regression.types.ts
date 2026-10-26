import { type FittedLinearRegression, ml } from '../../src/index.ts'
import type { Equal, Expect } from './helpers.ts'

const massSchema = ml
  .features({
    island: ml.category(['Biscoe', 'Dream', 'Torgersen']),
    flipperLengthMm: ml.number(),
    sex: ml.category(['male', 'female']),
    bodyMassG: ml.number()
  })
  .target('bodyMassG')
const { train, validation, test } = ml
  .dataset(massSchema, [])
  .split({ validation: 0.2, test: 0.2, seed: 1 })

const prepared = ml
  .pipeline(massSchema)
  .encode({ island: { reference: 'Biscoe' }, sex: { reference: 'female' } })
  .scale(['flipperLengthMm'])
  .fit(train)

const model = ml.linearRegression({ solver: 'normal-equation' })
const fitted = model.fit(prepared.transform(train).matrix())

type C = 'island=Dream' | 'island=Torgersen' | 'flipperLengthMm' | 'sex=male'
type _fitted = Expect<Equal<typeof fitted, FittedLinearRegression<C>>>
type _weights = Expect<Equal<keyof typeof fitted.weights, C>>

// Any block with the same columns can be predicted
const predictions = fitted.predict(prepared.transform(validation).matrix())
fitted.predict(prepared.transform(test).matrix())
type _predictions = Expect<Equal<typeof predictions, readonly number[]>>

// @ts-expect-error — a test block can't train a model
model.fit(prepared.transform(test).matrix())

// @ts-expect-error — nor can a validation block
model.fit(prepared.transform(validation).matrix())

// @ts-expect-error — 'island' and 'sex' need a reference category
model.fit(ml.pipeline(massSchema).fit(train).transform(train).matrix())

const islandMatrix = ml
  .pipeline(massSchema)
  .encode({ sex: { reference: 'female' } })
  .fit(train)
  .transform(train)
  .matrix()
// @ts-expect-error — 'island' still needs one
model.fit(islandMatrix)

const speciesSchema = ml
  .features({ species: ml.category(['Adelie', 'Gentoo']), x: ml.number() })
  .target('species')
const species = ml
  .dataset(speciesSchema, [])
  .split({ test: 0.5, seed: 1 }).train
// @ts-expect-error — linear regression predicts a number; 'species' is a category
model.fit(ml.pipeline(speciesSchema).fit(species).transform(species).matrix())

const pointSchema = ml.features({ x: ml.number() })
const points = ml.dataset(pointSchema, []).split({ test: 0.5, seed: 1 }).train
// @ts-expect-error — without a target there is nothing to predict
model.fit(ml.pipeline(pointSchema).fit(points).transform(points).matrix())

// @ts-expect-error — the solver has no default
ml.linearRegression({})

// @ts-expect-error — learningRate doesn't apply to the normal equation
ml.linearRegression({ solver: 'normal-equation', learningRate: 0.01 })
