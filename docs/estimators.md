# Estimators

## Contracts

| Concept     | Definition                                                           |
| ----------- | -------------------------------------------------------------------- |
| Estimator   | Learns something via `fit`. Doesn't have to predict (e.g., K-means). |
| Transformer | An estimator that learns a state and transforms data.                |
| Predictor   | An estimator with `predict`.                                         |
| Regressor   | A predictor of numeric values.                                       |
| Classifier  | A predictor of classes, possibly with `predictProba`.                |

These concepts are interfaces and types, not necessarily classes. The contracts live in `src/estimators/contracts.ts`.

## One API, multiple implementations

The same algorithm can have different implementations behind the same public API. Each implementation exposes only the options that apply to it:

```ts
ml.linearRegression({
  solver: 'gradient-descent',
  learningRate: 0.01,
  epochs: 1000
})
ml.linearRegression({ solver: 'normal-equation' })
// @ts-expect-error — learningRate doesn't apply to the closed-form solution
ml.linearRegression({ solver: 'normal-equation', learningRate: 0.01 })
```

We keep four things separate:
- **Algorithm:** linear regression, logistic regression, KNN…
- **Implementation:** educational, optimized.
- **Hyperparameters:** specific to the algorithm.
- **Implementation options:** exposed only when they apply.

Different implementations of the same algorithm follow the same `fit` and `predict` contract. Results may differ within documented and tested tolerances.

The educational implementation is central to the project, not a throwaway prototype. An optimized implementation (WASM, native code) only comes in when there is a concrete reason.

## Fitting on the matrix

In M4 an estimator stands on its own: it fits the [matrix](preprocessing.md#output) of a prepared block, and `.estimator()` joins it to the pipeline in M5. Predicting the body mass of the penguins, a number, so regression:

```ts
const massSchema = ml
  .features({
    island: ml.category(['Biscoe', 'Dream', 'Torgersen']),
    flipperLengthMm: ml.number().optional(),
    sex: ml.category(['male', 'female']).optional(),
    bodyMassG: ml.number()
  })
  .target('bodyMassG') // regression: inferred from a number

const pipeline = ml
  .pipeline(massSchema)
  .missing({ flipperLengthMm: 'median', sex: 'mode' })
  .encode({ island: { reference: 'Biscoe' }, sex: { reference: 'female' } })
  .scale(['flipperLengthMm'])

const prepared = pipeline.fit(train)
const model = ml.linearRegression({ solver: 'normal-equation' })

const fitted = model.fit(prepared.transform(train).matrix())
fitted.predict(prepared.transform(validation).matrix()) // number[], one per row
```

- **`fit`** takes the matrix of a `Train` only: the matrix keeps the role of its block, like `transform` does. Its target has to be a number. A one-hot field with every category doesn't compile, since an intercept makes it [linearly dependent](algorithms/linear-regression.md#linearly-dependent-columns).
- **`predict`** takes the matrix of any block with the same columns and returns one prediction per row, in row order.
- **The fitted model prints** its solver, the intercept, one weight per column by name, and the MSE on the training block; with gradient descent, also the loss before the first epoch and after the last.

```ts
// @ts-expect-error — a test block can't train a model
model.fit(prepared.transform(test).matrix())
// @ts-expect-error — linear regression predicts a number; 'species' is a category
model.fit(speciesTrainMatrix) // the training matrix of a schema with .target('species')
```

MSE is the loss here, reported on the training block. Scoring validation and test blocks belongs to `evaluate` in [M5](../README.md#milestones).

## Open questions

- How to represent unsupervised problems.
- Default implementation when the user doesn't pick one: should there be one, or is the choice required?
