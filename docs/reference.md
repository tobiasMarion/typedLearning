# API reference

A compact reference of what TL exports today, for someone (or an agent) who has to write a model with it without reading the rest of the docs. Everything comes from one import:

```ts
import { ml, type Infer } from 'tl'
```

TL is written in TypeScript and its types carry most of the rules: a block's role (`Train`, `Validation`, `Test`), the decisions still pending, the columns of the matrix. Run the type checker (`tsc --noEmit`) to see them; Node runs `.ts` files directly (`node model.ts`).

## 1. Schema

```ts
const rentalSchema = ml
  .features({
    listedAt: ml.timestamp(), // ISO string, Date or ms since 1970 → number (ms, UTC)
    sizeM2: ml.number(),
    rooms: ml.number().optional(), // missing values allowed: undefined, null, NaN
    furnished: ml.boolean(),
    district: ml.category(['north', 'south', 'center']), // literal union
    condition: ml.orderedCategory(['poor', 'fair', 'good']), // ranked, lowest first
    rent: ml.number()
  })
  .target('rent') // a required field; a number means regression

type Rental = Infer<typeof rentalSchema> // the record type
rentalSchema.parse(record) // validates one record, throws FeatureValidationError
rentalSchema.parseRows(rows) // validates many, issues grouped by field
```

A value outside the declared categories, an impossible date or a missing required value is an error that names the field and the rows.

## 2. Data and splits

```ts
const rentals = ml.dataset(rentalSchema, rows) // validates every row once

// Random split: seed is required; validation is optional; stratify takes a category field
const { train, validation, test } = rentals.split({
  validation: 0.2,
  test: 0.2,
  seed: 42
})
```

Data ordered by time is a series. Times must be distinct (or distinct per `group`, a required category field), and the split is chronological, with a required `gap`:

```ts
const listings = ml.series(rentalSchema, rows, { time: 'listedAt' }) // sorted by time
const { train, validation, test } = listings.split({
  validationFrom: '2025-01-01', // optional
  testFrom: '2025-04-01',
  gap: { steps: 0 } // rows dropped before each held-out block: { steps }, { days }, { months }…
})
// listings.split({ test: 0.2, seed: 42 }) doesn't compile: a random split of a series leaks the future
```

`train`, `validation` and `test` are `Train<S>`, `Validation<S>` and `Test<S>`. Each has `rows` (frozen records) and prints as a table. Only the original data has `split`.

Cross-validation cuts a training block into folds:

```ts
const folds = ml.kFold({ k: 5, seed: 42 }).folds(train) // a dataset's train only
const walk = ml
  .walkForward({
    validationFrom: '2024-07-01',
    validation: { months: 1 },
    train: 'expanding',
    gap: { steps: 0 }
  })
  .folds(seriesTrain) // a series' train only
for (const fold of folds) {
  fold.train // Train<S>
  fold.validation // Validation<S>
}
```

## 3. Pipeline and model

```ts
const model = ml
  .pipeline(rentalSchema)
  .missing({ rooms: 'median' }) // required for every optional field
  .encode({ district: { reference: 'north' } }) // one-hot fields need a reference next to an intercept
  .scale(['sizeM2', 'rooms'])
  .estimator(ml.linearRegression({ solver: 'normal-equation' }))

const fitted = model.fit(train) // only a Train
```

- **`missing`**: per optional field, `'mean'`, `'median'`, `'mode'`, `{ constant: value }`, `'error'`, or `{ knn: k, using: [fields] }`. Statistics come from the training block.
- **`encode`**: categories are one-hot by default, ordered categories one column with their position. `{ field: { reference: category } }` leaves the reference's column out; `{ field: 'one-hot' }` makes an ordered category one-hot.
- **`scale`**: standardizes numbers and ordered positions with the training mean and standard deviation.
- Timestamps aren't columns: derive numbers from them in the schema's data if you need them.
- **`ml.linearRegression`**: `{ solver: 'normal-equation' }`, or `{ solver: 'gradient-descent', learningRate, epochs }`. Always fits an intercept. Linearly dependent columns are an error that names them.

`.estimator()` doesn't compile while a decision is missing; the error says which:

```
Argument of type 'LinearRegression' is not assignable to parameter of type
"'rooms' still needs a missing-value policy: add .missing({ rooms: ... }) to the pipeline"
```

## 4. Evaluate and predict

```ts
fitted.evaluate(validation, [ml.mse, ml.mae, ml.r2]) // { mse: number, mae: number, r2: number }
fitted.evaluate(test, [ml.r2]) // a test block is evaluated once
fitted.predict(records) // readonly number[], one per record, in order
```

- **`evaluate`** takes a `Validation` or a `Test`, never a `Train`. A test block can be evaluated once per process: evaluating it again, with this model or another, throws. Compare models on a validation block or with cross-validation.
- **`predict`** takes typed records without the target and prepares them as the training block was. It doesn't validate them: parse external records first, with a schema that has every field but the target:

  ```ts
  const { rent: _, ...inputs } = rentalSchema.fields
  const inputSchema = ml.features(inputs)
  fitted.predict(inputSchema.parseRows(newRows))
  ```

- **Metrics**: `ml.mse`, `ml.mae`, `ml.r2`, each `(yTrue, yPred) => number`.
- **Inspection**: `console.log` any value. `fitted.estimator` has `intercept`, `weights` (by column name) and the training `mse`; `fitted.pipeline.transform(block)` gives the prepared rows, and `.matrix()` the numbers the estimator sees.
