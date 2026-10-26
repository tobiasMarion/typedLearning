# Preprocessing

Preprocessing turns typed records into the numeric matrix an estimator consumes. It is a builder that starts from the schema:

```ts
const penguinSchema = ml
  .features({
    species: ml.category(['Adelie', 'Chinstrap', 'Gentoo']),
    island: ml.category(['Biscoe', 'Dream', 'Torgersen']),
    billLengthMm: ml.number(),
    flipperLengthMm: ml.number(),
    bodyMassG: ml.number().optional(),
    sex: ml.category(['male', 'female']).optional()
  })
  .target('species')

const pipeline = ml
  .pipeline(penguinSchema)
  .missing({ sex: 'mode' })
  .scale(['billLengthMm', 'flipperLengthMm', 'bodyMassG'])
  .missing({
    bodyMassG: {
      knn: 5,
      using: ['billLengthMm', 'flipperLengthMm', 'island', 'sex']
    }
  })

const fitted = pipeline.fit(train) // only accepts a Train
const prepared = fitted.transform(test) // still a Test
console.log(prepared.matrix())
```

- **From the schema, never from data.** `ml.pipeline(schema)` describes what to do with records of that shape. Passing all the data to the pipeline would suggest preprocessing all of it (S4), and a pipeline loaded in production has no data around it.
- **Learned in `fit`, applied in `transform`.** Every statistic (means, medians, standard deviations, neighbors) comes from the training block only. `transform` applies that same state to any block, including the one it was fitted on.
- **Configuration and learned state are separate values.** `pipeline` holds what the user declared; `fitted` holds what `fit` produced. `fit` doesn't mutate the pipeline, so the same pipeline can be fitted once per cross-validation fold.
- **Steps run in the order they are written,** and each one sees the output of the previous one. Above, the KNN compares penguins on lengths that are already scaled.

The estimator joins the same builder in M5 (`.estimator(...)`, see [Pipeline](pipeline.md)).

## Target

The target is part of the schema, since it is part of the question the data answers:

```ts
const penguinSchema = ml.features({ ... }).target('species')
```

`.target(field)` returns a new schema with the same fields, so one set of fields can serve several questions. The target:
- decides the kind of problem: a `number` is regression; a `boolean`, `category` or `orderedCategory` is classification. A timestamp can't be a target;
- must be a required field. Rows without a label are a choice about the population, made before `ml.dataset` (see [Missing values](#missing-values));
- is out of reach of every step: `missing`, `encode` and `scale` don't accept it;
- comes out of the matrix apart, as `y`;
- is not part of the record `predict` takes, and production records don't need it.

## Column state

Every column other than the target carries three facts in its type, and each step changes one of them:

| Fact     | From the schema                                    | Changed by | Required by                                                          |
| -------- | -------------------------------------------------- | ---------- | -------------------------------------------------------------------- |
| complete | required fields are; optional ones aren't          | `missing`  | `matrix()`, and `.estimator()` in M5                                 |
| encoding | the default of its kind, see [Encoding](#encoding) | `encode`   | nothing: every column is numeric by default                          |
| scaled   | no                                                 | `scale`    | anything that measures distances: KNN imputation, and `ml.knn` in M6 |

Each method only accepts the columns that still lack its fact: `missing` takes incomplete fields, `scale` takes numbers and ordered positions that aren't scaled yet, and `encode` takes categories still on their default. Filling a required field, scaling twice or re-encoding is a type error that says why. Once every column is complete, scaled where it matters and encoded as wanted, the methods have nothing left to accept.

Only completeness is required everywhere. Scaling isn't: the normal equation of a linear regression gives the same predictions with or without it, and one-hot columns are usually left as they are. It is required where a wrong scale changes the result without a word: in a distance, a field in grams drowns one in millimetres.

## Missing values

Every optional field needs a policy:

```ts
.missing({
  bodyMassG: 'median', // 'mean' | 'median' | { constant: number } | { knn, using } | 'error'
  sex: 'mode' // 'mode' | { constant: category } | { knn, using } | 'error'
})
```

| Policy            | Fields               | Fills a missing value with                                                                                              |
| ----------------- | -------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `'mean'`          | numbers              | the training mean                                                                                                       |
| `'median'`        | numbers              | the training median                                                                                                     |
| `'mode'`          | categories, booleans | the most frequent training value, ties by declared order (`false` before `true`)                                        |
| `{ constant: v }` | any                  | `v`, which must be a value the field accepts (milliseconds for a timestamp); after `scale`, it is scaled with the field |
| `{ knn, using }`  | any                  | the nearest training rows, see [KNN imputation](#knn-imputation)                                                        |
| `'error'`         | any                  | nothing: `fit` or `transform` throws, naming the field and the row                                                      |

`missing` can be called more than once, so some fields can be filled before `scale` and others after. A policy sees the field as it is at that point: a median after `scale` is the median of the scaled values. It behaves the same in `fit`, in `transform` and, later, in `predict`. Statistics ignore missing values and are computed once, in `fit`. A field whose training values are all missing has no mean, median or mode: `fit` throws and names it.

**There is no `'drop'`.** Dropping rows changes the population the model describes, and doing it inside the pipeline goes wrong in every block: dropping in validation and test removes exactly the hard rows and makes the metric optimistic, and a single record at `predict` can't be dropped. That choice belongs at the boundary: make the field required in the schema and filter the rows before `ml.dataset`, so every block, and production, sees the same population.

### KNN imputation

```ts
.missing({ sex: 'mode' })
.scale(['billLengthMm', 'flipperLengthMm'])
.missing({
  bodyMassG: { knn: 5, using: ['billLengthMm', 'flipperLengthMm', 'island', 'sex'] }
})
```

A missing value is filled from the `knn` training rows closest to the row being filled:
- **Distance** is euclidean over the columns of the `using` fields, as they are at that point. Each of them must be complete and comparable: numbers and ordered positions must be scaled, and one-hot columns already are 0 or 1. A field that isn't is a type error that lists the fields usable at that point. Timestamps have no encoding, so they can't be in it either. The field being filled and the target can't be in `using`, and `using` has no default: which fields define a neighbor changes the result.
- **Neighbors** are training rows that have the value. In `fit` and `transform` alike they come from the training block only: a validation row never fills another one. Ties in distance go to the earlier row.
- **The value** is the neighbors' mean for a number, in the field's current scale, and their most frequent category for a category, ties by declared order.
- **Learned state:** the training rows that have the value, with their `using` columns. KNN keeps its training data; that is what the state is.

If fewer than `knn` training rows have the value, `fit` throws and names the field and the count.

## Encoding

Categories get a numeric encoding from the schema, with no step needed:

| Field                | Default encoding | Columns                                                                                 |
| -------------------- | ---------------- | --------------------------------------------------------------------------------------- |
| `ml.category`        | one-hot          | one per declared category, in declared order                                            |
| `ml.orderedCategory` | position         | one, with the position in the declared order (`'poor'` → 0, `'fair'` → 1, `'good'` → 2) |
| `ml.boolean`         | value            | one, with 1 for `true` and 0 for `false`                                                |
| `ml.number`          | value            | one, with the number                                                                    |
| `ml.timestamp`       | none yet         | none: a timestamp is left out of the matrix, and the fitted state says so               |

`encode` only overrides a default:

```ts
.encode({ bodyCondition: 'one-hot' }) // an orderedCategory as one-hot
```

`'one-hot'` turns an ordered category into one-hot columns: `'position'` is already its default, and an `ml.category` has no order, so it can't have one. A field's encoding can change only while nothing has used it: once it is scaled or in a KNN's `using`, `encode` no longer accepts it. The default shows in the hover of `ml.category` and `ml.orderedCategory` and in the printed fitted state.

**A reference category** is one-hot without the reference's column: a record of the reference has 0 in every column of the field.

```ts
.encode({ island: { reference: 'Biscoe' } }) // columns island=Dream, island=Torgersen
.encode({ bodyCondition: { reference: 'poor' } }) // an orderedCategory as one-hot without 'poor'
```

The reference has to be a declared category, so its type is the literal union of the field. Linear models need it: one-hot with every category, next to an intercept, makes the columns [linearly dependent](algorithms/linear-regression.md#linearly-dependent-columns), and `fit` of a linear regression doesn't compile until each one-hot field has a reference. The reference only changes which columns reach the matrix. A KNN still measures distances with every category, because without the reference's column the reference would be closer to every other category than they are to each other. So `encode` accepts a reference for an `ml.category` even after a KNN used it. The fitted state prints the reference: `island: one-hot (Dream, Torgersen; reference Biscoe)`.

**Columns come from the schema, not from `fit`.** One-hot columns are the declared categories, so every block, every fold and every production record gets the same columns in the same order, whatever categories a training block happened to hold. There is no policy for unknown categories: `parse` and `ml.dataset` reject a value the schema doesn't declare, so it can't reach `transform`. A category that is declared but absent from the training block, common in small cross-validation folds, is not an error: its column is all zero in training, and the fitted state reports it.

The encoder doesn't reject an absent category, because only the estimator knows whether it can learn without it: a linear regression can't (see [Linearly dependent columns](algorithms/linear-regression.md#linearly-dependent-columns)), so its `fit` names the field and the category.

## Scaling

```ts
.scale(['billLengthMm', 'flipperLengthMm', 'bodyMassG', 'bodyCondition'])
```

Standardizes numbers and ordered positions: `(x - mean) / std`, with the mean and the population standard deviation of the training block. In `rows`, a scaled ordered category becomes its scaled position, a number. An ordered position needs it as much as a number does: a scale of ten levels goes from 0 to 9. A one-hot category isn't accepted.

`scale` learns from the values that are there, so it accepts a field that still has missing values; they stay missing, and the field stays incomplete until a `missing` step. A field with standard deviation 0 in training can't be divided: it is only centered, so it becomes 0, and the fitted state reports it.

## Output

`transform` keeps the block's role: a `Train` stays a `Train`, a `Test` stays a `Test`, so the rest of the workflow still knows what the block may be used for. The result has two views of the same data:
- **`rows`:** prepared records, with missing values filled and numbers scaled. Optional fields with a policy become required. Categories keep their labels, except a scaled ordered category, which is its scaled position; encoding shows in the matrix.
- **`matrix()`:** the numbers the estimator consumes. `columns` lists the column names in schema order, typed as the union of their literals (`'billLengthMm' | 'island=Biscoe' | …`; TypeScript doesn't keep the order of an object's keys, so it can't be a tuple). `values` is `number[][]`, one row per record, and `y` holds the target of each record, or is `undefined` for a schema without one. `fullOneHot` lists the one-hot fields without a reference, which a linear model rejects. The matrix keeps the role of its block, so an estimator's `fit` takes the matrix of a `Train` only.

The matrix is public on purpose. It is exactly what the estimator will see, so being able to print it is part of trusting the model. It prints as a table like the [blocks](dataset.md), with `#` as the row's position in the original data. From `examples/penguins` (`npm run example`), where the pipeline fills three lengths with their median and `sex` with its mode, scales the numbers, and then fills `bodyMassG` from its 3 nearest training penguins:

```ts
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

console.log(pipeline.fit(train).transform(test).matrix())
```

```
test matrix: 2 rows, 10 columns, target 'species'
#  island=Biscoe  island=Dream  island=Torgersen  billLengthMm  billDepthMm  flipperLengthMm  bodyMassG  sex=male  sex=female  bodyCondition  species
2              0             0                 1        -1.286        0.357           -0.469     -1.169         0           1              1  Adelie
8              0             1                 0        -0.225        0.711           -0.974     -0.855         0           1              0  Chinstrap
```

The fitted pipeline prints its learned state: the value each policy fills with, how many rows KNN keeps, the means and standard deviations, the encoding of each field, and what was reported (categories absent from training, fields with no spread):

```
fitted pipeline
missing
  billLengthMm: median 48.25
  billDepthMm: median 17.65
  flipperLengthMm: median 194.5
  sex: mode 'male'
scale
  billLengthMm: mean 46.48, std 4.808
  billDepthMm: mean 17.29, std 1.977
  flipperLengthMm: mean 201.5, std 13.86
  bodyMassG: mean 4275, std 876.8
missing
  bodyMassG: knn 3 over billLengthMm, flipperLengthMm, island (8 training rows with a value)
encode
  island: one-hot (Biscoe, Dream, Torgersen)
  sex: one-hot (male, female)
  bodyCondition: position (poor < fair < good)
```

## Pending decisions

An incomplete column is a pending decision. The pipeline carries the pending fields in its type, and `matrix()` only exists when there are none; the error names the field:

```ts
const fitted = ml.pipeline(penguinSchema).fit(train)
fitted.transform(test).rows // fine: rows can still hold missing values
fitted.transform(test).matrix() // type error: 'bodyMassG' and 'sex' need a missing-value policy
```

`fit`, `transform` and `rows` always work, so a pipeline can be built and inspected one step at a time. The `.estimator()` of M5 uses the same condition.

## Temporal features

Lag and rolling features, the target horizon and the warm-up of the first rows of a series are not part of M3. They come back once linear and logistic regression work end to end. The open design is kept here:

```ts
.lag('close', [1, 5, 21])
.rolling('close', { window: 21, stat: 'std' }) // 'mean' | 'std' | 'min' | 'max' | 'sum'
```

- **`lag`** and **`rolling`** use only data up to `t`. With `group`, they never cross from one series into another.
- A pipeline with temporal steps only accepts `fit(train)` for a `Train` that came from a series; the error at `fit` names the reason.
- **Warm-up:** the first rows of each group don't have enough history for the lags and windows. That has to be a decision, not silent `NaN`s.
- **Horizon and gap:** a walk-forward `gap` smaller than the target horizon should be an error.

## Open questions

- Features derived from timestamps (year, month, day of the week), instead of leaving them out.
- KNN over incomplete `using` fields, with distances that skip missing values.
- Custom user-defined transformers.
- Whether series `horizon` and the target transform also belong in the schema.
- Exponentially weighted windows (`ewm`) and features that cross groups.
