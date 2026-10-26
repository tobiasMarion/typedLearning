# Pipeline

The pipeline is a guided builder: each method returns a new type, and that type exposes only the valid next steps.

```ts
const model = ml
  .pipeline(schema) // a schema with a target, see Preprocessing
  .missing({ age: 'median' })
  .scale(['age', 'income'])
  .estimator(ml.linearRegression({ solver: 'normal-equation' }))

const fitted = model.fit(train)
```

## Workflow

```
schema ─► pipeline ─► missing / encode / scale ─► estimator ─► fit ─► evaluate / predict
```

`fit` does the following:

1. fits the transformers on the training data;
2. transforms the training data;
3. fits the estimator;
4. stores the learned states.

M3 ships the builder without `.estimator()`: `pipeline.fit(train)` returns the fitted transformers, and their `transform` gives the prepared rows and the matrix (see [Preprocessing](preprocessing.md)).

`predict` takes a record that is already typed, without the target, reapplies the same transformations, and only then predicts. It doesn't validate the record again: external data is validated at the boundary, with `parse` or with the project's own schema (Zod, for example), and from then on the type carries the guarantee. A category the schema doesn't declare can't get past that boundary; missing values follow the policies chosen in the pipeline.

## Pending decisions

The pipeline carries in its type the set of fields that still lack a decision (for example, optional fields without a missing-value policy). `.estimator()` only exists when that set is `never`, the same condition as `matrix()` in [Preprocessing](preprocessing.md#pending-decisions):

```ts
interface Pipeline<F, Pending extends keyof F> {
  estimator<E extends Estimator>(
    this: Pipeline<F, never>,
    e: E
  ): ConfiguredModel<F, E>
}
```

The error message has to name the pending field. The ergonomics of that message are part of the contract.

## Configured vs. fitted model

- `ConfiguredModel`: holds the configuration and has `fit`.
- `FittedModel`: holds the learned states and has `predict`, `evaluate`, and `describe`.

`fit` doesn't mutate the configured model; it returns a new value.

## Backtest

On the training block of a `Series<F>`, `model.backtest(train, cv)` runs the whole pipeline once per walk-forward fold:

1. fits the transformers and the estimator on the fold's `Train<F>`;
2. predicts the fold's `Validation<F>`;
3. stores the predictions and metrics for that fold.

The result reports metrics per fold and aggregated, along with the out-of-sample predictions as a series. Nothing learned in one fold leaks into another.

## Incompatible composition

If a step references a field that doesn't exist or has the wrong type (for example, `scale` applied to a category, or any step applied to the target), the composition fails at compile time. A pipeline is built from a schema, not from data, so temporal steps (`lag`, `rolling`, a target `horizon`) are checked at `fit`: it only accepts a `Train` that came from a `Series<F>`.

## Open questions

- Synchronous or asynchronous `fit`.
- Branching steps (the equivalent of `ColumnTransformer`).
- `describe()`: the output format.
- One method for both kinds of cross-validation (walk-forward and k-fold), or one per kind of data.
