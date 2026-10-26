# TL

An experiment: **can types prevent the methodological errors that make machine learning models look better than they are?**

Leaking test data into training, splitting a time series at random, picking hyperparameters by their test score: the code runs, the model trains, and the reported score is wrong. Kapoor & Narayanan found errors like these in hundreds of published papers across seventeen fields. They are hard to see in review, because the code is correct as code.

TL is a Machine Learning library in TypeScript built to test the premise that a typed API can turn those errors into compile errors, or at least into errors at the call. It is also a sandbox for learning ML: every algorithm is written by hand, math first.

## The premise

TL bets on making the workflow explicit:

- each step exposes only the valid next steps, so `Train`, `Validation` and `Test` are different types and a test block can't be trained on;
- decisions that change the result (split strategy, seed, gap, missing values, solver) are required and have no default;
- time series are part of the design from the start: a random split of a series doesn't compile, and temporal splits require a gap;
- what the types can't check, such as external data, is validated at runtime, once, at the entry points.

Whether that works is the question, not the claim.

## How we test it

1. **Coverage.** [Methodological errors](docs/methodology.md) tracks the errors TL deals with and records, for each one, whether TL catches it at compile time, at runtime, by design, or not at all. Every claim there has a test in `test/methodology/`.
2. **Effect.** Once a model can be trained and evaluated end to end, coding agents get the same tasks, with errors planted in them and a holdout they never see, in three arms: scikit-learn, TL without types, and TL with the type checker in the agent's loop. The outcome is how much the reported score overstates the holdout score. Agents are a fair first subject: types are feedback in their loop, so the experiment also measures how much semantic types help an agent harness.
3. **People.** If the effect shows up with agents, a small study with people checks whether it holds.

Priorities: correctness, understanding, and interface quality. Performance comes later.

## End goal

Classifying the [Palmer Penguins](https://allisonhorst.github.io/palmerpenguins/) dataset. It has numeric and categorical fields, and some values are missing:

```ts
import { ml } from 'tl'

const penguinSchema = ml
  .features({
    species: ml.category(['Adelie', 'Chinstrap', 'Gentoo']),
    island: ml.category(['Biscoe', 'Dream', 'Torgersen']),
    billLengthMm: ml.number().optional(),
    billDepthMm: ml.number().optional(),
    flipperLengthMm: ml.number().optional(),
    bodyMassG: ml.number().optional(),
    sex: ml.category(['male', 'female']).optional()
  })
  .target('species') // classification: inferred from a category

const penguins = ml.dataset(penguinSchema, rows) // runtime validation
const { train, test } = penguins.split({ test: 0.2, seed: 42, stratify: 'species' })

const model = ml
  .pipeline(penguinSchema)
  .missing({
    billLengthMm: 'median', // required: every optional field needs a policy
    billDepthMm: 'median',
    flipperLengthMm: 'median',
    bodyMassG: 'median',
    sex: 'mode'
  }) // categories are one-hot encoded, with the columns the schema declares
  .scale(['billLengthMm', 'billDepthMm', 'flipperLengthMm', 'bodyMassG'])
  .estimator(ml.knn({ k: 5, distance: 'euclidean' }))

const fitted = model.fit(train) // only accepts Train<F>
fitted.evaluate(test, [ml.accuracy, ml.confusionMatrix]) // Validation<F> or Test<F>, never Train<F>

fitted.predict({
  island: 'Biscoe',
  billLengthMm: 47.5,
  billDepthMm: 15.0,
  flipperLengthMm: 218,
  bodyMassG: 4950,
  sex: 'male'
}) // 'Adelie' | 'Chinstrap' | 'Gentoo'

// @ts-expect-error — evaluating on training data is leakage
fitted.evaluate(train, [ml.accuracy]) // @ts-expect-error — mse is a regression metric; `species` is a category
fitted.evaluate(test, [ml.mse])
```

### Time series

Time-ordered data uses `ml.series` and gets temporal features and walk-forward validation through the same pipeline:

```ts
const priceSchema = ml.features({
  date: ml.timestamp(),
  ticker: ml.category(['PETR4', 'VALE3', 'ITUB4']),
  close: ml.number()
})

const prices = ml.series(priceSchema, rows, { time: 'date', group: 'ticker' })
const { train, test } = prices.split({ testFrom: '2025-01-01', gap: { steps: 1 } })

const model = ml
  .pipeline(priceSchema)
  .target('close', { transform: 'log-return', horizon: 1 })
  .lag('close', [1, 5, 21]) // backward-looking only
  .missing({ warmup: 'drop' }) // required: lags leave the first rows empty
  .estimator(ml.linearRegression({ solver: 'normal-equation' }))

const report = model.backtest(
  train,
  ml.walkForward({
    validationFrom: '2024-01-01',
    validation: { months: 1 },
    train: 'expanding',
    gap: { steps: 1 }
  })
) // @ts-expect-error — a random split on a time series is leakage
prices.split({ test: 0.2, seed: 42 })
```

> The names and signatures above show the project's direction. They are not a contract yet.

## Milestones

| #  | Milestone                       | Deliverables |
|----|---------------------------------|--------------|
| M1 | **Feature DSL** ✅              | Schema, `timestamp` fields, type inference, categories as literal unions, runtime validation, positive and negative type tests, Standard Schema support and `ml.fromSchema` for Zod/Valibot/ArkType schemas |
| M2 | **Dataset, split, and validation** ✅ | `dataset` and `series`; branded `Train`/`Validation`/`Test`; random split (seed, stratified) and chronological split (gap) into train/validation/test; cross-validation of the training block: k-fold, stratified k-fold, and walk-forward (expanding/rolling window, gap) |
| M3 | **Preprocessing** ✅            | Target in the schema; imputation (mean, median, mode, constant, KNN); one-hot and ordinal encoding with columns from the schema; standardization; learned state kept separate from configuration; the printable matrix the estimators consume |
| M4 | **Linear regression**           | Prediction, MSE, gradients, gradient descent; comparison against the closed-form solution |
| M5 | **Pipeline, backtest, and metrics** | Typed composition, pending-decision checks, `fit`/`evaluate`/`predict`, lag and rolling features and target horizon, `backtest` with refit per fold, MSE/MAE/R²/directional accuracy |
| M6 | **Second algorithm (MVP)**      | Logistic regression or KNN; hyperparameter search with cross-validation; documentation and examples |
| M7 | **Ship it to production**       | Refit the validated pipeline on all the data (a `Train` with no `Validation` or `Test`, so the final model can't report a score); learned state frozen with the model and never recomputed at `predict`; saving and loading a fitted pipeline; runtime validation of production records |

After the MVP: purged/embargoed k-fold, group k-fold and nested cross-validation, decision trees, neural networks with backpropagation, ensembles, and optimized implementations behind the same public API.

## Documentation

| Document | Contents |
|----------|----------|
| [Methodological errors](docs/methodology.md) | The errors TL tries to prevent, and the status of each one |
| [Design principles](docs/principles.md) | Explicit decisions, time series from the start, static + runtime typing |
| [Feature DSL](docs/features.md) | Field types, inference, validation, Zod compatibility through Standard Schema |
| [Dataset and validation](docs/dataset.md) | Splits, cross-validation, branded types, leakage prevention |
| [Preprocessing](docs/preprocessing.md) | Target, missing values, encoding, scaling, the prepared matrix |
| [Pipeline](docs/pipeline.md) | Guided workflow, configured vs. fitted model, pending decisions, backtest |
| [Estimators](docs/estimators.md) | Estimator/Predictor contracts, multiple implementations |
| [Algorithms](docs/algorithms/README.md) | Math derivation and notes for each algorithm |
| [Metrics](docs/metrics.md) | Regression, classification, and directional metrics |
| [Testing](docs/testing.md) | Numerical tests, reference oracles, type tests |

## Development

```sh
npm run check   # lint, typecheck, and tests
```

Changes reach `main` only through pull requests. CI runs `check`, the build, and the examples on every pull request, and the PR title follows [Conventional Commits](https://www.conventionalcommits.org), since it becomes the merge commit title. Dependabot opens weekly update PRs for npm and GitHub Actions.

## Folder structure

```
.
├── .github/                  # CI, PR title check, Dependabot
├── src/
│   ├── index.ts              # public `ml` namespace
│   ├── standard-schema.ts    # Standard Schema interfaces (copied from the spec)
│   ├── colors.ts             # ANSI colors for terminal output
│   ├── features/             # schema DSL, inference, runtime validation
│   ├── dataset/              # dataset, series, Train/Validation/Test types
│   ├── validation/           # cross-validation: k-fold, walk-forward
│   ├── preprocess/           # steps: missing (incl. KNN), encode, scale
│   ├── pipeline/             # guided builder, fitted model, backtest
│   ├── estimators/
│   │   ├── contracts.ts      # Estimator, Predictor, Fitted
│   │   └── linear-regression/
│   │       ├── index.ts      # public API
│   │       └── educational.ts
│   ├── metrics/
│   └── math/                 # hand-written numerical ops (vectors, matrices)
├── test/
│   ├── unit/                 # runtime and numerical tests
│   ├── types/                # type tests (expect-type, @ts-expect-error)
│   └── fixtures/             # synthetic data with known parameters
├── examples/                 # complete runnable workflows, one folder each
│   ├── penguins/             # index.ts (pipeline), data.json, bad-data.json
│   ├── air-passengers/       # monthly time series, walk-forward
│   └── scikit-learn/         # the penguins workflow in scikit-learn, as a reference (uv)
└── docs/
    └── algorithms/
```
