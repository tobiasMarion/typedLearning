# Testing

## Numerical tests

Each algorithm goes through these steps:
1. Derive and understand the math (recorded in [algorithms/](algorithms/README.md)).
2. Implement the educational version.
3. Generate synthetic data with known parameters (`test/fixtures/`).
4. Check the loss function and how it evolves during training.
5. Test convergence and sensitivity to scale, noise, and hyperparameters.
6. Compare against an **oracle**, such as a closed-form solution or a reference implementation.
7. Test edge cases and invalid inputs.
8. Integrate with the API and test a real end-to-end workflow.

Oracles are there to catch bugs. They don't replace the implementation being studied.

When the same algorithm has multiple implementations, test the shared contract and numerical agreement within justified tolerances.

## Type tests

They live in `test/types/` and run in CI. Each one covers:
- **positive cases:** the inferred type is the expected one (`expectTypeOf`);
- **negative cases:** invalid code doesn't compile (`@ts-expect-error`).

Some important negative cases:
- value of the wrong type or undeclared category;
- `fit(test)` or `evaluate(train)`;
- `.estimator()` called with pending decisions;
- an implementation option that doesn't apply to the chosen solver;
- a metric that doesn't match the problem type;
- a random split of a `Series<F>`, or k-fold on its training block;
- cross-validation of a `Validation<F>` or a `Test<F>`;
- calling `split` on a block (`Train<F>`, `Validation<F>`, or `Test<F>`);
- `stratify` on a field that isn't a category;
- `lag`/`rolling` on a non-temporal `Dataset<F>`;
- a walk-forward `gap` smaller than the target `horizon`.

## Look-ahead tests

Type tests don't catch every leak. For each temporal transformer, a runtime test changes the data **after** `t` and checks that the features at `t` stay the same. For walk-forward, a test checks that no training index of a fold is greater than or equal to a test index minus `gap`. For k-fold, a test checks that every row is held out exactly once and never trains in the same fold.

## Methodology tests

`test/methodology/` backs the matrix in [Methodological errors](methodology.md). Every error the library catches today has a test named by its ID: the compile-time ones in `errors.types.ts` with `@ts-expect-error`, the runtime and by-design ones in `errors.test.ts`. A status in the matrix changes in the same commit as its test.
