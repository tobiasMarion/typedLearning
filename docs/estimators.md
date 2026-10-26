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

## Open questions

- How to represent unsupervised problems.
- Default implementation when the user doesn't pick one: should there be one, or is the choice required?
