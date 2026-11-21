# Metrics

Metrics are pure functions `(yTrue, yPred) => number` with a name, and `fitted.evaluate(test, [...])` applies several at once and returns one number per name (see [Evaluation](pipeline.md#evaluation)):

```ts
ml.mse([3, 5], [2, 5]) // 0.5
fitted.evaluate(validation, [ml.mse, ml.r2]) // { mse: 1520.3, r2: 0.81 }
```

Both arrays need the same length and at least one value, or the metric throws a `RangeError`. R² compares the error with the spread of `yTrue`, so it throws too when every true value is the same: there is no spread to compare with.

A metric's type has to match the problem type: using a classification metric on a regressor is a compile-time error.

## Regression

| Metric                | Formula                                                                            | Status  |
| --------------------- | ---------------------------------------------------------------------------------- | ------- |
| `mse`                 | $\frac{1}{n}\sum (y - \hat y)^2$                                                   | M5      |
| `mae`                 | $\frac{1}{n}\sum \lvert y - \hat y \rvert$                                         | M5      |
| `r2`                  | $1 - \frac{\sum (y - \hat y)^2}{\sum (y - \bar y)^2}$                              | M5      |
| `directionalAccuracy` | $\frac{1}{n}\sum \mathbb{1}[\operatorname{sign}(y) = \operatorname{sign}(\hat y)]$ | planned |

On a backtest, every metric is reported per fold and aggregated across folds. The spread across folds matters as much as the mean.

## Classification

| Metric                        | Status    |
| ----------------------------- | --------- |
| `accuracy`                    | after MVP |
| `precision` / `recall` / `f1` | after MVP |
| `logLoss`                     | after MVP |
| confusion matrix              | after MVP |
