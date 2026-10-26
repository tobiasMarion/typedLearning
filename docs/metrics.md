# Metrics

Metrics are pure functions `(yTrue, yPred) => number`, and `fitted.evaluate(test, [...])` applies several at once.

A metric's type has to match the problem type: using a classification metric on a regressor is a compile-time error.

## Regression

| Metric                | Formula                                                                            | Status  |
| --------------------- | ---------------------------------------------------------------------------------- | ------- |
| `mse`                 | $\frac{1}{n}\sum (y - \hat y)^2$                                                   | planned |
| `mae`                 | $\frac{1}{n}\sum \lvert y - \hat y \rvert$                                         | planned |
| `r2`                  | $1 - \frac{\sum (y - \hat y)^2}{\sum (y - \bar y)^2}$                              | planned |
| `directionalAccuracy` | $\frac{1}{n}\sum \mathbb{1}[\operatorname{sign}(y) = \operatorname{sign}(\hat y)]$ | planned |

On a backtest, every metric is reported per fold and aggregated across folds. The spread across folds matters as much as the mean.

## Classification

| Metric                        | Status    |
| ----------------------------- | --------- |
| `accuracy`                    | after MVP |
| `precision` / `recall` / `f1` | after MVP |
| `logLoss`                     | after MVP |
| confusion matrix              | after MVP |
