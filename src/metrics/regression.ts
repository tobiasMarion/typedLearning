/**
 * Scores predictions against the true values. `N` is the name `evaluate`
 * reports it by; `Y` is what it compares: numbers for regression.
 */
export type Metric<N extends string, Y> = {
  (yTrue: readonly Y[], yPred: readonly Y[]): number
  readonly name: N
}

const check = (
  name: string,
  yTrue: readonly number[],
  yPred: readonly number[]
) => {
  if (yTrue.length !== yPred.length) {
    throw new RangeError(
      `${name}(): yTrue has ${yTrue.length} values and yPred has ${yPred.length}; they must have one value per record`
    )
  }
  if (yTrue.length === 0) {
    throw new RangeError(`${name}(): there are no values to score`)
  }
}

const metric = <N extends string>(
  name: N,
  score: (yTrue: readonly number[], yPred: readonly number[]) => number
): Metric<N, number> => {
  const fn = (yTrue: readonly number[], yPred: readonly number[]) => {
    check(name, yTrue, yPred)
    return score(yTrue, yPred)
  }
  Object.defineProperty(fn, 'name', { value: name })
  return Object.freeze(fn) as Metric<N, number>
}

const mean = (values: readonly number[]) =>
  values.reduce((sum, value) => sum + value, 0) / values.length

/** Mean squared error: the mean of the squared differences. Lower is better. */
export const mse = metric('mse', (yTrue, yPred) =>
  mean(yTrue.map((y, i) => (y - (yPred[i] as number)) ** 2))
)

/**
 * Mean absolute error: the mean of the absolute differences, in the units of
 * the target. Lower is better.
 */
export const mae = metric('mae', (yTrue, yPred) =>
  mean(yTrue.map((y, i) => Math.abs(y - (yPred[i] as number))))
)

/**
 * Coefficient of determination: 1 minus the squared error over the squared
 * spread of `yTrue` around its mean. 1 is perfect, 0 is as good as predicting
 * the mean, and it goes below 0 for worse.
 */
export const r2 = metric('r2', (yTrue, yPred) => {
  const average = mean(yTrue)
  const spread = yTrue.reduce((sum, y) => sum + (y - average) ** 2, 0)
  if (spread === 0) {
    throw new RangeError(
      `r2(): every true value is ${average}, so there is no spread to compare the error with. Use mse or mae`
    )
  }
  const error = yTrue.reduce(
    (sum, y, i) => sum + (y - (yPred[i] as number)) ** 2,
    0
  )
  return 1 - error / spread
})
