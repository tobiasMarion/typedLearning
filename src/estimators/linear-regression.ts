import { bold } from '../colors.ts'
import { printable } from '../dataset/describe.ts'
import { roleOf } from '../dataset/roles.ts'
import { type Layout, layoutOf, type Matrix } from '../pipeline/pipeline.ts'
import { show } from '../preprocess/step.ts'
import {
  column,
  type Dependence,
  dot,
  firstDependence,
  normalEquation,
  type Rows,
  solve,
  type Vector
} from './linear-algebra.ts'

/** How to find the weights. There is no default: the choice is explicit. */
export type LinearRegressionOptions =
  | {
      /** The closed-form solution: solves the normal equation `XᵀX β = Xᵀy`. */
      readonly solver: 'normal-equation'
    }
  | {
      /**
       * Starts every weight at zero and steps against the gradient of the
       * MSE, over the whole training block, once per epoch.
       */
      readonly solver: 'gradient-descent'
      /** The size of each step. Too high diverges; too low converges slowly. */
      readonly learningRate: number
      /** How many steps to take. */
      readonly epochs: number
    }

/** The error for a one-hot field with a column for every category. */
type NeedsReference<F extends string> = F extends string
  ? `'${F}' needs a reference category: add .encode({ ${F}: { reference: ... } }) to the pipeline`
  : never

// Type aliases, not interfaces: see Type quality in docs/principles.md.
/** A linear regression before training: the options and `fit`. */
export type LinearRegression = {
  readonly options: LinearRegressionOptions
  /**
   * Learns an intercept and one weight per column from the matrix of a
   * training block. The target has to be a number, and every one-hot field
   * needs a reference: next to the intercept, a column for every category
   * would make the columns linearly dependent.
   */
  fit<C extends string, F extends string>(
    matrix: [F] extends [never]
      ? Matrix<C, readonly number[], 'train', never>
      : Matrix<C, readonly number[], 'train', F> & {
          readonly fullOneHot: readonly NeedsReference<F>[]
        }
  ): FittedLinearRegression<C>
}

/** A linear regression with the weights it learned. */
export type FittedLinearRegression<C extends string> = {
  readonly options: LinearRegressionOptions
  /** The base value every prediction starts from. */
  readonly intercept: number
  /** One weight per column of the training matrix. */
  readonly weights: Readonly<Record<C, number>>
  /** The mean squared error on the training block. */
  readonly mse: number
  /**
   * Predicts one number per row of a matrix with the columns the model was
   * fitted on, in row order.
   */
  predict(matrix: Matrix<C, unknown>): readonly number[]
}

const solverName = (options: LinearRegressionOptions) =>
  options.solver === 'gradient-descent'
    ? `gradient descent, learningRate ${options.learningRate}, ${options.epochs} ${options.epochs === 1 ? 'epoch' : 'epochs'}`
    : 'normal equation'

const checkOptions = (options: LinearRegressionOptions) => {
  if (
    options.solver !== 'normal-equation' &&
    options.solver !== 'gradient-descent'
  ) {
    throw new RangeError(
      `linearRegression(): solver must be 'normal-equation' or 'gradient-descent', received ${JSON.stringify((options as { solver: unknown }).solver)}`
    )
  }
  if (options.solver === 'gradient-descent') {
    const { learningRate, epochs } = options
    if (!(Number.isFinite(learningRate) && learningRate > 0)) {
      throw new RangeError(
        `linearRegression(): learningRate must be a positive number, received ${learningRate}`
      )
    }
    if (!(Number.isInteger(epochs) && epochs > 0)) {
      throw new RangeError(
        `linearRegression(): epochs must be a positive integer, received ${epochs}`
      )
    }
  }
}

/** The mean squared error of `coefficients` over the rows of `[1 | X]`. */
const meanSquaredError = (rows: Rows, y: Vector, coefficients: Vector) => {
  const residuals = rows.map(
    (row, i) => dot(coefficients, row) - (y[i] as number)
  )
  return dot(residuals, residuals) / rows.length
}

/** The loss before the first step, kept to show how far training went. */
type Trace = { readonly firstLoss: number }

const normalEquationSolver = (rows: Rows, y: Vector) => {
  const { gram, moment } = normalEquation(rows, y)
  return { coefficients: solve(gram, moment), trace: undefined }
}

/**
 * Full-batch gradient descent from zero: β ← β − η ∇L, with
 * ∇L = (2/n) X₁ᵀ (X₁β − y). With a step that is too large the loss grows
 * instead of shrinking, which is an error rather than a model.
 */
const gradientDescentSolver = (
  rows: Rows,
  y: Vector,
  learningRate: number,
  epochs: number
): { coefficients: number[]; trace: Trace } => {
  const n = rows.length
  const coefficients = new Array<number>(rows[0]?.length ?? 0).fill(0)
  const firstLoss = meanSquaredError(rows, y, coefficients)
  for (let epoch = 1; epoch <= epochs; epoch++) {
    const residuals = rows.map(
      (row, i) => dot(coefficients, row) - (y[i] as number)
    )
    coefficients.forEach((value, j) => {
      const gradient =
        (2 / n) *
        rows.reduce(
          (sum, row, i) => sum + (row[j] as number) * (residuals[i] as number),
          0
        )
      coefficients[j] = value - learningRate * gradient
    })
    const loss = meanSquaredError(rows, y, coefficients)
    if (!Number.isFinite(loss)) {
      throw new RangeError(
        `fit(): gradient descent diverged: the loss became ${loss} at epoch ${epoch}. Lower learningRate (${learningRate}), or scale the columns`
      )
    }
  }
  const lastLoss = meanSquaredError(rows, y, coefficients)
  if (lastLoss > firstLoss) {
    throw new RangeError(
      `fit(): gradient descent diverged: the loss grew from ${show(firstLoss)} to ${show(lastLoss)} in ${epochs} epochs. Lower learningRate (${learningRate}), or scale the columns`
    )
  }
  return { coefficients, trace: { firstLoss } }
}

const quoted = (names: readonly string[]) =>
  names.map(name => `'${name}'`).join(', ')

/** The rows of `[1 | X]`: a column of ones for the intercept first. */
const withIntercept = (values: Rows): Rows => values.map(row => [1, ...row])

const checkMatrix = (matrix: unknown) => {
  const role = roleOf(matrix)
  if (role === undefined || layoutOf(matrix) === undefined) {
    throw new TypeError(
      'fit() expects the matrix of a prepared block: pipeline.fit(train).transform(train).matrix()'
    )
  }
  if (role !== 'train') {
    throw new TypeError(
      `fit() takes the matrix of a training block, and this one comes from a ${role} block`
    )
  }
  const { y, values, fullOneHot } = matrix as Matrix<
    string,
    readonly unknown[] | undefined
  >
  if (y === undefined) {
    throw new TypeError(
      'fit() needs a target: mark it in the schema with .target(field)'
    )
  }
  const label = y.find(value => typeof value !== 'number')
  if (label !== undefined) {
    throw new TypeError(
      `fit(): linear regression predicts a number, and the target holds ${JSON.stringify(label)}`
    )
  }
  if (fullOneHot.length > 0) {
    throw new TypeError(
      `fit(): ${quoted(fullOneHot)} ${fullOneHot.length === 1 ? 'has' : 'have'} a column for every category, which the intercept makes linearly dependent. Add .encode({ ${fullOneHot.map(key => `${key}: { reference: ... }`).join(', ')} }) to the pipeline`
    )
  }
  if (values.length === 0) {
    throw new RangeError('fit(): the training matrix has no rows')
  }
  return { values, y: y as readonly number[] }
}

const FIX_ABSENT =
  'Use fewer folds, a stratified split, or merge rare categories in the schema'

/**
 * Explains why the columns of `[1 | X]` can't give a single answer, naming
 * fields and categories when the layout knows them.
 */
const explainDependence = (layout: Layout, dependence: Dependence): string => {
  const name = (index: number) =>
    index === 0 ? 'the intercept' : `'${layout.columns[index - 1]?.name}'`
  const source = layout.columns[dependence.index - 1]
  if (dependence.of.length === 0) {
    if (source?.category !== undefined) {
      return `no training row has ${source.field} '${source.category}', so its column '${source.name}' is all zero and its weight could be anything. A record of that category would be predicted by a guess. ${FIX_ABSENT}`
    }
    return `column ${name(dependence.index)} is 0 in every training row, so its weight could be anything`
  }
  if (dependence.of.every(({ index }) => index === 0)) {
    return `column ${name(dependence.index)} has the same value in every training row, so it repeats the intercept`
  }
  const of = dependence.of.map(({ index }) => name(index))
  return `column ${name(dependence.index)} is a linear combination of ${of.slice(0, -1).join(', ')}${of.length > 1 ? ' and ' : ''}${of.at(-1)} in the training block, so infinitely many weights fit it equally well. Remove one of these columns`
}

/**
 * A one-hot field whose reference no training row has: its other columns add
 * up to the intercept.
 */
const absentReference = (layout: Layout, values: Rows) => {
  for (const [field, reference] of layout.references) {
    const indices = layout.columns.flatMap((c, j) =>
      c.field === field ? [j] : []
    )
    if (indices.length === 0) continue
    const seen = values.some(row => indices.every(j => row[j] === 0))
    if (!seen) {
      return `no training row has ${field} '${reference}', the reference, so the other ${field} columns add up to the intercept. ${FIX_ABSENT}, or pick a reference that training has`
    }
  }
  return undefined
}

const describeFitted = (
  options: LinearRegressionOptions,
  names: readonly string[],
  coefficients: readonly number[],
  mse: number,
  trace: Trace | undefined
) => {
  const labels = ['intercept', ...names]
  const width = Math.max(...labels.map(label => label.length))
  return [
    bold(`fitted linear regression (${solverName(options)})`),
    ...labels.map(
      (label, i) =>
        `  ${label.padEnd(width)}  ${show(coefficients[i] as number)}`
    ),
    `training MSE ${show(mse)}`,
    ...(trace === undefined || options.solver !== 'gradient-descent'
      ? []
      : [
          `loss ${show(trace.firstLoss)} before the first epoch, ${show(mse)} after epoch ${options.epochs}`
        ])
  ].join('\n')
}

const fitted = (
  options: LinearRegressionOptions,
  names: readonly string[],
  coefficients: readonly number[],
  mse: number,
  trace: Trace | undefined
) => {
  const [intercept, ...weights] = coefficients as [number, ...number[]]
  const predictRows = (values: Rows) =>
    values.map(row => intercept + dot(weights, row))
  return printable(
    {
      options,
      intercept,
      weights: Object.freeze(
        Object.fromEntries(names.map((name, j) => [name, weights[j]]))
      ),
      mse,
      predict: (matrix: {
        readonly columns: readonly string[]
        readonly values: Rows
      }) => {
        const same =
          matrix.columns.length === names.length &&
          matrix.columns.every((name, j) => name === names[j])
        if (!same) {
          throw new TypeError(
            `predict() expects the columns the model was fitted on: ${quoted(names)}; received ${quoted(matrix.columns)}`
          )
        }
        return Object.freeze(predictRows(matrix.values))
      }
    },
    () => describeFitted(options, names, coefficients, mse, trace)
  )
}

/**
 * A linear regression: predicts a number as an intercept plus a weighted sum
 * of the columns, with the weights that minimize the squared error on the
 * training block. The solver has no default.
 */
export function linearRegression(
  options: LinearRegressionOptions
): LinearRegression {
  checkOptions(options)
  const frozen = Object.freeze({ ...options })
  const model = {
    options: frozen,
    fit: (matrix: unknown) => {
      const { values, y } = checkMatrix(matrix)
      const layout = layoutOf(matrix) as Layout
      const rows = withIntercept(values)
      const width = (rows[0] as readonly number[]).length
      const columns = Array.from({ length: width }, (_, j) => column(rows, j))
      const problem =
        absentReference(layout, values) ??
        (() => {
          const dependence = firstDependence(columns)
          return dependence && explainDependence(layout, dependence)
        })()
      if (problem !== undefined) throw new RangeError(`fit(): ${problem}`)

      const { coefficients, trace } =
        frozen.solver === 'gradient-descent'
          ? gradientDescentSolver(rows, y, frozen.learningRate, frozen.epochs)
          : normalEquationSolver(rows, y)
      return fitted(
        frozen,
        (matrix as Matrix<string, unknown>).columns,
        coefficients,
        meanSquaredError(rows, y, coefficients),
        trace
      )
    }
  }
  return printable(model, () =>
    bold(`linear regression (${solverName(frozen)})`)
  ) as unknown as LinearRegression
}
