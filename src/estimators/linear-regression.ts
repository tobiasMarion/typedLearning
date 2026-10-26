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
  solve
} from './linear-algebra.ts'

/** How to find the weights. There is no default: the choice is explicit. */
export type LinearRegressionOptions = {
  /** The closed-form solution: solves the normal equation `XᵀX β = Xᵀy`. */
  readonly solver: 'normal-equation'
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
  options.solver.replace('-', ' ')

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
  mse: number
) => {
  const labels = ['intercept', ...names]
  const width = Math.max(...labels.map(label => label.length))
  return [
    bold(`fitted linear regression (${solverName(options)})`),
    ...labels.map(
      (label, i) =>
        `  ${label.padEnd(width)}  ${show(coefficients[i] as number)}`
    ),
    `training MSE ${show(mse)}`
  ].join('\n')
}

const fitted = (
  options: LinearRegressionOptions,
  names: readonly string[],
  coefficients: readonly number[],
  mse: number
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
    () => describeFitted(options, names, coefficients, mse)
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
  if (options.solver !== 'normal-equation') {
    throw new RangeError(
      `linearRegression(): solver must be 'normal-equation', received ${JSON.stringify(options.solver)}`
    )
  }
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

      const { gram, moment } = normalEquation(rows, y)
      const coefficients = solve(gram, moment)
      const residuals = rows.map(
        (row, i) => dot(coefficients, row) - (y[i] as number)
      )
      const mse = dot(residuals, residuals) / rows.length
      return fitted(
        frozen,
        (matrix as Matrix<string, unknown>).columns,
        coefficients,
        mse
      )
    }
  }
  return printable(model, () =>
    bold(`linear regression (${solverName(frozen)})`)
  ) as unknown as LinearRegression
}
