import type { Shape } from '../features/index.ts'
import { columnsOf } from './encode.ts'
import {
  type Context,
  type FittedStep,
  type Positions,
  type Row,
  rowName,
  type Step,
  show,
  toCurrent
} from './step.ts'

export type KnnPolicy = {
  readonly knn: number
  readonly using: readonly string[]
}

export type MissingPolicy =
  | 'mean'
  | 'median'
  | 'mode'
  | 'error'
  | { readonly constant: unknown }
  | KnnPolicy

const isPresent = (value: unknown) => value !== undefined

const isKnn = (policy: MissingPolicy): policy is KnnPolicy =>
  typeof policy === 'object' && 'knn' in policy

const mean = (values: readonly number[]) =>
  values.reduce((sum, x) => sum + x, 0) / values.length

const median = (values: readonly number[]) => {
  const sorted = [...values].sort((a, b) => a - b)
  const middle = Math.floor(sorted.length / 2)
  return sorted.length % 2 === 1
    ? (sorted[middle] as number)
    : ((sorted[middle - 1] as number) + (sorted[middle] as number)) / 2
}

/** The most frequent value; ties go to the one that comes first in `order`. */
export const mode = (values: readonly unknown[], order: readonly unknown[]) => {
  const counts = new Map<unknown, number>()
  for (const value of values) counts.set(value, (counts.get(value) ?? 0) + 1)
  let best: unknown
  let bestCount = 0
  for (const value of order) {
    const count = counts.get(value) ?? 0
    if (count > bestCount) {
      best = value
      bestCount = count
    }
  }
  return best
}

const label = (policy: MissingPolicy) =>
  typeof policy === 'string'
    ? policy
    : isKnn(policy)
      ? `knn ${policy.knn} over ${policy.using.join(', ')}`
      : 'constant'

const showValue = (value: unknown) =>
  typeof value === 'number'
    ? show(value)
    : typeof value === 'string'
      ? `'${value}'`
      : String(value)

/**
 * The order ties are broken in: declared categories, false before true, or,
 * for a scaled ordered category, its positions after scaling.
 */
const orderOf = (
  fields: Shape,
  key: string,
  context: Context,
  values: readonly unknown[]
): readonly unknown[] => {
  const field = fields[key]
  if (field === undefined) return values
  if (field.kind === 'boolean') return [false, true]
  if ('categories' in field) {
    const scale = context.scales.get(key)
    return scale === undefined
      ? field.categories
      : field.categories.map(c => toCurrent(c, scale, field.categories))
  }
  return values
}

/** How a policy fills one row, once fit learned what it needs. */
type Filler = {
  readonly fill: (row: Row) => unknown
  readonly describe: string
}

/**
 * KNN keeps the training rows that have the value. A missing value takes the
 * mean (numbers) or the mode (anything else) of the `knn` closest of them, by
 * euclidean distance over the columns of `using`, which the types require to
 * be complete and on comparable scales. Ties in distance go to the earlier row.
 */
const knnFiller = (
  fields: Shape,
  oneHot: ReadonlySet<string>,
  key: string,
  policy: KnnPolicy,
  rows: readonly Row[],
  context: Context
): Filler => {
  const using = new Set(policy.using)
  const columns = columnsOf(fields, undefined, oneHot).filter(column =>
    using.has(column.field)
  )
  const vector = (row: Row) => columns.map(column => column.value(row))
  const references = rows
    .filter(row => isPresent(row[key]))
    .map(row => ({ value: row[key], at: vector(row) }))
  if (references.length < policy.knn) {
    throw new RangeError(
      `fit(): '${key}' has ${references.length} training rows with a value, fewer than knn: ${policy.knn}`
    )
  }
  const numeric = fields[key]?.kind === 'number'
  const order = orderOf(
    fields,
    key,
    context,
    references.map(reference => reference.value)
  )

  return {
    fill: row => {
      const at = vector(row)
      const nearest = references
        .map((reference, i) => ({
          i,
          value: reference.value,
          distance: reference.at.reduce(
            (sum, x, c) => sum + (x - (at[c] as number)) ** 2,
            0
          )
        }))
        .sort((a, b) => a.distance - b.distance || a.i - b.i)
        .slice(0, policy.knn)
        .map(neighbor => neighbor.value)
      return numeric ? mean(nearest as number[]) : mode(nearest, order)
    },
    describe: `knn ${policy.knn} over ${policy.using.join(', ')} (${references.length} training rows with a value)`
  }
}

const statisticFiller = (
  fields: Shape,
  key: string,
  policy: 'mean' | 'median' | 'mode',
  rows: readonly Row[],
  context: Context
): Filler => {
  const values = rows.map(row => row[key]).filter(isPresent)
  if (values.length === 0) {
    throw new RangeError(
      `fit(): '${key}' has no values in the training block, so it has no ${policy}`
    )
  }
  const value =
    policy === 'mean'
      ? mean(values as number[])
      : policy === 'median'
        ? median(values as number[])
        : mode(values, orderOf(fields, key, context, values))
  return { fill: () => value, describe: `${policy} ${showValue(value)}` }
}

/** Fills missing values with what a policy learned in fit, or throws for 'error'. */
export const missingStep = (
  fields: Shape,
  oneHot: ReadonlySet<string>,
  policies: Readonly<Record<string, MissingPolicy>>
): Step => {
  for (const [key, policy] of Object.entries(policies)) {
    if (isKnn(policy) && !(Number.isInteger(policy.knn) && policy.knn > 0)) {
      throw new RangeError(
        `missing(): knn for '${key}' must be a positive integer, received ${policy.knn}`
      )
    }
    if (isKnn(policy) && policy.using.length === 0) {
      throw new RangeError(
        `missing(): knn for '${key}' needs at least one field in using`
      )
    }
  }

  return {
    label: `missing  ${Object.entries(policies)
      .map(([key, policy]) => `${key}: ${label(policy)}`)
      .join(', ')}`,
    fit(rows: readonly Row[], _at: Positions, context: Context): FittedStep {
      const fillers = new Map<string, Filler>()
      for (const [key, policy] of Object.entries(policies)) {
        if (policy === 'error') continue
        if (isKnn(policy)) {
          fillers.set(
            key,
            knnFiller(fields, oneHot, key, policy, rows, context)
          )
        } else if (typeof policy === 'object') {
          const field = fields[key]
          const categories =
            field !== undefined && 'categories' in field
              ? field.categories
              : undefined
          const value = toCurrent(
            policy.constant,
            context.scales.get(key),
            categories
          )
          fillers.set(key, {
            fill: () => value,
            describe: `constant ${showValue(value)}`
          })
        } else {
          fillers.set(key, statisticFiller(fields, key, policy, rows, context))
        }
      }

      return {
        transform: (rows, at) =>
          rows.map((row, i) => {
            const filled: Record<string, unknown> = { ...row }
            for (const key of Object.keys(policies)) {
              if (isPresent(row[key])) continue
              const filler = fillers.get(key)
              if (filler === undefined) {
                throw new TypeError(
                  `'${key}' is missing in ${rowName(at[i])}, and its missing-value policy is 'error'`
                )
              }
              filled[key] = filler.fill(row)
            }
            return Object.freeze(filled)
          }),
        describe: () =>
          [
            'missing',
            ...Object.keys(policies).map(
              key =>
                `  ${key}: ${fillers.get(key)?.describe ?? 'error when missing'}`
            )
          ].join('\n')
      }
    }
  }
}
