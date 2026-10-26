import type { Shape } from '../features/index.ts'
import {
  type Context,
  type FittedStep,
  type Row,
  type Scale,
  type Step,
  show,
  toCurrent
} from './step.ts'

/**
 * Standardizes numbers and ordered positions: (x - mean) / std, with the
 * population standard deviation of the training block. A field with no spread
 * in training is only centered.
 */
export const scaleStep = (fields: Shape, keys: readonly string[]): Step => ({
  label: `scale    ${keys.join(', ')}`,
  fit(rows: readonly Row[], _at, context: Context): FittedStep {
    const categoriesOf = (key: string) => {
      const field = fields[key]
      return field !== undefined && 'categories' in field
        ? (field.categories as readonly unknown[])
        : undefined
    }
    const scales = new Map<string, Scale>()
    for (const key of keys) {
      const categories = categoriesOf(key)
      const values = rows
        .map(row => row[key])
        .filter(value => value !== undefined)
        .map(value =>
          categories === undefined
            ? (value as number)
            : categories.indexOf(value)
        )
      if (values.length === 0) {
        throw new RangeError(
          `fit(): '${key}' has no values in the training block, so it can't be scaled`
        )
      }
      const mean = values.reduce((sum, x) => sum + x, 0) / values.length
      const variance =
        values.reduce((sum, x) => sum + (x - mean) ** 2, 0) / values.length
      const scale = { mean, std: Math.sqrt(variance) }
      scales.set(key, scale)
      context.scales.set(key, scale)
    }

    return {
      transform: rows =>
        rows.map(row => {
          const scaled: Record<string, unknown> = { ...row }
          for (const [key, scale] of scales) {
            if (row[key] === undefined) continue
            scaled[key] = toCurrent(row[key], scale, categoriesOf(key))
          }
          return Object.freeze(scaled)
        }),
      describe: () =>
        [
          'scale',
          ...[...scales].map(
            ([key, { mean, std }]) =>
              `  ${key}: mean ${show(mean)}, std ${show(std)}${
                std === 0 ? ' (no spread in training: only centered)' : ''
              }`
          )
        ].join('\n')
    }
  }
})
