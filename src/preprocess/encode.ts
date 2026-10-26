import type { AnyField, Shape } from '../features/index.ts'
import type { Row, Step } from './step.ts'

/** One column of the matrix and how to read it from a prepared record. */
export type Column = {
  readonly name: string
  readonly field: string
  readonly value: (row: Row) => number
}

export type Encoding = 'one-hot' | 'position' | 'value'

/** How a field becomes numbers: from its kind, unless `encode` overrode it. */
export const encodingOf = (
  field: AnyField,
  oneHot: boolean
): Encoding | undefined => {
  switch (field.kind) {
    case 'number':
    case 'boolean':
      return 'value'
    case 'category':
      return 'one-hot'
    case 'ordered-category':
      return oneHot ? 'one-hot' : 'position'
    case 'timestamp':
      return undefined
  }
}

/**
 * The columns of the matrix, in schema order. The target is apart, and
 * timestamps have no encoding yet, so neither is a column.
 */
export const columnsOf = (
  fields: Shape,
  target: string | undefined,
  oneHot: ReadonlySet<string>
): Column[] =>
  Object.entries(fields).flatMap(([key, field]): Column[] => {
    if (key === target) return []
    const encoding = encodingOf(field, oneHot.has(key))
    if (encoding === undefined) return []
    if (encoding === 'one-hot' && 'categories' in field) {
      return field.categories.map(category => ({
        name: `${key}=${category}`,
        field: key,
        value: row => (row[key] === category ? 1 : 0)
      }))
    }
    if (encoding === 'position' && 'categories' in field) {
      // A scaled ordered category is already its scaled position
      const categories: readonly unknown[] = field.categories
      return [
        {
          name: key,
          field: key,
          value: row =>
            typeof row[key] === 'number'
              ? row[key]
              : categories.indexOf(row[key])
        }
      ]
    }
    return [
      {
        name: key,
        field: key,
        value: row =>
          typeof row[key] === 'boolean'
            ? Number(row[key])
            : (row[key] as number)
      }
    ]
  })

/** How each field is encoded, and the one-hot columns training never set. */
export const describeEncoding = (
  fields: Shape,
  target: string | undefined,
  oneHot: ReadonlySet<string>,
  train: readonly Row[]
): string[] =>
  Object.entries(fields).flatMap(([key, field]) => {
    if (key === target) return []
    const encoding = encodingOf(field, oneHot.has(key))
    if (encoding === undefined) return [`  ${key}: left out (timestamp)`]
    if (encoding === 'value') return []
    const categories: readonly string[] =
      'categories' in field ? field.categories : []
    if (encoding === 'position') {
      return [`  ${key}: position (${categories.join(' < ')})`]
    }
    const absent = categories.filter(c => !train.some(row => row[key] === c))
    return [
      `  ${key}: one-hot (${categories.join(', ')})${
        absent.length === 0
          ? ''
          : `, not in training: ${absent.map(c => `'${c}'`).join(', ')}`
      }`
    ]
  })

/**
 * Records an encoding override. Encoding learns nothing, so the step leaves
 * rows as they are; the matrix reads the override from the configuration.
 */
export const encodeStep = (
  overrides: Readonly<Record<string, 'one-hot'>>
): Step => ({
  label: `encode   ${Object.entries(overrides)
    .map(([key, encoding]) => `${key}: ${encoding}`)
    .join(', ')}`,
  fit: () => ({ transform: rows => [...rows], describe: () => '' })
})
