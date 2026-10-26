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

/** What `encode` sets for a field: one-hot, with or without a reference. */
export type Override = 'one-hot' | { readonly reference: string }

/**
 * The columns of the matrix, in schema order. The target is apart, and
 * timestamps have no encoding yet, so neither is a column. A reference
 * category has no column; without `references`, as for distances, every
 * category has one.
 */
export const columnsOf = (
  fields: Shape,
  target: string | undefined,
  oneHot: ReadonlySet<string>,
  references: ReadonlyMap<string, string> = new Map()
): Column[] =>
  Object.entries(fields).flatMap(([key, field]): Column[] => {
    if (key === target) return []
    const encoding = encodingOf(field, oneHot.has(key))
    if (encoding === undefined) return []
    if (encoding === 'one-hot' && 'categories' in field) {
      const categories: readonly string[] = field.categories
      const reference = references.get(key)
      return categories
        .filter(category => category !== reference)
        .map(category => ({
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
  references: ReadonlyMap<string, string>,
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
    const reference = references.get(key)
    const columns =
      reference === undefined
        ? categories.join(', ')
        : `${categories.filter(c => c !== reference).join(', ')}; reference ${reference}`
    return [
      `  ${key}: one-hot (${columns})${
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
  fields: Shape,
  overrides: Readonly<Record<string, Override>>
): Step => {
  for (const [key, override] of Object.entries(overrides)) {
    const field = fields[key]
    const categories: readonly string[] =
      field !== undefined && 'categories' in field ? field.categories : []
    if (
      typeof override === 'object' &&
      !categories.includes(override.reference)
    ) {
      throw new RangeError(
        `encode(): the reference of '${key}' must be one of ${categories.map(c => `'${c}'`).join(', ')}, received '${override.reference}'`
      )
    }
  }
  return {
    label: `encode   ${Object.entries(overrides)
      .map(
        ([key, override]) =>
          `${key}: ${
            typeof override === 'object'
              ? `reference '${override.reference}'`
              : override
          }`
      )
      .join(', ')}`,
    fit: () => ({ transform: rows => [...rows], describe: () => '' })
  }
}
