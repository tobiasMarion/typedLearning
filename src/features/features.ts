import type { StandardSchemaV1 } from '../standard-schema.ts'
import type { CategoryField, OrderedCategoryField } from './fields.ts'
import type { InferShape, Shape } from './infer.ts'
import { FeatureValidationError, validateRows } from './validate.ts'

const FIELD_KINDS: ReadonlySet<unknown> = new Set([
  'number',
  'boolean',
  'category',
  'ordered-category',
  'timestamp'
])

// A type alias, not an interface: see Type quality in docs/principles.md.
export type Features<S extends Shape> = StandardSchemaV1<InferShape<S>> & {
  readonly fields: S
  /** Field names, in declaration order. */
  readonly keys: readonly Extract<keyof S, string>[]
  /** Type-only: use `typeof features.infer`. Reading it at runtime throws. */
  readonly infer: InferShape<S>
  /** Validates one record. Throws `FeatureValidationError`. */
  parse(record: unknown): InferShape<S>
  /** Validates many records, grouping issues by field. Throws `FeatureValidationError`. */
  parseRows(rows: readonly unknown[]): InferShape<S>[]
  // The valid keys are spelled out instead of named with an alias, so a wrong
  // field reports the valid ones (see Type quality in docs/principles.md).
  /**
   * Marks the field a model predicts and returns a new schema with the same
   * fields. A category or boolean makes it classification, a number regression.
   * The target must be a required field.
   */
  target<
    const K extends {
      [Key in keyof S]: S[Key] extends
        | { readonly kind: 'number' | 'boolean'; readonly isOptional: false }
        | CategoryField
        | OrderedCategoryField
        ? Key
        : never
    }[keyof S] &
      string
  >(key: K): Targeted<S, K>
}

/** A schema with a target, built with `features.target(key)`. */
export type Targeted<
  S extends Shape,
  K extends keyof S & string
> = Features<S> & {
  /** The field a model predicts. */
  readonly targetKey: K
}

const TARGET_KINDS: ReadonlySet<unknown> = new Set([
  'number',
  'boolean',
  'category',
  'ordered-category'
])

const assertTarget = (shape: Shape, key: unknown): void => {
  const field = typeof key === 'string' ? shape[key] : undefined
  if (field === undefined) {
    throw new TypeError(
      `features.target(): '${String(key)}' is not a field. Expected one of ${Object.keys(
        shape
      )
        .map(name => `'${name}'`)
        .join(' | ')}`
    )
  }
  if (!TARGET_KINDS.has(field.kind)) {
    throw new TypeError(
      `features.target(): '${key}' is a ${field.kind}. A target is a number, a boolean or a category`
    )
  }
  if (field.isOptional) {
    throw new TypeError(
      `features.target(): '${key}' is optional, and a target must be required. Make it required and filter the rows without it before ml.dataset`
    )
  }
}

const assertShape = (shape: unknown): void => {
  if (typeof shape !== 'object' || shape === null) {
    throw new TypeError('ml.features() expects an object of fields')
  }
  const entries = Object.entries(shape as Record<string, unknown>)
  if (entries.length === 0) {
    throw new TypeError('ml.features() needs at least one field')
  }
  for (const [name, field] of entries) {
    const kind =
      typeof field === 'object' && field !== null && 'kind' in field
        ? field.kind
        : undefined
    if (!FIELD_KINDS.has(kind)) {
      throw new TypeError(
        `ml.features(): '${name}' is not a field. Use ml.number(), ml.boolean(), ml.category([...]), ml.orderedCategory([...]) or ml.timestamp()`
      )
    }
  }
}

export const features = <const S extends Shape>(shape: S): Features<S> => {
  assertShape(shape)
  const keys = Object.freeze(Object.keys(shape)) as readonly Extract<
    keyof S,
    string
  >[]

  // The casts to InferShape<S> below are safe: values are only returned when
  // every declared field passed its check, which is exactly InferShape<S>.
  const parseRows = (rows: readonly unknown[]): InferShape<S>[] => {
    const { values, issues } = validateRows(shape, rows, { indexed: true })
    if (issues.length > 0) throw new FeatureValidationError(issues, parseRows)
    return values as InferShape<S>[]
  }

  const parse = (record: unknown): InferShape<S> => {
    const { values, issues } = validateRows(shape, [record], {
      indexed: false
    })
    if (issues.length > 0) throw new FeatureValidationError(issues, parse)
    return values[0] as InferShape<S>
  }

  const validate = (value: unknown): StandardSchemaV1.Result<InferShape<S>> => {
    const { values, issues } = validateRows(shape, [value], { indexed: false })
    if (issues.length > 0) {
      return {
        issues: issues.map(issue => ({
          message: issue.message,
          path: issue.field === undefined ? undefined : [issue.field]
        }))
      }
    }
    return { value: values[0] as InferShape<S> }
  }

  const schema: Features<S> = Object.freeze({
    fields: shape,
    keys,
    get infer(): never {
      throw new Error(
        '`infer` is type-only: use `typeof features.infer` instead of reading it'
      )
    },
    parse,
    parseRows,
    target,
    '~standard': Object.freeze({ version: 1, vendor: 'tl', validate })
  })

  // Function declarations are hoisted, so `schema` above can hold `target`.
  // The descriptors copy the `infer` getter instead of calling it.
  function target<K extends keyof S & string>(key: K): Targeted<S, K> {
    assertTarget(shape, key)
    return Object.freeze(
      Object.defineProperties(
        {},
        {
          ...Object.getOwnPropertyDescriptors(schema),
          targetKey: { value: key, enumerable: true }
        }
      )
    ) as Targeted<S, K>
  }

  return schema
}
