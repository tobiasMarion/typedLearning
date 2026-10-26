import { bold } from '../colors.ts'
import type { Features } from '../features/features.ts'
import type { InferShape, Shape } from '../features/infer.ts'
import { describeBlock, printable } from './describe.ts'

// One brand for all roles, with the role as its value. Passing a test block
// where a training block is expected then reports
// `Type '"test"' is not assignable to type '"train"'`. The symbol isn't
// exported, so only the library can create a block.
declare const role: unique symbol

// Where a training block was cut from, so cross-validation can tell ordered
// data from unordered data: k-fold on a series' training block would train on
// the future. Reports `Type '"series"' is not assignable to type '"dataset"'`.
declare const origin: unique symbol

/** What a block was cut from: a `Dataset` has no order, a `Series` is ordered by time. */
export type Origin = 'dataset' | 'series'

/** What a block may be used for. */
export type Role = 'train' | 'validation' | 'test'

// Type aliases, not interfaces: see Type quality in docs/principles.md.
/** Rows to fit on. `O` is what it was cut from; `Train<S>` accepts both. */
export type Train<S extends Shape, O extends Origin = Origin> = {
  readonly features: Features<S>
  readonly rows: readonly InferShape<S>[]
  readonly [role]: 'train'
  readonly [origin]: O
}

// Held-out blocks carry an origin too, though nothing reads it: without it, a
// test block passed as a training block reports the missing origin instead of
// `Type '"test"' is not assignable to type '"train"'`.
export type Test<S extends Shape> = {
  readonly features: Features<S>
  readonly rows: readonly InferShape<S>[]
  readonly [role]: 'test'
  readonly [origin]: Origin
}

export type Validation<S extends Shape> = {
  readonly features: Features<S>
  readonly rows: readonly InferShape<S>[]
  readonly [role]: 'validation'
  readonly [origin]: Origin
}

/**
 * The role brand alone, for values derived from a block, such as a prepared
 * block: a prepared test block is still a test block.
 */
export type RoleBrand<R extends Role> = { readonly [role]: R }

/** One cut of a training block: fit on `train`, score on `validation`. */
export type Fold<S extends Shape, O extends Origin = Origin> = {
  readonly train: Train<S, O>
  readonly validation: Validation<S>
}

// Position of each record in the dataset or series it was validated into.
// Records are created there and only shared afterwards, so a record keeps its
// number in every block and fold, and printed blocks can name their rows.
const positions = new WeakMap<object, number>()

export const numberRows = (rows: readonly object[]) => {
  rows.forEach((record, i) => {
    positions.set(record, i)
  })
}

// The role of every block, which preprocessing needs to keep it. Kept out of
// the types, where the role is a brand.
const roles = new WeakMap<object, Role>()

// The roles and origins are type-only brands, so a frozen { features, rows }
// is all a block needs at runtime, plus how it prints. These casts are the one
// place where blocks are created; the splits pass in a new array of rows in
// the order the block should keep.
const block = <S extends Shape>(
  role: Role,
  features: Features<S>,
  rows: InferShape<S>[]
) => {
  const frozen = Object.freeze(rows)
  const created = printable({ features, rows: frozen }, () =>
    describeBlock({
      role,
      fields: features.fields,
      rows: frozen as readonly Record<string, unknown>[],
      position: record => positions.get(record)
    })
  )
  roles.set(created, role)
  return created
}

/** The role of a block, or undefined for any other value. */
export const roleOf = (block: unknown): Role | undefined =>
  typeof block === 'object' && block !== null ? roles.get(block) : undefined

/** Position of a record in the dataset or series it came from. */
export const positionOf = (record: object): number | undefined =>
  positions.get(record)

/** Gives `record` the position of `source`, so both print with the same number. */
export const keepPosition = (source: object, record: object) => {
  const position = positions.get(source)
  if (position !== undefined) positions.set(record, position)
}

/** Records the role of a value derived from a block. */
export const markRole = <T extends object>(value: T, role: Role): T => {
  roles.set(value, role)
  return value
}

export const trainBlock = <S extends Shape, O extends Origin>(
  features: Features<S>,
  rows: InferShape<S>[]
) => block('train', features, rows) as Train<S, O>

// The time field of every training block cut from a series, which walk-forward
// needs to cut it again. Kept out of the type, so Train stays a plain alias.
const timeFields = new WeakMap<object, string>()

export const seriesTrainBlock = <S extends Shape>(
  features: Features<S>,
  rows: InferShape<S>[],
  time: string
) => {
  const train = trainBlock<S, 'series'>(features, rows)
  timeFields.set(train, time)
  return train
}

/** The time field of a series' training block, or undefined for any other value. */
export const timeFieldOf = (block: unknown): string | undefined =>
  typeof block === 'object' && block !== null
    ? timeFields.get(block)
    : undefined

export const validationBlock = <S extends Shape>(
  features: Features<S>,
  rows: InferShape<S>[]
) => block('validation', features, rows) as Validation<S>

export const testBlock = <S extends Shape>(
  features: Features<S>,
  rows: InferShape<S>[]
) => block('test', features, rows) as Test<S>

/** A fold that prints as its number and both of its blocks. */
export const fold = <S extends Shape, O extends Origin>(
  train: Train<S, O>,
  validation: Validation<S>,
  index: number,
  count: number
): Fold<S, O> =>
  printable({ train, validation }, () =>
    [
      bold(`fold ${index + 1} of ${count}`),
      String(train),
      String(validation)
    ].join('\n\n')
  )
