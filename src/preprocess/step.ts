// The steps of the builder work on plain records: the types that keep them
// apart live on Pipeline, and the builder only creates valid steps.

export type Row = Readonly<Record<string, unknown>>

/** Position of each row in the dataset or series, for error messages. */
export type Positions = readonly (number | undefined)[]

/** What earlier steps learned that a later step needs. */
export type Context = {
  /** Mean and standard deviation of each scaled field, from its `scale` step. */
  readonly scales: Map<string, Scale>
}

export type Scale = { readonly mean: number; readonly std: number }

export type Step = {
  /** The configuration, for printing. */
  readonly label: string
  /** Learns from `rows`, the output of the steps before it. */
  fit(rows: readonly Row[], at: Positions, context: Context): FittedStep
}

export type FittedStep = {
  /** Returns one new record per record given, in the same order. */
  transform(rows: readonly Row[], at: Positions): Row[]
  /** The learned state, for printing. */
  describe(): string
}

/** A number short enough to print. */
export const show = (value: number) => String(Number(value.toPrecision(4)))

/** `row 12`, or `the record` when it has no position. */
export const rowName = (position: number | undefined) =>
  position === undefined ? 'the record' : `row ${position}`

/**
 * A value in the units of a field after its `scale` step, if it had one: an
 * ordered category by its position. A field with no spread is only centered.
 */
export const toCurrent = (
  value: unknown,
  scale: Scale | undefined,
  categories: readonly unknown[] | undefined
): unknown => {
  if (scale === undefined) return value
  const x =
    categories === undefined ? (value as number) : categories.indexOf(value)
  return scale.std === 0 ? x - scale.mean : (x - scale.mean) / scale.std
}
