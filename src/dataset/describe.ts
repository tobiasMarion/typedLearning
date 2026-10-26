import { bold, dim, green, yellow } from '../colors.ts'

// Text diagrams of cross-validation folds. The folds array prints them through
// console.log (Node's inspect hook) and String(folds), so the folds can be
// checked by eye before anything trains on them.

const WIDTH = 48
const TRAIN = '█'
const VALIDATION = '▓'
const GAP = '░'

const legend = (gap: boolean) =>
  dim(
    `${green(TRAIN)} train  ${yellow(VALIDATION)} validation${gap ? `  ${GAP} dropped by the gap` : ''}`
  )

const pad = (text: string, width: number) => text.padStart(width)

export type KFoldDescription = {
  readonly k: number
  readonly seed: number
  readonly stratify: string | undefined
  readonly rows: number
  /** Fold of each row of the training block. */
  readonly fold: readonly number[]
  /** Class of each row, with stratify. */
  readonly classes: readonly string[] | undefined
}

// The validation rows of a fold are spread over the training block, so the
// bars draw the rows grouped by fold: fold i holds out the i-th stretch. Each
// stretch is as wide as its share of the rows.
export const describeKFold = (d: KFoldDescription): string => {
  const sizes = Array.from(
    { length: d.k },
    (_, f) => d.fold.filter(x => x === f).length
  )
  const edges = [0]
  for (const size of sizes) edges.push((edges.at(-1) as number) + size)
  const column = (row: number) => Math.round((row / d.rows) * WIDTH)

  const header = [
    `k-fold: ${d.k} folds of ${d.rows} training rows`,
    d.stratify === undefined ? undefined : `stratified by '${d.stratify}'`,
    `seed ${d.seed}`
  ]
    .filter(part => part !== undefined)
    .join(', ')

  const digits = String(d.rows).length
  const lines = sizes.map((size, f) => {
    const from = column(edges[f] as number)
    const to = column(edges[f + 1] as number)
    const bar =
      green(TRAIN.repeat(from)) +
      yellow(VALIDATION.repeat(to - from)) +
      green(TRAIN.repeat(WIDTH - to))
    let classes = ''
    if (d.classes !== undefined) {
      const counts = new Map<string, number>()
      d.fold.forEach((x, i) => {
        if (x !== f) return
        const value = d.classes?.[i] as string
        counts.set(value, (counts.get(value) ?? 0) + 1)
      })
      classes = `  ${dim(
        [...counts]
          .toSorted(([a], [b]) => a.localeCompare(b))
          .map(([value, count]) => `${value} ${count}`)
          .join(', ')
      )}`
    }
    return `fold ${pad(String(f + 1), String(d.k).length)}  ${bar}  ${pad(String(d.rows - size), digits)} train, ${pad(String(size), digits)} validation${classes}`
  })
  return [bold(header), ...lines, legend(false)].join('\n')
}

// Dates without a time of day print as dates, so a monthly series reads
// 2024-04-01 instead of 2024-04-01T00:00:00.000Z.
const formatter = (times: readonly number[]) => {
  const dateOnly = times.every(time => time % 86_400_000 === 0)
  return (time: number) => {
    const text = new Date(time).toISOString()
    return dateOnly ? text.slice(0, 10) : text.replace('.000Z', 'Z')
  }
}

export type WalkForwardFold = {
  /** Distinct times of the fold's training rows, ascending. */
  readonly train: readonly number[]
  readonly validation: readonly number[]
}

export type WalkForwardDescription = {
  readonly train: string
  readonly validation: string
  readonly gap: string
  /** Distinct times of the training block, ascending. */
  readonly times: readonly number[]
  readonly folds: readonly WalkForwardFold[]
}

// A timeline from the first to the last time of the training block. Each
// column is a stretch of time, painted by the rows that fall in it. Times
// before a rolling window or after the validation block are left blank.
export const describeWalkForward = (d: WalkForwardDescription): string => {
  const first = d.times[0] as number
  const last = d.times.at(-1) as number
  const span = Math.max(last - first, 1)
  const column = (time: number) =>
    Math.min(WIDTH - 1, Math.floor(((time - first) / span) * WIDTH))
  // Each time fills the columns until the next one, so a short series with
  // fewer times than columns draws solid bars instead of dotted ones
  const next = new Map(d.times.map((time, i) => [time, d.times[i + 1]]))
  const fill = (paint: string[], time: number, cell: string) => {
    const following = next.get(time)
    const from = column(time)
    const to = following === undefined ? WIDTH : column(following)
    for (let c = from; c < Math.max(to, from + 1); c++) paint[c] = cell
  }
  const format = formatter(d.times)
  const digits = String(d.folds.length).length

  const header = `walk-forward: ${d.folds.length} folds, ${d.train === 'expanding' ? 'expanding' : `rolling ${d.train}`} train, ${d.validation} validation blocks, gap of ${d.gap}`
  const indent = ' '.repeat(`fold ${'0'.repeat(digits)}  `.length)
  const space = WIDTH - format(first).length - format(last).length
  const axis = `${indent}${format(first)}${' '.repeat(Math.max(1, space))}${format(last)}`

  const lines = d.folds.map((fold, f) => {
    // Later layers win a column: the gap over train, validation over both
    const paint: string[] = Array.from({ length: WIDTH }, () => ' ')
    const validationStart = fold.validation[0] as number
    const lastTrain = fold.train.at(-1) as number
    for (const time of fold.train) fill(paint, time, TRAIN)
    const dropped = d.times.filter(
      time => time > lastTrain && time < validationStart
    )
    for (const time of dropped) fill(paint, time, GAP)
    for (const time of fold.validation) fill(paint, time, VALIDATION)
    // A gap shorter than a column would vanish under the validation block;
    // it keeps the column before it, so a gap is never drawn as no gap
    const before = column(validationStart) - 1
    if (dropped.length > 0 && !paint.includes(GAP) && before >= 0) {
      paint[before] = GAP
    }
    const bar = paint
      .map(cell =>
        cell === TRAIN ? green(cell) : cell === VALIDATION ? yellow(cell) : cell
      )
      .join('')
    const range = (times: readonly number[]) =>
      `${format(times[0] as number)}..${format(times.at(-1) as number)}`
    return `fold ${pad(String(f + 1), digits)}  ${bar}  train ${range(fold.train)}, validation ${range(fold.validation)}`
  })
  return [bold(header), dim(axis), ...lines, legend(true)].join('\n')
}

// Blocks print their rows as a table. Long blocks keep their first and last
// rows; block.rows has all of them.
const HEAD = 10
const TAIL = 5

export type BlockDescription = {
  readonly role: string
  /** Replaces `<role>: <n> rows` as the first line. */
  readonly title?: string
  readonly fields: { readonly [key: string]: { readonly kind: string } }
  readonly rows: readonly Record<string, unknown>[]
  /** Position of a record in the dataset or series it came from. */
  readonly position: (record: object) => number | undefined
}

// The # column is the row's position in the dataset or series, so the same row
// has the same number in every block and fold it lands in.
export const describeBlock = (d: BlockDescription): string => {
  const total = d.rows.length
  const header = bold(
    d.title ?? `${d.role}: ${total} ${total === 1 ? 'row' : 'rows'}`
  )
  if (total === 0) return header
  const shown =
    total > HEAD + TAIL + 1
      ? [...d.rows.slice(0, HEAD), undefined, ...d.rows.slice(-TAIL)]
      : d.rows

  const keys = Object.keys(d.fields)
  const columns = keys.map(key => {
    const kind = d.fields[key]?.kind
    const values = d.rows.map(row => row[key])
    const times = values.filter(value => typeof value === 'number')
    const time = kind === 'timestamp' ? formatter(times) : undefined
    return {
      key,
      right: kind === 'number' || kind === 'timestamp',
      text: (value: unknown) =>
        value === null || value === undefined
          ? 'null'
          : time !== undefined
            ? time(value as number)
            : String(value)
    }
  })

  const cells = shown.map(row =>
    row === undefined
      ? undefined
      : [
          String(d.position(row) ?? '?'),
          ...columns.map(column => column.text(row[column.key]))
        ]
  )
  const titles = ['#', ...keys]
  const widths = titles.map((title, c) =>
    Math.max(title.length, ...cells.map(cell => cell?.[c]?.length ?? 0))
  )
  const right = [true, ...columns.map(column => column.right)]
  const line = (cell: readonly string[], style = (text: string) => text) =>
    cell
      .map((text, c) => {
        const width = widths[c] as number
        const padded = right[c] ? text.padStart(width) : text.padEnd(width)
        // Missing values stand out from the ones that were there
        return c > 0 && text === 'null' ? dim(padded) : style(padded)
      })
      .join('  ')
      .trimEnd()

  const lines = cells.map(cell =>
    cell === undefined ? dim(`… ${total - HEAD - TAIL} more rows`) : line(cell)
  )
  return [header, dim(line(titles)), ...lines].join('\n')
}

/**
 * Freezes a value and lets it print its description: through console.log in
 * Node, and through String(value) anywhere. The hooks aren't enumerable, so
 * spreading, iterating and JSON.stringify see the value as it is.
 */
export const printable = <T extends object>(
  value: T,
  describe: () => string
): Readonly<T> => {
  for (const key of [Symbol.for('nodejs.util.inspect.custom'), 'toString']) {
    Object.defineProperty(value, key, { value: describe, enumerable: false })
  }
  return Object.freeze(value)
}
