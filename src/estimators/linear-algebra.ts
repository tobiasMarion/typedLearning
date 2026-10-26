// The few matrix operations the linear models need, written out by hand. A
// matrix is an array of rows; a vector is an array of numbers.

export type Vector = readonly number[]
export type Rows = readonly Vector[]

export const dot = (a: Vector, b: Vector) =>
  a.reduce((sum, x, i) => sum + x * (b[i] as number), 0)

/** Column `j` of `rows`. */
export const column = (rows: Rows, j: number): number[] =>
  rows.map(row => row[j] as number)

/** `Aᵀ A` and `Aᵀ y`, the two sides of the normal equation. */
export const normalEquation = (rows: Rows, y: Vector) => {
  const width = rows[0]?.length ?? 0
  const columns = Array.from({ length: width }, (_, j) => column(rows, j))
  return {
    gram: columns.map(a => columns.map(b => dot(a, b))),
    moment: columns.map(a => dot(a, y))
  }
}

/**
 * Solves `A x = b` by Gaussian elimination with partial pivoting: at each
 * step, the row with the largest pivot goes up, which keeps rounding errors
 * small. `A` has to be square and invertible.
 */
export const solve = (a: Rows, b: Vector): number[] => {
  const n = b.length
  const m = a.map((row, i) => [...row, b[i] as number])
  for (let k = 0; k < n; k++) {
    let pivot = k
    for (let i = k + 1; i < n; i++) {
      if (Math.abs(m[i]?.[k] as number) > Math.abs(m[pivot]?.[k] as number)) {
        pivot = i
      }
    }
    ;[m[k], m[pivot]] = [m[pivot] as number[], m[k] as number[]]
    const top = m[k] as number[]
    for (let i = k + 1; i < n; i++) {
      const row = m[i] as number[]
      const factor = (row[k] as number) / (top[k] as number)
      for (let j = k; j <= n; j++) {
        row[j] = (row[j] as number) - factor * (top[j] as number)
      }
    }
  }
  return backSubstitute(
    m.map(row => row.slice(0, n)),
    m.map(row => row[n] as number)
  )
}

/** A column that the columns before it already explain. */
export type Dependence = {
  /** Its index. */
  readonly index: number
  /**
   * The earlier columns it is a combination of, with their coefficients:
   * column `index` = Σ coefficient × column. Empty for an all-zero column.
   */
  readonly of: readonly {
    readonly index: number
    readonly coefficient: number
  }[]
}

/**
 * Finds the first column that is a linear combination of the columns before
 * it, by modified Gram-Schmidt: each column loses its projection on the
 * directions found so far, and a column with nothing left (relative to its
 * own length) adds no direction. Undefined when every column adds one.
 */
export const firstDependence = (
  columns: readonly Vector[],
  tolerance = 1e-9
): Dependence | undefined => {
  // The orthonormal directions so far, and how each kept column is built from
  // them: kept[k] = Σ r[i][k] q[i], with r upper triangular.
  const q: number[][] = []
  const r: number[][] = []
  const kept: number[] = []
  for (const [index, original] of columns.entries()) {
    const residual = [...original]
    const projections = q.map(direction => {
      const projection = dot(direction, residual)
      direction.forEach((x, i) => {
        residual[i] = (residual[i] as number) - projection * x
      })
      return projection
    })
    const length = Math.sqrt(dot(original, original))
    const left = Math.sqrt(dot(residual, residual))
    if (length === 0) return { index, of: [] }
    if (left <= tolerance * length) {
      // original = Σ projections[i] q[i], and q = kept columns × r⁻¹, so the
      // coefficients on the kept columns solve r c = projections.
      const coefficients = backSubstitute(r, projections)
      return {
        index,
        of: coefficients
          .map((coefficient, k) => ({
            index: kept[k] as number,
            coefficient
          }))
          .filter(({ coefficient }) => Math.abs(coefficient) > tolerance)
      }
    }
    for (const row of r) row.push(0)
    projections.forEach((projection, i) => {
      ;(r[i] as number[])[q.length] = projection
    })
    r.push([...new Array<number>(q.length).fill(0), left])
    q.push(residual.map(x => x / left))
    kept.push(index)
  }
  return undefined
}

/** Solves `U x = b` for an upper triangular `U`. */
const backSubstitute = (u: Rows, b: Vector): number[] => {
  const n = b.length
  const x = new Array<number>(n).fill(0)
  for (let i = n - 1; i >= 0; i--) {
    const row = u[i] as Vector
    let sum = b[i] as number
    for (let j = i + 1; j < n; j++) sum -= (row[j] as number) * (x[j] as number)
    x[i] = sum / (row[i] as number)
  }
  return x
}
