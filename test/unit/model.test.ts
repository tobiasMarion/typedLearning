import assert from 'node:assert/strict'
import { describe, test } from 'node:test'
import { ml } from '../../src/index.ts'

// biome-ignore lint/suspicious/noControlCharactersInRegex: ANSI escapes
const plain = (text: string) => text.replace(/\u001b\[\d+m/g, '')

// y = 10 + 3x + 5 when group is 'b', with no noise
const schema = ml
  .features({
    x: ml.number().optional(),
    group: ml.category(['a', 'b']),
    y: ml.number()
  })
  .target('y')
const rows = Array.from({ length: 20 }, (_, i) => {
  const group = i % 3 === 0 ? 'b' : 'a'
  const x = i % 7 === 3 ? undefined : i
  return {
    x,
    group,
    y: 10 + 3 * (x ?? 10) + (group === 'b' ? 5 : 0)
  } as const
})
const { train, test: heldOut } = ml
  .dataset(schema, rows)
  .split({ test: 0.25, seed: 1 })

const pipeline = ml
  .pipeline(schema)
  .missing({ x: { constant: 10 } })
  .encode({ group: { reference: 'a' } })
const regression = ml.linearRegression({ solver: 'normal-equation' })

describe('pipeline.estimator()', () => {
  const fitted = pipeline.estimator(regression).fit(train)

  test('fits the estimator on the matrix of the training block', () => {
    const prepared = pipeline.fit(train)
    const alone = regression.fit(prepared.transform(train).matrix())
    assert.deepEqual(fitted.estimator.weights, alone.weights)
    assert.equal(fitted.estimator.intercept, alone.intercept)
  })

  test('predicts records without the target, prepared like training', () => {
    const [a, b, filled] = fitted.predict([
      { x: 2, group: 'a' },
      { x: 2, group: 'b' },
      { group: 'a' }
    ])
    assert.ok(Math.abs((a as number) - 16) < 1e-9)
    assert.ok(Math.abs((b as number) - 21) < 1e-9)
    assert.ok(Math.abs((filled as number) - 40) < 1e-9)
  })

  test('predicts the records of a block like predict on its matrix', () => {
    const records = heldOut.rows.map(({ y: _, ...record }) => record)
    assert.deepEqual(
      fitted.predict(records),
      fitted.estimator.predict(fitted.pipeline.transform(heldOut).matrix())
    )
  })

  test('prints the pipeline and the estimator', () => {
    const model = pipeline.estimator(regression)
    assert.match(
      plain(String(model)),
      /pipeline for 'y'[\s\S]*linear regression \(normal equation\)/
    )
    assert.match(
      plain(String(fitted)),
      /fitted pipeline[\s\S]*fitted linear regression \(normal equation\)/
    )
  })

  test('needs a missing-value policy for every optional field', () => {
    const unfilled = ml.pipeline(schema) as unknown as {
      estimator(estimator: unknown): unknown
    }
    assert.throws(() => unfilled.estimator(regression), {
      name: 'TypeError',
      message:
        "estimator(): 'x' still need a missing-value policy. Add .missing({ ... }) to the pipeline"
    })
  })

  test('applies the checks of the estimator at fit', () => {
    const full = ml.pipeline(schema).missing({ x: 'median' }) as unknown as {
      estimator(estimator: unknown): { fit(train: unknown): unknown }
    }
    assert.throws(() => full.estimator(regression).fit(train), {
      name: 'TypeError',
      message: /'group' has a column for every category/
    })
  })
})
