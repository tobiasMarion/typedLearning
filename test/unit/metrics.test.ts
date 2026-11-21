import assert from 'node:assert/strict'
import { describe, test } from 'node:test'
import { ml } from '../../src/index.ts'

const close = (actual: number, expected: number) =>
  assert.ok(Math.abs(actual - expected) < 1e-12, `${actual} is not ${expected}`)

// The example of scikit-learn's documentation for each metric
const yTrue = [3, -0.5, 2, 7]
const yPred = [2.5, 0, 2, 8]

describe('regression metrics', () => {
  test('match scikit-learn', () => {
    close(ml.mse(yTrue, yPred), 0.375)
    close(ml.mae(yTrue, yPred), 0.5)
    close(ml.r2(yTrue, yPred), 0.9486081370449679)
  })

  test('are perfect on perfect predictions', () => {
    assert.equal(ml.mse(yTrue, yTrue), 0)
    assert.equal(ml.mae(yTrue, yTrue), 0)
    assert.equal(ml.r2(yTrue, yTrue), 1)
  })

  test('r2 is 0 for the mean and negative for worse', () => {
    close(
      ml.r2(
        yTrue,
        yTrue.map(() => 2.875)
      ),
      0
    )
    assert.ok(ml.r2(yTrue, [7, 2, -0.5, 3]) < 0)
  })

  test('carry their names', () => {
    assert.deepEqual(
      [ml.mse, ml.mae, ml.r2].map(metric => metric.name),
      ['mse', 'mae', 'r2']
    )
  })

  test('need one prediction per true value', () => {
    assert.throws(() => ml.mse([1, 2], [1]), {
      name: 'RangeError',
      message:
        'mse(): yTrue has 2 values and yPred has 1; they must have one value per record'
    })
    assert.throws(() => ml.mae([], []), {
      name: 'RangeError',
      message: 'mae(): there are no values to score'
    })
  })

  test('r2 needs spread in the true values', () => {
    assert.throws(() => ml.r2([4, 4], [4, 5]), {
      name: 'RangeError',
      message:
        'r2(): every true value is 4, so there is no spread to compare the error with. Use mse or mae'
    })
  })
})
