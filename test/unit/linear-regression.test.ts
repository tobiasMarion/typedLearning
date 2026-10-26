import assert from 'node:assert/strict'
import { describe, test } from 'node:test'
import { ml } from '../../src/index.ts'

// biome-ignore lint/suspicious/noControlCharactersInRegex: ANSI escapes
const plain = (text: string) => text.replace(/\u001b\[\d+m/g, '')

const close = (
  actual: number | undefined,
  expected: number,
  tolerance = 1e-9
) =>
  assert.ok(
    Math.abs((actual as number) - expected) < tolerance,
    `${actual} is not ${expected}`
  )

const model = ml.linearRegression({ solver: 'normal-equation' })

// y = 3 + 2a - b, with no noise: the normal equation recovers it exactly
const lineSchema = ml
  .features({ a: ml.number(), b: ml.number(), y: ml.number() })
  .target('y')
const lineRows = Array.from({ length: 12 }, (_, i) => {
  const a = i
  const b = (i * 7) % 5
  return { a, b, y: 3 + 2 * a - b }
})
const lines = ml.dataset(lineSchema, lineRows).split({ test: 0.25, seed: 1 })
const linePipeline = ml.pipeline(lineSchema).fit(lines.train)

describe('ml.linearRegression', () => {
  test('recovers the weights of noiseless data', () => {
    const fitted = model.fit(linePipeline.transform(lines.train).matrix())
    close(fitted.intercept, 3)
    close(fitted.weights.a, 2)
    close(fitted.weights.b, -1)
    close(fitted.mse, 0)
  })

  test('predicts held-out rows in order', () => {
    const fitted = model.fit(linePipeline.transform(lines.train).matrix())
    const predictions = fitted.predict(
      linePipeline.transform(lines.test).matrix()
    )
    assert.equal(predictions.length, lines.test.rows.length)
    lines.test.rows.forEach((row, i) => {
      close(predictions[i], row.y)
    })
    assert.ok(Object.isFrozen(predictions))
  })

  test('minimizes the squared error on noisy data', () => {
    // Moving any weight away from the solution can only raise the MSE
    const noisy = ml.dataset(
      lineSchema,
      lineRows.map((row, i) => ({ ...row, y: row.y + (i % 3) - 1 }))
    )
    const { train } = noisy.split({ test: 0.25, seed: 1 })
    const matrix = ml.pipeline(lineSchema).fit(train).transform(train).matrix()
    const fitted = model.fit(matrix)
    const mse = (intercept: number, a: number, b: number) =>
      matrix.values.reduce((sum, [x1, x2], i) => {
        const error =
          intercept +
          a * (x1 as number) +
          b * (x2 as number) -
          (matrix.y[i] as number)
        return sum + error ** 2
      }, 0) / matrix.values.length
    const { intercept } = fitted
    const { a, b } = fitted.weights
    close(fitted.mse, mse(intercept, a, b))
    for (const step of [-0.01, 0.01]) {
      assert.ok(mse(intercept + step, a, b) > fitted.mse)
      assert.ok(mse(intercept, a + step, b) > fitted.mse)
      assert.ok(mse(intercept, a, b + step) > fitted.mse)
    }
  })

  test('matches scikit-learn on the same data', () => {
    // LinearRegression().fit(X, y) in scikit-learn 1.9 on the first ten rows
    const schema = ml
      .features({
        flipper: ml.number(),
        male: ml.boolean(),
        dream: ml.boolean(),
        mass: ml.number()
      })
      .target('mass')
    const rows = [
      [181, 1, 0, 3750],
      [186, 0, 0, 3800],
      [195, 0, 0, 3250],
      [193, 0, 0, 3450],
      [192, 0, 1, 3500],
      [196, 1, 1, 3900],
      [193, 1, 1, 3650],
      [188, 0, 1, 3525],
      [211, 0, 0, 4500],
      [230, 1, 0, 5700],
      [200, 0, 0, 4000]
    ].map(([flipper, male, dream, mass]) => ({
      flipper,
      male: male === 1,
      dream: dream === 1,
      mass
    }))
    const split = (seed: number) =>
      ml.dataset(schema, rows).split({ test: 0.05, seed })
    // The last row is held out, so training has the ten rows above it
    const seed = Array.from({ length: 50 }, (_, s) => s).find(
      s => split(s).test.rows[0]?.flipper === 200
    ) as number
    const { train } = split(seed)
    const fitted = model.fit(
      ml.pipeline(schema).fit(train).transform(train).matrix()
    )
    close(fitted.intercept, -3994.8475902787313, 1e-6)
    close(fitted.weights.flipper, 39.84507300105268, 1e-6)
    close(fitted.weights.male, 382.1891162066915, 1e-6)
    close(fitted.weights.dream, -212.71225227699196, 1e-6)
  })

  test('reads each one-hot weight against the reference', () => {
    // Biscoe penguins weigh 4700 g, the others 3700 g
    const schema = ml
      .features({
        island: ml.category(['Biscoe', 'Dream', 'Torgersen']),
        mass: ml.number()
      })
      .target('mass')
    const islands = ['Biscoe', 'Dream', 'Torgersen'] as const
    const { train } = ml
      .dataset(
        schema,
        Array.from({ length: 12 }, (_, i) => {
          const island = islands[i % 3] as (typeof islands)[number]
          return { island, mass: island === 'Biscoe' ? 4700 : 3700 }
        })
      )
      .split({ test: 0.25, seed: 1, stratify: 'island' })
    const fitted = model.fit(
      ml
        .pipeline(schema)
        .encode({ island: { reference: 'Biscoe' } })
        .fit(train)
        .transform(train)
        .matrix()
    )
    close(fitted.intercept, 4700)
    close(fitted.weights['island=Dream'], -1000)
    close(fitted.weights['island=Torgersen'], -1000)
  })

  test('prints the solver, the weights and the training MSE', () => {
    assert.equal(plain(String(model)), 'linear regression (normal equation)')
    const fitted = model.fit(linePipeline.transform(lines.train).matrix())
    assert.match(
      plain(String(fitted)),
      /^fitted linear regression \(normal equation\)\n {2}intercept {2}3\n {2}a {10}2\n {2}b {10}-1\ntraining MSE \d/
    )
  })

  test('the model and the fitted model are frozen', () => {
    assert.ok(Object.isFrozen(model))
    const fitted = model.fit(linePipeline.transform(lines.train).matrix())
    assert.ok(Object.isFrozen(fitted))
    assert.ok(Object.isFrozen(fitted.weights))
  })
})

describe('gradient descent', () => {
  // y = 3 + 2a - b plus noise, on columns scaled by the pipeline
  const noisy = ml.dataset(
    lineSchema,
    lineRows.map((row, i) => ({ ...row, y: row.y + ((i * 5) % 3) - 1 }))
  )
  const { train } = noisy.split({ test: 0.25, seed: 1 })
  const matrix = ml
    .pipeline(lineSchema)
    .scale(['a', 'b'])
    .fit(train)
    .transform(train)
    .matrix()
  const exact = model.fit(matrix)

  test('agrees with the normal equation once it has run enough epochs', () => {
    const fitted = ml
      .linearRegression({
        solver: 'gradient-descent',
        learningRate: 0.1,
        epochs: 2000
      })
      .fit(matrix)
    close(fitted.intercept, exact.intercept, 1e-9)
    close(fitted.weights.a, exact.weights.a, 1e-9)
    close(fitted.weights.b, exact.weights.b, 1e-9)
    close(fitted.mse, exact.mse, 1e-9)
  })

  test('approaches the normal equation as the epochs grow', () => {
    const distance = (epochs: number) => {
      const fitted = ml
        .linearRegression({
          solver: 'gradient-descent',
          learningRate: 0.05,
          epochs
        })
        .fit(matrix)
      return Math.abs(fitted.weights.a - exact.weights.a)
    }
    assert.ok(distance(10) > distance(50))
    assert.ok(distance(50) > distance(200))
  })

  test('is deterministic: it starts from zero', () => {
    const fit = () =>
      ml
        .linearRegression({
          solver: 'gradient-descent',
          learningRate: 0.1,
          epochs: 30
        })
        .fit(matrix)
    assert.deepEqual(fit().weights, fit().weights)
  })

  test('prints the loss before and after training', () => {
    const fitted = ml
      .linearRegression({
        solver: 'gradient-descent',
        learningRate: 0.1,
        epochs: 100
      })
      .fit(matrix)
    const text = plain(String(fitted)).split('\n')
    assert.equal(
      text[0],
      'fitted linear regression (gradient descent, learningRate 0.1, 100 epochs)'
    )
    assert.match(
      text.at(-1) ?? '',
      /^loss \S+ before the first epoch, \S+ after epoch 100$/
    )
  })

  test('a step that is too large is an error, not a model', () => {
    const steep = (epochs: number) =>
      ml.linearRegression({
        solver: 'gradient-descent',
        learningRate: 5,
        epochs
      })
    assert.throws(
      () => steep(3).fit(matrix),
      /gradient descent diverged: the loss grew from \S+ to \S+ in 3 epochs\. Lower learningRate \(5\)/
    )
    assert.throws(
      () => steep(5000).fit(matrix),
      /gradient descent diverged: the loss became Infinity at epoch \d+\. Lower learningRate \(5\)/
    )
  })

  test('checks for dependent columns like the normal equation', () => {
    const comboSchema = ml
      .features({ a: ml.number(), b: ml.number(), y: ml.number() })
      .target('y')
    const { train } = ml
      .dataset(
        comboSchema,
        Array.from({ length: 6 }, (_, i) => ({ a: i, b: 2 * i, y: i }))
      )
      .split({ test: 0.2, seed: 1 })
    assert.throws(
      () =>
        ml
          .linearRegression({
            solver: 'gradient-descent',
            learningRate: 0.1,
            epochs: 10
          })
          .fit(ml.pipeline(comboSchema).fit(train).transform(train).matrix()),
      /column 'b' is a linear combination of 'a'/
    )
  })

  test('rejects a learning rate or epochs that make no sense', () => {
    assert.throws(
      () =>
        ml.linearRegression({
          solver: 'gradient-descent',
          learningRate: 0,
          epochs: 10
        }),
      /learningRate must be a positive number, received 0/
    )
    assert.throws(
      () =>
        ml.linearRegression({
          solver: 'gradient-descent',
          learningRate: 0.1,
          epochs: 1.5
        }),
      /epochs must be a positive integer, received 1.5/
    )
  })
})

describe('ml.linearRegression errors', () => {
  test('rejects an unknown solver', () => {
    assert.throws(
      () => ml.linearRegression({ solver: 'magic' } as never),
      /solver must be 'normal-equation' or 'gradient-descent', received "magic"/
    )
  })

  test('fits only the matrix of a training block', () => {
    assert.throws(
      () => model.fit(linePipeline.transform(lines.test).matrix() as never),
      /takes the matrix of a training block, and this one comes from a test block/
    )
    assert.throws(
      () => model.fit({ columns: [], values: [], y: [] } as never),
      /expects the matrix of a prepared block/
    )
  })

  test('needs a numeric target', () => {
    const schema = ml
      .features({ x: ml.number(), label: ml.category(['a', 'b']) })
      .target('label')
    const { train } = ml
      .dataset(schema, [
        { x: 1, label: 'a' },
        { x: 2, label: 'b' },
        { x: 3, label: 'a' }
      ])
      .split({ test: 0.3, seed: 1 })
    const matrix = ml.pipeline(schema).fit(train).transform(train).matrix()
    assert.throws(
      () => model.fit(matrix as never),
      /linear regression predicts a number, and the target holds "(a|b)"/
    )
  })

  test('asks for a reference when a one-hot field has every category', () => {
    const schema = ml
      .features({ island: ml.category(['Biscoe', 'Dream']), mass: ml.number() })
      .target('mass')
    const { train } = ml
      .dataset(schema, [
        { island: 'Biscoe', mass: 1 },
        { island: 'Dream', mass: 2 },
        { island: 'Biscoe', mass: 1 }
      ])
      .split({ test: 0.3, seed: 1 })
    const matrix = ml.pipeline(schema).fit(train).transform(train).matrix()
    assert.throws(
      () => model.fit(matrix as never),
      /'island' has a column for every category, which the intercept makes linearly dependent\. Add \.encode\(\{ island: \{ reference: \.\.\. \} \}\)/
    )
  })
})

describe('linearly dependent columns', () => {
  const schema = ml
    .features({
      island: ml.category(['Biscoe', 'Dream', 'Torgersen']),
      a: ml.number(),
      b: ml.number(),
      mass: ml.number()
    })
    .target('mass')
  // Torgersen is declared but never observed
  const rows = Array.from({ length: 10 }, (_, i) => ({
    island: i % 2 === 0 ? 'Biscoe' : 'Dream',
    a: i,
    b: (i * 3) % 7,
    mass: 3000 + 10 * i
  }))
  const { train } = ml.dataset(schema, rows).split({ test: 0.2, seed: 1 })

  test('a category absent from training names the field and the category', () => {
    const matrix = ml
      .pipeline(schema)
      .encode({ island: { reference: 'Biscoe' } })
      .fit(train)
      .transform(train)
      .matrix()
    assert.throws(
      () => model.fit(matrix),
      /no training row has island 'Torgersen', so its column 'island=Torgersen' is all zero and its weight could be anything\. A record of that category would be predicted by a guess\. Use fewer folds/
    )
  })

  test('an absent reference makes the other columns add up to the intercept', () => {
    const matrix = ml
      .pipeline(schema)
      .encode({ island: { reference: 'Torgersen' } })
      .fit(train)
      .transform(train)
      .matrix()
    assert.throws(
      () => model.fit(matrix),
      /no training row has island 'Torgersen', the reference, so the other island columns add up to the intercept/
    )
  })

  test('a column that repeats the intercept is named', () => {
    const constantSchema = ml
      .features({ a: ml.number(), b: ml.number(), y: ml.number() })
      .target('y')
    const { train } = ml
      .dataset(
        constantSchema,
        Array.from({ length: 6 }, (_, i) => ({ a: i, b: 5, y: i }))
      )
      .split({ test: 0.2, seed: 1 })
    const matrix = ml
      .pipeline(constantSchema)
      .fit(train)
      .transform(train)
      .matrix()
    assert.throws(
      () => model.fit(matrix),
      /column 'b' has the same value in every training row, so it repeats the intercept/
    )
  })

  test('a combination of other columns names them', () => {
    const comboSchema = ml
      .features({
        a: ml.number(),
        b: ml.number(),
        c: ml.number(),
        y: ml.number()
      })
      .target('y')
    const { train } = ml
      .dataset(
        comboSchema,
        Array.from({ length: 8 }, (_, i) => {
          const a = i
          const b = (i * 3) % 5
          return { a, b, c: 1 + a - 2 * b, y: i }
        })
      )
      .split({ test: 0.2, seed: 1 })
    const matrix = ml
      .pipeline(comboSchema)
      .scale(['a'])
      .fit(train)
      .transform(train)
      .matrix()
    assert.throws(
      () => model.fit(matrix),
      /column 'c' is a linear combination of the intercept, 'a' and 'b' in the training block, so infinitely many weights fit it equally well/
    )
  })

  test('predict expects the columns of the training matrix', () => {
    const fitted = model.fit(linePipeline.transform(lines.train).matrix())
    const other = ml.features({ a: ml.number(), y: ml.number() }).target('y')
    const { train, test: held } = ml
      .dataset(other, [
        { a: 1, y: 1 },
        { a: 2, y: 2 }
      ])
      .split({ test: 0.5, seed: 1 })
    const matrix = ml.pipeline(other).fit(train).transform(held).matrix()
    assert.throws(
      () => fitted.predict(matrix),
      /predict\(\) expects the columns the model was fitted on: 'a', 'b'; received 'a'/
    )
  })
})
