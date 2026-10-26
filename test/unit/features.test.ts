import assert from 'node:assert/strict'
import { describe, test } from 'node:test'
import { FeatureValidationError, ml } from '../../src/index.ts'

const penguinSchema = ml.features({
  species: ml.category(['Adelie', 'Chinstrap', 'Gentoo']),
  billLengthMm: ml.number().optional(),
  isAdult: ml.boolean(),
  observedAt: ml.timestamp()
})

const valid = {
  species: 'Adelie',
  billLengthMm: 39.1,
  isAdult: true,
  observedAt: new Date('2007-11-10')
}
// What parse returns for `valid`: the timestamp in milliseconds since 1970
const parsed = { ...valid, observedAt: 1194652800000 }

const issuesOf = (fn: () => unknown) => {
  try {
    fn()
  } catch (error) {
    assert.ok(error instanceof FeatureValidationError)
    return error.issues
  }
  assert.fail('expected a FeatureValidationError')
}

describe('fields', () => {
  test('category keeps the declaration order', () => {
    assert.deepEqual(ml.category(['b', 'a', 'c']).categories, ['b', 'a', 'c'])
  })

  test('category rejects duplicates', () => {
    assert.throws(() => ml.category(['a', 'a']), /'a' more than once/)
  })

  test('optional() returns a new field and keeps the original', () => {
    const required = ml.number()
    const optional = required.optional()
    assert.equal(required.isOptional, false)
    assert.equal(optional.isOptional, true)
  })
})

describe('ml.features', () => {
  test('keeps keys in declaration order', () => {
    assert.deepEqual(penguinSchema.keys, [
      'species',
      'billLengthMm',
      'isAdult',
      'observedAt'
    ])
  })

  test('rejects values that are not fields', () => {
    assert.throws(
      () => ml.features({ age: 3 } as never),
      /'age' is not a field/
    )
  })

  test('infer is type-only', () => {
    assert.throws(() => penguinSchema.infer, /type-only/)
  })
})

describe('target', () => {
  test('returns a new schema with the same fields', () => {
    const targeted = penguinSchema.target('species')
    assert.equal(targeted.targetKey, 'species')
    assert.equal(targeted.fields, penguinSchema.fields)
    assert.deepEqual(targeted.keys, penguinSchema.keys)
    assert.deepEqual(targeted.parse(valid), parsed)
    assert.equal('targetKey' in penguinSchema, false)
    assert.ok(Object.isFrozen(targeted))
  })

  test('keeps infer type-only', () => {
    assert.throws(() => penguinSchema.target('species').infer, /type-only/)
  })

  test('rejects a field that is not in the schema', () => {
    assert.throws(
      () => penguinSchema.target('weight' as never),
      /'weight' is not a field\. Expected one of 'species' \| 'billLengthMm'/
    )
  })

  test('rejects an optional field', () => {
    assert.throws(
      () => penguinSchema.target('billLengthMm' as never),
      /'billLengthMm' is optional, and a target must be required/
    )
  })

  test('rejects a timestamp', () => {
    assert.throws(
      () => penguinSchema.target('observedAt' as never),
      /'observedAt' is a timestamp/
    )
  })
})

describe('parse', () => {
  test('returns the record', () => {
    assert.deepEqual(penguinSchema.parse(valid), parsed)
  })

  test('drops undeclared fields', () => {
    assert.deepEqual(penguinSchema.parse({ ...valid, id: 7 }), parsed)
  })

  test('treats undefined, null and NaN as missing in optional fields', () => {
    for (const billLengthMm of [undefined, null, Number.NaN]) {
      assert.equal(
        penguinSchema.parse({ ...valid, billLengthMm }).billLengthMm,
        undefined
      )
    }
  })

  test('reports a missing required field with a hint', () => {
    const [issue] = issuesOf(() =>
      penguinSchema.parse({ ...valid, isAdult: null })
    )
    assert.equal(issue?.code, 'missing')
    assert.match(issue?.message ?? '', /'isAdult' is required.*\.optional\(\)/)
  })

  test('reports the wrong type', () => {
    const [issue] = issuesOf(() =>
      penguinSchema.parse({ ...valid, billLengthMm: '39.1' })
    )
    assert.equal(issue?.code, 'invalid_type')
    assert.match(issue?.message ?? '', /expected a finite number.*'39\.1'/)
  })

  test('rejects Infinity', () => {
    const [issue] = issuesOf(() =>
      penguinSchema.parse({ ...valid, billLengthMm: Number.POSITIVE_INFINITY })
    )
    assert.equal(issue?.code, 'invalid_type')
  })

  test('reports unknown categories with the expected ones', () => {
    const [issue] = issuesOf(() =>
      penguinSchema.parse({ ...valid, species: 'Emperor' })
    )
    assert.equal(issue?.code, 'unknown_category')
    assert.match(
      issue?.message ?? '',
      /'Emperor'.*'Adelie' \| 'Chinstrap' \| 'Gentoo'/
    )
  })

  test('converts timestamps to milliseconds since 1970', () => {
    for (const observedAt of [
      new Date('2007-11-10'),
      1194652800000,
      '2007-11-10',
      '2007-11-10T00:00:00Z'
    ]) {
      assert.equal(
        penguinSchema.parse({ ...valid, observedAt }).observedAt,
        1194652800000
      )
    }
  })

  test('reports invalid dates with the values received', () => {
    const [issue] = issuesOf(() =>
      penguinSchema.parse({ ...valid, observedAt: '2007-02-30' })
    )
    assert.equal(issue?.code, 'invalid_timestamp')
    assert.equal(
      issue?.message,
      "'observedAt' received dates that don't exist: '2007-02-30'"
    )
  })

  test('reports records that are not objects', () => {
    const [issue] = issuesOf(() => penguinSchema.parse([1, 2]))
    assert.equal(issue?.code, 'invalid_record')
  })
})

describe('parseRows', () => {
  test('groups issues by field and code', () => {
    const rows = Array.from({ length: 100 }, (_, index) => ({
      ...valid,
      billLengthMm: index % 10 === 0 ? 'NA' : 40
    }))
    const issues = issuesOf(() => penguinSchema.parseRows(rows))
    assert.equal(issues.length, 1)
    assert.deepEqual(issues[0]?.rows, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
    assert.deepEqual(issues[0]?.received, ['NA'])
    assert.match(
      issues[0]?.message ?? '',
      /in 10 rows \(index 0, 10, 20, 30, 40, …\)/
    )
  })

  test('reports every problem at once', () => {
    const issues = issuesOf(() =>
      penguinSchema.parseRows([
        { ...valid, species: 'Emperor' },
        { ...valid, isAdult: 'yes' }
      ])
    )
    assert.deepEqual(
      issues.map(issue => [issue.field, issue.code]),
      [
        ['species', 'unknown_category'],
        ['isAdult', 'invalid_type']
      ]
    )
  })
})

describe('FeatureValidationError stack', () => {
  const firstFrame = (fn: () => unknown): string => {
    try {
      fn()
    } catch (error) {
      assert.ok(error instanceof FeatureValidationError)
      const frame = error.stack?.split('\n').find(line => /^\s+at /.test(line))
      return frame ?? ''
    }
    assert.fail('expected a FeatureValidationError')
  }

  test('starts at the parse call', () => {
    assert.match(
      firstFrame(() => penguinSchema.parse({})),
      /features\.test\.ts/
    )
  })

  test('starts at the parseRows call', () => {
    assert.match(
      firstFrame(() => penguinSchema.parseRows([{}])),
      /features\.test\.ts/
    )
  })
})

describe('Standard Schema', () => {
  const { validate, vendor, version } = penguinSchema['~standard']

  test('exposes vendor and version', () => {
    assert.equal(vendor, 'tl')
    assert.equal(version, 1)
  })

  test('returns the value on success', () => {
    assert.deepEqual(validate(valid), { value: parsed })
  })

  test('returns issues with a path on failure', () => {
    const result = validate({ ...valid, species: 'Emperor' })
    assert.ok('issues' in result && result.issues !== undefined)
    assert.deepEqual(result.issues[0]?.path, ['species'])
  })
})
