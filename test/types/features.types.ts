import { type Infer, ml, type StandardSchemaV1 } from '../../src/index.ts'
import type { Equal, Expect } from './helpers.ts'

const penguinSchema = ml.features({
  species: ml.category(['Adelie', 'Chinstrap', 'Gentoo']),
  billLengthMm: ml.number().optional(),
  isAdult: ml.boolean(),
  observedAt: ml.timestamp(),
  sex: ml.category(['male', 'female']).optional()
})

type Penguin = typeof penguinSchema.infer

type _infersRecord = Expect<
  Equal<
    Penguin,
    {
      species: 'Adelie' | 'Chinstrap' | 'Gentoo'
      isAdult: boolean
      observedAt: number
      billLengthMm?: number | undefined
      sex?: 'male' | 'female' | undefined
    }
  >
>

type _inferHelper = Expect<Equal<Infer<typeof penguinSchema>, Penguin>>

type _standardOutput = Expect<
  Equal<StandardSchemaV1.InferOutput<typeof penguinSchema>, Penguin>
>

type _keys = Expect<
  Equal<
    (typeof penguinSchema.keys)[number],
    'species' | 'billLengthMm' | 'isAdult' | 'observedAt' | 'sex'
  >
>

type _categoriesKeepLiterals = Expect<
  Equal<
    (typeof penguinSchema.fields.species.categories)[number],
    'Adelie' | 'Chinstrap' | 'Gentoo'
  >
>

type _parseReturnsRecord = Expect<
  Equal<ReturnType<typeof penguinSchema.parse>, Penguin>
>

export const valid: Penguin = {
  species: 'Gentoo',
  isAdult: true,
  observedAt: 1194652800000
}

export const wrongType: Penguin = {
  species: 'Gentoo',
  isAdult: true,
  observedAt: 1194652800000,
  // @ts-expect-error — billLengthMm is a number
  billLengthMm: '28'
}

export const unknownCategory: Penguin = {
  // @ts-expect-error — 'Emperor' is not a declared category
  species: 'Emperor',
  isAdult: true,
  observedAt: 1194652800000
}

// @ts-expect-error — species is required
export const missingRequired: Penguin = {
  isAdult: true,
  observedAt: 1194652800000
}

// @ts-expect-error — at least one category is required
ml.category([])

// @ts-expect-error — categories are strings
ml.category([1, 2])

// @ts-expect-error — fields must be built with ml.number(), ml.category(), …
ml.features({ age: 3 })

// ── target ───────────────────────────────────────────────────────────────────

const targeted = penguinSchema.target('species')

type _targetKey = Expect<Equal<typeof targeted.targetKey, 'species'>>
type _targetKeepsRecord = Expect<Equal<typeof targeted.infer, Penguin>>

// A schema with a target is still a schema
ml.dataset(targeted, [])

penguinSchema.target('isAdult')

// @ts-expect-error — a target must be required
penguinSchema.target('billLengthMm')

// @ts-expect-error — a timestamp can't be a target
penguinSchema.target('observedAt')

// @ts-expect-error — 'weight' is not a field
penguinSchema.target('weight')
