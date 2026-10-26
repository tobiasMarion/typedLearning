# Feature DSL

The DSL defines the data schema using only the library. Users don't need to import Zod or any other validator.

```ts
const penguinSchema = ml.features({
  species: ml.category(['Adelie', 'Chinstrap', 'Gentoo']),
  billLengthMm: ml.number().optional(),
  isAdult: ml.boolean(),
  observedAt: ml.timestamp()
})

type Penguin = typeof penguinSchema.infer // or Infer<typeof penguinSchema>
// {
//   species: 'Adelie' | 'Chinstrap' | 'Gentoo'
//   billLengthMm?: number | undefined
//   isAdult: boolean
//   observedAt: number // milliseconds since 1970, UTC
// }

penguinSchema.parse(record) // one record
penguinSchema.parseRows(rows) // many records, issues grouped by field
```

Training data doesn't need `parseRows`: `ml.dataset` validates its rows. `parse` and `parseRows` validate records on their own, for example a form entry or a request that goes to `predict`, which trusts its typed input. A project that already validates with Zod (or another Standard Schema library) can keep doing so and derive the features with [`ml.fromSchema`](#zod-compatibility-through-standard-schema).

## Requirements

- Infer the record type from the schema.
- Represent categories as literal unions.
- Reject at compile time values of the wrong type (`age: '28'`) and categories that weren't declared.
- Validate external data at runtime, with messages that point to the field and the problem.
- Mark optional fields in the type, so the pipeline [requires a policy](preprocessing.md#missing-values) for them.
- Only required `timestamp` fields can be used as the `time` key of a [series](dataset.md#dataset-and-series), and only required `category` fields as its `group`.

## Field types

| Constructor               | Inferred type                           | Status |
| ------------------------- | --------------------------------------- | ------ |
| `ml.number()`             | `number`                                | done   |
| `ml.boolean()`            | `boolean`                               | done   |
| `ml.category([…])`        | literal union                           | done   |
| `ml.orderedCategory([…])` | literal union, ranked in list order     | done   |
| `ml.timestamp()`          | `number`, milliseconds since 1970 (UTC) | done   |
| `.optional()`             | `T \| undefined`                        | done   |

`ml.category([…])` needs at least one category and rejects duplicates, both at compile time and at runtime. Its declaration order is kept: it is the column order the encoder produces.

`.optional()` wraps the field type instead of flipping a flag on it, so hovers stay readable:

```ts
// typeof penguinSchema.fields
{
  readonly species: CategoryField<'Adelie' | 'Chinstrap' | 'Gentoo'>
  readonly billLengthMm: Optional<NumberField>
  readonly sex: Optional<CategoryField<'male' | 'female'>>
}
```

The runtime copy only changes `isOptional` to `true`. It has no `.optional()` of its own, so `ml.number().optional().optional()` is a compile-time error.

## Ordered categories

Some categories have a rank: `'poor' < 'fair' < 'good'`, `'cold' < 'warm' < 'hot'`. That order is a fact about the data, not a preprocessing choice, so it belongs in the schema:

```ts
const reviewSchema = ml.features({
  condition: ml.orderedCategory(['poor', 'fair', 'good']) // lowest to highest
})
```

The value type is the same literal union as `ml.category`, and validation is the same; error messages show the order (`one of 'poor' < 'fair' < 'good'`). The difference is for later steps: by default the encoder turns an ordered category into a single column with its position instead of one-hot columns (see [Encoding](preprocessing.md#encoding)), and a tree can split on `≤ 'fair'`. `ml.fromSchema` maps `z.enum` to `ml.category`; pass an `ml.orderedCategory` override when the order matters.

See `examples/penguins`, where `bodyCondition` is a graded scale from `'poor'` to `'good'` (`npm run example`).

## Target

`.target(field)` marks the field a model predicts and returns a new schema with the same fields:

```ts
const penguinSchema = ml.features({ ... }).target('species')
```

The target has to be a required `number`, `boolean`, `category` or `orderedCategory`, and its kind decides the problem: a number is regression, the others are classification. A wrong field is a type error that lists the valid ones, and the key is kept as `targetKey`. See [Preprocessing](preprocessing.md#target) for what it changes downstream.

## Runtime validation

- **Missing values:** `undefined`, `null`, and `NaN` count as missing. An optional field turns them into `undefined`; a required field reports them.
- **Numbers** must be finite. `Infinity` is rejected.
- **Timestamps** accept a valid `Date`, a number of milliseconds since 1970-01-01T00:00:00Z, or an ISO 8601 string (`'2025-01-31'`, `'2025-01-31T10:00:00-03:00'`, with `T` or a space before the time). They come out as milliseconds, the unit of `Date.getTime()`: a number can't be changed through a shared reference, compares with `<`, and is what the algorithms use. Dates that don't exist, such as `'2025-02-30'`, are rejected instead of rolled over.
- **Time zone:** an ISO string without an offset (`'2025-01-31T10:00'`) is read as **UTC**, so the same data gives the same timestamps on every machine. If the data was recorded in local time, every row is shifted by the same offset, which keeps the order and the intervals between rows. Add the offset (`-03:00`) to keep the real instant.
- **Undeclared fields** are dropped from the output, so a CSV can carry extra columns such as an id.
- **No coercion:** `'39.1'` is not a number. Parsing strings from CSV is part of loading data (see [Dataset](dataset.md)). Timestamps are the exception: dates have no JSON literal, so ISO 8601 strings are an accepted way to write them.

`parseRows` groups issues by field and problem, so 100k rows with one bad column produce one message, not 100k:

```
Invalid features: 3 problems found
  - 'species' received unknown categories in 1 row (index 1): 'Adelie '. Expected one of 'Adelie' | 'Chinstrap' | 'Gentoo'
  - 'bodyMassG' expected a finite number in 2 rows (index 1, 2), received 'NA'
  - 'bodyMassG' is required but missing in 1 row (index 3). Mark it with .optional() if missing values are expected
```

Errors are thrown as `FeatureValidationError`, whose `issues` hold the field, code, row indexes, and up to 3 distinct sample values.

The stack trace starts at the `parse`/`parseRows` call, so an uncaught error points at your line, not at the library. In a terminal the stack is colored (fields, received and expected values); `message` is always plain text, and `NO_COLOR` turns colors off. Run `npm run example:bad` to see it on a messy dataset.

## Zod compatibility through Standard Schema

**Decision:** the DSL and its validator are our own, with no runtime dependencies. Compatibility with Zod, Valibot, and ArkType goes through [Standard Schema](https://standardschema.dev), the shared interface those libraries implement.

**Why not build on Zod:**
- **Different jobs.** Zod describes the _shape of a value_. The DSL describes the _meaning of a column_: its kind, its category order, and whether it needs a missing-value policy. The pipeline types inspect that meaning. Building them on Zod's internal types would couple the most delicate part of the API to a dependency, and hover types would show `ZodObject<…>` instead of readable records.
- **Tabular errors.** Zod reports one issue per value. For datasets we want issues grouped per column.
- **Missing-value semantics.** In ML, `NaN` and `null` usually mean missing. Zod rejects `NaN` in `z.number()`.

**How compatibility works:**

| Direction      | Mechanism                                                                                                                                                             | Status |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ |
| TL → ecosystem | Features implement `~standard` (Standard Schema V1), so tools that accept Zod/Valibot/ArkType through Standard Schema accept them too                                 | done   |
| Zod → TL       | `ml.fromSchema(schema)` reads the types from `StandardSchemaV1.InferOutput` and the categories and required fields from Standard JSON Schema (`~standard.jsonSchema`) | done   |

```ts
const Penguin = z.object({
  species: z.enum(['Adelie', 'Chinstrap', 'Gentoo']),
  billLengthMm: z.number().optional(),
  notes: z.string(),
  observedAt: z.date()
})

const penguinSchema = ml.fromSchema(Penguin, {
  notes: ml.category(['calm', 'agitated']), // required: z.string() has no categories
  observedAt: ml.timestamp() // required: JSON Schema can't represent dates
})
```

**Mapping rules:**

| Zod                                         | TL                                      |
| ------------------------------------------- | --------------------------------------- |
| `z.number()`, `z.number().int()`            | `ml.number()`                           |
| `z.boolean()`                               | `ml.boolean()`                          |
| `z.enum([...])`, `z.literal('x')`           | `ml.category([...])`, in the same order |
| `.optional()`, `.nullable()`                | `.optional()`                           |
| anything else (`z.string()`, `z.date()`, …) | needs an override                       |

Fields that can't be mapped make the override **required at compile time**: leaving one out is a type error that names the missing field. An override can also replace a mapped field, for example to change the category order. An override for a field the schema doesn't have is a type error too.

The Standard Schema interfaces are copied into `src/standard-schema.ts`, as the spec recommends, so there is no dependency on `@standard-schema/spec`. Zod is a `devDependency` used only in compatibility tests.

**Known limits:**
- A TL schema is not a Zod schema: it works where Standard Schema is accepted, but not inside `z.object({...})`.
- JSON Schema can't represent `z.date()`, and `z.iso.datetime()` validates strings. Dates always need an `ml.timestamp()` override; the data can hold `Date` instances, milliseconds, or ISO strings, and comes out as milliseconds.
- Zod's published types declare `~standard` without `jsonSchema`, even though it exists at runtime, so `fromSchema` checks for it at runtime and explains what is missing.

## Open questions

- Distinguish `optional` from `nullable`. Today `null` is accepted at runtime as missing, but the type only says `| undefined`.
