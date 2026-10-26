import { dataset, kFold, series, walkForward } from './dataset/index.ts'
import {
  boolean,
  category,
  features,
  fromSchema,
  number,
  orderedCategory,
  timestamp
} from './features/index.ts'
import { pipeline } from './pipeline/index.ts'

export const ml = Object.freeze({
  features,
  fromSchema,
  number,
  boolean,
  category,
  orderedCategory,
  timestamp,
  dataset,
  series,
  kFold,
  walkForward,
  pipeline
})

/** The record type described by a `Features` schema. */
export type Infer<F extends { readonly infer: unknown }> = F['infer']

export type {
  ChronologicalSplitOptions,
  ChronologicalSplitWithValidationOptions,
  Dataset,
  Fold,
  Gap,
  KFold,
  KFoldOptions,
  Origin,
  RandomSplitOptions,
  RandomSplitWithValidationOptions,
  Role,
  RoleBrand,
  Series,
  SeriesOptions,
  Test,
  TimestampInput,
  Train,
  TrainTest,
  TrainValidationTest,
  Validation,
  WalkForward,
  WalkForwardOptions
} from './dataset/index.ts'
export type {
  AnyField,
  BooleanField,
  CategoryField,
  FeatureIssue,
  Features,
  IssueCode,
  NumberField,
  Optional,
  OrderedCategoryField,
  RequiredField,
  Shape,
  Targeted,
  TimestampField
} from './features/index.ts'
export { FeatureValidationError } from './features/index.ts'
export type {
  Columns,
  Fitted,
  Matrix,
  Pipeline,
  Prepared,
  PreparedRow
} from './pipeline/index.ts'
export type { StandardSchemaV1 } from './standard-schema.ts'
