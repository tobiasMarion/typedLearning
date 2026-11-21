import { type Metric, ml } from '../../src/index.ts'
import type { Equal, Expect } from './helpers.ts'

type _mse = Expect<Equal<typeof ml.mse, Metric<'mse', number>>>
type _name = Expect<Equal<typeof ml.r2.name, 'r2'>>
type _score = Expect<Equal<ReturnType<typeof ml.mae>, number>>

// @ts-expect-error — regression metrics score numbers
ml.mse(['Adelie'], ['Gentoo'])
