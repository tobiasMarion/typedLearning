# Methodological errors

Methodological errors leave the code running but make the reported score better than the model really is. This document tracks the ones the library deals with. The goal for each one is to prevent it, or at least make it harder to write.

The categories follow the leakage taxonomy of Kapoor & Narayanan, _Leakage and the reproducibility crisis in machine-learning-based science_ (Patterns, 2023), extended with errors in model selection and evaluation (Cawley & Talbot, _On over-fitting in model selection and subsequent selection bias in performance evaluation_, JMLR, 2010) and in time series (López de Prado, _Advances in Financial Machine Learning_, 2018).

Each error gets a status:

| Status        | Meaning                                                                  |
| ------------- | ------------------------------------------------------------------------ |
| **compile**   | The wrong code doesn't compile.                                          |
| **runtime**   | The wrong code throws at the call, with a message that names the fix.    |
| **by design** | There is no way to write it: the API does it the right way.              |
| **required**  | The user has to write the decision down; there is no default.            |
| **opt-in**    | A tool avoids it, but the default code doesn't use it and nothing warns. |
| **silent**    | The wrong code runs and reports a score.                                 |

_Today_ is the state of `main`. _Planned_ names the milestone that changes it; a dash means nothing is planned. Every status of _Today_ other than silent has a test with its ID in [`test/methodology/`](../test/methodology/).

## Train and test aren't separate

| ID  | Error                                                                                  | Today                                                                                          | Planned                                    |
| --- | -------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------ |
| S1  | Training on the test block, or evaluating on the training block                        | compile (roles)                                                                                | compile at `fit` (M3) and `evaluate` (M5)  |
| S2  | Splitting the test block again, to carve validation out of it                          | compile                                                                                        | —                                          |
| S3  | Choosing hyperparameters or models by their test score                                 | silent                                                                                         | compile: the search only sees `Train` (M6) |
| S4  | Fitting preprocessing (imputation, scaling, encoding) on all the data before the split | compile (`pipeline.fit` takes a `Train`, and the pipeline starts from a schema, not from data) | —                                          |
| S5  | Selecting features with all the data before the split                                  | silent                                                                                         | —                                          |
| S6  | Oversampling before the split, so copies of a row land in both blocks                  | silent                                                                                         | —                                          |
| S7  | The same record in train and test                                                      | silent                                                                                         | —                                          |

S1 is already caught by the types: a `Test` isn't assignable to `Train`. Nothing consumes a block until `fit` in M3, which is why the planned column repeats it.

## Train and test aren't independent

| ID  | Error                                                                                                | Today                                              | Planned                                            |
| --- | ---------------------------------------------------------------------------------------------------- | -------------------------------------------------- | -------------------------------------------------- |
| T1  | Random split or k-fold of a time series                                                              | compile, runtime (split); compile (k-fold)         | —                                                  |
| T2  | Time series split in the order the rows arrived, not by time                                         | by design (`ml.series` sorts)                      | —                                                  |
| T3  | Rows of several series stacked as one, so a time repeats                                             | runtime (`duplicate_time`)                         | —                                                  |
| T4  | No gap between train and test, so the last training labels look into the test period                 | required (`gap`, in the split and in walk-forward) | —                                                  |
| T5  | A gap shorter than the target's horizon                                                              | silent                                             | compile or runtime against the target horizon (M5) |
| T6  | Panel data cut at a different time per series, so one series trains on a period another is tested on | by design (one cut for every group)                | —                                                  |
| T7  | Features computed with future values, such as a centered rolling mean                                | silent                                             | by design: lag and rolling only look back (M5)     |
| T8  | The same entity (patient, user) in train and test                                                    | silent                                             | opt-in: group k-fold (after the MVP)               |

## Evaluation

| ID  | Error                                                               | Today             | Planned      |
| --- | ------------------------------------------------------------------- | ----------------- | ------------ |
| E1  | A split that can't be repeated, so a lucky one can't be told apart  | required (`seed`) | —            |
| E2  | A metric that doesn't match the target, such as MSE on class labels | —                 | compile (M5) |

## Data

| ID  | Error                                                                                     | Today                                                                               | Planned |
| --- | ----------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | ------- |
| D1  | Invalid values read silently: unknown categories, impossible dates such as `'2025-02-30'` | runtime                                                                             | —       |
| D2  | Missing values filled or dropped by a default nobody chose                                | runtime (required fields); required (a policy per optional field before `matrix()`) | —       |
