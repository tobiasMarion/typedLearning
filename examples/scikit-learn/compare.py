"""Repeats compare.ts with scikit-learn and compares the two.

Same rows in each block (read from TL's output), same preprocessing: median for
the missing numbers, the most frequent sex, standardization with the population
standard deviation, one-hot without the reference categories. Then it reports
how many decimal places the weights and the test predictions agree on, and how
long each library took.

Run it with `npm run compare:scikit-learn`, which runs compare.ts first.
"""

import json
import math
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

set_config(transform_output="pandas")

tl = json.loads((Path(tempfile.gettempdir()) / "tl-penguin-mass.json").read_text())
REPEATS = tl["repeats"]
NUMBERS = ["billLengthMm", "billDepthMm", "flipperLengthMm"]

rows = json.loads((Path(__file__).parent / "../penguins/all.json").read_text())
penguins = pd.DataFrame([row for row in rows if row["bodyMassG"] is not None])
X = penguins[["island", *NUMBERS, "sex"]]
y = penguins["bodyMassG"]
X_train, y_train = X.iloc[tl["train"]], y.iloc[tl["train"]]
X_test, y_test = X.iloc[tl["test"]], y.iloc[tl["test"]]

preprocess = ColumnTransformer(
    [
        (
            "numbers",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]
            ),
            NUMBERS,
        ),
        (
            "categories",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    (
                        "encode",
                        OneHotEncoder(
                            categories=[
                                ["Biscoe", "Dream", "Torgersen"],
                                ["female", "male"],
                            ],
                            drop=["Biscoe", "female"],
                            sparse_output=False,
                        ),
                    ),
                ]
            ),
            ["island", "sex"],
        ),
    ],
    verbose_feature_names_out=False,
)


def timed(run):
    """The median time of REPEATS runs, in milliseconds, after one warm-up run."""
    result = run()
    times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        result = run()
        times.append((time.perf_counter() - start) * 1000)
    return result, sorted(times)[REPEATS // 2]


def prepare():
    fitted = preprocess.fit(X_train)
    return fitted.transform(X_train), fitted.transform(X_test)


(train_matrix, test_matrix), prepare_ms = timed(prepare)
model, fit_ms = timed(lambda: LinearRegression().fit(train_matrix, y_train))
predictions = model.predict(test_matrix)


def tl_name(name):
    """island_Dream → island=Dream, as TL names one-hot columns."""
    for field in ("island", "sex"):
        if name.startswith(f"{field}_"):
            return f"{field}={name[len(field) + 1 :]}"
    return name


def decimals(a, b):
    """How many decimal places a and b agree on, rounding both."""
    for places in range(15, -1, -1):
        if round(a, places) == round(b, places):
            return places
    return 0


sk = {"intercept": model.intercept_}
sk.update({tl_name(n): w for n, w in zip(train_matrix.columns, model.coef_)})
ours = {"intercept": tl["normalEquation"]["intercept"], **tl["normalEquation"]["weights"]}
descent = {"intercept": tl["gradientDescent"]["intercept"], **tl["gradientDescent"]["weights"]}

assert set(sk) == set(ours), f"columns differ: {set(sk) ^ set(ours)}"

table = pd.DataFrame(
    {
        "scikit-learn": sk,
        "TL normal equation": {k: ours[k] for k in sk},
        "|difference|": {k: abs(ours[k] - sk[k]) for k in sk},
        "same decimals": {k: decimals(ours[k], sk[k]) for k in sk},
        "TL gradient descent": {k: descent[k] for k in sk},
        "GD same decimals": {k: decimals(descent[k], sk[k]) for k in sk},
    }
)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: f"{x:.10g}")
print(f"\n{len(X_train)} training rows, {len(X_test)} test rows\n")
print(table)

difference = np.abs(np.array(tl["normalEquation"]["predictions"]) - predictions)
print(
    f"\ntest predictions: largest |difference| {difference.max():.3g} g, "
    f"agree to {min(decimals(a, b) for a, b in zip(tl['normalEquation']['predictions'], predictions))} decimals"
)

ms = tl["ms"]
print(f"\nmedian of {REPEATS} runs, in ms")
print(
    pd.DataFrame(
        {
            "TL": [ms["preprocessing"], ms["normalEquation"]],
            "scikit-learn": [prepare_ms, fit_ms],
        },
        index=["preprocessing (fit + transform both blocks)", "linear regression fit"],
    ).round(3)
)
print(f"TL gradient descent (2000 epochs): {ms['gradientDescent']:.3f} ms")
