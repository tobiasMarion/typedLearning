"""Palmer Penguins body mass regression with scikit-learn, as a reference.

The M4 part of examples/penguins, written the way a careful scikit-learn user
would: rows without the target leave before the split, each one-hot field drops
a reference category so the intercept doesn't make the columns linearly
dependent, and the closed-form solution is compared with gradient descent.

Run it with `npm run example:scikit-learn:mass`; uv installs the dependencies in
pyproject.toml. seaborn downloads the dataset the first time.
"""

import pandas as pd
import seaborn as sns
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

set_config(transform_output="pandas")

SEED = 42
TARGET = "body_mass_g"
CATEGORICAL = {
    "island": ["Biscoe", "Dream", "Torgersen"],
    "sex": ["Female", "Male"],
}
# The reference of each field: its column is dropped, and the other weights
# read against it.
REFERENCE = {"island": "Biscoe", "sex": "Female"}

# ── Boundary: the target has to be present ────────────────────────────────────

penguins = sns.load_dataset("penguins").dropna(subset=[TARGET])
X = penguins[["island", "flipper_length_mm", "sex"]]
y = penguins[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=X["island"], random_state=SEED
)

# ── Preprocessing ─────────────────────────────────────────────────────────────

preprocess = ColumnTransformer(
    [
        (
            "numeric",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]
            ),
            ["flipper_length_mm"],
        ),
        (
            "categorical",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    (
                        "encode",
                        OneHotEncoder(
                            categories=list(CATEGORICAL.values()),
                            # Without drop, LinearRegression still answers:
                            # lstsq picks the smallest of infinitely many
                            # weights, silently.
                            drop=[REFERENCE[field] for field in CATEGORICAL],
                            handle_unknown="error",
                            sparse_output=False,
                        ),
                    ),
                ]
            ),
            list(CATEGORICAL),
        ),
    ],
    remainder="drop",
    verbose_feature_names_out=False,
)

# ── Closed form and gradient descent ──────────────────────────────────────────

exact = Pipeline([("preprocess", preprocess), ("regress", LinearRegression())])
exact.fit(X_train, y_train)

# SGDRegressor steps one row at a time, not over the whole block like TL's
# gradient descent, so it only gets close to the closed form.
stepped = Pipeline(
    [
        ("preprocess", preprocess),
        (
            "regress",
            SGDRegressor(
                penalty=None,
                learning_rate="constant",
                eta0=0.001,
                max_iter=5_000,
                tol=None,
                random_state=SEED,
            ),
        ),
    ]
)
stepped.fit(X_train, y_train)

columns = exact.named_steps["preprocess"].get_feature_names_out()
weights = pd.DataFrame(
    {
        "normal equation": [
            exact.named_steps["regress"].intercept_,
            *exact.named_steps["regress"].coef_,
        ],
        "gradient descent": [
            stepped.named_steps["regress"].intercept_[0],
            *stepped.named_steps["regress"].coef_,
        ],
    },
    index=["intercept", *columns],
)
print(weights.round(1))

# ── Predict the test block ────────────────────────────────────────────────────
# Scoring is M5's evaluate in TL; here only the predictions are shown.

print(pd.DataFrame({"predicted": exact.predict(X_test), "actual": y_test}).head())
