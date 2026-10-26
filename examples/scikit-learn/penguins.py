"""Palmer Penguins species classification with scikit-learn, as a reference.

The workflow TL is aiming for, written the way a careful scikit-learn user
would: every preprocessing step lives inside the pipeline, the test block is
held out before anything learns, hyperparameters are chosen by cross-validation
on the training block only, the test block is scored once, and the production
model is refitted on all the data.

Run it with `npm run example:scikit-learn`; uv installs the dependencies in pyproject.toml.
seaborn downloads the dataset the first time.
"""

import tempfile
from pathlib import Path

import joblib
import pandas as pd
import seaborn as sns
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Transformers return DataFrames with named columns, so the matrix the model
# sees can be printed and read.
set_config(transform_output="pandas")

SEED = 42

# ── 1. Schema: what each column is ────────────────────────────────────────────
# scikit-learn has no schema, so it is written by hand. Declaring categories
# (instead of learning them from the data) keeps one-hot columns stable across
# folds and production.

TARGET = "species"
SPECIES = ["Adelie", "Chinstrap", "Gentoo"]
NUMERIC = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
CATEGORICAL = {
    "island": ["Biscoe", "Dream", "Torgersen"],
    "sex": ["Male", "Female"],
}

# ── 2. Load and validate at the boundary ──────────────────────────────────────
# Population choices are made here, before the split: rows without a label
# can't be used to train or to score, so they leave now, for every block alike.

penguins = sns.load_dataset("penguins")
penguins = penguins.dropna(subset=[TARGET])

unknown = set(penguins[TARGET]) - set(SPECIES)
assert not unknown, f"undeclared species: {unknown}"
for column, categories in CATEGORICAL.items():
    unknown = set(penguins[column].dropna()) - set(categories)
    assert not unknown, f"undeclared {column}: {unknown}"

X = penguins[NUMERIC + list(CATEGORICAL)]
y = penguins[TARGET]

# ── 3. Split once: the test block is held out before anything learns ─────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

# ── 4. Preprocessing, declared but not fitted ─────────────────────────────────
# Nothing here has seen data. Each transformer learns its state (means, standard
# deviations, neighbors, most frequent category) only when the whole pipeline
# is fitted, so cross-validation refits it on each fold's training part.

numeric = Pipeline(
    [
        # StandardScaler ignores missing values when it learns and keeps them,
        # so scaling first lets KNN measure distances in comparable units.
        ("scale", StandardScaler()),
        ("impute", KNNImputer(n_neighbors=5)),
    ]
)

categorical = Pipeline(
    [
        ("impute", SimpleImputer(strategy="most_frequent")),
        (
            "encode",
            OneHotEncoder(
                categories=list(CATEGORICAL.values()),
                handle_unknown="error",
                sparse_output=False,
            ),
        ),
    ]
)

preprocess = ColumnTransformer(
    [
        ("numeric", numeric, NUMERIC),
        ("categorical", categorical, list(CATEGORICAL)),
    ],
    # Columns not listed above don't reach the model
    remainder="drop",
    verbose_feature_names_out=False,
)

# ── 5. The model: preprocessing and estimator in one object ───────────────────
# Fitting this object fits everything, in order, on the same data. There is no
# way to fit the scaler on one block and the classifier on another.

model = Pipeline(
    [
        ("preprocess", preprocess),
        ("classify", LogisticRegression(max_iter=1_000)),
    ]
)

# ── 6. Hyperparameter search on the training block only ───────────────────────
# GridSearchCV clones the unfitted model for every candidate and every fold,
# fits the clone on the fold's training part and scores it on the held-out
# part. That is why the declared model and the fitted model must be separate:
# the search needs the recipe, not one fitted copy.

search = GridSearchCV(
    model,
    param_grid={
        "preprocess__numeric__impute__n_neighbors": [3, 5, 10],
        "classify__C": [0.01, 0.1, 1.0, 10.0],
    },
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
    scoring="accuracy",
    # Refits the best candidate on the whole training block
    refit=True,
)
search.fit(X_train, y_train)

print("best parameters:", search.best_params_)
print(f"cross-validated accuracy: {search.best_score_:.3f}")

# ── 7. Look at what the model sees ────────────────────────────────────────────
# The fitted preprocessing applied to the test block, without the classifier:
# the same matrix predict() builds internally.

fitted = search.best_estimator_
matrix = fitted.named_steps["preprocess"].transform(X_test)
print(matrix.head())
print("scaler means:", fitted.named_steps["preprocess"]
      .named_transformers_["numeric"].named_steps["scale"].mean_)

# ── 8. Score the test block once, after every choice is made ──────────────────

y_pred = fitted.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred, labels=SPECIES))

# ── 9. Production: refit on all the data with the chosen parameters ───────────
# The test score above is the estimate. The final model learns from every row,
# so it has no score of its own.

final = model.set_params(**search.best_params_).fit(X, y)
path = Path(tempfile.gettempdir()) / "penguins.joblib"
joblib.dump(final, path)

# A production record goes through the same preprocessing, with the state
# learned in fit; nothing is recomputed from the new data.
loaded = joblib.load(path)
record = pd.DataFrame(
    [
        {
            "bill_length_mm": 47.5,
            "bill_depth_mm": 15.0,
            "flipper_length_mm": 218.0,
            "body_mass_g": None,
            "island": "Biscoe",
            "sex": "Male",
        }
    ]
)
print(loaded.predict(record))
