# Machine Learning Foundations Assignment

**Course:** Introduction to Artificial Intelligence  
**Type:** Major assignment  
**Estimated effort:** 18–22 hours

---

## 1. Learning goals

You will practise how to:

1. Derive evaluation metrics for classification and regression from first principles.
2. Train non-linear regression models with handcrafted feature maps and gradient descent / closed-form solutions.
3. Implement logistic and softmax regression for binary and multiclass classification without relying on scikit-learn.
4. Evaluate decision tree criteria (entropy, Gini, information gain, gain ratio) and select the best split.
5. Structure experiments, sanity-check results with public datasets, and reason about generalisation to hidden data.

---

## 2. Package contents

```
assignment/
├── README.md                # This document
├── requirements.txt         # Python dependencies (numpy, pandas, pytest)
├── data/                    # Datasets – do NOT edit
│   ├── metrics_classification.csv
│   ├── metrics_regression.csv
│   ├── regression_1d.csv
│   ├── regression_2d.csv
│   ├── logistic_binary.csv
│   ├── softmax_multiclass.csv
│   └── decision_tree.csv
├── code/                    # Your implementation area
│   ├── metrics_classification.py
│   ├── metrics_regression.py
│   ├── polynomial_transformer.py
│   ├── linear_regression.py
│   ├── regression.py
│   ├── logistic_softmax.py
│   └── decision_tree.py
├── local_test.py            # Convenience wrapper around public tests
└── public_tests/            # Basic validation (read-only)
    ├── test_metrics.py
    ├── test_regression.py
    ├── test_logistic_softmax.py
    └── test_decision_tree.py
```

---

## 3. Environment setup

Requires **Python 3.10 or newer**.

```bash
# from the assignment/ directory
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

All starter code assumes you run commands from the `assignment/` directory.

---

## 4. Working process

1. Open the files in `code/` and read every docstring and TODO. Each function already exposes the exact signature we will import during grading.
2. Implement the TODO blocks. Do **not** change function names, parameters, or return types.
3. After finishing a part, execute:

   ```bash
   # quick wrapper
   python local_test.py --part <n>

   # or run pytest directly
   pytest public_tests/test_<part>.py -v
   ```

   Public tests cover basic behaviour only. Hidden grading tests use stricter tolerances, additional datasets, and property-based checks.

4. Before submission, ensure there are no remaining `raise NotImplementedError` statements and that public tests pass.

---

## 5. Part-by-part requirements

### Part 1 – Evaluation Metrics (`code/metrics_classification.py`, `code/metrics_regression.py`, 20 pts)

Datasets: `metrics_classification.csv` and `metrics_regression.csv`.

- `accuracy_score(y_true, y_pred)`
- `precision_score(y_true, y_pred, positive_label)`
- `recall_score(y_true, y_pred, positive_label)`
- `f1_score(y_true, y_pred, positive_label)`
- `macro_f1_score(y_true, y_pred, labels)`
- `micro_f1_score(y_true, y_pred, labels)`
- `mean_absolute_error(y_true, y_pred)`
- `mean_squared_error(y_true, y_pred)`
- `root_mean_squared_error(y_true, y_pred)`
- `r2_score(y_true, y_pred)`
- `regression_report(y_true, y_pred)` – returns a dictionary aggregating the regression metrics above.

Implementations must work on generic Python sequences (lists, tuples, pandas Series). Do not rely on scikit-learn helpers.

### Part 2 – Non-linear Regression (`code/regression.py`, plus helpers, 15 pts)

Datasets: `regression_1d.csv` (single feature) and `regression_2d.csv` (surface).

- `polynomial_features(x, degree)` – returns a Vandermonde-style matrix including bias.
- `fit_polynomial_regression(x, y, degree)` – build features then solve weights via the closed-form linear regression helper.
- `predict_polynomial(x, weights)` – reuse `polynomial_features` to produce predictions.
- `fit_surface_regression(x1, x2, y)` – build the quadratic surface basis `[1, x1, x2, x1^2, x1*x2, x2^2]` and solve weights via the helper.
- `predict_surface(x1, x2, weights)` – produce predictions for arbitrary grids.

The helper modules `polynomial_transformer.py` and `linear_regression.py` provide class/function skeletons and docstrings for you to complete. Implement their TODO blocks and use these helpers instead of re-deriving the math elsewhere.

### Part 3 – Logistic & Softmax Regression (`code/logistic_softmax.py`, 35 pts)

Datasets: `logistic_binary.csv` and `softmax_multiclass.csv`.

- `class LogisticRegression`: implement `fit`, `predict_proba`, and `predict`.
- `class SoftmaxRegression`: implement `fit`, `predict_proba`, and `predict`.

Both models must:

1. Accept dense numpy arrays or Python lists for features.
2. Use gradient descent with cross-entropy loss.
3. Include L2 regularisation controlled via the provided `reg_strength` argument.
4. Return probabilities that sum to one per sample (after clipping stability epsilon).

Helper functions for sigmoid/softmax will be provided, but their usage is optional.

---

### Part 4 – Decision Tree Metrics (`code/decision_tree.py`, 15 pts)

Dataset: `data/decision_tree.csv`

- Implement `entropy(labels)`, `gini(labels)`, and `partition_dataset(df, feature)`.
- Implement `information_gain(df, feature, target="play")` and `gain_ratio(df, feature, target="play")`.
- Implement `best_split(df, candidate_features, target="play", criterion="gain_ratio")` that supports `"gain_ratio"`, `"information_gain"`, and `"gini"` (reduction).
- Public expected values live in `public_tests/expected/expected_decision_tree_*.csv`. Hidden tests reuse the same API on `decision_tree_hidden.csv`.

---

## 6. Testing and grading

- `python local_test.py` – runs all public tests and prints a summary.
- `pytest public_tests -v` – full verbose output.
- Public suite: 24+ sanity tests (coverage evenly split across the four parts). Hidden suite: additional datasets, randomised initialisations, and stricter tolerances.
- Partial credit is awarded per test case; implement defensively.

---

## 7. Submission checklist

1. `metrics_classification.py`, `metrics_regression.py`, `regression.py`, `logistic_softmax.py`, and `decision_tree.py` contain no `NotImplementedError`.
2. `pytest public_tests -v` passes.
3. Code is readable (docstrings, comments where needed, no dead code or print debugging).
4. Zip the following into `{StudentID}_mlfoundations.zip`:

   ```
   {StudentID}_mlfoundations/
   ├── metrics_classification.py
   ├── metrics_regression.py
   ├── polynomial_transformer.py
   ├── linear_regression.py
   ├── regression.py
   ├── logistic_softmax.py
   └── decision_tree.py
   ```

5. Upload the ZIP to the course LMS before the deadline (late policy: -10% per day, up to 2 days).

Do **not** include data files, public tests, or virtual environments.

---

## 8. Academic integrity

This is an individual assignment. Discussing concepts is permitted; sharing code or using external implementations is not. We automatically compare submissions against peers, past cohorts, and public repositories. Do not upload this assignment to public platforms (GitHub, forums, AI tools).

---

## 9. Getting help

Discuss questions directly with the course lecturer during class sessions.

---

*Last updated: November 2025*