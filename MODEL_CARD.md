# Model Card

**Model:** logreg (StandardScaler + Logistic Regression) · **Version:** 1.0.0 · **Framework:** scikit-learn 1.9.1

## Intended use
Educational capstone demonstrating ML model serving. Classifies breast-mass measurements as malignant or benign.
**Not a medical device and not for clinical decision-making.**

## Data
Breast Cancer Wisconsin (Diagnostic), 569 samples, 30 numeric features, 80/20 stratified split, seed 42.

## Model selection
5-fold CV ROC-AUC on the training split: logistic regression 0.9935, gradient boosting 0.9876, random forest 0.9866. Highest CV score was selected.

## Test performance (held-out 20%)
ROC-AUC 0.9954 · Accuracy 0.9825 · F1 0.9861

## Limitations
- Small dataset (114 test samples), so metrics have wide confidence intervals.
- No calibration check or subgroup/fairness analysis was done.
- Inputs far outside the training range are not rejected, only non-finite values are.
- Features must be in the exact order listed by `GET /model-info`.
