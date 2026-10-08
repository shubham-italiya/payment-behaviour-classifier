## Model comparison (5-fold cross-validation on the training set)

| Model | PR-AUC | ROC-AUC |
|---|---|---|
| gradient_boosting_class_weights | 0.556 ± 0.010 | 0.782 |
| gradient_boosting_smote | 0.528 ± 0.011 | 0.766 |
| logistic_regression | 0.508 ± 0.016 | 0.756 |
| logistic_regression_smote | 0.507 ± 0.017 | 0.757 |

## Held-out test set (6,000 customers, 22.1% paid late)

| Model | F1 | Precision | Recall | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| baseline (logistic regression, threshold 0.50) | 0.405 | 0.646 | 0.295 | 0.499 | 0.742 |
| best (gradient_boosting_class_weights, threshold 0.56) | 0.543 | 0.516 | 0.573 | 0.553 | 0.777 |
