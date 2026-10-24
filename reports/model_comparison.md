| run                        | model               |   AUC |   recall |   precision |    F1 |   accuracy |   specificity |   CV AUC |   train AUC | eligible   |
|:---------------------------|:--------------------|------:|---------:|------------:|------:|-----------:|--------------:|---------:|------------:|:-----------|
| gradient_boosting          | gradient_boosting   | 0.938 |    0.902 |       0.902 | 0.902 |      0.891 |         0.878 |    0.93  |       0.991 | True       |
| random_forest_default      | random_forest       | 0.936 |    0.931 |       0.896 | 0.913 |      0.902 |         0.866 |    0.928 |       1     | True       |
| random_forest_regularized  | random_forest       | 0.932 |    0.931 |       0.888 | 0.909 |      0.897 |         0.854 |    0.935 |       0.968 | True       |
| logreg_strong_reg_balanced | logistic_regression | 0.932 |    0.892 |       0.901 | 0.897 |      0.886 |         0.878 |    0.927 |       0.934 | True       |
| logreg_baseline            | logistic_regression | 0.932 |    0.922 |       0.895 | 0.908 |      0.897 |         0.866 |    0.928 |       0.935 | True       |
| xgboost_shallow            | xgboost             | 0.932 |    0.853 |       0.897 | 0.874 |      0.864 |         0.878 |    0.928 |       0.991 | True       |
| xgboost_deep               | xgboost             | 0.923 |    0.824 |       0.884 | 0.853 |      0.842 |         0.866 |    0.926 |       1     | False      |