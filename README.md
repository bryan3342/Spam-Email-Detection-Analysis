# Spam Email Detection Analysis

## Description

A collaborative notebook comparing machine-learning classifiers on the Kaggle Email Spam Classification Dataset. The repository also includes three PDF versions of the project analysis.

## Dependencies

Python 3, Jupyter Notebook or Google Colab, KaggleHub access, pandas, NumPy, Matplotlib, Seaborn, and scikit-learn. No pinned dependency file is included.

## Technologies

Python notebooks, KaggleHub, pandas, exploratory visualization, scikit-learn classification, PCA, and model evaluation.

## Setup

```bash
git clone https://github.com/bryan3342/Spam-Email-Detection-Analysis.git
cd Spam-Email-Detection-Analysis
python -m venv .venv
source .venv/bin/activate
python -m pip install notebook kagglehub pandas numpy matplotlib seaborn scikit-learn
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.

KaggleHub downloads `balaka18/email-spam-classification-dataset-csv`. Internet access and any authentication required by KaggleHub are needed.

## Features

- Spam and non-spam class distribution analysis.
- Word-count and word-frequency exploration.
- Stratified train/test splitting and class weighting.
- Logistic regression, random forest, Gaussian naive Bayes, SVM, MLP, and stacking classifiers.
- Confusion matrices, classification reports, ROC curves, and comparative metrics.

The notebook's saved outputs report strong test-set results, including a perfect random-forest ROC-AUC in one run. Those are results on this dataset and split, not a guarantee of performance on new email.

## Commands to run

```bash
jupyter notebook project3_final.ipynb
```

Run the cells in order. There is no packaged prediction service or automated test suite.

## Contributors

- [Bryan Mejia (@bryan3342)](https://github.com/bryan3342)
- Ahmed Ali

Credit follows human authors in Git history. The notebook describes broader collaboration, but the inspected Git history provides these two attributable names.
