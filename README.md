# Spam-Email-Detection-Analysis

# Customer Classification Analysis & Model Comparison

This repository contains the data analysis and machine learning models developed to address a classification problem, likely related to predicting customer behavior or status (e.g., churn prediction, lead conversion). The project focuses on building and evaluating two common classification algorithms: Logistic Regression and Random Forest, with a particular emphasis on handling class imbalance. This was a collaborative project with 3 other researchers.

## Project Overview

The primary goal of this project was to:
1.  Perform exploratory data analysis (EDA) to understand the dataset.
2.  Build robust classification models to predict a target variable.
3.  Compare the performance of Logistic Regression and Random Forest.
4.  Address the challenge of class imbalance in the dataset to ensure fair and accurate predictions across all classes.

## Key Features & Analysis Steps

* **Data Loading & Preprocessing:** Initial handling of the dataset, preparing it for modeling.
* **Class Imbalance Handling:** Implementation of `class_weight='balanced'` in both Logistic Regression and Random Forest models to mitigate bias towards the majority class.
* **Logistic Regression:**
    * Trained and evaluated both without and with class weighting.
    * Performance metrics (precision, recall, f1-score, accuracy) and confusion matrices were meticulously analyzed to show the impact of balancing.
    * ROC curve and AUC score were used for overall model evaluation.
* **Random Forest Classifier:**
    * Trained with `class_weight='balanced'` for robust performance.
    * Evaluated using a comprehensive classification report, confusion matrix, and ROC curve with AUC.
* **Model Comparison:** A direct comparison of the two models' effectiveness, particularly in handling the minority class.

## Results & Findings

* **Logistic Regression:** Showed significant improvement in identifying the minority class (recall increased from ~0.58 to ~0.80) when `class_weight='balanced'` was applied, while maintaining a strong overall AUC of **~0.931**.
* **Random Forest:** Achieved exceptional performance, with an **AUC of 1.00**, indicating a near-perfect ability to distinguish between classes. The confusion matrix also confirmed extremely high precision and recall for both classes (e.g., 715 true negatives, 287 true positives, with minimal false positives/negatives). This model demonstrated superior predictive power on this dataset.

## Repository Structure

* `Spam_Email_Classification_EDA.ipynb`: The main Jupyter Notebook containing all the code for data loading, preprocessing, model training, evaluation, and visualization.
* `README.md`: This file.
* `requirements.txt`: (To be added) Lists all necessary Python libraries for running the notebook.

## How to Run the Analysis

To run this analysis locally, please follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    *(You will need to create a `requirements.txt` file based on the libraries used in your notebook, e.g., pandas, scikit-learn, matplotlib, seaborn.)*
    ```bash
    pip install -r requirements.txt
    ```
4.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
5.  Open `project3_draft_final.ipynb` and run the cells.

## Technologies Used

* Python 3.x
* Jupyter Notebook
* Pandas (for data manipulation)
* NumPy (for numerical operations)
* Scikit-learn (for machine learning models and metrics)
* Matplotlib (for plotting)
* Seaborn (for enhanced visualizations)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details (if you plan to add one).

## Contact

For any questions or suggestions, feel free to open an issue or contact bryanmejiaeducation@gmail.com.
