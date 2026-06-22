# Task 2: Credit Risk Prediction & Loan Approval Classification

## 🎯 Task Objective
The primary objective of this project is to automate the credit risk assessment process for financial institutions. By evaluating applicant profiles (including continuous financial vectors like income and loan amount alongside categorical attributes like education and credit history), this project constructs a machine learning classifier to predict whether a loan application is likely to be **Approved** or **Default/Denied**.

---

## 🛠️ Your Approach & Methodology

The project follows a structured, production-grade end-to-end Data Science workflow broken down into five distinct phases:

1. **Dataset Understanding & Inspection:** * Loaded the source data (`loan_prediction.csv`) and reviewed its global attributes using structural metrics (`df.shape`, `df.info()`). 
   * Mapped the presence of missing elements across the feature space via null-matrix counts (`df.isnull().sum()`).

2. **Data Cleaning & Handling Missing Values:**
   * **Numerical Strategy:** Imputed missing values in continuous columns (e.g., `LoanAmount`) using the **median** value of the column. This keeps data imputation stable against right-skewed distributions and massive outlier boundaries.
   * **Categorical Strategy:** Imputed missing values in text features (e.g., `Gender`, `Married`, `Dependents`, `Self_Employed`, `Loan_Amount_Term`, and `Credit_History`) using the **mode** (highest frequency element). This method ensures zero row loss without introducing synthetic tracking errors.

3. **Exploratory Data Analysis (EDA):**
   * Generated frequency distributions and kernel density estimates (KDE) using Seaborn and Matplotlib to inspect structural skews in `LoanAmount` and `ApplicantIncome`.
   * Analyzed counts of categorical traits across educational layers to understand overall applicant profile demographics.

4. **Data Encoding & Structural Splitting:**
   * Transformed non-numeric text strings into standardized model-ready arrays using `LabelEncoder`.
   * Isolated features ($X$) from the class label ($y$, `Loan_Status`) and executed a robust **80/20 train-test partition** via `train_test_split` to shield evaluation from data leakage.

5. **Model Training & Diagnostics:**
   * Deployed a **Logistic Regression** classifier (configured to `max_iter=1000` to guarantee gradient convergence).
   * Validated performance against unseen data using global classification `accuracy_score` metrics and a visualized heatmap of the `confusion_matrix`.

---

## 📊 Results and Insights

* **Distribution Properties:** Initial plots of `ApplicantIncome` and `LoanAmount` revealed heavily right-skewed profiles with long tails. This confirmed that using a median imputation approach was correct, as a standard mean calculation would have been distorted by extreme high-income/high-loan outliers.
* **Model Accuracy:** The Logistic Regression algorithm maps a reliable linear decision boundary across the encoded financial attributes, yielding a robust, high-performing global accuracy metric on test-split validation points.
* **Error Analysis via Confusion Matrix:** The resulting heatmap provides clear insights into the model's true behavior:
  * **True Positives & True Negatives:** Confirms correct designations where the model precisely flags a credit-worthy applicant or identifies high-risk candidate profiles.
  * **Operational Security:** Looking closely at the False Positives (applicants labeled safe who actually default) allows risk adjustment teams to tweak threshold barriers, protecting the bank's operational capital from risk.

---

## 📁 Repository File Structure
* `loan_prediction.csv` - Raw input text dataset containing historical profile matrices.
* `Credit_Risk_Analysis.ipynb` - Clean, comment-supported Jupyter Notebook containing all processing transformations, code blocks, and output charts.
* `README.md` - Executive summary brief.
