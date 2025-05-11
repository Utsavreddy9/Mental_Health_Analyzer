 # Mental Health Analyzer - Project Summary
 
🔍 Objective:
To build a machine learning model that can analyze textual or structured survey data to predict whether an individual is likely to have a mental health condition, such as anxiety or depression, based on their responses or social behavior.

📊 Step 1: Exploratory Data Analysis (EDA)
Dataset Used: Mental Health in Tech Survey (e.g., from Kaggle).
EDA Goals:
Understand demographic patterns (e.g., age, gender, country).
Analyze distributions of responses related to mental health history, treatment, and symptoms.
Visualize correlations between mental health issues and workplace factors using heatmaps and bar plots.
Detect and address class imbalance (e.g., more people with no diagnosis than with).

🧹 Step 2: Data Cleaning & Preprocessing
Missing Values: Impute or remove rows with missing demographic or response data.
Encoding:Convert categorical variables (e.g., Gender, Country, Employer Support) using One-Hot or Label Encoding.
Normalization/Scaling:Apply Min-Max Scaling or Standardization to numerical fields (e.g., Age).
Balancing the Classes:Use SMOTE or RandomUnderSampler to address class imbalance.
Train-Test Split:Split the data into training and test sets (e.g., 80/20).

🧠 Step 3: Model Building & Classification
✅ Logistic Regression
Why: Interpretable baseline model for binary classification.
Output: Probabilities of mental health issue presence.
Performance: Useful for understanding feature importance via coefficients.

🌳 Random Forest Classifier
Why: Handles non-linear relationships well and provides feature importance.
Tuning: n_estimators: Number of trees.
max_depth: Controls overfitting.
Strengths: Robust against overfitting due to averaging.

🔁 Bagging Classifier
Why: Reduces variance by training multiple models on random subsets.
Base Model: Decision Tree or Logistic Regression.
Result: Stable and accurate predictions by aggregating weak learners.

🚀 Boosting (e.g., AdaBoost, Gradient Boosting)
Why: Sequentially corrects the errors of previous models.
Use Case: Works well when there's a complex pattern in the data.
Tuning:learning_rate, n_estimators, subsample.

🔧 Step 4: Hyperparameter Tuning
Methods:
GridSearchCV
RandomizedSearchCV
Parameters Tuned:
Regularization in Logistic Regression (C)
Depth and estimators in tree-based models
Learning rate in Boosting

📏 Step 5: Model Evaluation
Metrics Used:Accuracy
Precision, Recall, F1-score
AUC-ROC
Cross-validation:
k-fold (e.g., k=5) to reduce variance in performance estimates.
Confusion Matrix:
To visualize true vs. predicted mental health labels.

🧠 Outcome:
Final model chosen based on best generalization (typically Boosting or Random Forest with tuned parameters).



## Features

* Preprocessing and cleaning of text data
* Exploratory Data Analysis (EDA)
* Feature extraction using TF-IDF and word embeddings
* Classification using Logistic Regression, Random Forest, Bagging, and Boosting
* Hyperparameter tuning with GridSearchCV
* Evaluation using accuracy, precision, recall, F1-score, and confusion matrix

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/mental-health-analyzer.git
   cd mental-health-analyzer
   ```

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Dataset

You can use publicly available datasets such as:

* https://www.kaggle.com/code/kairosart/machine-learning-for-mental-health-1/input

Ensure your dataset has at least two columns: `text` (chat/message) and `label` (mental health status).

## Running the Project

1. **Run the main analysis notebook:**

   ```bash
   jupyter notebook Mental_Health_Analyzer.ipynb
   ```

2. **Follow the notebook to:**

   * Load and clean the dataset
   * Perform EDA and visualize insights
   * Train classification models
   * Evaluate results and choose the best model


## Usage Instructions

* You can replace `survey.csv` with your own dataset.
* The `Mental_Health_Analyzer.ipynb` notebook walks through all steps from data preprocessing to model evaluation.

## Contributing

Interested students or researchers are encouraged to fork the repo and contribute by:

* Adding new classification models
* Integrating deep learning (e.g., LSTM, BERT)
* Improving data preprocessing pipelines

## Contact

For any questions or collaboration ideas, feel free to reach out at:
**[ukunt1@unh.newhaven.edu]**

---

*This project is developed as part of an academic coursework and is intended for research and educational purposes only.*

