 # Mental Health Analyzer - Project Summary
 
🔍 Objective:
To build a machine learning model that can analyze textual or structured survey data to predict whether an individual is likely to have a mental health condition, such as anxiety or depression, based on their responses or social behavior.


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

