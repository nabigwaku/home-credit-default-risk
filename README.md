# Home Credit Default Risk – Probability of Default Modeling

## Project Overview

This project develops an end-to-end Probability of Default (PD) model using lending and credit bureau data from the Home Credit dataset.

The objective is to support data-driven loan approval decisions, reduce exposure to high-risk borrowers, and improve overall portfolio quality while maintaining responsible lending practices.

This project simulates how a credit risk analytics team would design, evaluate, and operationalize a default prediction model.


## Business Problem

Lenders must balance growth and risk.

- Approving high-risk applicants increases credit losses.
- Rejecting too many applicants reduces revenue and financial inclusion.

Problem Type: Binary Classification  
- 1 → Client defaulted  
- 0 → Client repaid successfully  

Goal:  
Estimate the Probability of Default (PD) for each applicant and translate predictions into structured credit decisions.


## Dataset

Source: Home Credit Default Risk (Kaggle)  
https://www.kaggle.com/c/home-credit-default-risk/data

The dataset includes:

- Applicant demographic and financial information  
- Historical credit bureau records  
- Previous Home Credit loan applications  
- Repayment and behavioral credit data  

Characteristics:
- Multi-table relational structure  
- High class imbalance (~8% default rate)  
- Significant missing values (lending complexity)  

Modeling is performed at the applicant level (SK_ID_CURR).


## Project Structure
```text
home-credit-default-risk/
│
├── data/
│   ├── raw/                 # Original Kaggle CSV files
│   └── processed/           # Cleaned and aggregated datasets
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_model_explainability.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   └── evaluate.py
│
├── reports/
│   ├── business_summary.md
│   └── model_results.md
│
├── requirements.txt
└── README.md
```


## Project Workflow

![Project Flow Diagram](<Project Flow.png>)

### Phase 1: Exploratory Data Analysis
- Default rate and class imbalance analysis  
- Missing value profiling  
- Credit amount, income, age, and risk ratio analysis  
- Initial portfolio risk insights  

### Phase 2: Feature Engineering
- Client-level aggregations from bureau data  
- Previous loan behavior summaries  
- Credit-to-income and annuity ratios  
- Handling missing values and outliers  

### Phase 3: Modeling
- Logistic Regression (interpretable baseline)  
- LightGBM (primary model)  
- Stratified cross-validation  
- ROC-AUC driven evaluation  

### Phase 4: Explainability
- SHAP values for global risk drivers  
- Local explanations for individual loan decisions  
- Model transparency for operational use  


## Models Used

- Logistic Regression (Baseline)  
- LightGBM (Final Model)  

Model selection prioritizes:
- Predictive performance  
- Stability  
- Explainability  


## Evaluation Metrics

- Primary: ROC-AUC  
- Secondary: Precision, Recall  
- Confusion Matrix at business decision thresholds  


## Business Decision Framework

Model outputs are interpreted as Probability of Default (PD) and mapped to operational actions:

| PD Range | Decision |
|----------|----------|
| < 5%     | Auto-Approve |
| 5–15%    | Manual Review |
| > 15%    | Reject |


## Business Impact

This modeling framework enables:

- Structured risk segmentation of applicants  
- Reduced exposure to high-risk borrowers  
- Transparent and explainable credit decisions  
- Improved portfolio monitoring using PD thresholds  


## Possible Future Enhancements

- Expected Loss modeling  
- Time-based validation  
- Scorecard-style credit scoring  
- API-based deployment  
- Monitoring dashboard for portfolio tracking  


## Author

Nabigwaku Edward Samuel  
Senior Data Analyst | Credit Risk & BI Analytics