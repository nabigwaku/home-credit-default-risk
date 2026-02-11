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
----

## Project Workflow

![Project Flow Diagram](../notes/project_workflow.png)

----

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