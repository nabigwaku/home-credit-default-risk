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