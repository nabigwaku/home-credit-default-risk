# Data README – Home Credit Default Risk

This folder contains the raw datasets used for the **Credit Risk / Default Prediction** project.  
The data comes from the **Home Credit Default Risk** Kaggle competition and represents lending, credit bureau, and repayment behavior.

The goal of the data is to predict whether a loan applicant will **default on their loan**.

----

# Data Relationship
![data re/s](../notes/data_relationship.png)

----

## 1. Main Application Data

### `application_train.csv`
**Primary dataset used for model training**

- Each row represents **one loan applicant**
- Contains demographic, financial, and loan-related information
- Includes the target variable

Key columns:
- `SK_ID_CURR` – Unique applicant ID
- `TARGET` – Default indicator  
  - `1` = Client defaulted  
  - `0` = Client repaid loan
- `AMT_INCOME_TOTAL` – Client income
- `AMT_CREDIT` – Loan amount
- `AMT_ANNUITY` – Loan annuity
- `DAYS_BIRTH` – Client age (in days, negative)
- `DAYS_EMPLOYED` – Employment length (in days)

This table is the **core of the project** and is used in EDA, feature engineering, and modeling.


### `application_test.csv`
**Loan applicants without known outcomes**

- Same structure as `application_train.csv`
- Does **not** contain the `TARGET` column
- Used for generating predictions in a real production scenario


## 2. Credit Bureau Data

### `bureau.csv`
**Client credit history from external credit bureaus**

- Each row represents a **previous or existing loan**
- One client (`SK_ID_CURR`) can have multiple bureau records

Key columns:
- `SK_ID_CURR` – Applicant ID
- `SK_ID_BUREAU` – Bureau loan ID
- `DAYS_CREDIT` – Days since loan was taken
- `AMT_CREDIT_SUM` – Total credit amount
- `CREDIT_DAY_OVERDUE` – Days overdue
- `AMT_CREDIT_SUM_OVERDUE` – Overdue amount

Used to capture **historical credit behavior** and repayment risk.


### `bureau_balance.csv`
**Monthly status of bureau loans**

- Granular, time-based credit behavior
- One bureau loan can have multiple monthly records

Key columns:
- `SK_ID_BUREAU` – Bureau loan ID
- `MONTHS_BALANCE` – Month relative to application date
- `STATUS` – Loan status (e.g. overdue, closed)

Primarily used for **advanced feature engineering**.


## 3. Previous Loan Applications

### `previous_application.csv`
**Client’s previous loan applications with Home Credit**

- Includes approved, refused, and cancelled loans
- One client can have multiple previous applications

Key columns:
- `SK_ID_CURR` – Applicant ID
- `AMT_APPLICATION` – Requested loan amount
- `AMT_CREDIT` – Approved loan amount
- `NAME_CONTRACT_STATUS` – Loan status (Approved, Refused, etc.)

Used to understand **client borrowing patterns and past decisions**.


## 4. Repayment & Card Behavior (Optional / Advanced)

### `installments_payments.csv`
- Repayment history of previous loans
- Used to assess payment discipline and delays

### `credit_card_balance.csv`
- Monthly credit card usage and balances
- Useful for behavioral risk features

### `POS_CASH_balance.csv`
- Point-of-sale and cash loan monthly snapshots
- Indicates short-term credit behavior

These tables are **optional** and mainly used for deeper feature engineering.


## 5. How the Data Is Used in This Project

- `application_train.csv` - Base dataset
- Bureau and previous loan tables - Aggregated to client level
- Engineered features - Used to train credit risk models
- Output - Probability of Default (PD) per applicant


## 6. Notes

- The dataset is **highly imbalanced** (~8% default rate)
- Missing values are expected and handled during preprocessing
- All modeling is done at the **client level (`SK_ID_CURR`)**


## 7. Data Source

Source:  
**Home Credit Default Risk – Kaggle**  
https://www.kaggle.com/c/home-credit-default-risk/data


Author:  
**Nabigwaku Edward Samuel**  
Credit Risk & Data Analytics
