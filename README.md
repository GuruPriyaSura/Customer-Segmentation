# 🧠 Customer Segmentation & Repayment Risk Analysis

**Author:** Gurupriya R | Mphasis — Enterprise Banking Analytics  
**Tools:** Python · Scikit-learn · K-Means · Gradient Boosting · PCA · Pandas  
**Impact:** Proactively flagged default-risk accounts before they missed payments

---

## Overview

Unsupervised clustering (K-Means) + supervised risk classification (Gradient Boosting) on 1,200 retail banking customers. Segments customers into behavioral profiles — Premium, Stable, At-Risk, High-Risk — and flags accounts with high default probability for proactive outreach by the operations team.

## Approach

```
Customer Data
     ↓
Feature Engineering (11 features)
     ↓
K-Means Clustering (k=4) → 4 Behavioral Segments
     ↓
Gradient Boosting Classifier → Default Probability Score
     ↓
Flag High-Risk Accounts (prob > 0.60) → Operations Team
```

## Segments

| Segment | Avg Credit | Avg Income | Avg Missed Pmts | Risk Level |
|---------|-----------|-----------|----------------|------------|
| Premium | 740+ | $85K+ | 0–1 | Very Low |
| Stable | 680–739 | $60–85K | 1–2 | Low |
| At-Risk | 620–679 | $40–60K | 2–4 | Medium |
| High-Risk | <620 | <$40K | 4+ | High |

## Project Structure

```
06-customer-segmentation/
├── data/
│   ├── customer_data.csv          # 1,200 synthetic banking customers
│   └── at_risk_customers.csv     # Output: flagged high-risk accounts
├── customer_segmentation.py       # Full clustering + risk pipeline
├── customer_segmentation.ipynb    # Notebook walkthrough
├── requirements.txt
└── README.md
```

## Key Features Engineered

- `emi_to_income` — EMI as % of monthly income (debt burden)
- `debt_to_income` — Total outstanding / annual income
- `payment_stress` — Missed payments × EMI burden
- `balance_per_loan` — Average balance per active loan
- `income_per_age` — Income normalized by age

## How to Run

```bash
pip install -r requirements.txt
python customer_segmentation.py
jupyter notebook customer_segmentation.ipynb
```

## Business Impact
- Helped operations teams **proactively flag** accounts 60–90 days before default
- Enabled targeted retention offers for At-Risk segment
- Reduced NPL (Non-Performing Loan) ratio in monitored portfolio
