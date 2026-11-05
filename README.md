# MUC02 — Credit Pre-Screening Engine: Loan Intake Optimisation

**Calibo AI Academy | Path 2 | KLU Batch | Phase 2 — Mini Use Case 02 (Supervised Learning Part 1)**  
**Industry:** BFSI — Retail Lending (Unsecured Personal Loans)

---

## 👥 Project Team

| Team Member | Role | Assigned Responsibilities |
|---|---|---|
| **Sai Surchit Das** | Data Analyst (Team Lead) | **Steps 5.2 & 5.3:** Approval driver analysis (<650 score floor), Feature Engineering (`FOIR`, `LTI`, `is_ntc`), and the 50% FOIR approval cliff analysis. |
| **R. Mahima Varshini** | Data Analyst | **CBIM Canvas & Steps 5.1 (A–D):** Data loading, data inspection, cleaning log, group-median income imputation, NTC retention, and salaried CTC/12 normalization. |
| **S. Vikas** | Model Builder & Policy Analyst | **Steps 5.4 – 5.7:** Logistic Regression pipeline, demonstrating 0.5 default cutoff risk violation, model coefficient interpretation, threshold sweep (0.85/0.90), fairness audit, and Credit Policy Briefing. |

---

## 🎯 Business Context & Problem Statement

A mid-sized private bank in India receives approximately **3,000 personal-loan applications per month**. Every application currently goes through a full manual credit review taking approximately **25 minutes of an analyst's time** (~1,250 hours/month total). Historically, nearly half (~47.06%) of all applications end up declined.

Manual analyst queues are currently congested by clear-cut decline profiles (e.g., severe indebtedness or poor credit history) that are evident within the first minute of evaluation. This delays decisions for creditworthy applicants and caps intake capacity.

### The Core Objective
Design and evaluate a supervised **Logistic Regression pre-screening classifier** to automatically filter and decline high-certainty rejections while strictly complying with the bank's operational risk boundary:
> **Hard Risk Constraint:** No more than **5.0%** of auto-declined applicants may be customers whom the credit underwriting team would have approved historically (**Wrongly Declined Rate ≤ 5.0%**).

---

## 📊 Dataset Summary

The analysis examines **12,000 historical personal-loan decisions** across a six-month observation window (April 2026 – September 2026) with 19 feature columns:
- **Historical Decisions:** 6,353 Approved (52.94%) vs. 5,647 Declined (47.06%).
- **Applicant Demographics & Employment:** Age (18–60), Gender (44% F, 56% M), City Tier (1, 2, 3), Employment Type (Salaried, Self-Employed, Business Owner, Gig/Contract), Monthly Income, and Job Tenure.
- **Credit & Debt Profile:** Existing EMI, CIBIL/Credit Score (300–900), Credit History Months, Active Loans, and Delinquencies in the past 24 months.
- **Loan Request Details:** Loan Purpose, Loan Amount (₹50k – ₹10L), Tenure (12–60 months), and Proposed EMI.

---

## 🛠️ Data Hygiene & Preprocessing Highlights

1. **441 Missing Monthly Incomes (3.68%):** Concentrated heavily in non-salaried roles (13.0% Business Owners, 10.8% Self-Employed). Dropping these would bias the intake against entrepreneurs. Missing values were imputed using employment-group medians learned exclusively from training data.
2. **1,039 Blank Credit Scores (8.66%):** 100% of these records had `credit_history_months = 0` (genuine New-to-Credit / NTC applicants, median age 29). To prevent exclusion, an explicit indicator flag `is_ntc = 1` was created, and scores were imputed with training median (726). All 12,000 rows were retained.
3. **12 Annual CTC Reporting Errors:** Twelve salaried applicants had reported monthly incomes > ₹6,00,000 (ranging from ₹6.18L to ₹16.56L). These were verified as annual CTCs entered into a monthly field and normalized by dividing by 12.

---

## ⚙️ Key Technical Findings & Analytical Insights

### 1. Affordability Cliff (FOIR Analysis)
Fixed Obligations to Income Ratio ($\text{FOIR} = \frac{\text{Existing EMI} + \text{Proposed EMI}}{\text{Monthly Income}}$) is the primary driver of credit rejection:
- **FOIR < 30%:** 65.5% approval rate
- **FOIR 30% – 40%:** 59.0% approval rate
- **FOIR 40% – 50%:** 47.9% approval rate
- **FOIR 50% – 55%:** 32.4% approval rate (*sharp drop*)
- **FOIR > 55%:** 15.6% approval rate (*approval collapse*)  
**Policy Recommendation:** Establish a standard underwriting cap at 50% FOIR; 50%–55% requires senior sign-off; >55% routed to decline.

### 2. Flaw of Default 0.50 Classification Threshold
Evaluating Logistic Regression at the naive 0.50 threshold yields an acceptable overall accuracy (79.08%, ROC-AUC 0.867), but produces a catastrophic credit failure:
- **225 historically approved applicants** were classified as auto-declined out of 1,077 auto-declines on test data (**20.89% Wrongly Declined Rate**).
- This severely breaches the bank's **≤ 5.0% risk limit**, rejecting 1 in 5 good customers.

### 3. Safe Threshold Sweep & Optimization
To satisfy the ≤ 5.0% risk constraint, the auto-decline decision threshold was swept from 0.50 to 0.95:
- **0.50 Cutoff:** 44.88% auto-declined | **20.89% wrong declines** (Violates policy)
- **0.80 Cutoff:** 23.46% auto-declined | **7.10% wrong declines** (Violates policy)
- **0.85 Cutoff:** 19.67% auto-declined | **5.93% wrong declines** (Borderline / depends on rounding)
- **0.90 Cutoff (Optimal & Selected):** **15.21% auto-declined** | **4.66% wrong declines** (Complies with ≤ 5.0% rule)
- **0.95 Cutoff:** 10.00% auto-declined | **2.08% wrong declines** (Overly conservative)

**Business Value at 0.90 Threshold:**
- **456 applications/month** safely auto-declined (~15.2% of intake volume).
- **~190 analyst hours freed monthly** (allowing analysts to focus on complex cases).
- Only ~21 good applications/month impacted (~4.66% false negative rate).

### 4. Fairness, Demographic & NTC Safeguards
- **Gender Consistency:** Approval rates (54.1% male vs 50.4% female) reflect underlying median income (₹55k vs ₹45k) and gig employment proportions (12% vs 21%). At the 0.90 threshold, auto-decline rates are equitable (15.0% male vs 15.6% female).
- **Mandatory NTC Safeguard:** Model accuracy for New-to-Credit applicants drops to **56.7%** (compared to 81.2% for scored applicants). **Operational Rule:** All NTC applicants (`is_ntc = 1`) are strictly bypassed from auto-decline and routed for human underwriting.

---

## 📂 Repository Contents

```text
├── credit_PreScreening_backup.ipynb             # Full end-to-end Python analytical notebook (Steps 5.1 - 5.7)
├── build_deck.py                                # Executive presentation generator script (python-pptx)
├── Mahima_Varshini_Individual_Reflection.html   # Print-ready Individual Reflection (R. Mahima Varshini)
├── Sai_Surchit_Das_Individual_Reflection.html   # Print-ready Individual Reflection (Sai Surchit Das)
├── S_Vikas_Individual_Reflection.html           # Print-ready Individual Reflection (S. Vikas)
├── MUC02_Credit_PreScreening_Dataset 1.csv      # Source intake dataset (12,000 records)
└── README.md                                    # Project documentation & executive briefing
```

---

## 🚀 How to Run & Reproduce

### 1. View / Print Individual Reflections to PDF
Open each `.html` file in Chrome or Edge and press `Ctrl + P` -> Select **"Save as PDF"** (Layout: Portrait, enable "Background graphics").

### 2. Run the Analysis Notebook
Open `credit_PreScreening_backup.ipynb` in VS Code or JupyterLab and execute all cells sequentially:
```bash
jupyter notebook credit_PreScreening_backup.ipynb
```

### 3. Generate the 12-Slide Executive Deck
```bash
python build_deck.py
```
This outputs `Credit_PreScreening_Deck.pptx` styled in the dark teal executive theme (`#0C2932`).

