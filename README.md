# Telecom Customer Retention: Revenue-at-Risk Scoring


---

## Business Problem

Most churn models stop at just predicting who will leave. That's an interesting number, but it doesn't tell the retention team what to actually do with the limited budget they are given. Is a customer who is 90% likely to churn but generates almost no revenue worth the same intervention as a customer who is 55% likely to churn but represents thousands of dollars in lifetime value? Thats the question my anayssins seeks to solve. 

** Which customers should we spend retention budget on to protect the most lifetime revenue per dollar spent?**

Rather than predicting churn alone, this analysis combines a calibrated churn probability with each customer's Customer Lifetime Value (CLV) to produce a **Revenue-at-Risk score**, then uses that score to tier customers into a value × risk matrix. The recommendation: don't spend retention budget equally across the customer base — concentrate it where the return is highest.

---

## Data & Tools

- **Dataset:** Telecom customer data (`Client.csv` and `Record.csv`), merged 1:1 on `Customer_ID` — ~100,000 customers, 100 features covering usage, revenue, equipment, demographics, and account behavior
- **Environment:** Google Colab (Python)
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn, XGBoost, LightGBM, CatBoost
- **Techniques used:** exploratory data analysis, feature engineering, label encoding, stratified train/test split, model bake-off (Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost), probability calibration, CLV modeling, customer segmentation, ROI analysis, and unsupervised clustering for root-cause explanation

---

## Approach 

**1. Data loading & merge**
Loaded `Client.csv` and `Record.csv` and merged on `Customer_ID` (a 1:1 join, each table has one row per customer), producing a combined dataset of 100,000 rows and 100 columns.

**2. Exploratory analysis**
Checked the churn class balance (50/50 — unusually balanced for telecom, where real-world churn is typically 5–10%), profiled missing data across columns, and examined individual features against churn. Equipment age (`eqpdays`) emerged as a clear candidate: churners skew toward older handsets in both the mean and the right tail of the distribution. A pairwise comparison of revenue, usage, tenure, and equipment age showed no single variable cleanly separates churners from stayers — confirming this is a multi-variable problem, not a one-feature threshold.

**3. Feature engineering**
Built 13 derived features designed to capture behavioral signals a raw column can't, including:
- Usage and revenue decay ratios (3-month vs. 6-month averages)
- Equipment age relative to customer tenure
- Failed-call and drop/block rates as shares of call attempts
- Customer care intensity 
- Overage revenue share and revenue-per-minute
- Inactive-subscriber ratio and revenue per month of tenure

**4. Preprocessing**
Dropped `Customer_ID` (a unique identifier with no predictive value), dropped columns with more than 40% missing data, and label-encoded categorical columns. Split the data with a stratified 70/30 train/test split, then imputed missing values using **training-set medians only** (fit after the split) to avoid data leakage from the test set into training.

**5. Model building**
Ran a bake-off across five classifiers: Logistic Regression, Random Forest, XGBoost, LightGBM, and CatBoost — evaluated on AUC and accuracy. **XGBoost performed best** (AUC = 0.6987), and was carried forward for calibration and scoring.

**6. Calibration**
Applied isotonic calibration (`CalibratedClassifierCV`) to the winning model so its output probabilities are genuine probabilities — a requirement for the CLV math in the next step, since a Revenue-at-Risk score built on miscalibrated probabilities would misstate dollar figures.

**7. Evaluation**
Reviewed the confusion matrix to understand the model's error trade-off (missed churners vs. false alarms) and used XGBoost's gain-based feature importance to identify which signals the model relies on most — with an eye toward which of those signals a business can actually act on (e.g., equipment age is actionable; tenure is not).

**8. From model to proposal: CLV and Revenue-at-Risk**
Converted each customer's monthly revenue into a Customer Lifetime Value using a 45% gross margin, a 36-month horizon, and a monthly discount rate of 0.8% (present-value annuity factor ≈ 31.17). Multiplied calibrated churn probability by CLV to produce each customer's **Revenue-at-Risk** score.

**9. Segmentation**
Split customers into four segments using a value threshold (top 40% of CLV) and a risk threshold (churn probability ≥ 0.50):

| Segment | Definition |
|---|---|
| Priority Retention | High value + high risk |
| Protect & Grow | High value + low risk |
| Standard Intervention | Low value + high risk |
| Monitor | Low value + low risk |

**10. ROI modeling**
Assigned an assumed intervention (offer cost and success rate) to each segment and calculated the net financial impact of running that intervention — campaign cost vs. revenue expected to be saved.

**11. Root-cause clustering**
Within the highest-priority segment, ran K-Means clustering on service-quality and equipment signals to split "Priority Retention" customers into distinct exit-reason profiles — so the retention team knows *which lever* to pull for which sub-group, not just who to call.

---

## Key Techniques Used

- **Feature engineering** 13 features built from raw usage, revenue, and service-quality columns to surface behavioral signal that raw values alone don't capture
- **Leak-safe preprocessing** — imputed missing values using training set medians computed *after* the train/test split, preventing test-set information from influencing training
- **Model bake-off** — compared five algorithms (Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost) on a consistent train/test split and selected the strongest performer by AUC
- **Probability calibration** — used isotonic regression (`CalibratedClassifierCV`) to convert raw model scores into true probabilities suitable for dollar-value calculations
- **CLV modeling** — converted monthly revenue into lifetime value using a discounted present-value annuity over a defined horizon, with margin and discount-rate assumptions stated explicitly
- **Revenue-at-Risk scoring** — multiplied calibrated churn probability by CLV to rank customers by dollar impact rather than churn likelihood alone
- **Value × risk segmentation** — tiered the customer base into four actionable segments instead of a single risk score
- **ROI analysis** — quantified the expected net financial impact of a retention campaign per segment, using assumed offer costs and success rates
- **Unsupervised clustering for explanation** — used K-Means on the highest-priority segment to separate customers by *why* they're at risk (equipment age vs. service quality vs. overage charges), turning a flat priority list into a set of distinct interventions

---

## Key Findings

1. **Churn in this dataset is close to 50/50**, unusually balanced for telecom — real-world churn is typically 5–10%. This makes accuracy a meaningful evaluation metric here, but that would not hold for a more realistically imbalanced dataset.
2. **No single feature cleanly separates churners from stayers.** The pairwise analysis of revenue, usage, tenure, and equipment age showed overlapping distributions — churn is a multi-variable pattern, not a single threshold.
3. **Equipment age is a meaningful and actionable signal.** Churners skew toward older handsets, and unlike tenure or demographics, equipment age is something the business can directly intervene on (e.g., proactive upgrade offers).
4. **XGBoost was the strongest model** in the bake-off (AUC = 0.6987), modestly ahead of LightGBM and CatBoost, and clearly ahead of Logistic Regression and Random Forest.
5. **Revenue-at-risk is highly concentrated.** Of $82.3M in total portfolio CLV, $40.3M is classified as revenue-at-risk — but targeted intervention on just the "Priority Retention" segment (19,132 customers) is projected to protect $5.57M in revenue for a $1.53M campaign cost, a 2.6x return.
6. **Not all high-risk customers are at risk for the same reason.** Clustering within the Priority Retention segment revealed distinct exit-reason profiles — one group with old equipment and low support-call volume, one group with very old equipment and long tenure, and a smaller but highest-value group driven by high support-call volume, dropped calls, and overage charges.

---

## Files

- `churn_revenue_at_risk_analysis.ipynb` — full notebook: EDA, feature engineering, model bake-off, calibration, CLV/Revenue-at-Risk scoring, segmentation, ROI analysis, and exit-reason clustering
- `revenue_at_risk_roi.csv` — exported ROI table by segment
- `high_value_exit_report.csv` — exit-reason cluster summary for the Priority Retention segment
