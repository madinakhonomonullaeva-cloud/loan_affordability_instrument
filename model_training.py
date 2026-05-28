"""
Monte Carlo Loan Risk Model — Dissertation Script
==================================================
Generates 100,000 synthetic Uzbekistan borrower profiles,
simulates 5-year budgets under inflation, labels defaults,
trains RandomForestClassifier, saves loan_risk_model.pkl.

Run once locally:
    pip install scikit-learn joblib numpy pandas
    python model_training.py

Output: loan_risk_model.pkl  → place next to app.py on Streamlit Cloud
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib

SEED = 42
N    = 100_000
np.random.seed(SEED)

# ── Step 1: Generate 100,000 synthetic borrowers ──────────────
print("═" * 60)
print("Step 1: Generating 100,000 synthetic borrowers...")
print("═" * 60)

# Income: log-normal, Uzbekistan context (UZS)
income = np.random.lognormal(mean=np.log(5_000_000), sigma=0.50, size=N)
income = np.clip(income, 1_500_000, 25_000_000)

# Loan amount: 1–10 months of income (realistic UZ consumer credit)
amount_ratio = np.random.lognormal(mean=np.log(4), sigma=0.6, size=N)
amount_ratio = np.clip(amount_ratio, 1.0, 12.0)
loan_amount  = np.clip(income * amount_ratio, 2_000_000, 80_000_000)

# Loan term: typical UZ market distribution
term = np.random.choice([6, 12, 18, 24, 36, 48, 60, 84, 120],
                        p=[0.04, 0.12, 0.10, 0.18, 0.22, 0.14, 0.12, 0.05, 0.03],
                        size=N)

# Annual interest rate: UZ banks 18–36%
rate = np.random.uniform(18.0, 36.0, size=N)

# Essential + flex expenses: 35–85% of income total
expense_ratio   = np.random.uniform(0.35, 0.85, size=N)
essential_exp   = income * expense_ratio * np.random.uniform(0.75, 0.90, size=N)
flex_exp        = income * expense_ratio * np.random.uniform(0.10, 0.25, size=N)

# Existing loan payments: 40% of people have them
has_exist    = np.random.rand(N) < 0.40
existing     = np.where(has_exist, income * np.random.uniform(0.05, 0.25, size=N), 0.0)

# Savings: 0–5 months of income
savings = income * np.random.uniform(0, 5, size=N)

# Qualitative flags
unstable     = (np.random.rand(N) < 0.20).astype(float)  # unstable salary
timing_risk  = (np.random.rand(N) < 0.30).astype(float)  # pays before salary day

df = pd.DataFrame({
    "income":        income,
    "loan_amount":   loan_amount,
    "term_months":   term,
    "interest_rate": rate,
    "essential_exp": essential_exp,
    "flex_exp":      flex_exp,
    "existing_loans":existing,
    "savings":       savings,
    "unstable":      unstable,
    "timing_risk":   timing_risk,
})

# ── Step 2: Compute monthly annuity payment ───────────────────
print("\nStep 2: Computing annuity payments...")

def annuity(p, r_pct, m):
    r = (r_pct / 100) / 12
    if r == 0:
        return p / m
    return p * r * (1 + r)**m / ((1 + r)**m - 1)

df["monthly_payment"] = [
    annuity(row.loan_amount, row.interest_rate, row.term_months)
    for row in df.itertuples()
]

# ── Step 3: Month-by-month simulation (3 macro scenarios) ─────
print("\nStep 3: 5-year budget simulation (3 macro scenarios × 100k borrowers)...")

# Scenario: (name, monthly_inflation, monthly_wage_growth, weight)
SCENARIOS = [
    ("baseline", 0.0082, 0.0055, 0.55),   # ~10% inflation, ~6.7% wage growth/yr
    ("stress",   0.0120, 0.0035, 0.30),   # ~15% inflation, ~4.3% wage growth/yr
    ("benign",   0.0045, 0.0080, 0.15),   # ~5.5% inflation, ~10% wage growth/yr
]

def simulate_batch(df_in, infl_mo, wage_mo):
    """Vectorised month-by-month simulation. Returns worst free-cash array."""
    n    = len(df_in)
    inc  = df_in["income"].values.copy().astype(float)
    ess  = (df_in["essential_exp"] + df_in["flex_exp"]).values.copy().astype(float)
    pmt  = df_in["monthly_payment"].values.astype(float)
    ext  = df_in["existing_loans"].values.astype(float)
    term = df_in["term_months"].values.astype(int)
    min_cash = np.full(n, np.inf)

    for m in range(1, int(term.max()) + 1):
        active = (m <= term)
        if m > 1:
            inc *= (1 + wage_mo)
            ess *= (1 + infl_mo)
        cash = inc - ess - pmt - ext
        cash_active = np.where(active, cash, np.inf)
        min_cash = np.minimum(min_cash, cash_active)
        if not active.any():
            break
    return min_cash

weighted_min_cash = np.zeros(N)
for name, infl, wage, w in SCENARIOS:
    print(f"   '{name}' scenario (weight {w:.0%})...")
    weighted_min_cash += w * simulate_batch(df, infl, wage)

df["min_free_cash"] = weighted_min_cash

# ── Step 4: Label defaults ─────────────────────────────────────
print("\nStep 4: Labelling defaults...")

df["initial_pti"] = df["monthly_payment"] / df["income"]
df["initial_tdb"] = (df["monthly_payment"] + df["existing_loans"]) / df["income"]

# Default if:
#   — worst-month free cash deeply negative (ran out of money), OR
#   — total debt burden initially > 65%, OR
#   — PTI > 55% AND no savings buffer
df["is_default"] = np.where(
    (df["min_free_cash"] < -df["monthly_payment"]) |
    (df["initial_tdb"]   >  0.65) |
    ((df["initial_pti"]  >  0.55) & (df["savings"] < df["monthly_payment"])),
    1, 0
).astype(int)

default_rate = df["is_default"].mean() * 100
print(f"   Default rate: {default_rate:.1f}%  (target: 20–50%)")

# ── Step 5: Train Random Forest ───────────────────────────────
print("\nStep 5: Training Random Forest (200 trees)...")

FEATURES = [
    "income", "loan_amount", "term_months", "interest_rate",
    "essential_exp", "flex_exp", "existing_loans", "savings",
    "unstable", "timing_risk",
]

X = df[FEATURES]
y = df["is_default"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)

model = RandomForestClassifier(
    n_estimators     = 200,
    max_depth        = 12,
    min_samples_leaf = 20,
    class_weight     = "balanced",
    random_state     = SEED,
    n_jobs           = -1,
)
model.fit(X_train, y_train)

# ── Step 6: Evaluate ──────────────────────────────────────────
print("\n" + "═" * 60)
print("MODEL RESULTS  (paste into dissertation Chapter 4)")
print("═" * 60)

y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_proba)
print(f"\nROC-AUC Score  :  {auc:.4f}")
print(f"Test samples   :  {len(y_test):,}")
print(f"\n{classification_report(y_test, y_pred, target_names=['No default','Default'])}")

cm = confusion_matrix(y_test, y_pred)
print(f"Confusion matrix:")
print(f"  TN = {cm[0,0]:,}   FP = {cm[0,1]:,}")
print(f"  FN = {cm[1,0]:,}   TP = {cm[1,1]:,}")

print("\nFeature importances:")
fi = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
for feat, imp in fi.items():
    bar = "█" * int(imp * 40)
    print(f"  {feat:<22}  {imp:.4f}  {bar}")

# ── Step 7: Save ──────────────────────────────────────────────
print("\nStep 7: Saving model...")
payload = {
    "model":        model,
    "features":     FEATURES,
    "version":      "1.0-montecarlo",
    "n_simulated":  N,
    "default_rate": round(default_rate, 2),
    "roc_auc":      round(auc, 4),
}
joblib.dump(payload, "loan_risk_model.pkl", compress=3)
print("✅  Saved: loan_risk_model.pkl")
print("    → Copy this file next to app.py on Streamlit Cloud\n")
