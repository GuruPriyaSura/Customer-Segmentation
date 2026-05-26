"""
Customer Segmentation & Repayment Risk Analysis
================================================
Author: Gurupriya R | Mphasis — Enterprise Banking Analytics
Description:
    K-Means clustering + Scikit-learn repayment risk classification on
    retail banking customer data. Segments customers into behavioral
    profiles and flags accounts at risk of default — enabling proactive
    outreach by the Mphasis operations team.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  Customer Segmentation & Repayment Risk Analysis")
print("  Gurupriya R | Mphasis Banking Analytics")
print("=" * 60)

# ── 1. Load Data ───────────────────────────────────────────────────────────────
df = pd.read_csv('data/customer_data.csv')
print(f"\n✅ Loaded {len(df):,} customer records")
print(f"   Risk Distribution:\n{df['repayment_risk'].value_counts().to_string()}\n")

# ── 2. Feature Engineering ─────────────────────────────────────────────────────
print("📐 Engineering features...")

le = LabelEncoder()
df['employment_enc']   = le.fit_transform(df['employment_type'])
df['risk_enc']         = le.fit_transform(df['repayment_risk'])  # Low=1, Med=2, High=0

df['emi_to_income']    = df['monthly_emi'] / (df['annual_income'] / 12)
df['debt_to_income']   = df['outstanding_balance'] / df['annual_income']
df['balance_per_loan'] = df['outstanding_balance'] / (df['total_loans'] + 1)
df['income_per_age']   = df['annual_income'] / df['age']
df['payment_stress']   = df['num_missed_payments'] * df['emi_to_income']

CLUSTER_FEATURES = [
    'credit_score', 'annual_income', 'outstanding_balance',
    'monthly_emi', 'num_missed_payments', 'total_loans',
    'avg_monthly_balance', 'emi_to_income', 'debt_to_income',
    'balance_per_loan', 'payment_stress',
]

RISK_FEATURES = CLUSTER_FEATURES + ['employment_enc', 'has_savings_account',
                                      'loan_tenure_months', 'age', 'income_per_age']

# ── 3. K-Means Clustering ──────────────────────────────────────────────────────
print("🔵 Running K-Means clustering (k=4)...")

scaler = StandardScaler()
X_cluster = scaler.fit_transform(df[CLUSTER_FEATURES])

# Elbow method
inertias = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_cluster)
    inertias.append(km.inertia_)

# Fit final model
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['segment'] = kmeans.fit_predict(X_cluster)

# Label segments by risk profile
seg_profiles = df.groupby('segment').agg(
    avg_credit=('credit_score','mean'),
    avg_income=('annual_income','mean'),
    avg_missed_payments=('num_missed_payments','mean'),
    avg_emi_to_income=('emi_to_income','mean'),
    count=('customer_id','count'),
).round(2).sort_values('avg_credit', ascending=False)

segment_labels = {
    seg_profiles.index[0]: 'Premium',
    seg_profiles.index[1]: 'Stable',
    seg_profiles.index[2]: 'At-Risk',
    seg_profiles.index[3]: 'High-Risk',
}
df['segment_label'] = df['segment'].map(segment_labels)

print("\n   Segment Profiles:")
for seg, label in segment_labels.items():
    row = seg_profiles.loc[seg]
    print(f"   [{label:10s}] Credit: {row['avg_credit']:.0f} | "
          f"Income: ${row['avg_income']:,.0f} | "
          f"Missed Pmts: {row['avg_missed_payments']:.1f} | "
          f"Count: {row['count']:.0f}")

# PCA for visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_cluster)
df['pca_1'] = X_pca[:, 0]
df['pca_2'] = X_pca[:, 1]

# ── 4. Repayment Risk Classification ──────────────────────────────────────────
print("\n🌲 Training repayment risk classifier...")

# Simplify to binary: High-Risk vs not
df['is_high_risk'] = (df['repayment_risk'] == 'High').astype(int)
X = df[RISK_FEATURES]
y = df['is_high_risk']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                 learning_rate=0.08, random_state=42)
gb.fit(X_train, y_train)

y_pred = gb.predict(X_test)
y_proba = gb.predict_proba(X_test)[:, 1]

print("\n   Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Standard','High-Risk']))

try:
    auc = roc_auc_score(y_test, y_proba)
    print(f"   AUC-ROC: {auc:.4f}")
except Exception:
    print("   AUC: N/A (single class in test set)")

# Feature importance
feat_imp = pd.Series(gb.feature_importances_, index=RISK_FEATURES).sort_values(ascending=False)

# Flag at-risk accounts
df['default_probability'] = gb.predict_proba(X)[:, 1]
at_risk = df[df['default_probability'] > 0.6][[
    'customer_id', 'segment_label', 'credit_score',
    'num_missed_payments', 'emi_to_income', 'default_probability'
]].sort_values('default_probability', ascending=False)

print(f"\n⚠️  High-Risk Accounts Flagged: {len(at_risk):,} ({len(at_risk)/len(df):.1%})")
at_risk.to_csv('data/at_risk_customers.csv', index=False)
print("   Saved → data/at_risk_customers.csv")

# ── 5. Dashboard ───────────────────────────────────────────────────────────────
print("\n📊 Generating dashboard...")

fig, axes = plt.subplots(2, 3, figsize=(16, 11))
fig.suptitle('Customer Segmentation & Repayment Risk Analysis\nGurupriya R | Mphasis Banking Analytics',
             fontsize=13, fontweight='bold')
fig.patch.set_facecolor('#fdf6ee')

COLORS = ['#3a8c8c','#5ab0b0','#e8924a','#e07060']

# 1. PCA Cluster Plot
for i, (seg, label) in enumerate(segment_labels.items()):
    mask = df['segment'] == seg
    axes[0,0].scatter(df[mask]['pca_1'], df[mask]['pca_2'],
                      c=COLORS[i], alpha=0.6, s=20, label=label)
axes[0,0].set_title('Customer Segments (PCA)', fontweight='bold')
axes[0,0].set_xlabel('PCA Component 1')
axes[0,0].set_ylabel('PCA Component 2')
axes[0,0].legend(fontsize=8)
axes[0,0].set_facecolor('#fff9f3')

# 2. Segment Size
seg_counts = df['segment_label'].value_counts()
axes[0,1].pie(seg_counts.values, labels=seg_counts.index,
              autopct='%1.0f%%', colors=COLORS, startangle=90, textprops={'fontsize':9})
axes[0,1].set_title('Segment Distribution', fontweight='bold')

# 3. Credit Score by Segment
df.boxplot(column='credit_score', by='segment_label', ax=axes[0,2],
           patch_artist=True, medianprops={'color':'white','linewidth':2})
axes[0,2].set_title('Credit Score by Segment', fontweight='bold')
axes[0,2].set_xlabel('Segment')
axes[0,2].set_ylabel('Credit Score')
plt.sca(axes[0,2])
plt.title('Credit Score by Segment')
axes[0,2].set_facecolor('#fff9f3')

# 4. Default Probability Distribution
axes[1,0].hist(df[df['is_high_risk']==0]['default_probability'], bins=30,
               alpha=0.7, color='#5ab0b0', label='Standard', edgecolor='white')
axes[1,0].hist(df[df['is_high_risk']==1]['default_probability'], bins=30,
               alpha=0.7, color='#e07060', label='High-Risk', edgecolor='white')
axes[1,0].set_title('Default Probability Distribution', fontweight='bold')
axes[1,0].set_xlabel('Default Probability')
axes[1,0].legend()
axes[1,0].set_facecolor('#fff9f3')

# 5. Top Feature Importances
top_feat = feat_imp.head(10).sort_values()
axes[1,1].barh(top_feat.index, top_feat.values, color='#e8924a', edgecolor='white')
axes[1,1].set_title('Top 10 Risk Predictors', fontweight='bold')
axes[1,1].set_xlabel('Feature Importance')
axes[1,1].set_facecolor('#fff9f3')
axes[1,1].grid(axis='x', alpha=0.3)

# 6. Repayment Risk by Segment
risk_cross = pd.crosstab(df['segment_label'], df['repayment_risk'], normalize='index')
risk_cross.plot(kind='bar', ax=axes[1,2], color=['#e07060','#5ab0b0','#e8924a'],
                edgecolor='white', linewidth=0.5)
axes[1,2].set_title('Repayment Risk Mix by Segment', fontweight='bold')
axes[1,2].set_xlabel('Segment')
axes[1,2].set_ylabel('Proportion')
axes[1,2].tick_params(axis='x', rotation=15)
axes[1,2].legend(title='Risk Level', fontsize=8)
axes[1,2].set_facecolor('#fff9f3')

plt.tight_layout()
plt.savefig('segmentation_dashboard.png', dpi=150, bbox_inches='tight', facecolor='#fdf6ee')
print("   Dashboard saved → segmentation_dashboard.png")
print(f"\n✅ Done! {len(at_risk):,} high-risk customers flagged for proactive outreach.")
