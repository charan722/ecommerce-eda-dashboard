import pandas as pd
from src.data_loader import get_data
from mlxtend.frequent_patterns import apriori, association_rules

print("Loading data...")
df = get_data()

# Cohort Analysis
print("Cohort Analysis...")
df_cohort = df[df["CustomerID"] != -1].copy()
df_cohort['InvoiceMonth'] = df_cohort['InvoiceDate'].dt.to_period('M')
df_cohort['CohortMonth'] = df_cohort.groupby('CustomerID')['InvoiceMonth'].transform('min')

def diff_month(d1, d2):
    return (d1.year - d2.year) * 12 + d1.month - d2.month

df_cohort['CohortIndex'] = diff_month(df_cohort['InvoiceMonth'].dt, df_cohort['CohortMonth'].dt) + 1
cohort_data = df_cohort.groupby(['CohortMonth', 'CohortIndex'])['CustomerID'].nunique().reset_index()
cohort_counts = cohort_data.pivot(index='CohortMonth', columns='CohortIndex', values='CustomerID')
cohort_counts.index = cohort_counts.index.astype(str)
retention = cohort_counts.divide(cohort_counts.iloc[:,0], axis=0).round(3) * 100
print(retention.head())

# RFM Analysis
print("RFM Analysis...")
snapshot_date = df_cohort['InvoiceDate'].max() + pd.Timedelta(days=1)
rfm = df_cohort.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalAmount': 'sum'
}).rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalAmount': 'Monetary'})

# Quantiles
r_labels = range(4, 0, -1)
f_labels = range(1, 5)
m_labels = range(1, 5)
try:
    r_groups = pd.qcut(rfm['Recency'], q=4, labels=r_labels)
    f_groups = pd.qcut(rfm['Frequency'].rank(method='first'), q=4, labels=f_labels)
    m_groups = pd.qcut(rfm['Monetary'], q=4, labels=m_labels)
    rfm = rfm.assign(R=r_groups, F=f_groups, M=m_groups)
    rfm['RFM_Segment_Concat'] = rfm['R'].astype(str) + rfm['F'].astype(str) + rfm['M'].astype(str)
    rfm['RFM_Score'] = rfm[['R', 'F', 'M']].sum(axis=1)
except Exception as e:
    print("Error in RFM:", e)
print(rfm.head())

# Basket Analysis (UK only)
print("Basket Analysis...")
df_uk = df[df['Country'] == 'United Kingdom']
basket = (df_uk.groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))

def encode_units(x):
    if x <= 0: return 0
    if x >= 1: return 1

basket_sets = basket.map(encode_units)
# drop invoices with 1 or less items to speed up
basket_sets = basket_sets[(basket_sets > 0).sum(axis=1) >= 2]
print(f"Basket shape after filtering: {basket_sets.shape}")
frequent_itemsets = apriori(basket_sets, min_support=0.03, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
print(rules.head())
