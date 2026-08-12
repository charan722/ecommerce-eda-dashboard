import streamlit as st
import plotly.express as px
import pandas as pd
from src.data_loader import get_data
from mlxtend.frequent_patterns import apriori, association_rules

# Page configuration
st.set_page_config(page_title="E-Commerce EDA Dashboard", layout="wide")

st.title("📊 E-Commerce Real-Time Exploratory Analytics")

# Cache data loading to optimize performance
@st.cache_data
def load_data():
    return get_data()

df = load_data()

# Sidebar: Cross-filtering controls
st.sidebar.header("Global Filters")
countries = st.sidebar.multiselect(
    "Select Country:",
    options=df["Country"].unique(),
    default=["United Kingdom", "Germany", "France"]
)

# Apply filters
filtered_df = df[df["Country"].isin(countries)] if countries else df

# Metric KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Revenue", f"${filtered_df['TotalAmount'].sum():,.2f}")
col2.metric("Total Orders", f"{filtered_df['InvoiceNo'].nunique():,}")
col3.metric("Avg Order Value", f"${filtered_df.groupby('InvoiceNo')['TotalAmount'].sum().mean():,.2f}")
col4.metric("Unique Customers", f"{filtered_df[filtered_df['CustomerID'] != -1]['CustomerID'].nunique():,}")

st.markdown("---")

# Visual Analytics Section
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Revenue Trends", 
    "📦 Top Products & Geography", 
    "🎯 RFM Segmentation", 
    "🔄 Cohort Analysis", 
    "🛒 Basket Analysis"
])

with tab1:
    st.subheader("Monthly Revenue Trajectory")
    monthly_rev = filtered_df.groupby("YearMonth")["TotalAmount"].sum().reset_index()
    fig_line = px.line(monthly_rev, x="YearMonth", y="TotalAmount", markers=True, 
                       title="Revenue over Time", labels={"TotalAmount": "Revenue ($)"})
    st.plotly_chart(fig_line, width="stretch")

with tab2:
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Top 10 Products by Sales")
        top_products = filtered_df.groupby("Description")["TotalAmount"].sum().nlargest(10).reset_index()
        fig_bar = px.bar(top_products, x="TotalAmount", y="Description", orientation="h",
                         color="TotalAmount", color_continuous_scale="Viridis")
        st.plotly_chart(fig_bar, width="stretch")
        
    with col_right:
        st.subheader("Revenue by Country")
        country_rev = filtered_df.groupby("Country")["TotalAmount"].sum().reset_index()
        fig_map = px.choropleth(country_rev, locations="Country", locationmode="country names",
                                color="TotalAmount", color_continuous_scale="Blues")
        st.plotly_chart(fig_map, width="stretch")

with tab3:
    st.subheader("RFM Customer Segmentation")
    st.markdown("Segments users based on Recency (days since last purchase), Frequency (total orders), and Monetary (total spend).")
    
    # Exclude guests
    df_rfm = filtered_df[filtered_df["CustomerID"] != -1].copy()
    if df_rfm.empty:
        st.warning("No registered customers in this selection.")
    else:
        snapshot_date = df_rfm['InvoiceDate'].max() + pd.Timedelta(days=1)
        rfm = df_rfm.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
            'InvoiceNo': 'nunique',
            'TotalAmount': 'sum'
        }).rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalAmount': 'Monetary'})
        
        # Calculate scores
        try:
            r_groups = pd.qcut(rfm['Recency'], q=4, labels=range(4, 0, -1))
            f_groups = pd.qcut(rfm['Frequency'].rank(method='first'), q=4, labels=range(1, 5))
            m_groups = pd.qcut(rfm['Monetary'], q=4, labels=range(1, 5))
            
            rfm = rfm.assign(R=r_groups, F=f_groups, M=m_groups)
            rfm['RFM_Score'] = rfm[['R', 'F', 'M']].sum(axis=1)
            
            # Segment Definition
            def segment_customer(score):
                if score >= 10: return 'Champions'
                elif score >= 8: return 'Loyal Customers'
                elif score >= 6: return 'Potential Loyalists'
                elif score >= 4: return 'At Risk'
                else: return 'Lost'
                
            rfm['Segment'] = rfm['RFM_Score'].apply(segment_customer)
            
            col_r1, col_r2 = st.columns([2, 1])
            with col_r1:
                segment_counts = rfm['Segment'].value_counts().reset_index()
                segment_counts.columns = ['Segment', 'Count']
                fig_tree = px.treemap(segment_counts, path=['Segment'], values='Count', 
                                      color='Count', color_continuous_scale='Blues',
                                      title='Customer Segments Overview')
                st.plotly_chart(fig_tree, width="stretch")
            with col_r2:
                st.dataframe(rfm[['Recency', 'Frequency', 'Monetary', 'Segment']].head(10), width="stretch")
        except Exception as e:
            st.error(f"Not enough data to calculate quantiles for this selection. (Error: {e})")

with tab4:
    st.subheader("Cohort Analysis / Retention Matrix")
    df_cohort = filtered_df[filtered_df["CustomerID"] != -1].copy()
    if df_cohort.empty:
        st.warning("No registered customers in this selection.")
    else:
        df_cohort['InvoiceMonth'] = df_cohort['InvoiceDate'].dt.to_period('M')
        df_cohort['CohortMonth'] = df_cohort.groupby('CustomerID')['InvoiceMonth'].transform('min')
        
        def diff_month(d1, d2):
            return (d1.year - d2.year) * 12 + d1.month - d2.month
            
        df_cohort['CohortIndex'] = diff_month(df_cohort['InvoiceMonth'].dt, df_cohort['CohortMonth'].dt) + 1
        cohort_data = df_cohort.groupby(['CohortMonth', 'CohortIndex'])['CustomerID'].nunique().reset_index()
        cohort_counts = cohort_data.pivot(index='CohortMonth', columns='CohortIndex', values='CustomerID')
        cohort_counts.index = cohort_counts.index.astype(str)
        retention = cohort_counts.divide(cohort_counts.iloc[:,0], axis=0) * 100
        
        fig_cohort = px.imshow(retention, 
                               labels=dict(x="Cohort Index (Months)", y="Cohort Month", color="Retention %"),
                               x=retention.columns,
                               y=retention.index,
                               text_auto=".1f",
                               aspect="auto",
                               color_continuous_scale="Viridis",
                               title="Monthly Customer Retention Rates")
        st.plotly_chart(fig_cohort, width="stretch")

with tab5:
    st.subheader("Basket Analysis (Cross-Selling)")
    st.markdown("Association rules uncovering which items are frequently bought together.")
    
    # Filter for computationally safe size - default to UK if selected, else take the top 10000 invoices
    df_basket = filtered_df[filtered_df['Country'] == 'United Kingdom'] if 'United Kingdom' in filtered_df['Country'].unique() else filtered_df
    
    # Identify top 500 most popular products to reduce matrix dimensionality (Memory optimization)
    top_products = df_basket['Description'].value_counts().head(500).index
    df_basket = df_basket[df_basket['Description'].isin(top_products)]
    
    # Limit to top 2000 invoices to prevent memory crash
    top_invoices = df_basket['InvoiceNo'].value_counts().head(2000).index
    df_basket = df_basket[df_basket['InvoiceNo'].isin(top_invoices)]
    
    if df_basket.empty:
        st.warning("Not enough data for basket analysis.")
    else:
        st.info("Market Basket Analysis is highly computationally intensive. Click the button below to generate cross-selling rules.")
        if st.button("Run Basket Analysis", type="primary"):
            with st.spinner("Computing association rules (this may take a minute)..."):
                # Pivot efficiently
                basket = (df_basket.groupby(['InvoiceNo', 'Description'])['Quantity']
                          .sum().unstack().reset_index().fillna(0)
                          .set_index('InvoiceNo'))
                
                # Convert to boolean for mlxtend
                basket_sets = (basket > 0).astype(bool)
                # Drop invoices with < 2 items
                basket_sets = basket_sets[(basket_sets.sum(axis=1)) >= 2]
                
                if basket_sets.shape[0] < 10:
                    st.warning("Not enough multi-item baskets to find association rules.")
                else:
                    frequent_itemsets = apriori(basket_sets, min_support=0.03, use_colnames=True)
                    if frequent_itemsets.empty:
                        st.info("No frequent itemsets found with 3% support. Try a larger dataset or different country.")
                    else:
                        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
                        rules = rules.sort_values(by='lift', ascending=False)
                        
                        # Formatting for display
                        rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
                        rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
                        rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
                        
                        st.dataframe(rules.head(20), width="stretch")
