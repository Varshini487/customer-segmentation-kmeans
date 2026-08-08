import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px

st.set_page_config(page_title="Customer Segmentation",layout="wide")
st.title("🧩 Customer Segmentation — K-Means + RFM")
st.caption("Synthetic transaction demo; replace with consented business data for real analysis.")

@st.cache_data
def make_transactions(n_customers=350,n_orders=5000):
    rng=np.random.default_rng(12); ids=rng.integers(1,n_customers+1,n_orders)
    dates=pd.Timestamp('2026-07-31')-pd.to_timedelta(rng.integers(0,365,n_orders),unit='D')
    return pd.DataFrame({'customer_id':ids,'date':dates,'quantity':rng.integers(1,6,n_orders),'price':rng.lognormal(3.4,.7,n_orders).round(2)})

tx=make_transactions(); tx['amount']=tx.quantity*tx.price; today=tx.date.max()+pd.Timedelta(days=1)
rfm=tx.groupby('customer_id').agg(last_purchase=('date','max'),frequency=('date','nunique'),monetary=('amount','sum')).reset_index()
rfm['recency']=(today-rfm.last_purchase).dt.days
rfm=rfm[['customer_id','recency','frequency','monetary']]
features=np.log1p(rfm[['recency','frequency','monetary']]); scaled=StandardScaler().fit_transform(features)

k=st.sidebar.slider('Number of segments (K)',2,8,5)
km=KMeans(n_clusters=k,n_init=20,random_state=42); rfm['cluster']=km.fit_predict(scaled)
sil=silhouette_score(scaled,rfm.cluster)
summary=rfm.groupby('cluster').agg(customers=('customer_id','count'),recency=('recency','mean'),frequency=('frequency','mean'),monetary=('monetary','mean')).reset_index()
summary['segment']=summary.apply(lambda r:'Champions' if r.monetary>=summary.monetary.quantile(.75) and r.recency<=summary.recency.quantile(.25) else ('At risk' if r.recency>=summary.recency.quantile(.75) else ('Loyal customers' if r.frequency>=summary.frequency.median() else 'New/Hibernating')),axis=1)

c1,c2,c3=st.columns(3); c1.metric('Customers',len(rfm)); c2.metric('Transactions',len(tx)); c3.metric('Silhouette score',f'{sil:.3f}')

t1,t2,t3=st.tabs(['📊 Segment overview','🔎 Customer lookup','📈 K selection'])
with t1:
    st.dataframe(summary.round(2),use_container_width=True)
    st.plotly_chart(px.scatter_3d(rfm,x='recency',y='frequency',z='monetary',color=rfm.cluster.astype(str),hover_data=['customer_id'],title='RFM customer clusters'),use_container_width=True)
    st.plotly_chart(px.bar(summary,x='segment',y='customers',color='segment',title='Customers by business segment'),use_container_width=True)
with t2:
    cid=st.number_input('Customer ID',1,int(rfm.customer_id.max()),1)
    row=rfm[rfm.customer_id==cid]
    if len(row):
        rr=row.iloc[0]; seg=summary.loc[summary.cluster==rr.cluster,'segment'].iloc[0]
        a,b,c=st.columns(3); a.metric('Recency (days)',int(rr.recency)); b.metric('Orders',int(rr.frequency)); c.metric('Total spend',f"${rr.monetary:,.2f}")
        st.success(f'Assigned segment: **{seg}**')
        st.write('Suggested action:', 'VIP rewards and referrals' if seg=='Champions' else 'Win-back offer and engagement campaign' if seg=='At risk' else 'Personalized onboarding or cross-sell test')
with t3:
    scores=[]
    for kk in range(2,9):
        labels=KMeans(n_clusters=kk,n_init=10,random_state=42).fit_predict(scaled); scores.append({'k':kk,'silhouette':silhouette_score(scaled,labels)})
    st.plotly_chart(px.line(pd.DataFrame(scores),x='k',y='silhouette',markers=True,title='Silhouette score by K'),use_container_width=True)
    st.info('Choose K using silhouette score, cluster stability, interpretability, and downstream campaign uplift—not the chart alone.')
