import streamlit as st
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px

st.set_page_config(page_title="RFM Segmentation", layout="wide")
st.title("🧩 Customer Segmentation — K-Means + RFM")
st.caption("Demo data is generated locally; replace it with your transaction export in production.")

@st.cache_data
def transactions(n=5000):
    rng=np.random.default_rng(4); customers=rng.integers(1,301,n); dates=pd.Timestamp("2026-07-31")-pd.to_timedelta(rng.integers(0,365,n),unit="D")
    return pd.DataFrame({"customer_id":customers,"date":dates,"quantity":rng.integers(1,6,n),"price":rng.gamma(2,35,n).round(2)})

tx=transactions(); tx["value"]=tx.quantity*tx.price; today=tx.date.max()+pd.Timedelta(days=1)
rfm=tx.groupby("customer_id").agg(recency=("date",lambda x:(today-x.max()).days),frequency=("date","count"),monetary=("value","sum")).reset_index()
X=np.log1p(rfm[["recency","frequency","monetary"]]); X=StandardScaler().fit_transform(X)
k=st.slider("Number of clusters",2,8,4); km=KMeans(n_clusters=k,n_init=20,random_state=42); rfm["cluster"]=km.fit_predict(X); score=silhouette_score(X,rfm.cluster)
st.metric("Silhouette score",f"{score:.3f}"); st.dataframe(rfm.head(20))

summary=rfm.groupby("cluster").agg(customers=("customer_id","count"),recency=("recency","mean"),frequency=("frequency","mean"),monetary=("monetary","mean")).reset_index(); st.subheader("Cluster profiles"); st.dataframe(summary.round(1)); st.plotly_chart(px.scatter(rfm,x="frequency",y="monetary",color="cluster",hover_data=["customer_id","recency"]),use_container_width=True)

def label(row):
    if row.monetary>=rfm.monetary.quantile(.75) and row.recency<=rfm.recency.quantile(.25): return "Champions"
    if row.recency>=rfm.recency.quantile(.75): return "At Risk"
    if row.frequency<=rfm.frequency.quantile(.25): return "New/Hibernating"
    return "Loyal"
rfm["segment"]=rfm.apply(label,axis=1); st.subheader("Recommended actions"); st.dataframe(rfm.groupby("segment").size().rename("customers")); st.info("Champions → VIP rewards | Loyal → cross-sell | At Risk → win-back offer | New/Hibernating → onboarding campaign")
