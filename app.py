import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px

st.set_page_config(page_title="Customer Segmentation",layout="wide"); st.title("🧩 Customer Segmentation — K-Means + RFM")
@st.cache_data
def transactions(n=1500):
    rng=np.random.default_rng(12); end=pd.Timestamp("2026-08-01")
    return pd.DataFrame({"customer_id":rng.integers(1000,1250,n),"date":end-pd.to_timedelta(rng.integers(0,365,n),unit="D"),"quantity":rng.integers(1,6,n),"unit_price":rng.lognormal(3.2,.6,n)})
raw=transactions(); raw["amount"]=raw.quantity*raw.unit_price
latest=raw.date.max()+pd.Timedelta(days=1)
rfm=raw.groupby("customer_id").agg(recency=("date",lambda x:(latest-x.max()).days),frequency=("date","count"),monetary=("amount","sum")).reset_index()
X=np.log1p(rfm[["recency","frequency","monetary"]]); X=StandardScaler().fit_transform(X)
k=st.slider("Number of clusters (K)",2,8,4); km=KMeans(n_clusters=k,n_init=20,random_state=42); rfm["cluster"]=km.fit_predict(X)
score=silhouette_score(X,rfm.cluster)
c1,c2,c3=st.columns(3); c1.metric("Customers",len(rfm)); c2.metric("Clusters",k); c3.metric("Silhouette",f"{score:.3f}")
profiles=rfm.groupby("cluster")[["recency","frequency","monetary"]].mean().round(1); st.subheader("Cluster profiles"); st.dataframe(profiles,use_container_width=True)
fig=px.scatter(rfm,x="frequency",y="monetary",color=rfm.cluster.astype(str),size="frequency",hover_data=["customer_id","recency"]); st.plotly_chart(fig,use_container_width=True)
st.info("Campaign idea: low recency/high frequency = win-back; low recency/high monetary = VIP loyalty; high recency/low frequency = reactivation.")
