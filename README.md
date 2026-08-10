# 🧩 Customer Segmentation with K-Means + RFM

An unsupervised-learning system that groups customers by **Recency, Frequency, and Monetary value** so marketing teams can personalize campaigns.

## How it works
1. Start from transaction-level customer, date, quantity, and price data.
2. Aggregate each customer into RFM features: days since last purchase, order count, and total spend.
3. Log-transform skewed values and standardize them.
4. Train K-Means and compare silhouette scores across candidate cluster counts.
5. Profile each cluster and map it to business labels such as Champions, Loyal, At Risk, or New.
6. Recommend a campaign for each segment.

## Tech stack
Python, Pandas, NumPy, Scikit-learn, Plotly, Streamlit

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Interview points
- RFM turns raw transactions into interpretable customer behavior signals.
- Scaling and log transforms are essential because K-Means is distance-based and spend is usually skewed.
- Choose K using silhouette score, stability, business interpretability, and campaign uplift—not one chart alone.
