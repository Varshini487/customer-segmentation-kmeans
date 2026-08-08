# 🧩 Customer Segmentation with K-Means + RFM

A practical unsupervised-learning project that groups customers by **Recency, Frequency, and Monetary value** so marketing teams can design more relevant campaigns.

## Segments
- **Champions** — recent, frequent, high-value buyers
- **Loyal customers** — repeat purchasers with strong engagement
- **At risk** — previously valuable but inactive
- **New customers** — recent, low-frequency buyers
- **Hibernating** — low activity and low value

## How it works
1. Start with transaction-level data containing customer ID, date, quantity, and price.
2. Aggregate each customer into RFM features: days since last purchase, order count, and total spend.
3. Log-transform/skew-correct the features and standardize them.
4. Use the elbow curve and silhouette score to choose K.
5. Fit K-Means, profile each cluster, and map clusters to business-friendly labels.
6. Export segment assignments for targeted marketing experiments.

## Tech stack
Python, Pandas, NumPy, Scikit-learn, Plotly, Streamlit

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Interview focus
The important production details are leakage-safe observation windows, cluster stability over time, and validating that segments lead to measurable uplift rather than only attractive visualizations.
