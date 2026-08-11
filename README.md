# 🧩 Customer Segmentation with K-Means + RFM

An unsupervised-learning project that turns transactions into actionable customer segments for targeted marketing.

## Pipeline
1. Load transaction records with customer, date, quantity, and price.
2. Aggregate Recency, Frequency, and Monetary value per customer.
3. Log-transform skewed values and standardize features.
4. Compare candidate cluster counts with silhouette score.
5. Profile clusters and assign business labels.
6. Recommend campaigns for each segment.

## Segments
- **Champions:** recent, frequent, high-value customers
- **Loyal:** regular customers with strong spend
- **At Risk:** previously valuable but becoming inactive
- **New/Hibernating:** low history or long time since purchase

## Interview talking points
- RFM creates interpretable features business teams can act on.
- Scaling and log transforms are crucial because K-Means is distance-based.
- The best K balances silhouette score, stability, interpretability, and campaign uplift.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
