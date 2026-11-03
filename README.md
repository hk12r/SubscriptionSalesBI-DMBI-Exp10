# SubScope: BI with Data Mining

## Run locally
    pip install -r requirements.txt
    streamlit run app.py

## Deploy free (Streamlit Community Cloud)
1. Push this folder to a GitHub repo.
2. share.streamlit.io -> New app -> pick the repo, main file `app.py` -> Deploy.

## Data: no uploading needed
On first launch the app generates 3,000 records (same plans, prices, cities and discount logic as the original file), runs ETL and builds a SQLite star schema. To include your original 150 rows, copy the CSV ONCE to `data/subscriptions.csv` before committing; it is merged automatically. The sidebar uploader is optional.

## Mapping to project parts
Problem: find high-value customers, predict churn, forecast revenue.
ETL + star schema (Warehouse tab) | RFM + CLV, k-means vs hierarchical with elbow/silhouette (Segments)
4 classifiers with ROC-AUC, F1, CV, confusion matrix, feature importance (Churn) | Holt-Winters forecast | Isolation Forest anomalies
Limitation: extra rows are synthetic, so model scores show the method, not real-world accuracy.
