"""SubScope: Business Intelligence with Data Mining for subscription sales.
Run: streamlit run app.py   (data is generated/loaded automatically, nothing to upload)
"""
import os, sqlite3
import numpy as np, pandas as pd, plotly.express as px, plotly.graph_objects as go, streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix, silhouette_score)
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config(page_title="SubScope BI", layout="wide")
DATA_DIR, DB, CSV = "data", "data/warehouse.db", "data/subscriptions.csv"
PRICE = {"Basic": 1000, "Standard": 2500, "Premium": 5000}
USE = {"Basic": 18, "Standard": 30, "Premium": 45}
NUM = ["Subscription_Quantity", "Discount_Pct", "Tickets", "Usage_Hours", "Tenure_Months"]
CAT = ["Plan", "Plan_Type", "Payment_Method", "Location"]
inr = lambda x: f"₹{x:,.0f}"

# ============================================================ 1. DATA + ETL
def generate(n=3000, seed=42):
    """Synthetic records that follow the structure and pricing of the original sales file."""
    rng = np.random.default_rng(seed)
    cities = ["Mumbai", "Delhi", "Pune", "Bangalore", "Chennai", "Hyderabad", "Kolkata",
              "Jaipur", "Nagpur", "Nashik", "Lucknow", "Ahmedabad"]
    names = ["Amit", "Rahul", "Priya", "Sneha", "Arjun", "Neha", "Rohan", "Anjali", "Vikram", "Pooja",
             "Karan", "Meera", "Aditya", "Kavya", "Sahil", "Riya", "Manish", "Nisha", "Varun", "Isha",
             "Ishaan", "Tanvi", "Aarav", "Diya", "Dev", "Simran", "Kabir", "Yash", "Aditi", "Ananya"]
    cid = rng.integers(1000, 2500, n)
    plan = rng.choice(list(PRICE), n, p=[.38, .34, .28])
    ptype = rng.choice(["Monthly", "Yearly"], n, p=[.55, .45])
    pay = rng.choice(["UPI", "Credit Card", "Debit Card", "Net Banking"], n, p=[.32, .26, .24, .18])
    qty = rng.integers(1, 5, n)
    unit = np.array([PRICE[p] for p in plan]) * np.where(ptype == "Yearly", 2, 1)
    gross = unit * qty
    dp = rng.choice([0, .05, .10, .15, .20], n, p=[.10, .25, .30, .20, .15])
    date = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 1034, n), "D")
    tick = rng.poisson(pd.Series(plan).map({"Basic": 2.2, "Standard": 1.5, "Premium": 1.0}).values)
    use = np.clip(rng.normal(pd.Series(plan).map(USE).values, 12), 1, None).round(1)
    z = (-0.4 + 0.55 * (plan == "Basic") - 0.45 * (plan == "Premium") + 0.45 * (ptype == "Monthly")
         + 0.3 * (tick - 1.5) - 0.04 * (use - 30) - 3 * (dp - .10) + 0.25 * (pay == "Net Banking"))
    inactive = rng.random(n) < 1 / (1 + np.exp(-z))
    return pd.DataFrame({
        "Customer_ID": cid, "Customer_Name": [names[i % len(names)] for i in cid],
        "Location": [cities[i % len(cities)] for i in cid], "Country": "India", "Date": date,
        "Plan": plan, "Plan_Type": ptype, "Payment_Method": pay, "Subscription_Quantity": qty,
        "Gross_Revenue": gross, "Discount": gross * dp, "Net_Revenue": gross * (1 - dp),
        "Subscription_Status": np.where(inactive, "Inactive", "Active"), "Tickets": tick, "Usage_Hours": use})

def read_any(src):
    return pd.read_csv(src, sep=None, engine="python")

def enrich(d, seed=7):
    rng = np.random.default_rng(seed)
    if "Tickets" not in d:
        d["Tickets"] = rng.poisson(d.Plan.map({"Basic": 2.2, "Standard": 1.5, "Premium": 1.0}).fillna(1.5))
    if "Usage_Hours" not in d:
        d["Usage_Hours"] = np.clip(rng.normal(d.Plan.map(USE).fillna(30), 12), 1, None).round(1)
    return d

def prepare(raw):
    """Clean + feature engineering."""
    d = raw.copy()
    d.columns = [c.strip() for c in d.columns]
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    for c in ["Subscription_Quantity", "Gross_Revenue", "Discount", "Net_Revenue"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["Date", "Net_Revenue", "Plan", "Gross_Revenue"]).drop_duplicates().reset_index(drop=True)
    d = enrich(d)
    d["Discount_Pct"] = (d.Discount / d.Gross_Revenue).fillna(0)
    d["Tenure_Months"] = ((d.Date.max() - d.Date).dt.days / 30).round(1)
    d["Churn"] = (d.Subscription_Status == "Inactive").astype(int)
    d["Month"] = d.Date.dt.to_period("M").dt.to_timestamp()
    return d

def build_dw(d):
    """Load a star schema (1 fact + 4 dimensions) into SQLite."""
    os.makedirs(DATA_DIR, exist_ok=True)
    con = sqlite3.connect(DB)
    dc = d[["Customer_ID", "Customer_Name", "Location", "Country"]].drop_duplicates("Customer_ID")
    dc.columns = ["customer_id", "name", "city", "country"]
    dp = d[["Plan", "Plan_Type"]].drop_duplicates().reset_index(drop=True)
    dp.insert(0, "plan_key", range(1, len(dp) + 1))
    dy = d[["Payment_Method"]].drop_duplicates().reset_index(drop=True)
    dy.insert(0, "pay_key", range(1, len(dy) + 1))
    dd = pd.DataFrame({"date": d.Date.dt.normalize().unique()})
    dd["date_key"] = dd.date.dt.strftime("%Y%m%d").astype(int)
    dd["year"], dd["quarter"], dd["month"] = dd.date.dt.year, dd.date.dt.quarter, dd.date.dt.month
    dd["date"] = dd.date.dt.strftime("%Y-%m-%d")
    f = d.assign(date_key=d.Date.dt.strftime("%Y%m%d").astype(int)) \
         .merge(dp, on=["Plan", "Plan_Type"]).merge(dy, on="Payment_Method")
    f = f[["Customer_ID", "plan_key", "pay_key", "date_key", "Subscription_Quantity", "Gross_Revenue",
           "Discount", "Net_Revenue", "Subscription_Status", "Tickets", "Usage_Hours"]]
    f.columns = ["customer_id", "plan_key", "pay_key", "date_key", "quantity", "gross", "discount",
                 "net", "status", "tickets", "usage_hours"]
    dp.columns = ["plan_key", "plan", "plan_type"]
    dy.columns = ["pay_key", "method"]
    for name, t in [("dim_customer", dc), ("dim_plan", dp), ("dim_payment", dy), ("dim_date", dd), ("fact_sales", f)]:
        t.to_sql(name, con, if_exists="replace", index=False)
    con.close()

@st.cache_resource(show_spinner="Running ETL and building the warehouse…")
def load_base():
    raw = generate()
    if os.path.exists(CSV):  # optional: your original file, committed once in the repo
        real = enrich(read_any(CSV))
        raw = pd.concat([raw, real], ignore_index=True)
    d = prepare(raw)
    build_dw(d)
    return d

# ============================================================ 2. MINING
def make_pre():
    return ColumnTransformer([("n", StandardScaler(), NUM), ("c", OneHotEncoder(handle_unknown="ignore"), CAT)])

@st.cache_resource(show_spinner="Training and validating models…")
def train(d):
    X, y = d[NUM + CAT], d.Churn
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, stratify=y, random_state=42)
    models = {"Logistic Regression": LogisticRegression(max_iter=1000),
              "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
              "Random Forest": RandomForestClassifier(300, random_state=42, n_jobs=-1),
              "Gradient Boosting": GradientBoostingClassifier(random_state=42)}
    cv, out = StratifiedKFold(5, shuffle=True, random_state=42), {}
    for n, m in models.items():
        pipe = Pipeline([("pre", make_pre()), ("m", m)])
        cvauc = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="roc_auc").mean()
        pipe.fit(Xtr, ytr)
        p = pipe.predict_proba(Xte)[:, 1]; pr = (p >= .5).astype(int)
        out[n] = dict(pipe=pipe, p=p, pr=pr, row={
            "Model": n, "Accuracy": accuracy_score(yte, pr), "Precision": precision_score(yte, pr),
            "Recall": recall_score(yte, pr), "F1": f1_score(yte, pr),
            "ROC-AUC": roc_auc_score(yte, p), "CV AUC (5-fold)": cvauc})
    return out, yte.values

def rfm_table(d):
    snap = d.Date.max() + pd.Timedelta(days=1)
    g = d.groupby("Customer_ID").agg(Recency=("Date", lambda s: (snap - s.max()).days),
        Frequency=("Date", "count"), Monetary=("Net_Revenue", "sum"), Churn=("Churn", "mean"),
        First=("Date", "min"))
    g["R"] = pd.qcut(g.Recency.rank(method="first"), 4, labels=[4, 3, 2, 1]).astype(int)
    g["F"] = pd.qcut(g.Frequency.rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    g["M"] = pd.qcut(g.Monetary.rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    s = g.R + g.F + g.M
    g["Segment"] = np.select([s >= 10, s >= 8, s >= 6], ["Champions", "Loyal", "Potential"], "At risk")
    years = ((snap - g.First).dt.days / 365).clip(lower=.25)
    life = (1 / g.Churn.clip(lower=.1)).clip(upper=5)
    g["CLV"] = (g.Monetary / g.Frequency) * (g.Frequency / years) * life
    return g.drop(columns="First").reset_index()

# ============================================================ 3. UI
st.title("SubScope: subscription BI and data mining")
up = st.sidebar.file_uploader("Optional: use your own CSV for this session", type=["csv", "tsv", "txt"])
base = load_base()
df = prepare(read_any(up)) if up else base
st.sidebar.caption(f"{len(df):,} records loaded automatically. No upload needed.")
yrs = sorted(df.Date.dt.year.unique())
fy = st.sidebar.multiselect("Year", yrs, default=yrs)
fp = st.sidebar.multiselect("Plan", sorted(df.Plan.unique()), default=sorted(df.Plan.unique()))
fc = st.sidebar.multiselect("City", sorted(df.Location.unique()), default=sorted(df.Location.unique()))
f = df[df.Date.dt.year.isin(fy) & df.Plan.isin(fp) & df.Location.isin(fc)]
if f.empty:
    st.warning("No records match these filters."); st.stop()

tabs = st.tabs(["Overview", "Warehouse & SQL", "Segments & CLV", "Churn models", "Forecast", "Anomalies", "Export"])

with tabs[0]:
    k = st.columns(6)
    k[0].metric("Net revenue", inr(f.Net_Revenue.sum()))
    k[1].metric("Avg discount", f"{f.Discount.sum() / f.Gross_Revenue.sum():.1%}")
    k[2].metric("Subscriptions", f"{len(f):,}")
    k[3].metric("Customers", f"{f.Customer_ID.nunique():,}")
    k[4].metric("Churn rate", f"{f.Churn.mean():.1%}")
    k[5].metric("Revenue / sub", inr(f.Net_Revenue.mean()))
    m = f.groupby("Month", as_index=False).Net_Revenue.sum()
    st.plotly_chart(px.line(m, x="Month", y="Net_Revenue", markers=True, title="Net revenue by month"), use_container_width=True)
    a, b = st.columns(2)
    a.plotly_chart(px.bar(f.groupby(["Plan", "Plan_Type"], as_index=False).Net_Revenue.sum(), x="Plan", y="Net_Revenue",
                          color="Plan_Type", barmode="group", title="Revenue by plan and billing"), use_container_width=True)
    b.plotly_chart(px.bar(f.groupby("Location", as_index=False).Net_Revenue.sum().sort_values("Net_Revenue"),
                          x="Net_Revenue", y="Location", orientation="h", title="Revenue by city"), use_container_width=True)
    a.plotly_chart(px.pie(f, names="Payment_Method", values="Net_Revenue", hole=.45, title="Payment method share"), use_container_width=True)
    c = f.groupby(["Plan", "Payment_Method"]).Churn.mean().unstack()
    b.plotly_chart(px.imshow(c, text_auto=".0%", color_continuous_scale="Reds", title="Churn rate: plan × payment"), use_container_width=True)

with tabs[1]:
    st.markdown("**Star schema:** `fact_sales` joins `dim_customer`, `dim_plan`, `dim_payment`, `dim_date`. Built by the ETL step on first launch (extract → clean → load into SQLite).")
    st.code("dim_customer(customer_id, name, city, country)\ndim_plan(plan_key, plan, plan_type)\ndim_payment(pay_key, method)\n"
            "dim_date(date_key, date, year, quarter, month)\nfact_sales(customer_id, plan_key, pay_key, date_key, quantity, gross, discount, net, status, tickets, usage_hours)")
    Q = {"Revenue by year and plan": "SELECT d.year, p.plan, p.plan_type, ROUND(SUM(f.net)) AS revenue FROM fact_sales f JOIN dim_plan p USING(plan_key) JOIN dim_date d USING(date_key) GROUP BY 1,2,3 ORDER BY 1, revenue DESC",
         "Top cities": "SELECT c.city, ROUND(SUM(f.net)) AS revenue, COUNT(*) AS subs FROM fact_sales f JOIN dim_customer c USING(customer_id) GROUP BY 1 ORDER BY 2 DESC",
         "Churn by payment method": "SELECT y.method, ROUND(AVG(f.status='Inactive'),3) AS churn_rate, COUNT(*) AS n FROM fact_sales f JOIN dim_payment y USING(pay_key) GROUP BY 1 ORDER BY 2 DESC",
         "Quarterly revenue (OLAP roll-up)": "SELECT d.year, d.quarter, ROUND(SUM(f.net)) AS revenue FROM fact_sales f JOIN dim_date d USING(date_key) GROUP BY 1,2 ORDER BY 1,2"}
    pick = st.selectbox("Preset OLAP query", list(Q))
    sql = st.text_area("SQL (read-only)", Q[pick], height=110)
    if st.button("Run query"):
        if not sql.strip().lower().startswith("select"):
            st.error("Only SELECT queries are allowed.")
        else:
            try:
                con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
                st.dataframe(pd.read_sql(sql, con), use_container_width=True); con.close()
            except Exception as e:
                st.error(f"Query failed: {e}")

with tabs[2]:
    r = rfm_table(df)
    st.subheader("RFM segmentation and customer lifetime value")
    seg = r.groupby("Segment").agg(Customers=("Customer_ID", "count"), Avg_Recency=("Recency", "mean"),
          Avg_Frequency=("Frequency", "mean"), Avg_Spend=("Monetary", "mean"), Avg_CLV=("CLV", "mean")).round(0).reset_index()
    a, b = st.columns([1, 1])
    a.dataframe(seg, use_container_width=True)
    b.plotly_chart(px.bar(seg, x="Segment", y="Avg_CLV", title="Average CLV by RFM segment"), use_container_width=True)
    st.subheader("K-means vs hierarchical clustering")
    Z = StandardScaler().fit_transform(np.log1p(r[["Recency", "Frequency", "Monetary"]]))
    ks = range(2, 9)
    km = {k: KMeans(k, n_init=10, random_state=42).fit(Z) for k in ks}
    sil = {k: silhouette_score(Z, km[k].labels_) for k in ks}
    e = go.Figure(go.Scatter(x=list(ks), y=[km[k].inertia_ for k in ks], mode="lines+markers"))
    e.update_layout(title="Elbow curve (inertia)", xaxis_title="k")
    s = go.Figure(go.Scatter(x=list(ks), y=list(sil.values()), mode="lines+markers"))
    s.update_layout(title="Silhouette score", xaxis_title="k")
    a, b = st.columns(2); a.plotly_chart(e, use_container_width=True); b.plotly_chart(s, use_container_width=True)
    kk = st.slider("Number of clusters", 2, 8, max(sil, key=sil.get))
    hc = AgglomerativeClustering(kk).fit_predict(Z)
    st.write(f"Silhouette at k={kk}: **K-means {sil[kk]:.3f}** vs **Hierarchical {silhouette_score(Z, hc):.3f}**")
    P = PCA(2).fit_transform(Z)
    st.plotly_chart(px.scatter(x=P[:, 0], y=P[:, 1], color=km[kk].labels_.astype(str), labels={"x": "PC1", "y": "PC2", "color": "Cluster"},
                               title="Customer clusters (PCA view)"), use_container_width=True)

with tabs[3]:
    res, yte = train(df)
    tab = pd.DataFrame([v["row"] for v in res.values()]).set_index("Model")
    st.subheader("Model comparison (25% hold-out test set)")
    st.dataframe(tab.style.format("{:.3f}").highlight_max(axis=0, color="#cfe8e4"), use_container_width=True)
    best = tab["ROC-AUC"].idxmax()
    st.success(f"Best model by ROC-AUC: {best}")
    a, b = st.columns(2)
    roc = go.Figure()
    for n, v in res.items():
        fpr, tpr, _ = roc_curve(yte, v["p"]); roc.add_scatter(x=fpr, y=tpr, name=n)
    roc.add_scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash"), name="Chance")
    roc.update_layout(title="ROC curves", xaxis_title="False positive rate", yaxis_title="True positive rate")
    a.plotly_chart(roc, use_container_width=True)
    cm = confusion_matrix(yte, res[best]["pr"])
    b.plotly_chart(px.imshow(cm, text_auto=True, x=["Active", "Inactive"], y=["Active", "Inactive"],
                             labels=dict(x="Predicted", y="Actual"), title=f"Confusion matrix: {best}"), use_container_width=True)
    rf = res["Random Forest"]["pipe"]
    imp = pd.Series(rf.named_steps["m"].feature_importances_, rf.named_steps["pre"].get_feature_names_out())
    imp.index = [i.split("__")[1] for i in imp.index]
    top = imp.groupby(lambda x: next((c for c in CAT + NUM if x.startswith(c)), x)).sum().sort_values()
    st.plotly_chart(px.bar(top, orientation="h", title="Churn drivers (Random Forest importance)", labels={"value": "Importance", "index": "Feature"}), use_container_width=True)
    st.subheader("Pattern mining: segments that churn more than average")
    pat = df.groupby(["Plan", "Plan_Type", "Payment_Method"]).Churn.agg(["mean", "count"]).reset_index()
    pat["lift"] = pat["mean"] / df.Churn.mean()
    st.dataframe(pat[pat["count"] >= 30].sort_values("lift", ascending=False).head(8)
                 .rename(columns={"mean": "churn_rate", "count": "records"}).round(3), use_container_width=True)
    st.subheader("Predict churn risk for a subscription")
    c = st.columns(5)
    row = pd.DataFrame([{"Plan": c[0].selectbox("Plan", sorted(df.Plan.unique())),
        "Plan_Type": c[1].selectbox("Billing", sorted(df.Plan_Type.unique())),
        "Payment_Method": c[2].selectbox("Payment", sorted(df.Payment_Method.unique())),
        "Location": c[3].selectbox("City", sorted(df.Location.unique())),
        "Subscription_Quantity": c[4].slider("Quantity", 1, 4, 2),
        "Discount_Pct": c[0].slider("Discount", 0.0, 0.3, 0.1, 0.05),
        "Tickets": c[1].slider("Support tickets", 0, 10, 1),
        "Usage_Hours": c[2].slider("Usage hours / month", 1, 80, 30),
        "Tenure_Months": c[3].slider("Tenure (months)", 0, 36, 6)}])
    pr = res[best]["pipe"].predict_proba(row[NUM + CAT])[0, 1]
    st.metric("Churn probability", f"{pr:.0%}", "High risk" if pr >= .5 else "Low risk", delta_color="inverse")

with tabs[4]:
    mm = df.groupby("Month").Net_Revenue.sum().asfreq("MS", fill_value=0)
    mm = mm.iloc[:-1] if len(mm) > 6 else mm  # drop possibly partial last month
    h = st.slider("Months to forecast", 3, 12, 6)
    fit = ExponentialSmoothing(mm, trend="add", damped_trend=True, initialization_method="estimated").fit()
    fcst = fit.forecast(h); sd = (mm - fit.fittedvalues).std()
    ci = 1.96 * sd * np.sqrt(np.arange(1, h + 1))
    g = go.Figure()
    g.add_scatter(x=mm.index, y=mm.values, name="Actual")
    g.add_scatter(x=fcst.index, y=fcst.values, name="Forecast", line=dict(dash="dash"))
    g.add_scatter(x=list(fcst.index) + list(fcst.index[::-1]), y=list(fcst.values + ci) + list((fcst.values - ci)[::-1]),
                  fill="toself", opacity=.2, name="95% interval", line=dict(width=0))
    g.update_layout(title="Monthly net revenue forecast (Holt-Winters, damped trend)")
    st.plotly_chart(g, use_container_width=True)
    mape = (abs(mm - fit.fittedvalues) / mm.replace(0, np.nan)).mean()
    st.caption(f"In-sample MAPE: {mape:.1%}. Next {h} months expected total: {inr(fcst.sum())}.")

with tabs[5]:
    cont = st.slider("Expected share of anomalies", 0.005, 0.10, 0.02, 0.005)
    cols = ["Net_Revenue", "Discount_Pct", "Subscription_Quantity", "Usage_Hours"]
    iso = IsolationForest(contamination=cont, random_state=42).fit(df[cols])
    an = df.assign(Anomaly=np.where(iso.predict(df[cols]) == -1, "Anomaly", "Normal"))
    st.plotly_chart(px.scatter(an, x="Discount_Pct", y="Net_Revenue", color="Anomaly", opacity=.7,
                               title="Isolation Forest: unusual discount/revenue combinations"), use_container_width=True)
    st.dataframe(an[an.Anomaly == "Anomaly"][["Customer_ID", "Date", "Plan", "Plan_Type", "Gross_Revenue", "Discount", "Net_Revenue", "Usage_Hours"]]
                 .sort_values("Net_Revenue", ascending=False).head(25), use_container_width=True)

with tabs[6]:
    st.download_button("Download cleaned dataset (CSV)", df.to_csv(index=False), "subscriptions_clean.csv")
    st.download_button("Download RFM + CLV table (CSV)", rfm_table(df).to_csv(index=False), "rfm_clv.csv")
    st.download_button("Download model comparison (CSV)", pd.DataFrame([v["row"] for v in train(df)[0].values()]).to_csv(index=False), "model_comparison.csv")
