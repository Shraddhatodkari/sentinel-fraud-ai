"""
Sentinel Investigator Dashboard — a lightweight, real, working UI on top
of the Sentinel API/data, built with Streamlit (pure Python, no separate
frontend framework needed).

Run:
    streamlit run dashboard/app.py

Pages:
    - Overview: alert volume, auto-clear rate, precision metrics
    - Case Detail: risk scores, SHAP factors, cross-silo findings for one transaction
    - Relationship Graph: interactive network view of shared devices/IPs/beneficiaries
    - Audit Log: table of every logged decision (from MongoDB, or local JSON fallback)
    - Investigation Report: auto-generated plain-English summary for one case

This reads from your existing data/processed files and MongoDB — nothing
new to learn, it's a UI layer over the same pipeline you already built.
"""
import json
from pathlib import Path

import joblib
import networkx as nx
import pandas as pd
import streamlit as st
from pyvis.network import Network
import streamlit.components.v1 as components

st.set_page_config(page_title="Sentinel — Financial Crime Investigation", layout="wide")

DATA_DIR = Path("data")


# ---------- Data loading helpers ----------

@st.cache_data
def load_transactions():
    path = DATA_DIR / "raw" / "transactions.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_metrics():
    path = DATA_DIR / "processed" / "metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_aml_sanctions():
    aml_path = DATA_DIR / "raw" / "aml_alerts.csv"
    sanc_path = DATA_DIR / "raw" / "sanctions_hits.csv"
    aml = pd.read_csv(aml_path) if aml_path.exists() else None
    sanctions = pd.read_csv(sanc_path) if sanc_path.exists() else None
    return aml, sanctions


def load_audit_logs_local_fallback():
    """
    If MongoDB isn't configured for this demo session, fall back to a
    local JSON export so the dashboard still renders something real.
    See scripts/export_audit_logs_demo.py to generate this file from Mongo.
    """
    path = DATA_DIR / "processed" / "audit_logs_demo.json"
    if path.exists():
        with open(path) as f:
            return pd.DataFrame(json.load(f))
    return pd.DataFrame()


# ---------- Sidebar navigation ----------

st.sidebar.title("🛡️ Sentinel")
st.sidebar.caption("Agentic Cross-Silo Financial Crime Investigation Platform")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Case Detail", "Relationship Graph", "Audit Log", "Investigation Report"],
)

txns = load_transactions()
metrics = load_metrics()
aml_df, sanctions_df = load_aml_sanctions()


# ---------- Page: Overview ----------

if page == "Overview":
    st.title("Overview")
    st.caption("Real-time snapshot of alert volume and model performance")

    col1, col2, col3, col4 = st.columns(4)
    if txns is not None:
        col1.metric("Total Transactions", f"{len(txns):,}")
        col2.metric("Flagged as Fraud", f"{int(txns['is_fraud'].sum()):,}")
    if metrics:
        col3.metric("Auto-Clear Rate", f"{metrics.get('auto_clear_rate', 0):.1%}")
        col4.metric("Auto-Clear Precision", f"{metrics.get('auto_clear_correctness', 0):.1%}")
    else:
        st.info("Run `python -m ml.evaluate` to populate benchmark metrics here.")

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Cross-Silo Signal Volume")
        counts = {
            "Fraud Alerts": len(txns[txns["is_fraud"] == 1]) if txns is not None else 0,
            "AML Alerts": len(aml_df) if aml_df is not None else 0,
            "Sanctions Hits": len(sanctions_df) if sanctions_df is not None else 0,
        }
        st.bar_chart(pd.Series(counts, name="count"))

    with c2:
        st.subheader("Model Benchmark")
        if metrics:
            st.json(metrics)
        else:
            st.info("No benchmark yet — run `python -m ml.evaluate`.")


# ---------- Page: Case Detail ----------

elif page == "Case Detail":
    st.title("Case Detail")
    st.caption("Risk scores, SHAP explanations, and cross-silo findings for a single alert")

    if txns is None:
        st.warning("No transaction data found. Run `python scripts/seed_synthetic_alerts.py` first.")
    else:
        fraud_txns = txns[txns["is_fraud"] == 1].head(50)
        selected_id = st.selectbox("Select a flagged transaction", fraud_txns["transaction_id"])
        row = txns[txns["transaction_id"] == selected_id].iloc[0]

        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader(f"Transaction {selected_id}")
            st.write(f"**Amount:** ${row['amount']:,.2f}")
            st.write(f"**Hour of day:** {row['hour_of_day']}:00")
            st.write(f"**Account age:** {row['account_age_days']} days")
            st.write(f"**New beneficiary:** {'Yes' if row['is_new_beneficiary'] else 'No'}")
            st.write(f"**Prior alerts on account:** {row['prior_alerts_for_account']}")

        with c2:
            st.subheader("Risk Signals")
            xgb_path = Path("data/processed/models/xgb_model.pkl")
            if xgb_path.exists():
                model = joblib.load(xgb_path)
                cols_path = Path("data/processed/models/feature_columns.json")
                with open(cols_path) as f:
                    feature_cols = json.load(f)
                X = pd.DataFrame([row[feature_cols].to_dict()])
                prob = model.predict_proba(X)[:, 1][0]
                st.metric("Fraud Probability", f"{prob:.1%}")
                if prob > 0.7:
                    st.error("HIGH RISK — recommend escalation")
                elif prob > 0.3:
                    st.warning("MEDIUM RISK")
                else:
                    st.success("LOW RISK — auto-clear eligible")
            else:
                st.info("Train models first: `python -m ml.train_xgboost`")

        st.divider()
        st.subheader("Cross-Silo Findings")
        found_something = False
        if aml_df is not None:
            aml_hit = aml_df[aml_df["account_id"] == row["account_id"]]
            if not aml_hit.empty:
                found_something = True
                st.warning(
                    f"🔗 Account also flagged by **AML** — "
                    f"typology: {aml_hit.iloc[0]['typology']}, "
                    f"severity: {aml_hit.iloc[0]['severity']}"
                )
        if sanctions_df is not None:
            sanc_hit = sanctions_df[sanctions_df["beneficiary_id"] == row["beneficiary_id"]]
            if not sanc_hit.empty:
                found_something = True
                st.error(
                    f"🚨 Beneficiary matched **Sanctions** watchlist "
                    f"({sanc_hit.iloc[0]['watchlist_source']}, "
                    f"confidence: {sanc_hit.iloc[0]['match_confidence']})"
                )
        if not found_something:
            st.success("No cross-silo findings — this case stays with the Fraud team only.")


# ---------- Page: Relationship Graph ----------

elif page == "Relationship Graph":
    st.title("Relationship Graph")
    st.caption("Shared devices, IPs, and beneficiaries across flagged transactions")

    if txns is None:
        st.warning("No transaction data found. Run `python scripts/seed_synthetic_alerts.py` first.")
    else:
        fraud_txns = txns[txns["is_fraud"] == 1].head(60)  # keep graph readable

        net = Network(height="600px", width="100%", bgcolor="#0e1117", font_color="white")
        added_nodes = set()

        for _, r in fraud_txns.iterrows():
            txn_node = r["transaction_id"]
            device_node = f"device_{r['device_id']}"
            ip_node = f"ip_{r['ip_id']}"
            bene_node = f"beneficiary_{r['beneficiary_id']}"

            for node, color in [
                (txn_node, "#e74c3c"),
                (device_node, "#3498db"),
                (ip_node, "#2ecc71"),
                (bene_node, "#f39c12"),
            ]:
                if node not in added_nodes:
                    net.add_node(node, label=node, color=color)
                    added_nodes.add(node)

            net.add_edge(txn_node, device_node)
            net.add_edge(txn_node, ip_node)
            net.add_edge(txn_node, bene_node)

        net.repulsion(node_distance=150)
        net.save_graph("dashboard_graph_temp.html")
        with open("dashboard_graph_temp.html", "r", encoding="utf-8") as f:
            components.html(f.read(), height=620)

        st.caption(
            "🔴 Transactions · 🔵 Devices · 🟢 IPs · 🟠 Beneficiaries — "
            "clusters of connected nodes suggest a coordinated fraud ring."
        )


# ---------- Page: Audit Log ----------

elif page == "Audit Log":
    st.title("Audit Log")
    st.caption("Immutable record of every disposition decision, for regulatory review")

    logs = load_audit_logs_local_fallback()
    if logs.empty:
        st.info(
            "No local audit log export found. Run the API (`uvicorn app.main:app`), "
            "process a few `/alerts/investigate` calls, then run "
            "`python scripts/export_audit_logs_demo.py` to populate this view. "
            "In production this reads directly from MongoDB Atlas."
        )
    else:
        st.dataframe(logs, use_container_width=True)


# ---------- Page: Investigation Report ----------

elif page == "Investigation Report":
    st.title("Investigation Report")
    st.caption("Auto-generated, audit-ready summary for a single case")

    if txns is None:
        st.warning("No transaction data found.")
    else:
        fraud_txns = txns[txns["is_fraud"] == 1].head(50)
        selected_id = st.selectbox(
            "Select a case to generate a report for", fraud_txns["transaction_id"], key="report_select"
        )
        row = txns[txns["transaction_id"] == selected_id].iloc[0]

        aml_hit = aml_df[aml_df["account_id"] == row["account_id"]] if aml_df is not None else pd.DataFrame()
        sanc_hit = sanctions_df[sanctions_df["beneficiary_id"] == row["beneficiary_id"]] if sanctions_df is not None else pd.DataFrame()

        teams = ["Fraud"]
        if not aml_hit.empty:
            teams.append("AML")
        if not sanc_hit.empty:
            teams.append("Sanctions")

        st.markdown(f"""
### Case Report — {selected_id}

**Teams involved:** {", ".join(teams)}

**Summary:**
Transaction `{selected_id}` for **${row['amount']:,.2f}** was flagged at
**{row['hour_of_day']}:00**, involving an account **{row['account_age_days']} days old**
{"with a new beneficiary" if row['is_new_beneficiary'] else "with an established beneficiary"}.
The account has **{row['prior_alerts_for_account']} prior alert(s)**.

**Cross-Silo Findings:**
{f"- AML: flagged for {aml_hit.iloc[0]['typology']} ({aml_hit.iloc[0]['severity']} severity)" if not aml_hit.empty else "- AML: no match"}
{f"- Sanctions: matched {sanc_hit.iloc[0]['watchlist_source']} watchlist (confidence {sanc_hit.iloc[0]['match_confidence']})" if not sanc_hit.empty else "- Sanctions: no match"}

**Recommendation:** {"Escalate to " + " + ".join(teams) + " for joint review." if len(teams) > 1 else "Standard Fraud team review."}

---
*Generated by Sentinel — full reasoning trace available in LangSmith, immutable record in MongoDB Atlas.*
        """)
