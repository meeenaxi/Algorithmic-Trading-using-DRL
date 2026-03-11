import streamlit as st
import pandas as pd
from serve_model import get_allocation_for_latest_row
from evaluate import run_clean_evaluation
from config import TICKERS

st.title("📈 RL-Based Stock Allocation System")

uploaded = st.file_uploader("Upload stock price CSV for prediction")

if uploaded:
    df = pd.read_csv(uploaded, index_col='Date', parse_dates=True)

    st.write("### Preview:")
    st.write(df.tail())

    st.write("### Calculating Recommended Allocation...")

    weights = get_allocation_for_latest_row(df)

    allocation = pd.DataFrame({"Ticker": TICKERS, "Weight": weights})
    st.write(allocation)

    st.write("### Pie Chart")
    st.plotly_chart(
        allocation.set_index("Ticker").plot.pie(y="Weight", use_container_width=True)
    )

st.write("---")
if st.button("Run Evaluation (Backtest)"):
    run_clean_evaluation()
    st.success("Evaluation complete. Check training_outputs folder!")
