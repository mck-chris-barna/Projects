import streamlit as st
import pandas as pd
import io

# Your local module that already does the heavy lifting
import sentiment_from_csv  # make sure this file/module is in the repo

st.set_page_config(page_title="CSV Sentiment Analyzer", page_icon="🧪", layout="wide")
st.title("CSV Sentiment Analyzer")
st.caption("Upload a CSV ➜ pick the text column ➜ download results.")

st.markdown("### 1) Upload your CSV")
uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded:
    # Read CSV
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    st.success("File loaded.")
    with st.expander("Preview (first 50 rows)"):
        st.dataframe(df.head(50), use_container_width=True)

    # Pick the text column (default to 'Text' if present)
    text_cols = [c for c in df.columns if df[c].dtype == object]
    default_ix = text_cols.index("Text") if "Text" in text_cols else 0 if text_cols else None

    if not text_cols:
        st.error("No text-like columns found. Add a column with free text (e.g., 'Text').")
        st.stop()

    st.markdown("### 2) Choose the text column")
    text_col = st.selectbox("Column containing the text to analyze", text_cols, index=default_ix)

    # Run analysis
    st.markdown("### 3) Run sentiment analysis")
    if st.button("Analyze"):
        with st.spinner("Running sentiment…"):
            # Ensure the text column is string and non-null
            df[text_col] = df[text_col].fillna("").astype(str)
            results = sentiment_from_csv.run_sentiment_analysis(df, text_column=text_col)

        st.success("Done!")
        st.markdown("#### Results (first 100 rows)")
        st.dataframe(results.head(100), use_container_width=True)

        # Quick summary if your function returns columns like net/pos/neg/etc.
        num_rows = len(results)
        cols = [c for c in results.columns if "sentiment" in c.lower()]
        st.caption(f"Processed {num_rows:,} rows. Sentiment columns detected: {', '.join(cols) or '—'}")

        # Download button
        csv_bytes = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download results as CSV",
            data=csv_bytes,
            file_name="sentiment_outputs.csv",
            mime="text/csv",
        )

else:
    st.info("Upload a CSV to get started.")
