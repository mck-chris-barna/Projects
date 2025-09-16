import streamlit as st
import pandas as pd
import io
import re
import uuid
from datetime import datetime, timezone

# Your local module that already does the heavy lifting
import sentiment_from_csv  # make sure this file/module is in the repo

# ---------------------------
# CONFIG
# ---------------------------
APP_TITLE = "CSV Sentiment Analyzer"
README_PATH = "README_app.md"   # <-- put your doc here in the repo root
EMAIL_REQUIRED_DOMAIN = None    # e.g., "acme.com" if you want to restrict; else None

st.set_page_config(page_title=APP_TITLE, page_icon="🧪", layout="wide")

# ---------------------------
# AUTH / LOGGING HELPERS
# ---------------------------
APP_PASSWORD = st.secrets.get("APP_PASSWORD", None)
SHEET_ID = st.secrets.get("SHEET_ID", None)
GCP_SA_INFO = st.secrets.get("gcp", None)

def _valid_email(e: str) -> bool:
    if not e:
        return False
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", e):
        return False
    if EMAIL_REQUIRED_DOMAIN and not e.lower().endswith("@" + EMAIL_REQUIRED_DOMAIN.lower()):
        return False
    return True

@st.cache_resource(show_spinner=False)
def _get_sheet():
    """Authorize and return the first worksheet of the Google Sheet (requires secrets)."""
    if not (SHEET_ID and GCP_SA_INFO):
        return None  # allow local dev without logging
    import gspread
    from google.oauth2.service_account import Credentials
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_info(GCP_SA_INFO, scopes=scopes)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(SHEET_ID)
    return sh.sheet1

def _log_email(email_val: str, status: str, details: str = ""):
    """
    status examples: "success", "bad_password", "invalid_email"
    """
    sheet = _get_sheet()
    if sheet is None:
        return  # silently skip if not configured
    try:
        sheet.append_row([
            datetime.now(timezone.utc).isoformat(),
            email_val or "",
            st.session_state.get("sid", ""),
            status,
            details
        ])
    except Exception as e:
        # Avoid breaking the app on logging errors
        st.toast(f"⚠️ Could not log email: {e}", icon="⚠️")

def _auth_gate():
    """Light-weight gate: capture email + shared password; log attempts."""
    if "sid" not in st.session_state:
        st.session_state["sid"] = str(uuid.uuid4())

    st.title(APP_TITLE)
    st.caption("This gate is not security; it exists to log access emails and usage.")

    with st.form("auth"):
        email = st.text_input("Email", placeholder="you@company.com")
        pw = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Enter")

    if submitted:
        if not _valid_email(email):
            st.error("Please enter a valid email.")
            _log_email(email, "invalid_email", "format/domain")
        elif APP_PASSWORD and pw != APP_PASSWORD:
            st.error("Incorrect password.")
            _log_email(email, "bad_password")
        else:
            _log_email(email, "success")
            st.success("Access granted.")
            st.session_state["authed"] = True

    return st.session_state.get("authed", False)

# ---------------------------
# README RENDERING
# ---------------------------
def render_readme():
    """Render README_app.md if present; otherwise nudge the user to add it."""
    st.divider()
    st.subheader("📘 Read Me")
    try:
        with open(README_PATH, "r", encoding="utf-8") as f:
            md = f.read()
        with st.expander("View / Hide", expanded=True):
            st.markdown(md, unsafe_allow_html=False)
    except FileNotFoundError:
        st.info(
            f"Add a Markdown file named `{README_PATH}` in your repo to show your documentation here."
        )

# ---------------------------
# APP START (AUTH FIRST)
# ---------------------------
authed = _auth_gate()
if not authed:
    st.stop()

# ---------------------------
# MAIN APP (YOUR ORIGINAL UI)
# ---------------------------
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

        # ---- README appears AFTER analysis output ----
        render_readme()

else:
    st.info("Upload a CSV to get started.")
