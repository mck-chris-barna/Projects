import streamlit as st
import pandas as pd
import re
import uuid
from datetime import datetime, timezone

# your local sentiment module
import sentiment_from_csv

# ---------------------------
# CONFIG
# ---------------------------
APP_TITLE = "CSV Sentiment Analyzer"
README_PATH = "README_app.md"
EMAIL_REQUIRED_DOMAIN = None  # set like "acme.com" if you want to restrict

st.set_page_config(page_title=APP_TITLE, page_icon="🧪", layout="wide")

# ---------------------------
# SECRETS CHECK
# ---------------------------
APP_PASSWORD = st.secrets.get("APP_PASSWORD", None)
SHEET_ID = st.secrets.get("SHEET_ID", None)
GCP_SA_INFO = st.secrets.get("gcp", None)

ok_pw = APP_PASSWORD is not None
ok_sheet = SHEET_ID is not None
ok_gcp = GCP_SA_INFO is not None
st.sidebar.markdown("**Secrets loaded:** " + ("✅" if (ok_pw and ok_sheet and ok_gcp) else "❌"))

# ---------------------------
# GOOGLE SHEETS HELPERS
# ---------------------------
@st.cache_resource(show_spinner=False)
def _get_sheet():
    import gspread
    from google.oauth2.service_account import Credentials
    scopes = ["https://www.googleapis.com/auth/spreadsheets",
              "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(GCP_SA_INFO, scopes=scopes)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(SHEET_ID)
    ws = sh.sheet1
    # add header if sheet empty
    if not ws.get_all_values():
        ws.append_row(["timestamp_utc","email","session_id","status","details"])
    return ws

def _log_email(email_val: str, status: str, details: str = ""):
    try:
        ws = _get_sheet()
        ws.append_row([
            datetime.now(timezone.utc).isoformat(),
            email_val or "",
            st.session_state.get("sid", ""),
            status,
            details
        ])
    except Exception as e:
        st.error("Google Sheets logging failed:")
        st.exception(e)

# sidebar test button
if st.sidebar.button("🔌 Test Google Sheets logging"):
    _log_email("healthcheck@example.com", "manual", "sidebar test")
    st.sidebar.success("Tried to append a test row to the Sheet.")

# ---------------------------
# AUTH GATE
# ---------------------------
def _valid_email(e: str) -> bool:
    if not e:
        return False
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", e):
        return False
    if EMAIL_REQUIRED_DOMAIN and not e.lower().endswith("@" + EMAIL_REQUIRED_DOMAIN.lower()):
        return False
    return True

def _auth_gate():
    if "sid" not in st.session_state:
        st.session_state["sid"] = str(uuid.uuid4())

    st.title(APP_TITLE)
    st.caption("This gate logs your email before allowing access.")

    with st.form("auth"):
        email = st.text_input("Email", placeholder="you@company.com")
        pw = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Enter")

    if submitted:
        if not _valid_email(email):
            st.error("Please enter a valid email.")
            _log_email(email, "invalid_email")
        elif APP_PASSWORD and pw != APP_PASSWORD:
            st.error("Incorrect password.")
            _log_email(email, "bad_password")
        else:
            _log_email(email, "success")
            st.success("Access granted.")
            st.session_state["authed"] = True

    return st.session_state.get("authed", False)

# ---------------------------
# README RENDER
# ---------------------------
def render_readme():
    st.divider()
    st.subheader("📘 Read Me")
    try:
        with open(README_PATH, "r", encoding="utf-8") as f:
            st.markdown(f.read())
    except FileNotFoundError:
        st.info(f"Add a file `{README_PATH}` to show documentation here.")

# ---------------------------
# MAIN APP
# ---------------------------
if not _auth_gate():
    st.stop()

st.markdown("### 1) Upload your CSV")
uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    st.success("File loaded.")
    with st.expander("Preview (first 50 rows)"):
        st.dataframe(df.head(50), width="stretch")

    text_cols = [c for c in df.columns if df[c].dtype == object]
    default_ix = text_cols.index("Text") if "Text" in text_cols else 0 if text_cols else None

    if not text_cols:
        st.error("No text-like columns found. Add a column with free text (e.g., 'Text').")
        st.stop()

    st.markdown("### 2) Choose the text column")
    text_col = st.selectbox("Column containing the text to analyze", text_cols, index=default_ix)

    st.markdown("### 3) Run sentiment analysis")
    if st.button("Analyze"):
        with st.spinner("Running sentiment…"):
            df[text_col] = df[text_col].fillna("").astype(str)
            results = sentiment_from_csv.run_sentiment_analysis(df, text_column=text_col)
            st.session_state["results_df"] = results

    if "results_df" in st.session_state:
        results = st.session_state["results_df"]
        st.success("Done!")
        st.dataframe(results.head(100), width="stretch")
        csv_bytes = results.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download results as CSV", csv_bytes, "sentiment_outputs.csv", "text/csv")
        render_readme()
else:
    st.info("Upload a CSV to get started.")
