import streamlit as st
import requests
import pandas as pd
from datetime import datetime

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="KYC Dedup Agent", page_icon="🏦", layout="wide")

# ---------------------------
# API Base
# ---------------------------
API_BASE = "http://localhost:8000"
HEADERS = {"Content-Type": "application/json"}

def api_call(method, endpoint, data=None, params=None):
    url = f"{API_BASE}{endpoint}"
    try:
        if method.upper() == "GET":
            resp = requests.get(url, params=params, headers=HEADERS, timeout=10)
        elif method.upper() == "POST":
            resp = requests.post(url, json=data, headers=HEADERS, timeout=10)
        elif method.upper() == "DELETE":
            resp = requests.delete(url, headers=HEADERS, timeout=10)
        return resp
    except Exception as e:
        st.error(f"Connection error: {e}")
        return None

# ---------------------------
# Top Navigation
# ---------------------------
st.title("🏦 KYC Dedup Agent")
st.markdown("---")

nav = st.selectbox(
    "Choose a section",
    [
        "🔍 KYC Dedup Check",
        "💰 Apply for Loan",
        "👤 Customer Profile",
        "📋 All Customers",
        "🚫 Blacklist Management",
        "📜 Dedup Audit Log"
    ]
)

st.markdown("---")

# ---------------------------
# PAGE: KYC Dedup Check
# ---------------------------
if nav == "🔍 KYC Dedup Check":
    st.header("🔍 KYC Deduplication Check")
    with st.form("dedup_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Full Name *")
            dob = st.date_input(
                "Date of Birth *",
                value=datetime.now(),
                min_value=datetime(1960, 1, 1),
                max_value=datetime.now()
            )
            pan = st.text_input("PAN *", max_chars=10).upper()
        with col2:
            phone = st.text_input("Phone Number *", max_chars=10)
            aadhaar = st.text_input("Aadhaar Number *", max_chars=12)
            address = st.text_area("Address *")
        submitted = st.form_submit_button("Check Duplicate", type="primary")
        if submitted:
            payload = {
                "wakes_on": "kyc.dedup_requested",
                "reads": {
                    "name": name,
                    "dob": dob.strftime("%Y-%m-%d"),
                    "pan": pan,
                    "phone": phone,
                    "aadhaar_number": aadhaar,
                    "address": address
                }
            }
            resp = api_call("POST", "/api/v1/kyc/dedup", data=payload)
            if resp and resp.status_code == 200:
                st.success("✅ Check completed")
                st.json(resp.json())
            elif resp:
                st.error(f"❌ Error {resp.status_code}: {resp.text}")

# ---------------------------
# PAGE: Apply for Loan (with Auto‑generated Account Number)
# ---------------------------
elif nav == "💰 Apply for Loan":
    st.header("💰 Apply for a New Loan")
    
    with st.form("loan_form"):
        st.subheader("👤 Customer Information")
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Full Name *")
            dob = st.date_input(
                "Date of Birth *",
                value=datetime.now(),
                min_value=datetime(1960, 1, 1),
                max_value=datetime.now()
            )
            aadhaar = st.text_input("Aadhaar Number *", max_chars=12)
            pan = st.text_input("PAN *", max_chars=10).upper()
        with col2:
            mobile = st.text_input("Mobile Number *", max_chars=10)
            email = st.text_input("Email (optional)")
            address = st.text_area("Address *")
        
        st.divider()
        st.subheader("💰 Loan Details")
        
        col3, col4 = st.columns(2)
        with col3:
            loan_type = st.selectbox("Loan Type *", ["Home Loan", "Personal Loan", "Car Loan", "Business Loan", "Education Loan", "Gold Loan"])
            loan_amount = st.number_input("Loan Amount (₹) *", min_value=1000, step=1000, help="Enter the amount you want to borrow")
        with col4:
            interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=30.0, step=0.1)
            term_months = st.number_input("Term (months)", min_value=6, max_value=360, step=6)
        
        # Clear note about auto‑generated account number
        st.info("📌 **Loan Account Number** will be **auto‑generated** by the system (e.g., LN0001, LN0002, …). You don't need to enter it.")

        submitted = st.form_submit_button("Apply Loan", type="primary")
        if submitted:
            if not all([name, dob, aadhaar, pan, mobile, address, loan_type, loan_amount]):
                st.warning("Please fill in all required fields.")
            else:
                payload = {
                    "name": name,
                    "dob": dob.strftime("%Y-%m-%d"),
                    "aadhaar_number": aadhaar,
                    "pan": pan,
                    "mobile_number": mobile,
                    "email": email or None,
                    "address": address,
                    "loan_type": loan_type,
                    "loan_amount": loan_amount,
                    "interest_rate": interest_rate or None,
                    "loan_term_months": term_months or None
                }
                resp = api_call("POST", "/api/v1/loan/apply", data=payload)
                if resp and resp.status_code == 200:
                    st.success("✅ Loan applied successfully!")
                    st.json(resp.json())
                elif resp:
                    st.error(f"❌ Error {resp.status_code}: {resp.text}")

# ---------------------------
# PAGE: Customer Profile
# ---------------------------
elif nav == "👤 Customer Profile":
    st.header("👤 Customer Profile")
    identifier = st.text_input("Enter Aadhaar / Mobile / PAN")
    if st.button("Fetch Profile", type="primary"):
        resp = api_call("GET", f"/api/v1/customer/{identifier}")
        if resp and resp.status_code == 200:
            data = resp.json()
            if data.get("status") == "SUCCESS":
                st.success(f"Customer: {data['customer']['name']}")
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Details")
                    st.json(data["customer"])
                with col2:
                    st.subheader("Loans")
                    if data["loans"]:
                        st.dataframe(pd.DataFrame(data["loans"]))
                    else:
                        st.info("No loans")
                st.metric("Total Loans", data["total_loans"])
            else:
                st.warning("Customer not found")
        elif resp and resp.status_code == 404:
            st.warning("Customer not found")

# ---------------------------
# PAGE: All Customers
# ---------------------------
elif nav == "📋 All Customers":
    st.header("📋 All Customers")
    resp = api_call("GET", "/api/v1/customers")
    if resp and resp.status_code == 200:
        customers = resp.json().get("data", [])
        if customers:
            st.dataframe(pd.DataFrame(customers))
        else:
            st.info("No customers")
    else:
        st.error("Could not fetch customers. Ensure the endpoint exists.")

# ---------------------------
# PAGE: Blacklist Management
# ---------------------------
elif nav == "🚫 Blacklist Management":
    st.header("🚫 Blacklist Management")

    with st.expander("📋 View Current Blacklist", expanded=True):
        resp = api_call("GET", "/api/v1/blacklist")
        if resp and resp.status_code == 200:
            data = resp.json().get("data", [])
            if data:
                st.dataframe(pd.DataFrame(data))
            else:
                st.info("No blacklist records.")
        else:
            st.error("Failed to fetch blacklist")

    with st.expander("➕ Add to Blacklist"):
        with st.form("add_blacklist_form"):
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input("Name *")
                dob = st.date_input(
                    "DOB *",
                    value=datetime.now(),
                    min_value=datetime(1960, 1, 1),
                    max_value=datetime.now()
                )
                pan = st.text_input("PAN", max_chars=10).upper()
                aadhaar = st.text_input("Aadhaar", max_chars=12)
            with col2:
                mobile = st.text_input("Mobile", max_chars=10)
                reason = st.text_area("Reason *")
                source = st.text_input("Source (e.g., Internal, RBI)")
            submitted = st.form_submit_button("Add to Blacklist", type="primary")
            if submitted:
                if not name or not dob or not reason:
                    st.warning("Name, DOB, and Reason are required.")
                else:
                    payload = {
                        "name": name,
                        "dob": dob.strftime("%Y-%m-%d"),
                        "pan": pan or None,
                        "aadhaar_number": aadhaar or None,
                        "mobile_number": mobile or None,
                        "reason": reason,
                        "source": source or None
                    }
                    resp = api_call("POST", "/api/v1/blacklist/add", data=payload)
                    if resp and resp.status_code == 200:
                        st.success("✅ Blacklist record added.")
                    else:
                        st.error(f"Failed: {resp.text if resp else 'Unknown error'}")

    with st.expander("🗑️ Remove from Blacklist"):
        blacklist_id = st.number_input("Blacklist ID to remove", min_value=1, step=1)
        if st.button("Remove", type="primary"):
            resp = api_call("DELETE", f"/api/v1/blacklist/remove/{blacklist_id}")
            if resp and resp.status_code == 200:
                st.success("✅ Removed.")
            elif resp and resp.status_code == 404:
                st.warning("Record not found.")
            else:
                st.error("Failed to remove.")

# ---------------------------
# PAGE: Dedup Audit Log
# ---------------------------
elif nav == "📜 Dedup Audit Log":
    st.header("📜 Deduplication Audit Log")
    limit = st.slider("Records to show", 10, 100, 50)
    resp = api_call("GET", f"/api/v1/dedup/results?limit={limit}")
    if resp and resp.status_code == 200:
        logs = resp.json().get("data", [])
        if logs:
            st.dataframe(pd.DataFrame(logs))
        else:
            st.info("No logs yet.")
    else:
        st.error("Could not fetch logs.")