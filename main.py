import os
import re
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from contextlib import contextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.fuzz import partial_ratio


# =============================================================================
# LOGGING WITH COLORS
# =============================================================================

class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# FASTAPI APP & CORS
# =============================================================================

app = FastAPI(
    title="KYC Deduplication & Loan Management System",
    description="Enhanced with exact match, conflict detection, auto loan terms",
    version="4.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


# =============================================================================
# DATABASE CONNECTION
# =============================================================================

db_pool = SimpleConnectionPool(
    minconn=1,
    maxconn=10,
    dsn=os.environ.get(
        "DATABASE_URL",
        "host=localhost dbname=postgres user=postgres password=Root port=5432"
    )
)

@contextmanager
def get_db_connection():
    conn = db_pool.getconn()
    try:
        yield conn
    finally:
        db_pool.putconn(conn)


# =============================================================================
# CONFIGURATION
# =============================================================================

class DedupeWeights:
    PAN = 0.35
    AADHAAR = 0.25
    NAME = 0.15
    DOB = 0.12
    MOBILE = 0.08
    ADDRESS = 0.05
    
    BLACKLIST_MULTIPLIER = 1.5
    FUZZY_NAME_THRESHOLD = 0.85
    ADDRESS_THRESHOLD = 0.70
    
    EXACT_MATCH = 1.0
    HIGH_CONFIDENCE = 0.85
    MEDIUM_CONFIDENCE = 0.70
    LOW_CONFIDENCE = 0.50
    WEAK_MATCH = 0.30


# ---- Auto Loan Terms ----
LOAN_TERMS = {
    "Home Loan": {"interest_rate": 8.5, "term_months": 240},
    "Personal Loan": {"interest_rate": 12.0, "term_months": 36},
    "Car Loan": {"interest_rate": 9.0, "term_months": 60},
    "Business Loan": {"interest_rate": 11.0, "term_months": 120},
    "Education Loan": {"interest_rate": 7.5, "term_months": 84},
    "Gold Loan": {"interest_rate": 10.0, "term_months": 12},
}

def get_loan_terms(loan_type: str) -> Tuple[float, int]:
    default = {"interest_rate": 10.0, "term_months": 36}
    terms = LOAN_TERMS.get(loan_type, default)
    return terms["interest_rate"], terms["term_months"]


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class ApplicantReads(BaseModel):
    name: str = Field(..., min_length=2, max_length=255)
    dob: str = Field(..., example="1992-05-12")
    pan: str = Field(..., min_length=10, max_length=10)
    phone: str = Field(..., min_length=10, max_length=15)
    aadhaar_number: str = Field(..., min_length=12, max_length=12)
    address: str = Field(..., min_length=5, max_length=500)
    
    @validator('dob')
    def validate_dob(cls, v):
        try:
            dob_date = datetime.strptime(v, '%Y-%m-%d')
            if dob_date > datetime.now():
                raise ValueError("Date of birth cannot be in future")
            return v
        except ValueError:
            raise ValueError("Invalid DOB format. Use YYYY-MM-DD")
    
    @validator('pan')
    def validate_pan(cls, v):
        v = v.upper().strip()
        pattern = r'^[A-Z]{5}[0-9]{4}[A-Z]{1}$'
        if not re.match(pattern, v):
            raise ValueError(f"Invalid PAN format: {v}")
        return v
    
    @validator('phone')
    def validate_phone(cls, v):
        cleaned = ''.join(filter(str.isdigit, v))
        if len(cleaned) < 10:
            raise ValueError("Phone number must have at least 10 digits")
        return cleaned[-10:]
    
    @validator('aadhaar_number')
    def validate_aadhaar(cls, v):
        v = v.strip()
        if not v.isdigit():
            raise ValueError("Aadhaar must contain only digits")
        if len(v) != 12:
            raise ValueError("Aadhaar must be exactly 12 digits")
        return v
    
    def normalize(self) -> Dict[str, Any]:
        return {
            'name': self.name.strip().lower(),
            'dob': self.dob,
            'pan': self.pan.upper().strip(),
            'mobile_number': ''.join(filter(str.isdigit, self.phone))[-10:],
            'aadhaar_number': self.aadhaar_number.strip(),
            'address': self.address.strip().lower()
        }


class KycEventPayload(BaseModel):
    wakes_on: str = Field("kyc.dedup_requested")
    reads: ApplicantReads


class LoanApplication(BaseModel):
    name: str = Field(..., min_length=2, max_length=255)
    dob: str = Field(..., example="1992-05-12")
    aadhaar_number: str = Field(..., min_length=12, max_length=12)
    pan: str = Field(..., min_length=10, max_length=10)
    mobile_number: str = Field(..., min_length=10, max_length=15)
    email: Optional[str] = Field(None, max_length=100)
    address: str = Field(..., min_length=5, max_length=500)
    loan_type: str = Field(..., example="Home Loan")
    loan_amount: float = Field(..., gt=0)
    loan_account_no: Optional[str] = Field(None, example="HL001")
    interest_rate: Optional[float] = Field(None, gt=0, le=30)
    loan_term_months: Optional[int] = Field(None, gt=0, le=360)


# =============================================================================
# FIELD COMPARISON FUNCTIONS
# =============================================================================

def compare_field(field_type: str, value1: str, value2: str) -> float:
    if not value1 or not value2:
        return 0.0
    
    v1 = str(value1).strip()
    v2 = str(value2).strip()
    
    if field_type == 'pan':
        v1_clean = re.sub(r'[^A-Z0-9]', '', v1.upper())
        v2_clean = re.sub(r'[^A-Z0-9]', '', v2.upper())
        return 1.0 if v1_clean == v2_clean else 0.0
    
    elif field_type == 'aadhaar_number':
        return 1.0 if v1 == v2 else 0.0
    
    elif field_type == 'name':
        similarity = JaroWinkler.similarity(v1.lower(), v2.lower())
        return similarity if similarity >= DedupeWeights.FUZZY_NAME_THRESHOLD else 0.0
    
    elif field_type == 'dob':
        try:
            for fmt in ['%Y-%m-%d', '%d-%m-%Y', '%m/%d/%Y', '%d/%m/%Y']:
                try:
                    d1 = datetime.strptime(v1, fmt)
                    d2 = datetime.strptime(v2, fmt)
                    return 1.0 if d1 == d2 else 0.0
                except:
                    continue
            return 1.0 if v1 == v2 else 0.0
        except:
            return 1.0 if v1 == v2 else 0.0
    
    elif field_type == 'mobile_number':
        v1_clean = re.sub(r'\D', '', v1)[-10:]
        v2_clean = re.sub(r'\D', '', v2)[-10:]
        return 1.0 if v1_clean == v2_clean else 0.0
    
    elif field_type == 'address':
        similarity = partial_ratio(v1.lower(), v2.lower()) / 100.0
        return similarity if similarity >= DedupeWeights.ADDRESS_THRESHOLD else 0.0
    
    return 0.0


def calculate_cumulative_score(applicant: Dict, db_record: Dict, is_blacklist: bool = False) -> tuple:
    score = 0.0
    matched_fields = []
    
    field_weights = {
        'pan': DedupeWeights.PAN,
        'aadhaar_number': DedupeWeights.AADHAAR,
        'name': DedupeWeights.NAME,
        'dob': DedupeWeights.DOB,
        'mobile_number': DedupeWeights.MOBILE,
        'address': DedupeWeights.ADDRESS
    }
    
    for field, weight in field_weights.items():
        if field in applicant and field in db_record:
            match_score = compare_field(field, applicant[field], db_record[field])
            if match_score > 0:
                field_score = weight * match_score
                score += field_score
                matched_fields.append(field)
    
    if is_blacklist and score > 0:
        score = min(score * DedupeWeights.BLACKLIST_MULTIPLIER, 1.0)
    
    return score, matched_fields


# =============================================================================
# KYC DEDUP DATABASE FUNCTIONS
# =============================================================================

def search_blacklist(cursor, applicant: Dict) -> List[Dict]:
    matches = []
    
    cursor.execute("""
        SELECT 
            'BLACKLIST_DB' as source,
            blacklist_id::text as id,
            name,
            reason,
            pan,
            aadhaar_number,
            dob,
            mobile_number
        FROM blacklist_record 
        WHERE pan = %s 
           OR aadhaar_number = %s 
           OR mobile_number = %s
           OR (aadhaar_number = %s AND dob = %s)
        LIMIT 10;
    """, (
        applicant['pan'], 
        applicant['aadhaar_number'], 
        applicant['mobile_number'],
        applicant['aadhaar_number'], 
        applicant['dob']
    ))
    
    records = cursor.fetchall()
    for record in records:
        record_dict = dict(record)
        record_dict['address'] = ''
        score, matched_fields = calculate_cumulative_score(applicant, record_dict, is_blacklist=True)
        if score > 0:
            matches.append({
                'record': record_dict,
                'score': score,
                'matched_fields': matched_fields,
                'source': 'BLACKLIST'
            })
    
    return matches


def search_customers_kyc(cursor, applicant: Dict) -> List[Dict]:
    matches = []
    
    cursor.execute("""
        SELECT 
            'CUSTOMER_DB' as source,
            customer_id::text as id,
            name,
            address,
            pan,
            aadhaar_number,
            dob,
            mobile_number
        FROM existing_customers_rec 
        WHERE pan = %s OR (aadhaar_number = %s AND dob = %s)
        LIMIT 10;
    """, (applicant['pan'], applicant['aadhaar_number'], applicant['dob']))
    
    records = cursor.fetchall()
    for record in records:
        record_dict = dict(record)
        score, matched_fields = calculate_cumulative_score(applicant, record_dict, is_blacklist=False)
        if score > 0:
            matches.append({
                'record': record_dict,
                'score': score,
                'matched_fields': matched_fields,
                'source': 'CUSTOMER'
            })
    
    cursor.execute("""
        SELECT 
            'CUSTOMER_DB' as source,
            customer_id::text as id,
            name,
            address,
            pan,
            aadhaar_number,
            dob,
            mobile_number
        FROM existing_customers_rec 
        WHERE mobile_number = %s
        LIMIT 5;
    """, (applicant['mobile_number'],))
    
    records = cursor.fetchall()
    for record in records:
        record_dict = dict(record)
        if any(m['record']['id'] == record_dict['id'] for m in matches):
            continue
        score, matched_fields = calculate_cumulative_score(applicant, record_dict, is_blacklist=False)
        if score > 0 and score < 0.90:
            matches.append({
                'record': record_dict,
                'score': score,
                'matched_fields': matched_fields,
                'source': 'CUSTOMER'
            })
    
    return matches


def fuzzy_name_search_kyc(cursor, applicant: Dict, existing_matches: List) -> List[Dict]:
    matches = []
    
    cursor.execute("""
        SELECT 
            'CUSTOMER_DB' as source,
            customer_id::text as id,
            name,
            address,
            pan,
            aadhaar_number,
            dob,
            mobile_number
        FROM existing_customers_rec 
        WHERE dob = %s
        LIMIT 20;
    """, (applicant['dob'],))
    
    candidates = cursor.fetchall()
    
    for candidate in candidates:
        candidate_dict = dict(candidate)
        if any(m['record']['id'] == candidate_dict['id'] for m in existing_matches):
            continue
        
        db_name = candidate_dict["name"].strip().lower()
        name_score = JaroWinkler.similarity(applicant['name'], db_name)
        
        if name_score >= DedupeWeights.FUZZY_NAME_THRESHOLD:
            score = name_score * DedupeWeights.NAME
            matches.append({
                'record': candidate_dict,
                'score': score,
                'matched_fields': ['name'],
                'source': 'CUSTOMER'
            })
    
    return matches


def check_customer_loans_kyc(cursor, customer_id: int) -> Dict:
    cursor.execute("""
        SELECT 
            COUNT(*) as loan_count,
            ARRAY_AGG(loan_account_no) as loan_accounts,
            ARRAY_AGG(loan_type) as loan_types,
            ARRAY_AGG(loan_status) as loan_statuses
        FROM loan_accounts 
        WHERE customer_id = %s
    """, (customer_id,))
    
    result = cursor.fetchone()
    if result and result['loan_count'] > 0:
        return {
            'has_loans': True,
            'loan_count': result['loan_count'],
            'loan_accounts': result['loan_accounts'],
            'loan_types': result['loan_types'],
            'loan_statuses': result['loan_statuses'],
            'has_multiple_loans': result['loan_count'] > 1
        }
    return {
        'has_loans': False,
        'loan_count': 0,
        'loan_accounts': [],
        'loan_types': [],
        'loan_statuses': [],
        'has_multiple_loans': False
    }


def store_dedup_result(cursor, customer_id: Optional[int], matched_customer_id: Optional[int], 
                       match_score: float, result_type: str, explanation: str):
    cursor.execute("""
        INSERT INTO deduplication_results 
        (customer_id, matched_customer_id, match_score, result_type, explanation)
        VALUES (%s, %s, %s, %s, %s)
    """, (customer_id, matched_customer_id, match_score, result_type, explanation))


# =============================================================================
# VERDICT DECISION ENGINE
# =============================================================================

def determine_verdict_kyc(confidence: float, is_blacklist: bool, has_matches: bool, 
                          loan_info: Dict = None, matched_fields: List = None) -> dict:
    if not has_matches:
        return {
            'heading': '✅ CLEAR',
            'status': 'CLEAR – No matching customer found in the system.',
            'verdict': 'NO_MATCH',
            'action': 'You may proceed with the KYC process as this appears to be a new customer.',
            'header_class': 'success',
            'confidence': 0.0
        }
    
    if is_blacklist and confidence >= 0.70:
        return {
            'heading': '⛔ BLACKLISTED',
            'status': 'BLACKLISTED – This customer is flagged in the blacklist database.',
            'verdict': 'BLACKLISTED_FRAUD',
            'action': 'Immediate rejection required. Please do not proceed with this application.',
            'header_class': 'error',
            'confidence': confidence
        }
    
    # Check strong identifiers
    strong_identifiers = ['pan', 'aadhaar_number', 'mobile_number']
    has_strong_match = any(field in matched_fields for field in strong_identifiers) if matched_fields else False
    
    if has_strong_match:
        confidence = 1.0
        if loan_info and loan_info.get('has_multiple_loans', False):
            return {
                'heading': '👤 EXISTING CUSTOMER',
                'status': f'EXISTING CUSTOMER – Customer found by PAN/Aadhaar/Mobile with {loan_info["loan_count"]} active loans.',
                'verdict': 'SAME_CUSTOMER_MULTIPLE_LOANS',
                'action': f'This customer already has {loan_info["loan_count"]} loans. Please review before proceeding.',
                'header_class': 'warning',
                'confidence': confidence
            }
        elif loan_info and loan_info.get('has_loans', False):
            return {
                'heading': '👤 EXISTING CUSTOMER',
                'status': f'EXISTING CUSTOMER – Customer found by PAN/Aadhaar/Mobile with {loan_info["loan_count"]} active loan.',
                'verdict': 'EXISTING_CUSTOMER_SINGLE_LOAN',
                'action': f'This customer already has {loan_info["loan_count"]} loan. Please review before adding new loan.',
                'header_class': 'warning',
                'confidence': confidence
            }
        else:
            return {
                'heading': '👤 EXISTING CUSTOMER',
                'status': 'EXISTING CUSTOMER – Customer found by PAN/Aadhaar/Mobile match.',
                'verdict': 'EXISTING_CUSTOMER_STRONG_MATCH',
                'action': 'This customer is already registered. Please proceed accordingly.',
                'header_class': 'warning',
                'confidence': confidence
            }
    
    if loan_info and loan_info.get('has_multiple_loans', False) and confidence >= 0.70:
        return {
            'heading': '👤 EXISTING CUSTOMER',
            'status': f'EXISTING CUSTOMER – This customer already exists in the system with {loan_info["loan_count"]} active loans.',
            'verdict': 'SAME_CUSTOMER_MULTIPLE_LOANS',
            'action': f'This customer already has {loan_info["loan_count"]} loans. Please review the existing loan portfolio before proceeding.',
            'header_class': 'warning',
            'confidence': confidence
        }
    
    if loan_info and loan_info.get('has_loans', False) and confidence >= 0.70:
        return {
            'heading': '👤 EXISTING CUSTOMER',
            'status': f'EXISTING CUSTOMER – This customer already exists in the system with {loan_info["loan_count"]} active loan.',
            'verdict': 'EXISTING_CUSTOMER_SINGLE_LOAN',
            'action': f'This customer already has {loan_info["loan_count"]} loan. Please review before adding new loan.',
            'header_class': 'warning',
            'confidence': confidence
        }
    
    if confidence >= DedupeWeights.EXACT_MATCH:
        return {
            'heading': '⚠️ EXACT MATCH',
            'status': 'EXACT MATCH – An exact duplicate customer record was found in the system.',
            'verdict': 'EXACT_MATCH',
            'action': 'Auto-reject this application as an exact ID match was found with confidence of 100%.',
            'header_class': 'error',
            'confidence': confidence
        }
    elif confidence >= DedupeWeights.HIGH_CONFIDENCE:
        return {
            'heading': '⚠️ HIGH CONFIDENCE DUPLICATE',
            'status': 'HIGH CONFIDENCE MATCH – A very strong duplicate match was detected.',
            'verdict': 'HIGH_CONFIDENCE_MATCH',
            'action': 'Reject this application and send it for manual verification immediately.',
            'header_class': 'error',
            'confidence': confidence
        }
    elif confidence >= DedupeWeights.MEDIUM_CONFIDENCE:
        return {
            'heading': '👤 EXISTING CUSTOMER',
            'status': 'EXISTING CUSTOMER – A potential existing customer match was found.',
            'verdict': 'EXISTING_CUSTOMER_MEDIUM',
            'action': 'This appears to be an existing customer. Please verify and proceed accordingly.',
            'header_class': 'warning',
            'confidence': confidence
        }
    elif confidence >= DedupeWeights.LOW_CONFIDENCE:
        return {
            'heading': '👤 EXISTING CUSTOMER',
            'status': 'EXISTING CUSTOMER – A possible existing customer match was found.',
            'verdict': 'EXISTING_CUSTOMER_LOW',
            'action': 'This may be an existing customer. Request additional verification documents before proceeding.',
            'header_class': 'warning',
            'confidence': confidence
        }
    elif confidence >= DedupeWeights.WEAK_MATCH:
        return {
            'heading': '🚩 FLAGGED',
            'status': 'WEAK MATCH – A minimal match was found, requiring monitoring.',
            'verdict': 'WEAK_MATCH',
            'action': 'Flag this application for monitoring while allowing KYC to proceed.',
            'header_class': 'info',
            'confidence': confidence
        }
    else:
        return {
            'heading': '✅ CLEAR',
            'status': 'CLEAR – No significant matches were found in the system.',
            'verdict': 'NO_MATCH',
            'action': 'You may proceed with the KYC process.',
            'header_class': 'success',
            'confidence': confidence
        }


# =============================================================================
# LOAN MANAGEMENT FUNCTIONS
# =============================================================================

def find_customer_by_aadhaar(cursor, aadhaar_number: str) -> Optional[Dict]:
    cursor.execute("""
        SELECT customer_id, name, dob, aadhaar_number, pan, mobile_number, email, address
        FROM existing_customers_rec 
        WHERE aadhaar_number = %s
    """, (aadhaar_number,))
    return cursor.fetchone()


def find_customer_by_mobile(cursor, mobile_number: str) -> Optional[Dict]:
    cursor.execute("""
        SELECT customer_id, name, dob, aadhaar_number, pan, mobile_number, email, address
        FROM existing_customers_rec 
        WHERE mobile_number = %s
    """, (mobile_number,))
    return cursor.fetchone()


def find_customer_by_pan(cursor, pan: str) -> Optional[Dict]:
    cursor.execute("""
        SELECT customer_id, name, dob, aadhaar_number, pan, mobile_number, email, address
        FROM existing_customers_rec 
        WHERE pan = %s
    """, (pan,))
    return cursor.fetchone()


def get_customer_loans(cursor, customer_id: int) -> List[Dict]:
    """Get all loans for a customer with proper column names"""
    cursor.execute("""
        SELECT 
            loan_id,
            loan_account_no,
            loan_type,
            loan_amount,
            interest_rate,
            loan_term_months,
            loan_status,
            application_date,
            approval_date,
            disbursement_date
        FROM loan_accounts 
        WHERE customer_id = %s
        ORDER BY application_date DESC
    """, (customer_id,))
    return cursor.fetchall()


def create_customer(cursor, customer_data: Dict) -> int:
    cursor.execute("""
        INSERT INTO existing_customers_rec 
        (name, dob, aadhaar_number, pan, mobile_number, email, address)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING customer_id
    """, (
        customer_data['name'],
        customer_data['dob'],
        customer_data['aadhaar_number'],
        customer_data['pan'],
        customer_data['mobile_number'],
        customer_data.get('email'),
        customer_data['address']
    ))
    return cursor.fetchone()['customer_id']


def create_loan(cursor, customer_id: int, loan_data: Dict) -> int:
    cursor.execute("""
        INSERT INTO loan_accounts 
        (customer_id, loan_account_no, loan_type, loan_amount, 
         interest_rate, loan_term_months, loan_status, application_date)
        VALUES (%s, %s, %s, %s, %s, %s, 'ACTIVE', CURRENT_DATE)
        RETURNING loan_id
    """, (
        customer_id,
        loan_data['loan_account_no'],
        loan_data['loan_type'],
        loan_data['loan_amount'],
        loan_data['interest_rate'],
        loan_data['loan_term_months']
    ))
    return cursor.fetchone()['loan_id']


def check_blacklist_loan(cursor, aadhaar_number: str, mobile_number: str, pan: str) -> Optional[Dict]:
    cursor.execute("""
        SELECT blacklist_id, name, reason
        FROM blacklist_record 
        WHERE aadhaar_number = %s OR mobile_number = %s OR pan = %s
    """, (aadhaar_number, mobile_number, pan))
    return cursor.fetchone()


# =============================================================================
# ENHANCED LOAN APPLICATION
# =============================================================================

def find_exact_customer(cursor, applicant: Dict) -> Optional[Dict]:
    cursor.execute("""
        SELECT customer_id, name, dob, aadhaar_number, pan, mobile_number, email, address
        FROM existing_customers_rec
        WHERE name = %s
          AND dob = %s
          AND aadhaar_number = %s
          AND pan = %s
          AND mobile_number = %s
          AND address = %s
    """, (
        applicant['name'],
        applicant['dob'],
        applicant['aadhaar_number'],
        applicant['pan'],
        applicant['mobile_number'],
        applicant['address']
    ))
    return cursor.fetchone()


def check_conflicts(cursor, applicant: Dict) -> Dict:
    """
    Check for conflicts with existing customer records.
    Only matches on EXACT identifiers (PAN, Aadhaar, Mobile).
    """
    conflict_messages = []
    matched_customer_id = None
    conflict_details = []
    
    # Clean mobile number
    cleaned_mobile = re.sub(r'\D', '', applicant.get('mobile_number', ''))[-10:]
    
    # Check PAN - EXACT match only (case-insensitive)
    if applicant.get('pan'):
        cursor.execute("""
            SELECT customer_id, name, pan 
            FROM existing_customers_rec 
            WHERE UPPER(pan) = UPPER(%s)
        """, (applicant['pan'].strip(),))
        pan_result = cursor.fetchone()
        if pan_result:
            conflict_messages.append(f"⚠️ PAN {applicant['pan']} already exists for customer: {pan_result['name']}")
            matched_customer_id = pan_result['customer_id']
            conflict_details.append({
                'field': 'pan',
                'value': applicant['pan'],
                'existing_customer_id': pan_result['customer_id'],
                'existing_customer_name': pan_result['name']
            })
    
    # Check Aadhaar - EXACT match only
    if applicant.get('aadhaar_number'):
        cursor.execute("""
            SELECT customer_id, name, aadhaar_number 
            FROM existing_customers_rec 
            WHERE aadhaar_number = %s
        """, (applicant['aadhaar_number'].strip(),))
        aadhaar_result = cursor.fetchone()
        if aadhaar_result:
            conflict_messages.append(f"⚠️ Aadhaar {applicant['aadhaar_number']} already exists for customer: {aadhaar_result['name']}")
            if not matched_customer_id:
                matched_customer_id = aadhaar_result['customer_id']
            conflict_details.append({
                'field': 'aadhaar_number',
                'value': applicant['aadhaar_number'],
                'existing_customer_id': aadhaar_result['customer_id'],
                'existing_customer_name': aadhaar_result['name']
            })
    
    # Check Mobile - EXACT match only (cleaned)
    if applicant.get('mobile_number') and cleaned_mobile:
        cursor.execute("""
            SELECT customer_id, name, mobile_number 
            FROM existing_customers_rec 
            WHERE mobile_number = %s
        """, (cleaned_mobile,))
        mobile_result = cursor.fetchone()
        if mobile_result:
            conflict_messages.append(f"⚠️ Mobile {cleaned_mobile} already exists for customer: {mobile_result['name']}")
            if not matched_customer_id:
                matched_customer_id = mobile_result['customer_id']
            conflict_details.append({
                'field': 'mobile_number',
                'value': cleaned_mobile,
                'existing_customer_id': mobile_result['customer_id'],
                'existing_customer_name': mobile_result['name']
            })
    
    return {
        "conflict_messages": conflict_messages, 
        "matched_customer_id": matched_customer_id,
        "conflict_details": conflict_details,
        "has_conflict": len(conflict_messages) > 0
    }


# =============================================================================
# API ENDPOINTS
# =============================================================================

# ---- KYC DEDUP ----

@app.post("/api/v1/kyc/dedup")
def process_dedup_api(event: KycEventPayload):
    applicant = event.reads.normalize()
    
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD}🔍 KYC DEDUPLICATION CHECK{Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"{Colors.BLUE}📌 Applicant: {applicant['name']}{Colors.RESET}")
    print(f"{Colors.BLUE}📌 PAN: {applicant['pan']}{Colors.RESET}")
    print(f"{Colors.BLUE}📌 Aadhaar: {applicant['aadhaar_number']}{Colors.RESET}")
    print(f"{Colors.BLUE}📌 Phone: {applicant['mobile_number']}{Colors.RESET}")
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # 1. Check Blacklist
            blacklist_matches = search_blacklist(cursor, applicant)
            if blacklist_matches:
                store_dedup_result(
                    cursor,
                    None,
                    None,
                    1.0,
                    'BLACKLISTED_FRAUD',
                    f"Blacklist match: {blacklist_matches[0]['record']['reason']}"
                )
                conn.commit()
                
                print(f"\n{Colors.RED}{Colors.BOLD}{'='*60}{Colors.RESET}")
                print(f"{Colors.RED}{Colors.BOLD}⛔ BLACKLISTED - Immediate rejection required{Colors.RESET}")
                print(f"{Colors.RED}{Colors.BOLD}{'='*60}{Colors.RESET}")
                print(f"{Colors.YELLOW}Reason:{Colors.RESET} {blacklist_matches[0]['record']['reason']}")
                
                return {
                    "heading": "⛔ BLACKLISTED",
                    "status": "BLACKLISTED – This customer is flagged in the blacklist database.",
                    "verdict": "BLACKLISTED_FRAUD",
                    "action": "Immediate rejection required. Please do not proceed with this application.",
                    "header_class": "error",
                    "reason": blacklist_matches[0]['record']['reason'],
                    "confidence": 1.0,
                    "customer_id": None,
                    "customer_name": None
                }
            
            # 2. Search for matches
            customer_matches = search_customers_kyc(cursor, applicant)
            fuzzy_matches = fuzzy_name_search_kyc(cursor, applicant, customer_matches)
            
            all_matches = customer_matches + fuzzy_matches
            all_matches.sort(key=lambda x: x['score'], reverse=True)
            
            final_confidence = max([m['score'] for m in all_matches]) if all_matches else 0.0
            matched_customer_id = None
            loan_info = None
            matched_fields = []
            
            # 3. If matches found
            if all_matches:
                best_match = all_matches[0]
                matched_customer_id = int(best_match['record']['id'])
                matched_fields = best_match['matched_fields']
                loan_info = check_customer_loans_kyc(cursor, matched_customer_id)
                
                store_dedup_result(
                    cursor,
                    None,
                    matched_customer_id,
                    round(final_confidence, 2),
                    'DUPLICATE_FOUND',
                    f"Found {len(all_matches)} match(es), best score: {final_confidence:.2f}"
                )
                conn.commit()
                
                print(f"\n{Colors.GREEN}{Colors.BOLD}{'='*60}{Colors.RESET}")
                print(f"{Colors.GREEN}{Colors.BOLD}✅ MATCH FOUND!{Colors.RESET}")
                print(f"{Colors.GREEN}{Colors.BOLD}{'='*60}{Colors.RESET}")
                print(f"{Colors.YELLOW}📌 Customer ID:{Colors.RESET} {matched_customer_id}")
                print(f"{Colors.YELLOW}📌 Name:{Colors.RESET} {best_match['record']['name']}")
                print(f"{Colors.YELLOW}📌 Confidence:{Colors.RESET} {final_confidence:.2%}")
                print(f"{Colors.YELLOW}📌 Matched Fields:{Colors.RESET} {', '.join(matched_fields)}")
                if loan_info and loan_info.get('has_loans', False):
                    print(f"{Colors.YELLOW}📌 Total Loans:{Colors.RESET} {loan_info['loan_count']}")
                    print(f"{Colors.YELLOW}📌 Loan Accounts:{Colors.RESET} {', '.join(loan_info['loan_accounts'])}")
                
                verdict = determine_verdict_kyc(
                    final_confidence, 
                    False, 
                    bool(all_matches), 
                    loan_info,
                    matched_fields
                )
                
                final_confidence = verdict.get('confidence', final_confidence)
                
                response = {
                    "heading": verdict['heading'],
                    "status": verdict['status'],
                    "verdict": verdict['verdict'],
                    "action": verdict['action'],
                    "header_class": verdict['header_class'],
                    "confidence": round(final_confidence, 2),
                    "customer_id": matched_customer_id,
                    "customer_name": best_match['record']['name'],
                    "matched_fields": matched_fields,
                    "match_count": len(all_matches)
                }
                
                if loan_info and loan_info.get('has_loans', False):
                    response["loan_details"] = {
                        "loan_count": loan_info['loan_count'],
                        "loan_accounts": loan_info['loan_accounts'],
                        "loan_types": loan_info['loan_types'],
                        "loan_statuses": loan_info['loan_statuses']
                    }
                
                print(f"\n{Colors.MAGENTA}{Colors.BOLD}📋 Verdict:{Colors.RESET} {verdict['heading']}")
                print(f"{Colors.MAGENTA}{Colors.BOLD}📋 Status:{Colors.RESET} {verdict['status']}")
                print(f"{Colors.MAGENTA}{Colors.BOLD}📋 Action:{Colors.RESET} {verdict['action']}")
                print(f"{Colors.CYAN}{Colors.BOLD}{'='*60}{Colors.RESET}")
                
                return response
            
            # 4. No matches found
            store_dedup_result(
                cursor,
                None,
                None,
                0.0,
                'NEW_CUSTOMER',
                'No matching customer found'
            )
            conn.commit()
            
            print(f"\n{Colors.GREEN}{Colors.BOLD}{'='*60}{Colors.RESET}")
            print(f"{Colors.GREEN}{Colors.BOLD}✅ CLEAR - No matches found{Colors.RESET}")
            print(f"{Colors.GREEN}{Colors.BOLD}{'='*60}{Colors.RESET}")
            print(f"{Colors.YELLOW}📌 Result:{Colors.RESET} This appears to be a new customer.")
            print(f"{Colors.YELLOW}📌 Action:{Colors.RESET} Proceed with KYC process.")
            
            return {
                "heading": "✅ CLEAR",
                "status": "CLEAR – No matching customer found in the system.",
                "verdict": "NO_MATCH",
                "action": "You may proceed with the KYC process as this appears to be a new customer.",
                "header_class": "success",
                "confidence": 0.0,
                "customer_id": None,
                "customer_name": None,
                "matched_fields": [],
                "match_count": 0
            }
            
    except psycopg2.Error as e:
        logger.error(f"Database error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        logger.error(f"Processing error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


# ---- LOAN APPLICATION ----

@app.post("/api/v1/loan/apply")
def apply_loan(application: LoanApplication):
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Clean and normalize applicant data
            cleaned_mobile = re.sub(r'\D', '', application.mobile_number)[-10:]
            
            applicant = {
                'name': application.name.strip().lower(),
                'dob': application.dob,
                'aadhaar_number': application.aadhaar_number.strip(),
                'pan': application.pan.upper().strip(),
                'mobile_number': cleaned_mobile,
                'email': application.email,
                'address': application.address.strip().lower()
            }
            
            # Log the applicant details for debugging
            print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*60}{Colors.RESET}")
            print(f"{Colors.CYAN}{Colors.BOLD}📝 LOAN APPLICATION{Colors.RESET}")
            print(f"{Colors.CYAN}{Colors.BOLD}{'='*60}{Colors.RESET}")
            print(f"{Colors.BLUE}📌 Name: {application.name}{Colors.RESET}")
            print(f"{Colors.BLUE}📌 PAN: {applicant['pan']}{Colors.RESET}")
            print(f"{Colors.BLUE}📌 Aadhaar: {applicant['aadhaar_number']}{Colors.RESET}")
            print(f"{Colors.BLUE}📌 Mobile: {applicant['mobile_number']}{Colors.RESET}")
            print(f"{Colors.BLUE}📌 Loan Type: {application.loan_type}{Colors.RESET}")
            print(f"{Colors.BLUE}📌 Loan Amount: ₹{application.loan_amount:,.2f}{Colors.RESET}")
            
            # 1. Check Blacklist
            blacklist = check_blacklist_loan(
                cursor, 
                applicant['aadhaar_number'], 
                applicant['mobile_number'], 
                applicant['pan']
            )
            if blacklist:
                return {
                    "heading": "⛔ BLACKLISTED",
                    "status": "REJECTED – This customer is blacklisted.",
                    "verdict": "BLACKLISTED",
                    "action": "Immediate rejection required. Customer is blacklisted.",
                    "header_class": "error",
                    "customer_id": None,
                    "customer_name": None,
                    "reason": blacklist['reason'],
                    "confidence": 1.0
                }
            
            # 2. Check if customer already exists with EXACT match
            # First check if any identifier already exists
            existing_customer = None
            
            # Check by PAN
            cursor.execute("""
                SELECT customer_id, name, pan, aadhaar_number, mobile_number
                FROM existing_customers_rec 
                WHERE UPPER(pan) = UPPER(%s)
            """, (applicant['pan'],))
            pan_customer = cursor.fetchone()
            if pan_customer:
                existing_customer = pan_customer
                print(f"{Colors.YELLOW}⚠️ Found existing customer by PAN: {pan_customer['name']} (ID: {pan_customer['customer_id']}){Colors.RESET}")
            
            # Check by Aadhaar if not found by PAN
            if not existing_customer:
                cursor.execute("""
                    SELECT customer_id, name, pan, aadhaar_number, mobile_number
                    FROM existing_customers_rec 
                    WHERE aadhaar_number = %s
                """, (applicant['aadhaar_number'],))
                aadhaar_customer = cursor.fetchone()
                if aadhaar_customer:
                    existing_customer = aadhaar_customer
                    print(f"{Colors.YELLOW}⚠️ Found existing customer by Aadhaar: {aadhaar_customer['name']} (ID: {aadhaar_customer['customer_id']}){Colors.RESET}")
            
            # Check by Mobile if not found by PAN or Aadhaar
            if not existing_customer:
                cursor.execute("""
                    SELECT customer_id, name, pan, aadhaar_number, mobile_number
                    FROM existing_customers_rec 
                    WHERE mobile_number = %s
                """, (applicant['mobile_number'],))
                mobile_customer = cursor.fetchone()
                if mobile_customer:
                    existing_customer = mobile_customer
                    print(f"{Colors.YELLOW}⚠️ Found existing customer by Mobile: {mobile_customer['name']} (ID: {mobile_customer['customer_id']}){Colors.RESET}")
            
            # If customer exists, add loan to existing customer
            if existing_customer:
                customer_id = existing_customer['customer_id']
                
                # Check if loan account number already exists
                if application.loan_account_no:
                    loan_account_no = application.loan_account_no.strip()
                    cursor.execute("SELECT loan_id FROM loan_accounts WHERE loan_account_no = %s", (loan_account_no,))
                    if cursor.fetchone():
                        return {
                            "status": "ERROR", 
                            "message": f"Loan account {loan_account_no} already exists",
                            "customer_id": customer_id,
                            "customer_name": existing_customer['name'],
                            "confidence": 1.0
                        }
                else:
                    cursor.execute("SELECT COALESCE(MAX(loan_id), 0) + 1 AS next_id FROM loan_accounts")
                    next_id = cursor.fetchone()['next_id']
                    loan_account_no = f"LN{next_id:04d}"
                
                # Get loan terms
                interest_rate, term_months = get_loan_terms(application.loan_type)
                if application.interest_rate:
                    interest_rate = application.interest_rate
                if application.loan_term_months:
                    term_months = application.loan_term_months
                
                # Create loan
                loan_data = {
                    'loan_account_no': loan_account_no,
                    'loan_type': application.loan_type,
                    'loan_amount': application.loan_amount,
                    'interest_rate': interest_rate,
                    'loan_term_months': term_months
                }
                loan_id = create_loan(cursor, customer_id, loan_data)
                updated_loans = get_customer_loans(cursor, customer_id)
                conn.commit()
                
                loan_count = len(updated_loans)
                
                return {
                    "heading": "👤 EXISTING CUSTOMER",
                    "status": f"EXISTING CUSTOMER – New loan added to existing customer. Total loans: {loan_count}",
                    "verdict": "EXISTING_CUSTOMER_NEW_LOAN",
                    "action": f"Loan successfully added to customer ID {customer_id}. Total loans now: {loan_count}",
                    "header_class": "warning",
                    "customer_id": customer_id,
                    "customer_name": existing_customer['name'],
                    "confidence": 1.0,
                    "customer": existing_customer,
                    "new_loan": {
                        "loan_id": loan_id,
                        "loan_account_no": loan_account_no,
                        "loan_type": application.loan_type,
                        "loan_amount": application.loan_amount,
                        "interest_rate": interest_rate,
                        "loan_term_months": term_months
                    },
                    "all_loans": updated_loans,
                    "total_loans": loan_count
                }
            
            # 3. No existing customer found - Create new customer
            print(f"{Colors.GREEN}✅ No existing customer found. Creating new customer...{Colors.RESET}")
            
            # Get loan terms
            interest_rate, term_months = get_loan_terms(application.loan_type)
            if application.interest_rate:
                interest_rate = application.interest_rate
            if application.loan_term_months:
                term_months = application.loan_term_months
            
            # Generate loan account number
            if application.loan_account_no:
                loan_account_no = application.loan_account_no.strip()
                cursor.execute("SELECT loan_id FROM loan_accounts WHERE loan_account_no = %s", (loan_account_no,))
                if cursor.fetchone():
                    return {
                        "status": "ERROR", 
                        "message": f"Loan account {loan_account_no} already exists",
                        "customer_id": None,
                        "customer_name": None,
                        "confidence": 0.0
                    }
            else:
                cursor.execute("SELECT COALESCE(MAX(loan_id), 0) + 1 AS next_id FROM loan_accounts")
                next_id = cursor.fetchone()['next_id']
                loan_account_no = f"LN{next_id:04d}"
            
            # Create customer
            customer_id = create_customer(cursor, applicant)
            print(f"{Colors.GREEN}✅ Customer created with ID: {customer_id}{Colors.RESET}")
            
            # Create loan
            loan_data = {
                'loan_account_no': loan_account_no,
                'loan_type': application.loan_type,
                'loan_amount': application.loan_amount,
                'interest_rate': interest_rate,
                'loan_term_months': term_months
            }
            loan_id = create_loan(cursor, customer_id, loan_data)
            loans = get_customer_loans(cursor, customer_id)
            conn.commit()
            
            print(f"{Colors.GREEN}✅ Loan created with ID: {loan_id}{Colors.RESET}")
            print(f"{Colors.CYAN}{Colors.BOLD}{'='*60}{Colors.RESET}")
            
            return {
                "heading": "✅ NEW CUSTOMER",
                "status": "NEW CUSTOMER – Customer created successfully with the loan application.",
                "verdict": "NEW_CUSTOMER",
                "action": f"New customer created with loan account {loan_account_no}. Total loans: 1",
                "header_class": "success",
                "customer_id": customer_id,
                "customer_name": application.name,
                "confidence": 0.0,
                "customer": {
                    "customer_id": customer_id,
                    "name": application.name,
                    "dob": application.dob,
                    "aadhaar_number": applicant['aadhaar_number'],
                    "pan": applicant['pan'],
                    "mobile_number": applicant['mobile_number'],
                    "email": application.email,
                    "address": application.address
                },
                "new_loan": {
                    "loan_id": loan_id,
                    "loan_account_no": loan_account_no,
                    "loan_type": application.loan_type,
                    "loan_amount": application.loan_amount,
                    "interest_rate": interest_rate,
                    "loan_term_months": term_months
                },
                "all_loans": loans,
                "total_loans": len(loans)
            }
            
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ---- CUSTOMER PROFILE ----

@app.get("/api/v1/customer/{identifier}")
def get_customer_profile(identifier: str):
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get customer details
            cursor.execute("""
                SELECT customer_id, name, dob, aadhaar_number, pan, mobile_number, email, address
                FROM existing_customers_rec 
                WHERE aadhaar_number = %s OR mobile_number = %s OR UPPER(pan) = UPPER(%s)
            """, (identifier, identifier, identifier))
            
            customer = cursor.fetchone()
            if not customer:
                raise HTTPException(status_code=404, detail="Customer not found")
            
            # Get all loans for this customer
            loans = get_customer_loans(cursor, customer['customer_id'])
            
            return {
                "status": "SUCCESS",
                "customer": customer,
                "loans": loans,
                "total_loans": len(loans),
                "has_multiple_loans": len(loans) > 1
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Customer profile error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ---- ALL CUSTOMERS ----

@app.get("/api/v1/customers")
def get_all_customers():
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("SELECT * FROM existing_customers_rec ORDER BY customer_id")
            customers = cursor.fetchall()
            return {"status": "success", "data": customers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# BLACKLIST MANAGEMENT ENDPOINTS
# =============================================================================

@app.get("/api/v1/blacklist")
def get_blacklist():
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("SELECT * FROM blacklist_record ORDER BY flagged_at DESC")
            records = cursor.fetchall()
            return {"status": "success", "data": records}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/blacklist/add")
def add_blacklist(
    name: str,
    dob: str,
    pan: Optional[str] = None,
    aadhaar_number: Optional[str] = None,
    mobile_number: Optional[str] = None,
    reason: Optional[str] = None,
    source: Optional[str] = None
):
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("""
                INSERT INTO blacklist_record 
                (name, dob, pan, aadhaar_number, mobile_number, reason, source, verification_date)
                VALUES (%s, %s, %s, %s, %s, %s, %s, CURRENT_DATE)
                RETURNING blacklist_id
            """, (name, dob, pan, aadhaar_number, mobile_number, reason, source))
            blacklist_id = cursor.fetchone()['blacklist_id']
            conn.commit()
            return {
                "status": "success",
                "message": "Blacklist record added successfully.",
                "blacklist_id": blacklist_id
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/blacklist/remove/{blacklist_id}")
def remove_blacklist(blacklist_id: int):
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("DELETE FROM blacklist_record WHERE blacklist_id = %s RETURNING blacklist_id", (blacklist_id,))
            deleted = cursor.fetchone()
            if not deleted:
                raise HTTPException(status_code=404, detail="Blacklist record not found")
            conn.commit()
            return {
                "status": "success",
                "message": f"Blacklist record {blacklist_id} removed successfully."
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---- AUDIT LOG ----

@app.get("/api/v1/dedup/results")
def get_dedup_results(limit: int = 50):
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("""
                SELECT * FROM deduplication_results 
                ORDER BY created_at DESC 
                LIMIT %s
            """, (limit,))
            results = cursor.fetchall()
            return {"status": "success", "data": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---- DEBUG ENDPOINT ----

@app.get("/api/v1/debug/check-identifier")
def check_identifier(identifier_type: str, value: str):
    """Debug endpoint to check if an identifier exists in the system"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            if identifier_type == "pan":
                cursor.execute("SELECT customer_id, name, pan FROM existing_customers_rec WHERE UPPER(pan) = UPPER(%s)", (value,))
            elif identifier_type == "aadhaar":
                cursor.execute("SELECT customer_id, name, aadhaar_number FROM existing_customers_rec WHERE aadhaar_number = %s", (value,))
            elif identifier_type == "mobile":
                cleaned = re.sub(r'\D', '', value)[-10:]
                cursor.execute("SELECT customer_id, name, mobile_number FROM existing_customers_rec WHERE mobile_number = %s", (cleaned,))
            else:
                return {"error": "Invalid identifier type. Use: pan, aadhaar, mobile"}
            
            result = cursor.fetchone()
            if result:
                return {
                    "exists": True,
                    "customer_id": result['customer_id'],
                    "customer_name": result['name'],
                    "identifier_value": result.get(identifier_type) if identifier_type in result else value
                }
            else:
                return {
                    "exists": False,
                    "message": f"No customer found with this {identifier_type}: {value}"
                }
    except Exception as e:
        return {"error": str(e)}


# ---- HEALTH CHECK ----

@app.get("/api/v1/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/")
def root():
    return {
        "service": "KYC Deduplication & Loan Management System",
        "version": "4.0.0",
        "endpoints": {
            "kyc_dedup": "POST /api/v1/kyc/dedup",
            "apply_loan": "POST /api/v1/loan/apply",
            "customer_profile": "GET /api/v1/customer/{identifier}",
            "customers": "GET /api/v1/customers",
            "blacklist": "GET /api/v1/blacklist",
            "blacklist_add": "POST /api/v1/blacklist/add",
            "blacklist_remove": "DELETE /api/v1/blacklist/remove/{blacklist_id}",
            "dedup_results": "GET /api/v1/dedup/results",
            "debug_check": "GET /api/v1/debug/check-identifier?identifier_type=pan&value=ABCDE1234F",
            "health": "GET /api/v1/health",
            "docs": "GET /docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)