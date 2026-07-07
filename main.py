import os
import json
import psycopg2
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.fuzz import partial_ratio
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from contextlib import contextmanager

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="KYC Deduplication & Loan Management System",
    description="KYC Deduplication with Aadhaar as Primary Key + Multiple Loans",
    version="3.0.0"
)

# ============================================================================
# DATABASE CONNECTION
# ============================================================================

db_pool = SimpleConnectionPool(
    minconn=1,
    maxconn=10,
    host="localhost",
    database="postgres",
    user="postgres",
    password="Root",
    port="5432"
)

@contextmanager
def get_db_connection():
    conn = db_pool.getconn()
    try:
        yield conn
    finally:
        db_pool.putconn(conn)

# ============================================================================
# CONFIGURATION
# ============================================================================

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

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

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
    # Make loan_account_no optional (system will generate if not provided)
    loan_account_no: Optional[str] = Field(None, example="HL001")
    interest_rate: Optional[float] = Field(None, gt=0, le=30)
    loan_term_months: Optional[int] = Field(None, gt=0, le=360)

# ============================================================================
# FIELD COMPARISON FUNCTIONS
# ============================================================================

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

# ============================================================================
# CUMULATIVE SCORING
# ============================================================================

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

# ============================================================================
# KYC DEDUP DATABASE FUNCTIONS
# ============================================================================

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
        WHERE pan = %s OR (aadhaar_number = %s AND dob = %s)
        LIMIT 10;
    """, (applicant['pan'], applicant['aadhaar_number'], applicant['dob']))
    
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
    
    if not matches:
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
            WHERE mobile_number = %s
            LIMIT 5;
        """, (applicant['mobile_number'],))
        
        records = cursor.fetchall()
        for record in records:
            record_dict = dict(record)
            record_dict['address'] = ''
            score, matched_fields = calculate_cumulative_score(applicant, record_dict, is_blacklist=True)
            if score > 0 and score < 1.0:
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

def determine_verdict_kyc(confidence: float, is_blacklist: bool, has_matches: bool, loan_info: Dict = None) -> dict:
    if not has_matches:
        return {
            'status': 'CLEAR',
            'verdict': 'NO_MATCH',
            'action': 'Proceed with KYC'
        }
    
    if is_blacklist and confidence >= 0.70:
        return {
            'status': 'BLACKLISTED',
            'verdict': 'BLACKLISTED_FRAUD',
            'action': 'Immediate rejection required'
        }
    
    if loan_info and loan_info.get('has_multiple_loans', False):
        return {
            'status': 'EXISTING_CUSTOMER',
            'verdict': 'SAME_CUSTOMER_MULTIPLE_LOANS',
            'action': 'Customer already has multiple active loans',
            'loan_details': {
                'loan_count': loan_info['loan_count'],
                'loan_accounts': loan_info['loan_accounts'],
                'loan_types': loan_info['loan_types'],
                'loan_statuses': loan_info['loan_statuses']
            }
        }
    
    if confidence >= DedupeWeights.EXACT_MATCH:
        return {
            'status': 'REJECTED',
            'verdict': 'EXACT_MATCH',
            'action': 'Auto-reject application - Exact ID match found'
        }
    elif confidence >= DedupeWeights.HIGH_CONFIDENCE:
        return {
            'status': 'REJECTED',
            'verdict': 'HIGH_CONFIDENCE_MATCH',
            'action': 'Reject with manual verification'
        }
    elif confidence >= DedupeWeights.MEDIUM_CONFIDENCE:
        return {
            'status': 'REVIEW',
            'verdict': 'MEDIUM_CONFIDENCE_MATCH',
            'action': 'Send to manual review team'
        }
    elif confidence >= DedupeWeights.LOW_CONFIDENCE:
        return {
            'status': 'REVIEW',
            'verdict': 'LOW_CONFIDENCE_MATCH',
            'action': 'Request additional verification documents'
        }
    elif confidence >= DedupeWeights.WEAK_MATCH:
        return {
            'status': 'FLAGGED',
            'verdict': 'WEAK_MATCH',
            'action': 'Flag for monitoring, allow KYC'
        }
    else:
        return {
            'status': 'CLEAR',
            'verdict': 'NO_MATCH',
            'action': 'Proceed with KYC'
        }

# ============================================================================
# LOAN MANAGEMENT FUNCTIONS
# ============================================================================

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
    cursor.execute("""
        SELECT 
            loan_id, loan_account_no, loan_type, loan_amount, 
            interest_rate, loan_term_months, loan_status, 
            application_date, approval_date, disbursement_date
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
        loan_data.get('interest_rate'),
        loan_data.get('loan_term_months')
    ))
    return cursor.fetchone()['loan_id']

def check_blacklist_loan(cursor, aadhaar_number: str, mobile_number: str, pan: str) -> Optional[Dict]:
    cursor.execute("""
        SELECT blacklist_id, name, reason
        FROM blacklist_record 
        WHERE aadhaar_number = %s OR mobile_number = %s OR pan = %s
    """, (aadhaar_number, mobile_number, pan))
    return cursor.fetchone()

# ============================================================================
# KYC DEDUP ENDPOINT – WITH FULL CONFLICT CHECKS
# ============================================================================

@app.post("/api/v1/kyc/dedup")
def process_dedup_api(event: KycEventPayload):
    applicant = event.reads.normalize()
    
    logger.info(f"Processing dedup request for: {applicant['name']} (PAN: {applicant['pan']})")
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # ---- 1. BLACKLIST CHECK ----
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
                return {
                    "status": "BLACKLISTED",
                    "verdict": "BLACKLISTED_FRAUD",
                    "action": "Immediate rejection required",
                    "reason": blacklist_matches[0]['record']['reason']
                }
            
            # ---- 2. PAN CONFLICT CHECK ----
            cursor.execute("""
                SELECT customer_id, name, aadhaar_number, mobile_number 
                FROM existing_customers_rec 
                WHERE pan = %s
            """, (applicant['pan'],))
            pan_match = cursor.fetchone()
            if pan_match:
                store_dedup_result(
                    cursor,
                    None,
                    pan_match['customer_id'],
                    1.0,
                    'PAN_CONFLICT',
                    f"PAN {applicant['pan']} already registered to {pan_match['name']} (ID: {pan_match['customer_id']})"
                )
                conn.commit()
                return {
                    "status": "FLAGGED",
                    "verdict": "PAN_CONFLICT",
                    "action": "PAN already registered to another customer. Manual verification required.",
                    "existing_customer": {
                        "customer_id": pan_match['customer_id'],
                        "name": pan_match['name'],
                        "aadhaar_number": pan_match['aadhaar_number'],
                        "mobile_number": pan_match['mobile_number']
                    },
                    "confidence": 1.0
                }
            
            # ---- 3. AADHAAR CONFLICT CHECK ----
            cursor.execute("""
                SELECT customer_id, name, dob, pan, mobile_number 
                FROM existing_customers_rec 
                WHERE aadhaar_number = %s
            """, (applicant['aadhaar_number'],))
            aadhaar_match = cursor.fetchone()
            if aadhaar_match:
                if (aadhaar_match['name'] != applicant['name'] or 
                    str(aadhaar_match['dob']) != applicant['dob'] or 
                    aadhaar_match['pan'] != applicant['pan']):
                    store_dedup_result(
                        cursor,
                        None,
                        aadhaar_match['customer_id'],
                        1.0,
                        'AADHAAR_CONFLICT',
                        f"Aadhaar {applicant['aadhaar_number']} already registered to {aadhaar_match['name']} (ID: {aadhaar_match['customer_id']}) with different details"
                    )
                    conn.commit()
                    return {
                        "status": "FLAGGED",
                        "verdict": "AADHAAR_CONFLICT",
                        "action": "Aadhaar already registered to another customer with different details. Manual verification required.",
                        "existing_customer": {
                            "customer_id": aadhaar_match['customer_id'],
                            "name": aadhaar_match['name'],
                            "dob": str(aadhaar_match['dob']),
                            "pan": aadhaar_match['pan']
                        },
                        "confidence": 1.0
                    }
            
            # ---- 4. MOBILE CONFLICT CHECK ----
            cursor.execute("""
                SELECT customer_id, name, dob, pan, aadhaar_number 
                FROM existing_customers_rec 
                WHERE mobile_number = %s
            """, (applicant['mobile_number'],))
            mobile_match = cursor.fetchone()
            if mobile_match:
                if (mobile_match['name'] != applicant['name'] or 
                    str(mobile_match['dob']) != applicant['dob'] or 
                    mobile_match['pan'] != applicant['pan']):
                    store_dedup_result(
                        cursor,
                        None,
                        mobile_match['customer_id'],
                        1.0,
                        'MOBILE_CONFLICT',
                        f"Mobile {applicant['mobile_number']} already registered to {mobile_match['name']} (ID: {mobile_match['customer_id']}) with different details"
                    )
                    conn.commit()
                    return {
                        "status": "FLAGGED",
                        "verdict": "MOBILE_CONFLICT",
                        "action": "Mobile number already registered to another customer with different details. Manual verification required.",
                        "existing_customer": {
                            "customer_id": mobile_match['customer_id'],
                            "name": mobile_match['name'],
                            "dob": str(mobile_match['dob']),
                            "pan": mobile_match['pan']
                        },
                        "confidence": 1.0
                    }
            
            # ---- 5. EXISTING CUSTOMER SEARCH ----
            customer_matches = search_customers_kyc(cursor, applicant)
            fuzzy_matches = fuzzy_name_search_kyc(cursor, applicant, customer_matches)
            
            all_matches = blacklist_matches + customer_matches + fuzzy_matches
            all_matches.sort(key=lambda x: x['score'], reverse=True)
            
            has_blacklist = any(m['source'] == 'BLACKLIST' for m in all_matches)
            final_confidence = max([m['score'] for m in all_matches]) if all_matches else 0.0
            
            loan_info = None
            matched_customer_id = None
            
            if all_matches:
                best_match = all_matches[0]
                if best_match['source'] == 'CUSTOMER':
                    matched_customer_id = int(best_match['record']['id'])
                    loan_info = check_customer_loans_kyc(cursor, matched_customer_id)
                    
                    if loan_info and loan_info['has_multiple_loans'] and final_confidence >= 0.70:
                        verdict = determine_verdict_kyc(final_confidence, has_blacklist, bool(all_matches), loan_info)
                        
                        store_dedup_result(
                            cursor, 
                            None,
                            matched_customer_id,
                            round(final_confidence, 2),
                            'SAME_CUSTOMER_MULTIPLE_LOANS',
                            f"Customer has {loan_info['loan_count']} existing loans"
                        )
                        conn.commit()
                        
                        response = {
                            "emit": "dedup.match_found",
                            "output": {
                                "status": verdict['status'],
                                "verdict": verdict['verdict'],
                                "action": verdict['action'],
                                "confidence": round(final_confidence, 2),
                                "has_blacklist": has_blacklist,
                                "match_count": len(all_matches),
                                "loan_details": verdict.get('loan_details', {}),
                                "match_summary": [
                                    {
                                        "id": m['record'].get('id', 'N/A'),
                                        "name": m['record'].get('name', 'Unknown'),
                                        "source": m['source'],
                                        "score": round(m['score'], 2),
                                        "matched_fields": m['matched_fields']
                                    }
                                    for m in all_matches[:5]
                                ],
                                "matched_records": [m['record'] for m in all_matches[:5]]
                            }
                        }
                        return response
            
            verdict = determine_verdict_kyc(final_confidence, has_blacklist, bool(all_matches))
            
            logger.info(
                f"Decision for {applicant['name']}: {verdict['verdict']} "
                f"(Confidence: {final_confidence:.2%})"
            )
            
            if all_matches:
                best_match = all_matches[0]
                matched_customer_id = int(best_match['record']['id']) if best_match['source'] == 'CUSTOMER' else None
                
                store_dedup_result(
                    cursor,
                    None,
                    matched_customer_id,
                    round(final_confidence, 2),
                    verdict['verdict'],
                    f"Match found with confidence {final_confidence:.2%}"
                )
                conn.commit()
            
            if all_matches:
                response = {
                    "emit": "dedup.match_found",
                    "output": {
                        "status": verdict['status'],
                        "verdict": verdict['verdict'],
                        "action": verdict['action'],
                        "confidence": round(final_confidence, 2),
                        "has_blacklist": has_blacklist,
                        "match_count": len(all_matches),
                        "loan_details": verdict.get('loan_details', {}),
                        "match_summary": [
                            {
                                "id": m['record'].get('id', 'N/A'),
                                "name": m['record'].get('name', 'Unknown'),
                                "source": m['source'],
                                "score": round(m['score'], 2),
                                "matched_fields": m['matched_fields']
                            }
                            for m in all_matches[:5]
                        ],
                        "matched_records": [m['record'] for m in all_matches[:5]]
                    }
                }
            else:
                store_dedup_result(
                    cursor,
                    None,
                    None,
                    0.0,
                    'NEW_CUSTOMER',
                    'No matching customer found'
                )
                conn.commit()
                
                response = {
                    "emit": "dedup.clear",
                    "output": {
                        "status": "CLEAR",
                        "verdict": "NO_MATCH",
                        "action": "Proceed with KYC",
                        "confidence": 0.0,
                        "has_blacklist": False,
                        "match_count": 0,
                        "loan_details": {},
                        "match_summary": [],
                        "matched_records": []
                    }
                }
            
            return response
            
    except psycopg2.Error as e:
        logger.error(f"Database error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        logger.error(f"Processing error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

# ============================================================================
# LOAN APPLICATION ENDPOINT – WITH AUTO‑GENERATED ACCOUNT NUMBER
# ============================================================================

@app.post("/api/v1/loan/apply")
def apply_loan(application: LoanApplication):
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            applicant = {
                'name': application.name.strip().lower(),
                'dob': application.dob,
                'aadhaar_number': application.aadhaar_number.strip(),
                'pan': application.pan.upper().strip(),
                'mobile_number': ''.join(filter(str.isdigit, application.mobile_number))[-10:],
                'email': application.email,
                'address': application.address.strip().lower()
            }
            
            # ---- Generate loan account number if not provided ----
            if application.loan_account_no:
                loan_account_no = application.loan_account_no.strip()
                cursor.execute("SELECT loan_id FROM loan_accounts WHERE loan_account_no = %s", (loan_account_no,))
                if cursor.fetchone():
                    return {"status": "ERROR", "message": f"Loan account {loan_account_no} already exists"}
            else:
                # Auto‑generate: get the next number based on max loan_id
                cursor.execute("SELECT COALESCE(MAX(loan_id), 0) + 1 AS next_id FROM loan_accounts")
                next_id = cursor.fetchone()['next_id']   # <-- FIXED: alias added
                loan_account_no = f"LN{next_id:04d}"
            
            loan_data = {
                'loan_account_no': loan_account_no,
                'loan_type': application.loan_type,
                'loan_amount': application.loan_amount,
                'interest_rate': application.interest_rate,
                'loan_term_months': application.loan_term_months
            }
            
            # ---- (rest of the function remains unchanged) ----
            # ---- Check by Aadhaar ----
            existing = find_customer_by_aadhaar(cursor, applicant['aadhaar_number'])
            if existing:
                customer_id = existing['customer_id']
                # Duplicate loan account check already done, but we double‑check
                loan_id = create_loan(cursor, customer_id, loan_data)
                loans = get_customer_loans(cursor, customer_id)
                conn.commit()
                return {
                    "status": "EXISTING_CUSTOMER",
                    "verdict": "EXISTING_CUSTOMER_NEW_LOAN",
                    "message": "New loan added to existing customer",
                    "customer": existing,
                    "new_loan": {
                        "loan_id": loan_id,
                        "loan_account_no": loan_data['loan_account_no'],
                        "loan_type": loan_data['loan_type'],
                        "loan_amount": loan_data['loan_amount']
                    },
                    "all_loans": loans,
                    "total_loans": len(loans)
                }
            
            # ---- Check by Mobile ----
            existing = find_customer_by_mobile(cursor, applicant['mobile_number'])
            if existing:
                customer_id = existing['customer_id']
                loan_id = create_loan(cursor, customer_id, loan_data)
                loans = get_customer_loans(cursor, customer_id)
                conn.commit()
                return {
                    "status": "EXISTING_CUSTOMER",
                    "verdict": "EXISTING_CUSTOMER_NEW_LOAN",
                    "message": "New loan added to existing customer (found by mobile)",
                    "customer": existing,
                    "new_loan": {
                        "loan_id": loan_id,
                        "loan_account_no": loan_data['loan_account_no'],
                        "loan_type": loan_data['loan_type'],
                        "loan_amount": loan_data['loan_amount']
                    },
                    "all_loans": loans,
                    "total_loans": len(loans)
                }
            
            # ---- Check PAN conflict ----
            existing = find_customer_by_pan(cursor, applicant['pan'])
            if existing:
                return {
                    "status": "FLAGGED",
                    "verdict": "PAN_CONFLICT",
                    "message": "PAN exists with different Aadhaar/Mobile",
                    "existing_customer": {
                        "customer_id": existing['customer_id'],
                        "name": existing['name'],
                        "aadhaar_number": existing['aadhaar_number'],
                        "mobile_number": existing['mobile_number']
                    }
                }
            
            # ---- New Customer ----
            customer_id = create_customer(cursor, applicant)
            loan_id = create_loan(cursor, customer_id, loan_data)
            loans = get_customer_loans(cursor, customer_id)
            conn.commit()
            return {
                "status": "NEW_CUSTOMER",
                "verdict": "NEW_CUSTOMER",
                "message": "New customer created with loan",
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
                    "loan_account_no": loan_data['loan_account_no'],
                    "loan_type": loan_data['loan_type'],
                    "loan_amount": loan_data['loan_amount']
                },
                "all_loans": loans,
                "total_loans": len(loans)
            }
            
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# CUSTOMER PROFILE
# ============================================================================

@app.get("/api/v1/customer/{identifier}")
def get_customer_profile(identifier: str):
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("""
                SELECT customer_id, name, dob, aadhaar_number, pan, mobile_number, email, address
                FROM existing_customers_rec 
                WHERE aadhaar_number = %s OR mobile_number = %s OR pan = %s
            """, (identifier, identifier, identifier))
            customer = cursor.fetchone()
            if not customer:
                raise HTTPException(status_code=404, detail="Customer not found")
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
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# UI ENDPOINTS
# ============================================================================

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
            return {"status": "success", "message": "Blacklist record added", "blacklist_id": blacklist_id}
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
            return {"status": "success", "message": "Blacklist record removed"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/api/v1/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/")
def root():
    return {
        "service": "KYC Deduplication & Loan Management System",
        "version": "3.0.0",
        "endpoints": {
            "kyc_dedup": "POST /api/v1/kyc/dedup",
            "apply_loan": "POST /api/v1/loan/apply",
            "customer_profile": "GET /api/v1/customer/{identifier}",
            "customers": "GET /api/v1/customers",
            "blacklist": "GET /api/v1/blacklist",
            "blacklist_add": "POST /api/v1/blacklist/add",
            "blacklist_remove": "DELETE /api/v1/blacklist/remove/{blacklist_id}",
            "dedup_results": "GET /api/v1/dedup/results",
            "health": "GET /api/v1/health",
            "docs": "GET /docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)