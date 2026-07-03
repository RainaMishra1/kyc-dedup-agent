import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.fuzz import partial_ratio
import re
from datetime import datetime
from typing import List, Dict, Any, Optional

# ============================================================================
# DATABASE CONNECTION
# ============================================================================

def get_db_connection():
    return psycopg2.connect(
        host="localhost",
        database="postgres",
        user="postgres",
        password="Root",
        port="5432"
    )

# ============================================================================
# CONFIGURATION
# ============================================================================

class DedupeWeights:
    # Primary Identifiers (Highest Importance)
    AADHAAR = 0.40      # 12-digit Aadhaar
    MOBILE = 0.25       # 10-digit Mobile
    PAN = 0.20          # 10-digit PAN
    
    # Secondary Identifiers
    NAME = 0.10         # Fuzzy name match
    DOB = 0.05          # Date of Birth
    
    TOTAL = 1.0
    
    # Thresholds
    EXACT_MATCH = 1.0
    HIGH_CONFIDENCE = 0.85
    MEDIUM_CONFIDENCE = 0.70
    LOW_CONFIDENCE = 0.50
    WEAK_MATCH = 0.30
    
    # Fuzzy Thresholds
    FUZZY_NAME_THRESHOLD = 0.85
    
    # Old weights (for backward compatibility)
    PAN_OLD = 0.35
    AADHAAR_LAST4_OLD = 0.25
    NAME_OLD = 0.15
    DOB_OLD = 0.12
    PHONE_OLD = 0.08
    ADDRESS_OLD = 0.05
    BLACKLIST_MULTIPLIER = 1.5
    ADDRESS_THRESHOLD = 0.70

# ============================================================================
# FIELD COMPARISON FUNCTIONS (For KYC Dedup)
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
    
    elif field_type == 'aadhaar_last4':
        v1_last4 = v1[-4:] if len(v1) >= 4 else v1
        v2_last4 = v2[-4:] if len(v2) >= 4 else v2
        return 1.0 if v1_last4 == v2_last4 else 0.0
    
    elif field_type == 'aadhaar_full':
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
    
    elif field_type == 'phone' or field_type == 'mobile':
        v1_clean = re.sub(r'\D', '', v1)[-10:]
        v2_clean = re.sub(r'\D', '', v2)[-10:]
        return 1.0 if v1_clean == v2_clean else 0.0
    
    elif field_type == 'address':
        similarity = partial_ratio(v1.lower(), v2.lower()) / 100.0
        return similarity if similarity >= DedupeWeights.ADDRESS_THRESHOLD else 0.0
    
    return 0.0

# ============================================================================
# CUMULATIVE SCORING (For KYC Dedup - Old Method)
# ============================================================================

def calculate_cumulative_score(applicant: Dict, db_record: Dict, is_blacklist: bool = False) -> tuple:
    score = 0.0
    matched_fields = []
    
    field_weights = {
        'pan': DedupeWeights.PAN_OLD,
        'aadhaar_last4': DedupeWeights.AADHAAR_LAST4_OLD,
        'name': DedupeWeights.NAME_OLD,
        'dob': DedupeWeights.DOB_OLD,
        'phone': DedupeWeights.PHONE_OLD,
        'address': DedupeWeights.ADDRESS_OLD
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
# NEW DEDUP SCORING (For Loan Apply - Aadhaar + Mobile)
# ============================================================================

def calculate_dedup_score(applicant: Dict, db_record: Dict) -> tuple:
    score = 0.0
    matched_fields = []
    match_type = None
    
    # 1. Aadhaar Match (Primary - 40% weight)
    if applicant.get('aadhaar_number') and db_record.get('aadhaar_number'):
        if applicant['aadhaar_number'] == db_record['aadhaar_number']:
            score += DedupeWeights.AADHAAR
            matched_fields.append('aadhaar_number')
            match_type = 'EXACT_AADHAAR'
    
    # 2. Mobile Match (25% weight)
    if applicant.get('mobile_number') and db_record.get('mobile_number'):
        if applicant['mobile_number'] == db_record['mobile_number']:
            score += DedupeWeights.MOBILE
            matched_fields.append('mobile_number')
            if not match_type:
                match_type = 'EXACT_MOBILE'
    
    # 3. PAN Match (20% weight)
    if applicant.get('pan') and db_record.get('pan'):
        if applicant['pan'] == db_record['pan']:
            score += DedupeWeights.PAN
            matched_fields.append('pan')
            if not match_type:
                match_type = 'EXACT_PAN'
    
    # 4. Fuzzy Name Match (10% weight)
    if applicant.get('name') and db_record.get('name'):
        name_similarity = JaroWinkler.similarity(
            applicant['name'].lower(), 
            db_record['name'].lower()
        )
        if name_similarity >= DedupeWeights.FUZZY_NAME_THRESHOLD:
            score += DedupeWeights.NAME * name_similarity
            matched_fields.append('name_fuzzy')
            if not match_type:
                match_type = 'FUZZY_NAME'
    
    # 5. DOB Match (5% weight)
    if applicant.get('dob') and db_record.get('dob'):
        if applicant['dob'] == db_record['dob']:
            score += DedupeWeights.DOB
            matched_fields.append('dob')
            if not match_type:
                match_type = 'EXACT_DOB'
    
    return round(score, 2), matched_fields, match_type

# ============================================================================
# LOAN FUNCTIONS
# ============================================================================

def check_customer_loans(cursor, customer_id: int) -> Dict:
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

def find_similar_customers(cursor, applicant: Dict) -> List[Dict]:
    matches = []
    
    # 1. Exact Aadhaar Match
    if applicant.get('aadhaar_number'):
        cursor.execute("""
            SELECT customer_id, name, dob, aadhaar_number, pan, mobile_number, email, address
            FROM existing_customers_rec 
            WHERE aadhaar_number = %s
        """, (applicant['aadhaar_number'],))
        records = cursor.fetchall()
        for record in records:
            score, fields, match_type = calculate_dedup_score(applicant, dict(record))
            if score > 0:
                matches.append({
                    'record': dict(record),
                    'score': score,
                    'matched_fields': fields,
                    'match_type': match_type or 'EXACT_AADHAAR'
                })
    
    # 2. Exact Mobile Match
    if applicant.get('mobile_number'):
        cursor.execute("""
            SELECT customer_id, name, dob, aadhaar_number, pan, mobile_number, email, address
            FROM existing_customers_rec 
            WHERE mobile_number = %s
        """, (applicant['mobile_number'],))
        records = cursor.fetchall()
        for record in records:
            if any(m['record']['customer_id'] == record['customer_id'] for m in matches):
                continue
            score, fields, match_type = calculate_dedup_score(applicant, dict(record))
            if score > 0:
                matches.append({
                    'record': dict(record),
                    'score': score,
                    'matched_fields': fields,
                    'match_type': match_type or 'EXACT_MOBILE'
                })
    
    # 3. PAN Match
    if applicant.get('pan'):
        cursor.execute("""
            SELECT customer_id, name, dob, aadhaar_number, pan, mobile_number, email, address
            FROM existing_customers_rec 
            WHERE pan = %s
        """, (applicant['pan'],))
        records = cursor.fetchall()
        for record in records:
            if any(m['record']['customer_id'] == record['customer_id'] for m in matches):
                continue
            score, fields, match_type = calculate_dedup_score(applicant, dict(record))
            if score > 0:
                matches.append({
                    'record': dict(record),
                    'score': score,
                    'matched_fields': fields,
                    'match_type': match_type or 'EXACT_PAN'
                })
    
    # 4. Fuzzy Name + DOB Match
    if applicant.get('name') and applicant.get('dob'):
        cursor.execute("""
            SELECT customer_id, name, dob, aadhaar_number, pan, mobile_number, email, address
            FROM existing_customers_rec 
            WHERE dob = %s
        """, (applicant['dob'],))
        records = cursor.fetchall()
        for record in records:
            if any(m['record']['customer_id'] == record['customer_id'] for m in matches):
                continue
            score, fields, match_type = calculate_dedup_score(applicant, dict(record))
            if score > 0 and score >= DedupeWeights.WEAK_MATCH:
                matches.append({
                    'record': dict(record),
                    'score': score,
                    'matched_fields': fields,
                    'match_type': match_type or 'FUZZY_NAME_DOB'
                })
    
    return sorted(matches, key=lambda x: x['score'], reverse=True)

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

def check_blacklist(cursor, aadhaar_number: str, mobile_number: str, pan: str) -> Optional[Dict]:
    cursor.execute("""
        SELECT blacklist_id, name, reason
        FROM blacklist_record 
        WHERE aadhaar_number = %s OR mobile_number = %s OR pan = %s
    """, (aadhaar_number, mobile_number, pan))
    return cursor.fetchone()

def store_dedup_result(cursor, customer_id: Optional[int], matched_customer_id: Optional[int], 
                       match_score: float, result_type: str, explanation: str):
    cursor.execute("""
        INSERT INTO deduplication_results 
        (customer_id, matched_customer_id, match_score, result_type, explanation)
        VALUES (%s, %s, %s, %s, %s)
    """, (customer_id, matched_customer_id, match_score, result_type, explanation))

def determine_dedup_verdict(score: float, match_type: str, has_loans: bool, loan_count: int) -> dict:
    if score >= DedupeWeights.EXACT_MATCH:
        if has_loans:
            return {
                'status': 'EXISTING_CUSTOMER',
                'verdict': 'EXISTING_CUSTOMER_NEW_LOAN',
                'action': 'Customer exists. Adding new loan to existing profile.',
                'confidence': score
            }
        return {
            'status': 'EXACT_CUSTOMER_MATCH',
            'verdict': 'EXACT_CUSTOMER_MATCH',
            'action': 'Exact match found. Customer already exists.',
            'confidence': score
        }
    elif score >= DedupeWeights.HIGH_CONFIDENCE:
        return {
            'status': 'HIGH_CONFIDENCE_MATCH',
            'verdict': 'HIGH_CONFIDENCE_MATCH',
            'action': 'High confidence match. Manual verification recommended.',
            'confidence': score
        }
    elif score >= DedupeWeights.MEDIUM_CONFIDENCE:
        return {
            'status': 'MEDIUM_CONFIDENCE_MATCH',
            'verdict': 'MEDIUM_CONFIDENCE_MATCH',
            'action': 'Medium confidence match. Manual verification required.',
            'confidence': score
        }
    elif score >= DedupeWeights.LOW_CONFIDENCE:
        return {
            'status': 'POSSIBLE_DUPLICATE',
            'verdict': 'POSSIBLE_DUPLICATE',
            'action': 'Possible duplicate found. Verification required.',
            'confidence': score
        }
    else:
        return {
            'status': 'NEW_CUSTOMER',
            'verdict': 'NEW_CUSTOMER',
            'action': 'No match found. Creating new customer profile.',
            'confidence': score
        }

# ============================================================================
# MAIN PROCESSING FUNCTIONS
# ============================================================================

def process_loan_application(application_data: Dict) -> Dict:
    """Process loan application with deduplication"""
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        applicant = {
            'name': application_data['name'].strip().lower(),
            'dob': application_data['dob'],
            'aadhaar_number': application_data['aadhaar_number'].strip(),
            'pan': application_data['pan'].upper().strip(),
            'mobile_number': ''.join(filter(str.isdigit, application_data['mobile_number']))[-10:],
            'email': application_data.get('email'),
            'address': application_data['address'].strip().lower()
        }
        
        loan_data = {
            'loan_account_no': application_data['loan_account_no'],
            'loan_type': application_data['loan_type'],
            'loan_amount': application_data['loan_amount'],
            'interest_rate': application_data.get('interest_rate'),
            'loan_term_months': application_data.get('loan_term_months')
        }
        
        # STEP 1: Check Blacklist
        blacklist = check_blacklist(
            cursor, 
            applicant['aadhaar_number'], 
            applicant['mobile_number'], 
            applicant['pan']
        )
        
        if blacklist:
            return {
                "status": "REJECTED",
                "message": "Customer is BLACKLISTED",
                "reason": blacklist['reason'],
                "blacklist_id": blacklist['blacklist_id']
            }
        
        # STEP 2: Find by Aadhaar (Primary)
        existing = find_customer_by_aadhaar(cursor, applicant['aadhaar_number'])
        
        if existing:
            customer_id = existing['customer_id']
            loans = get_customer_loans(cursor, customer_id)
            
            cursor.execute("SELECT loan_id FROM loan_accounts WHERE loan_account_no = %s", 
                         (loan_data['loan_account_no'],))
            if cursor.fetchone():
                return {
                    "status": "ERROR",
                    "message": f"Loan account {loan_data['loan_account_no']} already exists"
                }
            
            loan_id = create_loan(cursor, customer_id, loan_data)
            updated_loans = get_customer_loans(cursor, customer_id)
            
            store_dedup_result(
                cursor,
                None,
                customer_id,
                1.0,
                'EXISTING_CUSTOMER_NEW_LOAN',
                f"Customer found by Aadhaar. Added new loan. Total loans: {len(updated_loans)}"
            )
            conn.commit()
            
            return {
                "status": "EXISTING_CUSTOMER",
                "verdict": "EXISTING_CUSTOMER_NEW_LOAN",
                "message": f"New loan added to existing customer profile",
                "customer": {
                    "customer_id": existing['customer_id'],
                    "name": existing['name'],
                    "dob": existing['dob'],
                    "aadhaar_number": existing['aadhaar_number'],
                    "pan": existing['pan'],
                    "mobile_number": existing['mobile_number'],
                    "email": existing['email'],
                    "address": existing['address']
                },
                "new_loan": {
                    "loan_id": loan_id,
                    "loan_account_no": loan_data['loan_account_no'],
                    "loan_type": loan_data['loan_type'],
                    "loan_amount": loan_data['loan_amount'],
                    "status": "ACTIVE"
                },
                "all_loans": updated_loans,
                "total_loans": len(updated_loans)
            }
        
        # STEP 3: Find by Mobile (Secondary)
        existing = find_customer_by_mobile(cursor, applicant['mobile_number'])
        
        if existing:
            customer_id = existing['customer_id']
            loans = get_customer_loans(cursor, customer_id)
            
            cursor.execute("SELECT loan_id FROM loan_accounts WHERE loan_account_no = %s", 
                         (loan_data['loan_account_no'],))
            if cursor.fetchone():
                return {
                    "status": "ERROR",
                    "message": f"Loan account {loan_data['loan_account_no']} already exists"
                }
            
            loan_id = create_loan(cursor, customer_id, loan_data)
            updated_loans = get_customer_loans(cursor, customer_id)
            
            store_dedup_result(
                cursor,
                None,
                customer_id,
                0.85,
                'EXISTING_CUSTOMER_NEW_LOAN',
                f"Customer found by Mobile. Added new loan. Total loans: {len(updated_loans)}"
            )
            conn.commit()
            
            return {
                "status": "EXISTING_CUSTOMER",
                "verdict": "EXISTING_CUSTOMER_NEW_LOAN",
                "message": f"Customer found by mobile number. New loan added.",
                "customer": {
                    "customer_id": existing['customer_id'],
                    "name": existing['name'],
                    "dob": existing['dob'],
                    "aadhaar_number": existing['aadhaar_number'],
                    "pan": existing['pan'],
                    "mobile_number": existing['mobile_number'],
                    "email": existing['email'],
                    "address": existing['address']
                },
                "new_loan": {
                    "loan_id": loan_id,
                    "loan_account_no": loan_data['loan_account_no'],
                    "loan_type": loan_data['loan_type'],
                    "loan_amount": loan_data['loan_amount'],
                    "status": "ACTIVE"
                },
                "all_loans": updated_loans,
                "total_loans": len(updated_loans)
            }
        
        # STEP 4: Find by PAN (Conflict Check)
        existing = find_customer_by_pan(cursor, applicant['pan'])
        
        if existing:
            return {
                "status": "FLAGGED",
                "verdict": "PAN_CONFLICT",
                "message": "PAN already exists with different Aadhaar or Mobile",
                "existing_customer": {
                    "customer_id": existing['customer_id'],
                    "name": existing['name'],
                    "aadhaar_number": existing['aadhaar_number'],
                    "mobile_number": existing['mobile_number']
                },
                "action": "Manual verification required"
            }
        
        # STEP 5: Find Similar Customers (Dedup)
        similar = find_similar_customers(cursor, applicant)
        
        if similar and similar[0]['score'] >= DedupeWeights.MEDIUM_CONFIDENCE:
            best_match = similar[0]
            verdict = determine_dedup_verdict(
                best_match['score'], 
                best_match['match_type'],
                False,
                0
            )
            
            store_dedup_result(
                cursor,
                None,
                best_match['record']['customer_id'],
                best_match['score'],
                best_match['match_type'],
                f"Possible duplicate found with score {best_match['score']}"
            )
            conn.commit()
            
            return {
                "status": verdict['status'],
                "verdict": verdict['verdict'],
                "message": "Possible duplicate customer found",
                "matches": [
                    {
                        "customer_id": m['record']['customer_id'],
                        "name": m['record']['name'],
                        "aadhaar_number": m['record']['aadhaar_number'],
                        "mobile_number": m['record']['mobile_number'],
                        "score": m['score'],
                        "matched_fields": m['matched_fields'],
                        "match_type": m['match_type']
                    }
                    for m in similar[:5]
                ],
                "action": verdict['action']
            }
        
        # STEP 6: Create New Customer
        customer_id = create_customer(cursor, applicant)
        loan_id = create_loan(cursor, customer_id, loan_data)
        loans = get_customer_loans(cursor, customer_id)
        
        store_dedup_result(
            cursor,
            customer_id,
            None,
            0.0,
            'NEW_CUSTOMER',
            'New customer created with first loan'
        )
        conn.commit()
        
        return {
            "status": "NEW_CUSTOMER",
            "verdict": "NEW_CUSTOMER",
            "message": "New customer created with loan",
            "customer": {
                "customer_id": customer_id,
                "name": application_data['name'],
                "dob": application_data['dob'],
                "aadhaar_number": applicant['aadhaar_number'],
                "pan": applicant['pan'],
                "mobile_number": applicant['mobile_number'],
                "email": application_data.get('email'),
                "address": application_data['address']
            },
            "new_loan": {
                "loan_id": loan_id,
                "loan_account_no": loan_data['loan_account_no'],
                "loan_type": loan_data['loan_type'],
                "loan_amount": loan_data['loan_amount'],
                "status": "ACTIVE"
            },
            "all_loans": loans,
            "total_loans": len(loans)
        }
        
    except Exception as e:
        print(f"Error processing loan: {str(e)}")
        return {"status": "ERROR", "message": str(e)}
    finally:
        cursor.close()
        conn.close()

def get_customer_profile(identifier: str) -> Dict:
    """Get customer by Aadhaar, Mobile, or PAN"""
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        cursor.execute("""
            SELECT customer_id, name, dob, aadhaar_number, pan, mobile_number, email, address
            FROM existing_customers_rec 
            WHERE aadhaar_number = %s OR mobile_number = %s OR pan = %s
        """, (identifier, identifier, identifier))
        
        customer = cursor.fetchone()
        if not customer:
            return {"status": "ERROR", "message": "Customer not found"}
        
        loans = get_customer_loans(cursor, customer['customer_id'])
        
        return {
            "status": "SUCCESS",
            "customer": customer,
            "loans": loans,
            "total_loans": len(loans),
            "has_multiple_loans": len(loans) > 1
        }
        
    except Exception as e:
        return {"status": "ERROR", "message": str(e)}
    finally:
        cursor.close()
        conn.close()

# ============================================================================
# OLD KYC DEDUP PROCESSOR (For Backward Compatibility)
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
            aadhaar_last4,
            dob,
            phone
        FROM blacklist_record 
        WHERE pan = %s OR (aadhaar_last4 = %s AND dob = %s)
        LIMIT 10;
    """, (applicant['pan'], applicant['aadhaar_last4'], applicant['dob']))
    
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
                aadhaar_last4,
                dob,
                phone
            FROM blacklist_record 
            WHERE phone = %s
            LIMIT 5;
        """, (applicant['phone'],))
        
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

def search_customers_old(cursor, applicant: Dict) -> List[Dict]:
    matches = []
    
    cursor.execute("""
        SELECT 
            'CUSTOMER_DB' as source,
            customer_id::text as id,
            name,
            address,
            pan,
            aadhaar_last4,
            dob,
            phone
        FROM existing_customers_rec 
        WHERE pan = %s OR (aadhaar_last4 = %s AND dob = %s)
        LIMIT 10;
    """, (applicant['pan'], applicant['aadhaar_last4'], applicant['dob']))
    
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
            aadhaar_last4,
            dob,
            phone
        FROM existing_customers_rec 
        WHERE phone = %s
        LIMIT 5;
    """, (applicant['phone'],))
    
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

def fuzzy_name_search_old(cursor, applicant: Dict, existing_matches: List) -> List[Dict]:
    matches = []
    
    cursor.execute("""
        SELECT 
            'CUSTOMER_DB' as source,
            customer_id::text as id,
            name,
            address,
            pan,
            aadhaar_last4,
            dob,
            phone
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
            score = name_score * DedupeWeights.NAME_OLD
            matches.append({
                'record': candidate_dict,
                'score': score,
                'matched_fields': ['name'],
                'source': 'CUSTOMER'
            })
    
    return matches

def determine_verdict_old(confidence: float, is_blacklist: bool, has_matches: bool, loan_info: Dict = None) -> dict:
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
    
    if confidence >= 1.0:
        return {
            'status': 'REJECTED',
            'verdict': 'EXACT_MATCH',
            'action': 'Auto-reject application - Exact ID match found'
        }
    elif confidence >= 0.85:
        return {
            'status': 'REJECTED',
            'verdict': 'HIGH_CONFIDENCE_MATCH',
            'action': 'Reject with manual verification'
        }
    elif confidence >= 0.70:
        return {
            'status': 'REVIEW',
            'verdict': 'MEDIUM_CONFIDENCE_MATCH',
            'action': 'Send to manual review team'
        }
    elif confidence >= 0.50:
        return {
            'status': 'REVIEW',
            'verdict': 'LOW_CONFIDENCE_MATCH',
            'action': 'Request additional verification documents'
        }
    elif confidence >= 0.30:
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

def process_dedup_old(event_payload):
    """Old KYC Dedup processor for backward compatibility"""
    applicant = event_payload["reads"]
    
    input_name = applicant["name"].strip().lower()
    input_dob = applicant["dob"]
    input_pan = applicant["pan"].strip().upper()
    input_phone = ''.join(filter(str.isdigit, applicant["phone"]))[-10:] 
    input_aadhaar = applicant["aadhaar_last4"].strip()
    input_address = applicant["address"].strip().lower()
    
    applicant_dict = {
        'name': input_name,
        'dob': input_dob,
        'pan': input_pan,
        'phone': input_phone,
        'aadhaar_last4': input_aadhaar,
        'address': input_address
    }

    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    try:
        all_matches = []
        final_confidence = 0.0
        source = None
        match_reason = []
        loan_info = None
        matched_customer_id = None
        
        # Check Blacklist
        blacklist_matches = search_blacklist(cursor, applicant_dict)
        for match in blacklist_matches:
            all_matches.append(match)
            final_confidence = max(final_confidence, match['score'])
            match_reason.append("Blacklist match")
            source = 'BLACKLIST'
        
        # Check Existing Customers
        customer_matches = search_customers_old(cursor, applicant_dict)
        for match in customer_matches:
            all_matches.append(match)
            final_confidence = max(final_confidence, match['score'])
            match_reason.append("Customer match")
            source = 'CUSTOMER'
            if not matched_customer_id:
                matched_customer_id = int(match['record']['id'])
        
        # Fuzzy Name Search
        fuzzy_matches = fuzzy_name_search_old(cursor, applicant_dict, customer_matches)
        for match in fuzzy_matches:
            all_matches.append(match)
            final_confidence = max(final_confidence, match['score'])
            match_reason.append("Fuzzy name match")
            source = 'CUSTOMER'
        
        all_matches.sort(key=lambda x: x['score'], reverse=True)
        has_blacklist = any(m['source'] == 'BLACKLIST' for m in all_matches)
        
        if matched_customer_id:
            loan_info = check_customer_loans(cursor, matched_customer_id)
            
            if loan_info and loan_info['has_multiple_loans'] and final_confidence >= 0.70:
                store_dedup_result(
                    cursor,
                    None,
                    matched_customer_id,
                    round(final_confidence, 2),
                    'SAME_CUSTOMER_MULTIPLE_LOANS',
                    f"Customer has {loan_info['loan_count']} existing loans"
                )
                conn.commit()
                
                verdict = determine_verdict_old(final_confidence, has_blacklist, bool(all_matches), loan_info)
                
                return {
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
        
        verdict = determine_verdict_old(final_confidence, has_blacklist, bool(all_matches), loan_info)
        
        if all_matches:
            store_dedup_result(
                cursor,
                None,
                matched_customer_id,
                round(final_confidence, 2),
                verdict['verdict'],
                f"Match found with confidence {final_confidence:.2%}"
            )
            conn.commit()
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
        
        if all_matches:
            return {
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
            return {
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

    except Exception as e:
        print(f"Error in deduplication: {str(e)}")
        return {
            "emit": "dedup.error",
            "output": {
                "status": "ERROR",
                "verdict": "SYSTEM_ERROR",
                "confidence": 0.0,
                "error": str(e)
            }
        }
    finally:
        cursor.close()
        conn.close()

# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("WORKER.PY - KYC Dedup + Loan Management")
    print("="*60)
    
    # Test Loan Application
    print("\n1. Testing Loan Application...")
    test_loan = {
        "name": "Rahul Sharma",
        "dob": "1992-05-12",
        "aadhaar_number": "123456789012",
        "pan": "ABCDE1234F",
        "mobile_number": "9876543210",
        "email": "rahul@email.com",
        "address": "Mumbai, Maharashtra",
        "loan_type": "Home Loan",
        "loan_amount": 4500000,
        "loan_account_no": "HL001",
        "interest_rate": 8.5,
        "loan_term_months": 240
    }
    result = process_loan_application(test_loan)
    print(json.dumps(result, indent=2))
    
    print("\n" + "="*60)
    print("✅ WORKER.PY READY!")
    print("="*60)