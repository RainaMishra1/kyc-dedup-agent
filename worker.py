import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.fuzz import partial_ratio
import re
from datetime import datetime

# 1. Database Connection Helper
def get_db_connection():
    return psycopg2.connect(
        host="localhost",
        database="postgres",  # Changed from kyc_dedup_db to postgres
        user="postgres",
        password="Root",
        port="5432"
    )
# 2. Field Weight Configuration
class DedupeWeights:
    # Primary identifiers (highest uniqueness)
    PAN = 0.35
    AADHAAR_LAST4 = 0.25
    
    # Secondary identifiers (medium uniqueness)
    NAME = 0.15
    DOB = 0.12
    
    # Tertiary identifiers (low uniqueness, often change)
    PHONE = 0.08
    ADDRESS = 0.05
    
    # Total = 1.0 (100%)
    
    # Thresholds
    BLACKLIST_MULTIPLIER = 1.5
    FUZZY_NAME_THRESHOLD = 0.85
    ADDRESS_THRESHOLD = 0.70

# 3. Field Comparison Functions
def compare_fields(field_type, value1, value2):
    """
    Returns match confidence (0.0 to 1.0) for a specific field.
    """
    if not value1 or not value2:
        return 0.0
    
    # Normalize values
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
    
    elif field_type == 'phone':
        v1_clean = re.sub(r'\D', '', v1)[-10:]
        v2_clean = re.sub(r'\D', '', v2)[-10:]
        return 1.0 if v1_clean == v2_clean else 0.0
    
    elif field_type == 'address':
        similarity = partial_ratio(v1.lower(), v2.lower()) / 100.0
        return similarity if similarity >= DedupeWeights.ADDRESS_THRESHOLD else 0.0
    
    return 0.0

# 4. Calculate Cumulative Score for a Single Record
def calculate_record_score(applicant, db_record, is_blacklist=False):
    """
    Calculate cumulative confidence score for a single database record.
    Returns: (score, matched_fields)
    """
    score = 0.0
    matched_fields = []
    
    field_weights = {
        'pan': DedupeWeights.PAN,
        'aadhaar_last4': DedupeWeights.AADHAAR_LAST4,
        'name': DedupeWeights.NAME,
        'dob': DedupeWeights.DOB,
        'phone': DedupeWeights.PHONE,
        'address': DedupeWeights.ADDRESS
    }
    
    for field, weight in field_weights.items():
        if field in applicant and field in db_record:
            match_score = compare_fields(field, applicant[field], db_record[field])
            if match_score > 0:
                score += weight * match_score
                matched_fields.append({
                    'field': field,
                    'similarity': round(match_score * 100, 2),
                    'weight': round(weight * 100, 2)
                })
    
    if is_blacklist and score > 0:
        score = min(score * DedupeWeights.BLACKLIST_MULTIPLIER, 1.0)
    
    return score, matched_fields

# 5. Check Customer Loans
def check_customer_loans(cursor, customer_id):
    """Check if a customer has multiple loan accounts"""
    cursor.execute("""
        SELECT 
            COUNT(*) as loan_count,
            ARRAY_AGG(loan_account_no) as loan_accounts,
            ARRAY_AGG(loan_type) as loan_types,
            ARRAY_AGG(loan_status) as loan_statuses,
            ARRAY_AGG(loan_amount) as loan_amounts
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
            'loan_amounts': result['loan_amounts'],
            'has_multiple_loans': result['loan_count'] > 1
        }
    return {
        'has_loans': False,
        'loan_count': 0,
        'loan_accounts': [],
        'loan_types': [],
        'loan_statuses': [],
        'loan_amounts': [],
        'has_multiple_loans': False
    }

# 6. Store Deduplication Result
def store_dedup_result(cursor, customer_id, matched_customer_id, match_score, result_type, explanation):
    """Store deduplication result in deduplication_results table"""
    cursor.execute("""
        INSERT INTO deduplication_results 
        (customer_id, matched_customer_id, match_score, result_type, explanation)
        VALUES (%s, %s, %s, %s, %s)
    """, (customer_id, matched_customer_id, match_score, result_type, explanation))

# 7. Main Deduplication Engine
def process_dedup(event_payload):
    applicant = event_payload["reads"]
    
    # Normalize inputs
    input_name = applicant["name"].strip().lower()
    input_dob = applicant["dob"]
    input_pan = applicant["pan"].strip().upper()
    input_phone = ''.join(filter(str.isdigit, applicant["phone"]))[-10:] 
    input_aadhaar = applicant["aadhaar_last4"].strip()

    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    try:
        all_matches = []
        final_confidence = 0.0
        source = None
        match_reason = []
        loan_info = None
        matched_customer_id = None
        
        # ------------------------------------------------------------------
        # STAGE 1: CHECK BLACKLIST DATABASE (Using correct table name: blacklist_record)
        # ------------------------------------------------------------------
        
        # Check 1: Blacklist with PAN or Aadhaar+DOB
        cursor.execute("""
            SELECT 
                'BLACKLIST_DB' as source, 
                blacklist_id as id, 
                name, 
                reason,
                pan,
                aadhaar_last4,
                dob,
                phone
            FROM blacklist_record
            WHERE pan = %s OR (aadhaar_last4 = %s AND dob = %s);
        """, (input_pan, input_aadhaar, input_dob))
        
        bl_hard_matches = cursor.fetchall()
        
        for record in bl_hard_matches:
            record_dict = dict(record)
            record_dict['address'] = ''  # Blacklist doesn't have address
            score, matched_fields = calculate_record_score(applicant, record_dict, is_blacklist=True)
            if score > 0:
                all_matches.append({
                    'record': record_dict,
                    'score': score,
                    'matched_fields': matched_fields,
                    'source': 'BLACKLIST'
                })
                final_confidence = max(final_confidence, score)
                match_reason.append(f"Blacklist match: {record['reason']}")
                source = 'BLACKLIST'
        
        # Check 2: Blacklist with Phone Only (if no hard matches found)
        if not bl_hard_matches:
            cursor.execute("""
                SELECT 
                    'BLACKLIST_DB' as source, 
                    blacklist_id as id, 
                    name, 
                    reason,
                    pan,
                    aadhaar_last4,
                    dob,
                    phone
                FROM blacklist_record
                WHERE phone = %s;
            """, (input_phone,))
            
            bl_soft_matches = cursor.fetchall()
            
            for record in bl_soft_matches:
                record_dict = dict(record)
                record_dict['address'] = ''
                score, matched_fields = calculate_record_score(applicant, record_dict, is_blacklist=True)
                if score > 0 and score < 1.0:
                    all_matches.append({
                        'record': record_dict,
                        'score': score,
                        'matched_fields': matched_fields,
                        'source': 'BLACKLIST'
                    })
                    final_confidence = max(final_confidence, score)
                    match_reason.append(f"Blacklist phone match: {record['reason']}")
                    source = 'BLACKLIST'
        
        # ------------------------------------------------------------------
        # STAGE 2: CHECK EXISTING CUSTOMER DATABASE
        # ------------------------------------------------------------------
        
        # Check 3: Customer with PAN or Aadhaar+DOB
        cursor.execute("""
            SELECT 
                'CUSTOMER_DB' as source, 
                customer_id as id, 
                name, 
                address,
                pan,
                aadhaar_last4,
                dob,
                phone
            FROM existing_customers_rec
            WHERE pan = %s OR (aadhaar_last4 = %s AND dob = %s);
        """, (input_pan, input_aadhaar, input_dob))
        
        cust_hard_matches = cursor.fetchall()
        
        for record in cust_hard_matches:
            record_dict = dict(record)
            score, matched_fields = calculate_record_score(applicant, record_dict, is_blacklist=False)
            if score > 0:
                all_matches.append({
                    'record': record_dict,
                    'score': score,
                    'matched_fields': matched_fields,
                    'source': 'CUSTOMER'
                })
                final_confidence = max(final_confidence, score)
                match_reason.append("Customer match via PAN or Aadhaar+DOB")
                source = 'CUSTOMER'
                matched_customer_id = record['id']
        
        # Check 4: Customer with Phone Only
        cursor.execute("""
            SELECT 
                'CUSTOMER_DB' as source, 
                customer_id as id, 
                name, 
                address,
                pan,
                aadhaar_last4,
                dob,
                phone
            FROM existing_customers_rec
            WHERE phone = %s;
        """, (input_phone,))
        
        cust_soft_matches = cursor.fetchall()
        
        for record in cust_soft_matches:
            record_dict = dict(record)
            if any(m['record']['id'] == record_dict['id'] for m in all_matches):
                continue
            score, matched_fields = calculate_record_score(applicant, record_dict, is_blacklist=False)
            if score > 0 and score < 0.90:
                all_matches.append({
                    'record': record_dict,
                    'score': score,
                    'matched_fields': matched_fields,
                    'source': 'CUSTOMER'
                })
                final_confidence = max(final_confidence, score)
                match_reason.append("Customer match via phone")
                source = 'CUSTOMER'
                if not matched_customer_id:
                    matched_customer_id = record['id']
        
        # ------------------------------------------------------------------
        # STAGE 3: FUZZY NAME MATCHING (Same DOB)
        # ------------------------------------------------------------------
        cursor.execute("""
            SELECT 
                'CUSTOMER_DB' as source, 
                customer_id as id, 
                name, 
                address,
                pan,
                aadhaar_last4,
                dob,
                phone
            FROM existing_customers_rec
            WHERE dob = %s;
        """, (input_dob,))
        
        dob_candidates = cursor.fetchall()
        
        for candidate in dob_candidates:
            if any(m['record']['id'] == candidate['id'] for m in all_matches):
                continue
                
            db_name = candidate["name"].strip().lower()
            name_score = JaroWinkler.similarity(input_name, db_name)
            
            if name_score >= DedupeWeights.FUZZY_NAME_THRESHOLD:
                score = name_score * DedupeWeights.NAME
                
                all_matches.append({
                    'record': dict(candidate),
                    'score': score,
                    'matched_fields': [{
                        'field': 'name (fuzzy)',
                        'similarity': round(name_score * 100, 2),
                        'weight': round(DedupeWeights.NAME * 100, 2)
                    }],
                    'source': 'CUSTOMER'
                })
                final_confidence = max(final_confidence, score)
                match_reason.append(f"Fuzzy name match: {round(name_score * 100, 2)}% similarity")
                source = 'CUSTOMER'
                if not matched_customer_id:
                    matched_customer_id = candidate['id']

        # ------------------------------------------------------------------
        # STAGE 4: CHECK LOAN ACCOUNTS FOR MATCHED CUSTOMER
        # ------------------------------------------------------------------
        
        if matched_customer_id:
            loan_info = check_customer_loans(cursor, matched_customer_id)
            
            if loan_info and loan_info['has_multiple_loans'] and final_confidence >= 0.70:
                store_dedup_result(
                    cursor,
                    None,
                    matched_customer_id,
                    round(final_confidence, 2),
                    'SAME_CUSTOMER_MULTIPLE_LOANS',
                    f"Customer has {loan_info['loan_count']} existing loans. Loan accounts: {', '.join(loan_info['loan_accounts'])}"
                )
                conn.commit()
                
                response = {
                    "emit": "dedup.match_found",
                    "output": {
                        "status": "EXISTING_CUSTOMER",
                        "verdict": "SAME_CUSTOMER_MULTIPLE_LOANS",
                        "confidence": round(final_confidence * 100, 2),
                        "source": source,
                        "match_reason": match_reason + [f"Customer has {loan_info['loan_count']} active loans"],
                        "loan_details": {
                            "loan_count": loan_info['loan_count'],
                            "loan_accounts": loan_info['loan_accounts'],
                            "loan_types": loan_info['loan_types'],
                            "loan_statuses": loan_info['loan_statuses'],
                            "loan_amounts": loan_info['loan_amounts']
                        },
                        "matched_records": [m['record'] for m in all_matches[:5]],
                        "detailed_matches": all_matches[:5] if all_matches else []
                    }
                }
                return response

        # ------------------------------------------------------------------
        # STAGE 5: DETERMINE FINAL VERDICT
        # ------------------------------------------------------------------
        
        all_matches.sort(key=lambda x: x['score'], reverse=True)
        
        if final_confidence >= 1.0:
            status = "REJECTED"
            verdict = "EXACT_MATCH"
            confidence = 1.0
        elif final_confidence >= 0.85:
            status = "REJECTED"
            verdict = "HIGH_CONFIDENCE_MATCH"
            confidence = final_confidence
        elif final_confidence >= 0.70:
            status = "REVIEW"
            verdict = "MEDIUM_CONFIDENCE_MATCH"
            confidence = final_confidence
        elif final_confidence >= 0.50:
            status = "REVIEW"
            verdict = "LOW_CONFIDENCE_MATCH"
            confidence = final_confidence
        elif final_confidence >= 0.30:
            status = "FLAGGED"
            verdict = "WEAK_MATCH"
            confidence = final_confidence
        else:
            status = "CLEAR"
            verdict = "NO_MATCH"
            confidence = 0.0
        
        if source == 'BLACKLIST' and final_confidence > 0:
            status = "REJECTED"
            verdict = "BLACKLISTED"
            confidence = max(confidence, 0.70)
        
        if all_matches:
            result_type = verdict
            if verdict == "BLACKLISTED":
                explanation = f"Blacklist match with confidence {confidence:.2%}"
            else:
                explanation = f"Match found with confidence {confidence:.2%}. Action: {status}"
            
            store_dedup_result(
                cursor,
                None,
                matched_customer_id,
                round(confidence, 2),
                result_type,
                explanation
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
        
        response = {
            "emit": "dedup.match_found" if all_matches else "dedup.clear",
            "output": {
                "status": status,
                "verdict": verdict,
                "confidence": round(confidence * 100, 2),
                "source": source,
                "match_reason": match_reason if match_reason else ["No matches found"],
                "loan_details": loan_info if loan_info else {},
                "matched_records": [m['record'] for m in all_matches[:5]],
                "detailed_matches": all_matches[:5] if all_matches else []
            }
        }
        
        return response

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

# 8. Test Scenarios
if __name__ == "__main__":
    test_scenarios = {
        "1. Multiple Loans - Rahul Sharma": {
            "reads": {
                "name": "Rahul Sharma", 
                "dob": "1992-05-12", 
                "pan": "ABCDE1234F", 
                "phone": "9876543210", 
                "aadhaar_last4": "5678", 
                "address": "Mumbai, Maharashtra"
            }
        },
        "2. Blacklist Match - Nirav Gupta": {
            "reads": {
                "name": "Nirav Gupta", 
                "dob": "1975-02-14", 
                "pan": "TREWA7890Q", 
                "phone": "9006006677", 
                "aadhaar_last4": "9006", 
                "address": "Hub"
            }
        },
        "3. Multiple Loans - Aarav Mehta": {
            "reads": {
                "name": "Aarav Mehta", 
                "dob": "1991-03-18", 
                "pan": "ABCDE1111A", 
                "phone": "9876543211", 
                "aadhaar_last4": "1001", 
                "address": "Mumbai, Maharashtra"
            }
        },
        "4. Single Loan - Priya Patel": {
            "reads": {
                "name": "Priya Patel", 
                "dob": "1995-08-15", 
                "pan": "KJGHS9911Z", 
                "phone": "9811223344", 
                "aadhaar_last4": "9012", 
                "address": "Mumbai, Maharashtra"
            }
        },
        "5. Clean User - Sachin Tendulkar": {
            "reads": {
                "name": "Sachin Tendulkar", 
                "dob": "1973-04-24", 
                "pan": "SRTPA1111A", 
                "phone": "9999988888", 
                "aadhaar_last4": "1973", 
                "address": "Mumbai"
            }
        }
    }

    print("="*70)
    print("DEDUPLICATION ENGINE WITH CUMULATIVE SCORING & LOAN CHECK")
    print("="*70)
    
    for name, payload in test_scenarios.items():
        print(f"\n▶ {name}")
        print("-" * 50)
        result = process_dedup(payload)
        print(f"Emit: {result['emit']}")
        print(f"Status: {result['output']['status']}")
        print(f"Verdict: {result['output']['verdict']}")
        print(f"Confidence: {result['output']['confidence']}%")
        print(f"Reasons: {', '.join(result['output']['match_reason'])}")
        
        if result['output'].get('loan_details') and result['output']['loan_details'].get('has_loans'):
            loan_details = result['output']['loan_details']
            print(f"Loan Details:")
            print(f"  - Total Loans: {loan_details['loan_count']}")
            print(f"  - Loan Accounts: {', '.join(loan_details['loan_accounts'])}")
            print(f"  - Loan Types: {', '.join(loan_details['loan_types'])}")
            if loan_details.get('has_multiple_loans'):
                print(f"  - ⚠️ MULTIPLE LOANS DETECTED")
        
        if result['output']['matched_records']:
            print(f"Matched Records: {len(result['output']['matched_records'])} found")
            for i, record in enumerate(result['output']['matched_records'][:2], 1):
                print(f"  {i}. ID: {record.get('id', 'N/A')} - {record.get('name', 'Unknown')}")
        
        if result['output']['detailed_matches']:
            print("Detailed Scores:")
            for match in result['output']['detailed_matches'][:2]:
                print(f"  Score: {round(match['score'] * 100, 2)}%")
                for field in match.get('matched_fields', []):
                    print(f"    - {field['field']}: {field['similarity']}% (weight: {field['weight']}%)")
        
        print("-" * 50)