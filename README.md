# KYC Deduplication Engine 
A real-time KYC deduplication and fraud detection system built with FastAPI and PostgreSQL. 
It identifies duplicate customers by comparing PAN, Aadhaar (last 4), Name, DOB, Phone, and Address using cumulative confidence scoring. 
The system integrates with loan accounts to detect customers with multiple loans (SAME_CUSTOMER_MULTIPLE_LOANS), checks against blacklist records (BLACKLISTED_FRAUD), and stores all deduplication results for audit. 
Simply clone, install dependencies (fastapi, uvicorn, psycopg2-binary, rapidfuzz), setup PostgreSQL tables, and run `uvicorn main:app --reload`. 
Test via Swagger UI at `http://localhost:8000/docs` or POST to `/api/v1/kyc/dedup` with KYC data. 
