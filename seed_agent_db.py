import psycopg, os

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def seed_database():
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            print("[*] Creating corporate financial audit tables...")
            cur.execute("DROP TABLE IF EXISTS corporate_transactions;")
            cur.execute("""
                CREATE TABLE corporate_transactions (
                    transaction_id SERIAL PRIMARY KEY,
                    department TEXT NOT NULL,
                    vendor TEXT NOT NULL,
                    amount_usd NUMERIC(12, 2) NOT NULL,
                    is_flagged_audit BOOLEAN DEFAULT FALSE,
                    transaction_date DATE NOT NULL
                );
            """)

            sample_data = [
                ("Engineering", "AWS Cloud Services", 14500.00, False, "2026-01-15"),
                ("Engineering", "Datadog Monitoring", 3200.50, False, "2026-01-18"),
                ("Engineering", "GitHub Enterprise", 2100.00, False, "2026-02-01"),
                ("Legal", "Baker & McKenzie LLP", 45000.00, True, "2026-02-10"),
                ("Marketing", "Google Ads Global", 28000.00, False, "2026-02-12"),
                ("Marketing", "Meta Ad Operations", 19500.00, True, "2026-02-14"),
                ("Operations", "WeWork London Lease", 12000.00, False, "2026-02-20"),
                ("Security", "CrowdStrike Falcon", 8500.00, False, "2026-03-01"),
                ("Legal", "Compliance Auditing UK", 18500.00, True, "2026-03-05")
            ]

            cur.executemany("""
                INSERT INTO corporate_transactions 
                (department, vendor, amount_usd, is_flagged_audit, transaction_date)
                VALUES (%s, %s, %s, %s, %s);
            """, sample_data)
            
            conn.commit()
            print("[+] Seeded 9 transactions into 'corporate_transactions'.")

if __name__ == "__main__":
    seed_database()