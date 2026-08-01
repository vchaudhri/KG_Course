import csv
import time
import psycopg
from exercise_02_11_benchmark import BenchmarkResult

# ----------------------------------------------------
# Configuration
# ----------------------------------------------------

HOST = "localhost"
PORT = 5432
DATABASE = "hr_demo"
USER = "postgres"
PASSWORD = "<password>"

EMPLOYEE_CSV = "Employee.csv"
DEPARTMENT_CSV = "Department.csv"
EMPLOYEE_DEPT_CSV = "Employee_Department.csv"


def create_database():
    # ----------------------------------------------------
    # Connect
    # ----------------------------------------------------

    conn = psycopg.connect(
        host=HOST,
        port=PORT,
        dbname=DATABASE,
        user=USER,
        password=PASSWORD
    )

    cur = conn.cursor()

    # ----------------------------------------------------
    # Drop existing tables
    # ----------------------------------------------------

    cur.execute("DROP TABLE IF EXISTS Employee_Department")
    cur.execute("DROP TABLE IF EXISTS Employee")
    cur.execute("DROP TABLE IF EXISTS Department")

    # ----------------------------------------------------
    # Create tables
    # ----------------------------------------------------

    cur.execute("""
    CREATE TABLE Employee (
        Employee_ID INTEGER PRIMARY KEY,
        FIRST_NAME VARCHAR(50),
        LAST_NAME VARCHAR(50)
    )
    """)

    cur.execute("""
    CREATE TABLE Department (
        DEPARTMENT_ID INTEGER PRIMARY KEY,
        DEPARTMENT_NAME VARCHAR(100)
    )
    """)

    cur.execute("""
    CREATE TABLE Employee_Department (
        EMPLOYEE_ID INTEGER REFERENCES Employee(Employee_ID),
        DEPARTMENT_ID INTEGER REFERENCES Department(DEPARTMENT_ID)
    )
    """)

    # ----------------------------------------------------
    # Load Employees
    # ----------------------------------------------------

    with open(EMPLOYEE_CSV, newline='', encoding="utf-8") as f:

        reader = csv.DictReader(f)

        rows = [
            (
                int(r["EMPLOYEE_ID"]),
                r["FIRST_NAME"],
                r["LAST_NAME"]
            )
            for r in reader
        ]

    cur.executemany("""
    INSERT INTO Employee
    VALUES (%s,%s,%s)
    """, rows)

    # ----------------------------------------------------
    # Load Departments
    # ----------------------------------------------------

    with open(DEPARTMENT_CSV, newline='', encoding="utf-8") as f:

        reader = csv.DictReader(f)

        rows = [
            (
                int(r["DEPARTMENT_ID"]),
                r["DEPARTMENT_NAME"]
            )
            for r in reader
        ]

    cur.executemany("""
    INSERT INTO Department
    VALUES (%s,%s)
    """, rows)

    # ----------------------------------------------------
    # Load Employee_Department
    # ----------------------------------------------------

    with open(EMPLOYEE_DEPT_CSV, newline='', encoding="utf-8") as f:

        reader = csv.DictReader(f)

        rows = []

        for r in reader:

            dept = r["DEPARTMENT_ID"].strip()

            rows.append((
                int(r["EMPLOYEE_ID"]),
                int(dept) if dept else None
            ))

    cur.executemany("""
    INSERT INTO Employee_Department
    VALUES (%s,%s)
    """, rows)

    conn.commit()

    print("Database loaded successfully.")

# ----------------------------------------------------
# Benchmark query
# ----------------------------------------------------

query = """
SELECT
    Employee.FIRST_NAME || ' ' || Employee.LAST_NAME AS Name
FROM Employee
JOIN Employee_Department
ON Employee.Employee_ID = Employee_Department.EMPLOYEE_ID
JOIN Department
ON Department.DEPARTMENT_ID = Employee_Department.DEPARTMENT_ID
WHERE Department.DEPARTMENT_NAME = 'IT'
"""

def benchmark():

    conn = psycopg.connect(
        host=HOST,
        port=PORT,
        dbname=DATABASE,
        user=USER,
        password=PASSWORD
    )

    cur = conn.cursor()

    NUM_RUNS = 10000

    start = time.perf_counter()

    for _ in range(NUM_RUNS):
        cur.execute(query)
        rows = cur.fetchall()

    end = time.perf_counter()

    cur.close()
    conn.close()

    return BenchmarkResult(
        database="PostgreSQL",
        rows=len(rows),
        average_ms=(end - start) * 1000 / NUM_RUNS
    )


if __name__ == "__main__":
    create_database()
    result = benchmark()
    print(result)
