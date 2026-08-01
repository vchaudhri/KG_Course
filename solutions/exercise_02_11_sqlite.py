import sqlite3
import csv
import time
from exercise_02_11_benchmark import BenchmarkResult

DB_NAME = "hr.db"

EMPLOYEE_CSV = "Employee.csv"
EMPLOYEE_DEPT_CSV = "Employee_Department.csv"
DEPARTMENT_CSV = "Department.csv"

# ----------------------------------------------------
# Create database
# ----------------------------------------------------


def create_database():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    # Drop tables if they already exist
    cur.executescript("""
    DROP TABLE IF EXISTS Employee;
    DROP TABLE IF EXISTS Employee_Department;
    DROP TABLE IF EXISTS Department;
    """)

    # Create tables
    cur.executescript("""
    CREATE TABLE Employee (
        Employee_ID INTEGER PRIMARY KEY,
        FIRST_NAME TEXT,
        LAST_NAME TEXT
    );

    CREATE TABLE Department (
        DEPARTMENT_ID INTEGER PRIMARY KEY,
        DEPARTMENT_NAME TEXT
    );

    CREATE TABLE Employee_Department (
        EMPLOYEE_ID INTEGER,
        DEPARTMENT_ID INTEGER,
        PRIMARY KEY (EMPLOYEE_ID, DEPARTMENT_ID),
        FOREIGN KEY (EMPLOYEE_ID)
            REFERENCES Employee(Employee_ID),
        FOREIGN KEY (DEPARTMENT_ID)
            REFERENCES Department(DEPARTMENT_ID)
    );
    """)

    # ----------------------------------------------------
    # Load Employee table
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
    VALUES (?, ?, ?)
    """, rows)

    # ----------------------------------------------------
    # Load Department table
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
    VALUES (?, ?)
    """, rows)

    # ----------------------------------------------------
    # Load Employee_Department table
    # ----------------------------------------------------

    with open(EMPLOYEE_DEPT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        rows = []

        for r in reader:
            if not r["EMPLOYEE_ID"].strip():
                continue

            dept = r["DEPARTMENT_ID"].strip()

            rows.append((
                int(r["EMPLOYEE_ID"]),
                int(dept) if dept else None
            ))

    print(f"Loaded {len(rows)} employee-department rows.")

    cur.executemany("""
    INSERT INTO Employee_Department (EMPLOYEE_ID, DEPARTMENT_ID)
    VALUES (?, ?)
    """, rows)

    cur.execute("SELECT COUNT(*) FROM Employee_Department")
    print("Rows in Employee_Department:", cur.fetchone()[0])

    conn.commit()
    conn.close

# ----------------------------------------------------
# Execute query
# ----------------------------------------------------

query = """
SELECT
    Employee.FIRST_NAME || ' ' || Employee.LAST_NAME AS Name
FROM Employee
LEFT JOIN Employee_Department
    ON Employee.Employee_ID = Employee_Department.EMPLOYEE_ID
LEFT JOIN Department
    ON Department.DEPARTMENT_ID = Employee_Department.DEPARTMENT_ID
WHERE Department.DEPARTMENT_NAME = 'IT'
ORDER BY Employee.LAST_NAME;
"""

def benchmark(show_results=False):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    NUM_RUNS = 10000
    start = time.perf_counter()
    for _ in range(NUM_RUNS):
        rows = list(cur.execute(query))
    end = time.perf_counter()
    if show_results:
        print("Employees in the IT department:\n")
        for row in rows:
            print(row[0])
    conn.close()
    return BenchmarkResult(
        database="SQLite",
        rows=len(rows),
        average_ms=(end - start) * 1000 / NUM_RUNS
    )


if __name__ == "__main__":
    create_database()
    result = benchmark()
    print(result)