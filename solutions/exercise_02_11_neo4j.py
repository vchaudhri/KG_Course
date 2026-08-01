from neo4j import GraphDatabase
import csv
import time
from exercise_02_11_benchmark import BenchmarkResult
import os, sys

URI = "bolt://localhost:7687"
USERNAME = "neo4j"
PASSWORD = os.getenv("NEO4J_PASSWORD")
if PASSWORD is None:
    sys.exit("Please set the NEO4J_PASSWORD environment variable.")


EMPLOYEE_CSV = "Employee.csv"
DEPARTMENT_CSV = "Department.csv"
EMPLOYEE_DEPT_CSV = "Employee_Department.csv"

driver = GraphDatabase.driver(
    URI,
    auth=(USERNAME, PASSWORD)
)

def create_database():
    with driver.session() as session:

        # ----------------------------------------------------
        # Delete existing graph
        # ----------------------------------------------------

        session.run("MATCH (n) DETACH DELETE n")

        # ----------------------------------------------------
        # Load Employees
        # ----------------------------------------------------

        with open(EMPLOYEE_CSV, newline='', encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for r in reader:
                session.run("""
                    CREATE (:Employee {
                        employee_id: $id,
                        first_name: $first,
                        last_name: $last
                    })
                """,
                id=int(r["EMPLOYEE_ID"]),
                first=r["FIRST_NAME"],
                last=r["LAST_NAME"])

        # ----------------------------------------------------
        # Load Departments
        # ----------------------------------------------------

        with open(DEPARTMENT_CSV, newline='', encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for r in reader:
                session.run("""
                    CREATE (:Department {
                        department_id: $id,
                        name: $name
                    })
                """,
                id=int(r["DEPARTMENT_ID"]),
                name=r["DEPARTMENT_NAME"])

        # ----------------------------------------------------
        # Create WORKS_IN edges
        # ----------------------------------------------------

        with open(EMPLOYEE_DEPT_CSV, newline='', encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for r in reader:

                dept = r["DEPARTMENT_ID"].strip()

                if dept == "":
                    continue

                session.run("""
                    MATCH (e:Employee {employee_id:$eid})
                    MATCH (d:Department {department_id:$did})
                    CREATE (e)-[:WORKS_IN]->(d)
                """,
                eid=int(r["EMPLOYEE_ID"]),
                did=int(dept))


        session.run("""
            CREATE INDEX employee_id_idx IF NOT EXISTS
            FOR (e:Employee)
            ON (e.employee_id)
        """)

        session.run("""
            CREATE INDEX department_id_idx IF NOT EXISTS
            FOR (d:Department)
            ON (d.department_id)
        """)

        session.run("""
            CREATE INDEX department_name_idx IF NOT EXISTS
            FOR (d:Department)
            ON (d.name)
        """)

def benchmark(show_results=False):
    with driver.session() as session:
        query = """
        MATCH (d:Department {name: 'IT'})<-[:WORKS_IN]-(p:Employee)
        RETURN p.first_name AS first,
               p.last_name AS last
        """

        NUM_RUNS = 10000

        start = time.perf_counter()

        for _ in range(NUM_RUNS):
            result = list(session.run(query))

        end = time.perf_counter()

        if show_results:
            print("Employees in IT\n")

            for r in result:
                print(r["first"], r["last"])

            print()
            print(f"Returned {len(result)} rows")
            print(f"Average execution time: {(end-start)*1000/NUM_RUNS:.6f} ms")

    driver.close()

    return BenchmarkResult(
        database="Neo4j",
        rows=len(result),
        average_ms=(end - start) * 1000 / NUM_RUNS
    )



if __name__ == "__main__":
    create_database()
    result = benchmark()
    print(result)