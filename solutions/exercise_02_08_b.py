#!/usr/bin/env python3
"""
Property Graph Example using Neo4j

This script accompanies Section 3 of the chapter on Property Graphs.

Requirements
------------
pip install neo4j tabulate

Update the URI, username, and password below before running.
"""

from neo4j import GraphDatabase
from tabulate import tabulate


# ----------------------------------------------------------------------
# Connection information
# ----------------------------------------------------------------------

URI = "bolt://localhost:7687"
AUTH = ("<userid>", "<password>")
driver = GraphDatabase.driver(
    URI,
    auth=AUTH
)


# ----------------------------------------------------------------------
# Utility routine
# ----------------------------------------------------------------------

def run_query(query):

    print("=" * 72)
    print("neo4j>")
    print(query.strip())
    print()

    with driver.session() as session:
        result = session.run(query)

        records = list(result)
        columns = result.keys()

        if columns:
            rows = [[record[c] for c in columns] for record in records]
            print(tabulate(rows,
                           headers=columns,
                           tablefmt="github"))
        else:
            summary = result.consume()
            c = summary.counters

            print("Summary")
            print("-------")

            if c.nodes_created:
                print(f"Nodes created: {c.nodes_created}")

            if c.relationships_created:
                print(f"Relationships created: {c.relationships_created}")

            if c.properties_set:
                print(f"Properties set: {c.properties_set}")

        print()


# ----------------------------------------------------------------------
# Clean database
# ----------------------------------------------------------------------

run_query("""
MATCH (n)
DETACH DELETE n
""")


# ----------------------------------------------------------------------
# Create nodes
# We are using the design in Figure 3    
# ----------------------------------------------------------------------

run_query("""
CREATE
    (art:Person {name:'art'}),
    (bob:Person {name:'bob'}),
    (bea:Person {name:'bea', age:23}),
    (cal:Person {name:'cal'}),
    (cam:Person {name:'cam'}),
    (coe:Person {name:'coe'}),
    (cory:Person {name:'cory'}),
    (seattle:City {name:'seattle'})
""")


# ----------------------------------------------------------------------
# Create relationships
# ----------------------------------------------------------------------

run_query("""
MATCH
    (art:Person {name:'art'}),
    (bob:Person {name:'bob'}),
    (bea:Person {name:'bea'}),
    (cal:Person {name:'cal'}),
    (cam:Person {name:'cam'}),
    (coe:Person {name:'coe'}),
    (cory:Person {name:'cory'}),
    (seattle:City {name:'seattle'})
CREATE
    (art)-[:KNOWS {since:2005}]->(bob),
    (art)-[:KNOWS {since:2012}]->(bea),
    (bob)-[:KNOWS]->(cal),
    (bob)-[:KNOWS]->(cam),
    (bea)-[:KNOWS]->(coe),
    (bea)-[:KNOWS]->(cory),
    (bea)-[:BASED_NEAR]->(seattle)
""")


# ----------------------------------------------------------------------
# Query 1
# ----------------------------------------------------------------------

run_query("""
MATCH (p1:Person {name:'art'})-[:KNOWS]->(p2:Person)
RETURN p2.name AS Friend
""")


# ----------------------------------------------------------------------
# Query 2
# ----------------------------------------------------------------------

run_query("""
MATCH (p1:Person {name:'art'})-[r:KNOWS {since:2010}]->(p2:Person)
RETURN p2.name AS Friend
""")


# ----------------------------------------------------------------------
# Query 3
# ----------------------------------------------------------------------

run_query("""
MATCH (p1:Person {name:'art'})-[r:KNOWS]->(p2:Person)
WHERE r.since <= 2010
RETURN p2.name AS Friend
""")


# ----------------------------------------------------------------------
# Query 4
# ----------------------------------------------------------------------

run_query("""
MATCH (p:Person {name:'bea'})-[:BASED_NEAR]->(c:City)
RETURN c.name AS City
""")


driver.close()
