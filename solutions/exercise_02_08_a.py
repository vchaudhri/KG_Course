"""
Exercise 2.8a

Creating and Querying an RDF Knowledge Graph using RDFLib.

This program demonstrates how to

1. Create an RDF knowledge graph from Turtle data.
2. Execute SPARQL queries.
3. Display the query results.
"""
from rdflib import Graph


# ---------------------------------------------------------------------
# Sample RDF data in Turtle format
# ---------------------------------------------------------------------
rdf_data = """
@prefix foaf: <http://xmlns.com/foaf/0.1/> .
@prefix ex:   <http://example.org/> .

ex:art  foaf:knows ex:bob .
ex:art  foaf:knows ex:bea .

ex:bob  foaf:knows ex:cal .
ex:bob  foaf:knows ex:cam .

ex:bea  foaf:knows ex:coe .
ex:bea  foaf:knows ex:cory .

ex:bea  foaf:age 23 .
ex:bea  foaf:based_near _:o1 .
"""


# ---------------------------------------------------------------------
# Create the RDF graph and load the data
# ---------------------------------------------------------------------
g = Graph()
g.parse(data=rdf_data, format="turtle")
print(f"Number of triples in the graph: {len(g)}")


# ---------------------------------------------------------------------
# SPARQL Query 1
#
# Find all people that Art knows.
# ---------------------------------------------------------------------
query1 = """
SELECT ?person
WHERE {
    <http://example.org/art>
        <http://xmlns.com/foaf/0.1/knows>
        ?person .
}
"""

print("\nPeople known by Art")
print("-------------------")
results = g.query(query1)
for row in results:
    print(row.person)


# ---------------------------------------------------------------------
# SPARQL Query 2
#
# Find everyone known by Art together with the people they know.
# ---------------------------------------------------------------------

query2 = """
SELECT ?person ?person1
WHERE {

    <http://example.org/art>
        <http://xmlns.com/foaf/0.1/knows>
        ?person .

    ?person
        <http://xmlns.com/foaf/0.1/knows>
        ?person1 .
}
"""
print("\nFriends of Art's Friends")
print("------------------------")
results = g.query(query2)
for row in results:
    print(f"{row.person} -> {row.person1}")


# ---------------------------------------------------------------------
# Helper function
#
# Extract the local name from a URI.
#
# Example:
#     http://example.org/bob
#
# becomes
#
#     bob
# ---------------------------------------------------------------------

def local_name(uri):
    return str(uri).split("/")[-1]


# ---------------------------------------------------------------------
# Display Query 1 using local names
# ---------------------------------------------------------------------

print("\nPeople known by Art (readable output)")
print("-------------------------------------")
results = g.query(query1)
for row in results:
    print(local_name(row.person))

# ---------------------------------------------------------------------
# Display Query 2 using local names
# ---------------------------------------------------------------------

print("\nFriends of Art's Friends (readable output)")
print("------------------------------------------")
results = g.query(query2)
for row in results:
    print(f"{local_name(row.person):5} -> {local_name(row.person1)}")
