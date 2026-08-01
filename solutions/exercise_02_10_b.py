from rdflib import (
    Graph,
    Namespace,
    URIRef,
    Literal,
    BNode,
)
from rdflib.namespace import RDF


########################################################################
# Property Graph Classes
########################################################################

class Node:

    def __init__(self, id):
        self.id = id
        self.labels = set()
        self.properties = {}


class Edge:

    def __init__(self, source, label, target):
        self.source = source
        self.label = label
        self.target = target
        self.properties = {}


class PropertyGraph:

    def __init__(self):
        self.nodes = {}
        self.edges = []

    def get_node(self, id):

        if id not in self.nodes:
            self.nodes[id] = Node(id)

        return self.nodes[id]

    def add_edge(self, source, label, target):

        edge = Edge(source, label, target)
        self.edges.append(edge)

        return edge


########################################################################
# Property Graph -> RDF Translator
########################################################################

class PropertyGraphToRDF:

    def __init__(self):

        self.graph = Graph()

        self.EX = Namespace("http://example.org/")

        self.graph.bind("ex", self.EX)
        self.graph.bind("rdf", RDF)

    ####################################################################
    # Utility Functions
    ####################################################################

    def uri(self, name):
        """Return a URI for a graph object."""
        return self.EX[name]

    def literal(self, value):
        """Convert a Python value into an RDF literal."""
        return Literal(value)

    ####################################################################
    # Translation
    ####################################################################

    def translate(self, property_graph):

        self.graph = Graph()

        self.graph.bind("ex", self.EX)
        self.graph.bind("rdf", RDF)

        #
        # Translate nodes
        #

        for node in property_graph.nodes.values():
            self.translate_node(node)

        #
        # Translate edges
        #

        for edge in property_graph.edges:
            self.translate_edge(edge)

        return self.graph

    ####################################################################
    # Translate one node
    ####################################################################

    def translate_node(self, node):

        subject = self.uri(node.id)

        #
        # Labels become rdf:type assertions
        #

        for label in node.labels:

            self.graph.add(
                (
                    subject,
                    RDF.type,
                    self.uri(label),
                )
            )

        #
        # Properties become RDF triples
        #

        for key, value in node.properties.items():

            self.graph.add(
                (
                    subject,
                    self.uri(key),
                    self.literal(value),
                )
            )

    ####################################################################
    # Translate one edge
    ####################################################################

    def translate_edge(self, edge):

        subject = self.uri(edge.source)
        predicate = self.uri(edge.label)
        object = self.uri(edge.target)

        #
        # Simple relationship
        #

        if len(edge.properties) == 0:

            self.graph.add(
                (
                    subject,
                    predicate,
                    object,
                )
            )

            return

        #
        # Relationship with properties
        # Use RDF reification.
        #

        statement = BNode()

        self.graph.add((statement, RDF.type, RDF.Statement))
        self.graph.add((statement, RDF.subject, subject))
        self.graph.add((statement, RDF.predicate, predicate))
        self.graph.add((statement, RDF.object, object))

        for key, value in edge.properties.items():

            self.graph.add(
                (
                    statement,
                    self.uri(key),
                    self.literal(value),
                )
            )

    ####################################################################
    # Output
    ####################################################################

    def serialize(self, format="turtle"):

        return self.graph.serialize(format=format)


########################################################################
# Example
########################################################################

pg = PropertyGraph()

#
# Nodes
#

art = pg.get_node("art")
art.labels.add("Person")
art.properties["name"] = "Art"

bea = pg.get_node("bea")
bea.labels.add("Person")
bea.properties["age"] = 23

#
# Relationship
#

edge = pg.add_edge("art", "knows", "bea")
edge.properties["since"] = 2010

#
# Translate
#

translator = PropertyGraphToRDF()

rdf_graph = translator.translate(pg)

print(translator.serialize())


