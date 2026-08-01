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
# Property Graph -> Turtle-star Translator
########################################################################

class PropertyGraphToTurtleStar:

    def __init__(self):

        self.output = []

    ####################################################################
    # Utility methods
    ####################################################################

    def uri(self, name):
        """Return a prefixed URI."""
        return f"ex:{name}"

    def literal(self, value):
        """Convert a Python value into a Turtle literal."""

        if isinstance(value, str):
            return f'"{value}"'

        return str(value)

    ####################################################################
    # Main translation
    ####################################################################

    def translate(self, property_graph):

        self.output = []

        #
        # Prefix declarations
        #

        self.output.append("@prefix ex: <http://example.org/> .")
        self.output.append(
            "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> ."
        )
        self.output.append("")

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

        return "\n".join(self.output)

    ####################################################################
    # Translate one node
    ####################################################################

    def translate_node(self, node):

        subject = self.uri(node.id)

        #
        # Labels become rdf:type triples.
        #

        for label in sorted(node.labels):

            self.output.append(
                f"{subject} rdf:type {self.uri(label)} ."
            )

        #
        # Properties become RDF triples.
        #

        for key, value in node.properties.items():

            self.output.append(
                f"{subject} {self.uri(key)} {self.literal(value)} ."
            )

        self.output.append("")

    ####################################################################
    # Translate one edge
    ####################################################################

    def translate_edge(self, edge):

        subject = self.uri(edge.source)
        predicate = self.uri(edge.label)
        object = self.uri(edge.target)

        #
        # Emit the RDF triple.
        #

        self.output.append(
            f"{subject} {predicate} {object} ."
        )

        #
        # Emit RDF-star annotations.
        #

        if edge.properties:

            quoted = f"<< {subject} {predicate} {object} >>"

            for key, value in edge.properties.items():

                self.output.append(
                    f"{quoted} {self.uri(key)} {self.literal(value)} ."
                )

        self.output.append("")

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

translator = PropertyGraphToTurtleStar()

print(translator.translate(pg))  