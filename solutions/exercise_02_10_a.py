from rdflib import Graph, URIRef, Literal, BNode
from collections import defaultdict


class RDFTranslator:
    """
    Translate RDF into a Cypher-like property graph.
    """

    def __init__(self):
        self.graph = Graph()

        # RDF term -> printable node id
        self.node_names = {}

        # blank node numbering
        self.next_blank = 1

    ####################################################################
    # Parsing
    ####################################################################

    def parse(self, data, format="turtle"):
        self.graph = Graph()
        self.graph.parse(data=data, format=format)

    ####################################################################
    # Name resolution
    ####################################################################

    def node_name(self, term):

        if term in self.node_names:
            return self.node_names[term]

        if isinstance(term, URIRef):
            uri = str(term)

            if "#" in uri:
                name = uri.split("#")[-1]
            else:
                name = uri.rsplit("/", 1)[-1]

        elif isinstance(term, BNode):
            name = f"_b{self.next_blank}"
            self.next_blank += 1

        else:
            raise ValueError("Expected URI or blank node")

        self.node_names[term] = name
        return name

    def predicate_name(self, pred):

        uri = str(pred)

        if "#" in uri:
            return uri.split("#")[-1]

        return uri.rsplit("/", 1)[-1]

    ####################################################################
    # Translation
    ####################################################################

    def property_graph(self):

        node_properties = defaultdict(dict)
        edges = []

        for s, p, o in self.graph:

            subject = self.node_name(s)
            predicate = self.predicate_name(p)

            if isinstance(o, Literal):
                node_properties[subject][predicate] = o.toPython()

            else:
                object_node = self.node_name(o)
                edges.append((subject, predicate, object_node))

        return node_properties, edges

    ####################################################################
    # Pretty printer
    ####################################################################

    def print_cypher(self):

        node_properties, edges = self.property_graph()

        # Nodes participating in edges
        nodes = set()

        for s, _, o in edges:
            nodes.add(s)
            nodes.add(o)

        nodes.update(node_properties.keys())

        printed = set()

        def node_text(node):

            props = node_properties.get(node, {})

            if not props:
                return f"({node})"

            prop_string = ", ".join(
                f'{k}: {repr(v)}'
                for k, v in sorted(props.items())
            )

            return f"({node} {{{prop_string}}})"

        for s, p, o in edges:

            left = node_text(s) if s not in printed else f"({s})"
            printed.add(s)

            right = node_text(o) if o not in printed else f"({o})"
            printed.add(o)

            print(f"{left}-[:{p}]->{right}")


def test ():
    rdf = """
    @prefix foaf: <http://xmlns.com/foaf/0.1/> .
    @prefix ex:   <http://example.org/> .

    ex:art foaf:knows ex:bob .
    ex:art foaf:knows ex:bea .

    ex:bob foaf:knows ex:cal .
    ex:bob foaf:knows ex:cam .

    ex:bea foaf:knows ex:coe .
    ex:bea foaf:knows ex:cory .
    ex:bea foaf:age 23 .
    ex:bea foaf:based_near _:o1 .

    _:o1 ex:city "Seattle" .
    _:o1 ex:state "WA" .
    _:o1 ex:country "USA" .
    """
    translator = RDFTranslator()
    translator.parse(rdf)
    translator.print_cypher()


test()