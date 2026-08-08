import csv
import pandas as pd
from neo4j import GraphDatabase
import os
from pathlib import Path

from snp import properties_to_query

# URI examples: "neo4j://localhost", "neo4j+s://xxx.databases.neo4j.io"
URI = "neo4j://localhost"
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
AUTH = ("neo4j", NEO4J_PASSWORD)
driver = GraphDatabase.driver(URI, auth=AUTH)

pd.set_option('display.max_rows', None)

WORKING_DIR = Path(__file__).parent 
TICKER_TO_WID_MAPPING = WORKING_DIR / "ticker_to_wid_mapping_enhanced.csv"
FINREFLECTDATA_PARQUET = WORKING_DIR / "snp100_finreflectkg.parquet"
PERSON_TO_WID_MAPPING = WORKING_DIR / "person_to_wid_mapping.csv"
WIKIDATA_COMPANY_DATA = WORKING_DIR / "wikidata_company_data.csv"
PERSON_TO_WIKIDATA_RECONCILIATION = WORKING_DIR / "person_to_wikidata_reconciliation.csv"
COMPANY_TO_WIKIDATA_RECONCILIATION = WORKING_DIR / "company_to_wikidata_reconciliation.csv"


def load_company_mapping():
    companies = {}
    with open(TICKER_TO_WID_MAPPING, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            companies[row["ticker"]] = {
                "wikidata_id": row["wikidata_id"],
                "name": row["companyLabel"]
            }
    return companies


def clear_neo4j_database():
    with driver.session(database="finkg") as session:
        result = session.run("MATCH (n) DETACH DELETE n")
        summary = result.consume()
        print("Nodes deleted:", summary.counters.nodes_deleted)
        print("Relationships deleted:", summary.counters.relationships_deleted)


def create_snp_100_company_nodes():
    print("Creating S&P 100 company nodes in Neo4j")
    companies_mapping = load_company_mapping()
    companies = [
        {
            "ticker": ticker,
            "wikidata_id": company["wikidata_id"],
            "name": company["name"]
        }
        for ticker, company in companies_mapping.items()
    ]
    neo4j_command = """
        UNWIND $companies as company
        MERGE (c:Entity:Company {wikidata_id: company.wikidata_id })
        SET c.name = company.name,
            c.ticker = company.ticker
        """
    with driver.session(database="finkg") as session:
        result = session.run(
            neo4j_command,
            companies=companies
        )
        summary = result.consume()
        print(summary.counters)


import re
def to_rel_type(label: str) -> str:
    """
    Convert a Wikidata property label into a Neo4j relationship type.
    Examples:
        "product or material produced" -> "PRODUCT_OR_MATERIAL_PRODUCED"
        "chief executive officer" -> "CHIEF_EXECUTIVE_OFFICER"
        "part of" -> "PART_OF"
    """
    label = label.upper()
    label = re.sub(r'[^A-Z0-9]+', '_', label)
    label = re.sub(r'_+', '_', label)
    return label.strip('_')


def create_snp_100_edges ():
    print("Creating edges for S&P 100 companies in Neo4j")
    total_nodes_created = 0
    total_relationships_created = 0
    total_properties_set = 0
    df = pd.read_csv(WIKIDATA_COMPANY_DATA)
    df = df.fillna("")
    rows = df.to_dict(orient="records")
    for p in properties_to_query:
        print(f"Processing {p["id"]}")
        relationship_type = to_rel_type(p["label"])
        target_type = p["target_type"]
        property_rows = [
            row
            for row in rows
            if row["property"] == p["id"]
        ]
        neo4j_command = f"""
            UNWIND $property_rows as row
            MERGE (c:Entity:Company {{wikidata_id: row.company }})
            MERGE (o:Entity:{target_type} {{wikidata_id: row.mainValue }})
            MERGE (c)-[r:{relationship_type}{{factStmt: row.statement}}]->(o)
            SET o.name = row.mainValueLabel

            FOREACH (_ IN CASE WHEN row.qualifierProperty ENDS WITH "P39" AND
                                    row.qualifierValue IS NOT NULL THEN [1] ELSE [] END |
                SET r.position_held = row.qualifierValue
            )
            
            FOREACH (_ IN CASE WHEN row.qualifierProperty ENDS WITH "P155" AND
                                    row.qualifierValue IS NOT NULL THEN [1] ELSE [] END |
                SET r.follows = row.qualifierValue
            )
             
            FOREACH (_ IN CASE WHEN row.qualifierProperty ENDS WITH "P459" AND
                                    row.qualifierValue IS NOT NULL THEN [1] ELSE [] END |
                SET r.determination_method_or_standard = row.qualifierValue
            )
                                     
            FOREACH (_ IN CASE WHEN row.qualifierProperty ENDS WITH "P518" AND
                                    row.qualifierValue IS NOT NULL THEN [1] ELSE [] END |
                SET r.applies_to_part = row.qualifierValue)       

            FOREACH (_ IN CASE WHEN row.qualifierProperty ENDS WITH "P580" AND
                                    row.qualifierValue IS NOT NULL THEN [1] ELSE [] END |
                SET r.start_Time = row.qualifierValue
            )   
            
            FOREACH (_ IN CASE WHEN row.qualifierProperty ENDS WITH "P582" AND
                                    row.qualifierValue IS NOT NULL THEN [1] ELSE [] END |
                SET r.end_time = row.qualifierValue
            )

            FOREACH (_ IN CASE WHEN row.qualifierProperty ENDS WITH "P585" AND
                                    row.qualifierValue IS NOT NULL THEN [1] ELSE [] END |
                SET r.point_in_time = row.qualifierValue
                )
                
            FOREACH (_ IN CASE WHEN row.qualifierProperty ENDS WITH "P1001" AND
                                    row.qualifierValue IS NOT NULL THEN [1] ELSE [] END |
                SET r.applies_to_jurisdiction = row.qualifierValue
                )
                
            FOREACH (_ IN CASE WHEN row.qualifierProperty ENDS WITH "P1012" AND
                                    row.qualifierValue IS NOT NULL THEN [1] ELSE [] END |
                SET r.includes = row.qualifierValue)   
                
            FOREACH (_ IN CASE WHEN row.qualifierProperty ENDS WITH "P1013" AND
                                    row.qualifierValue IS NOT NULL THEN [1] ELSE [] END |
                SET r.criterion_used = row.qualifierValue)   
                            
            FOREACH (_ IN CASE WHEN row.qualifierProperty ENDS WITH "P1107" AND
                                    row.qualifierValue IS NOT NULL THEN [1] ELSE [] END |
                SET r.proportion = row.qualifierValue
                )
                
                FOREACH (_ IN CASE WHEN row.qualifierProperty ENDS WITH "P1352" AND
                                    row.qualifierValue IS NOT NULL THEN [1] ELSE [] END |
                SET r.ranking = row.qualifierValue
                )
                
            FOREACH (_ IN CASE WHEN row.qualifierProperty ENDS WITH "P1365" AND
                                    row.qualifierValue IS NOT NULL THEN [1] ELSE [] END |
                SET r.replaces = row.qualifierValue
                )
                
            FOREACH (_ IN CASE WHEN row.qualifierProperty ENDS WITH "P1366" AND
                                    row.qualifierValue IS NOT NULL THEN [1] ELSE [] END |
                SET r.replaced_by = row.qualifierValue
                )
                
                
            FOREACH (_ IN CASE WHEN row.qualifierProperty ENDS WITH "P1480" AND
                                    row.qualifierValue IS NOT NULL THEN [1] ELSE [] END |
                SET r.sourcing_circumstances = row.qualifierValue
                )
                
            FOREACH (_ IN CASE WHEN row.qualifierProperty ENDS WITH "P1534" AND
                                    row.qualifierValue IS NOT NULL THEN [1] ELSE [] END |
                SET r.end_cause = row.qualifierValue
                )
                
                FOREACH (_ IN CASE WHEN row.qualifierProperty ENDS WITH "P1545" AND
                                    row.qualifierValue IS NOT NULL THEN [1] ELSE [] END |
                SET r.series_ordinal = row.qualifierValue
                )
                
            FOREACH (_ IN CASE WHEN row.qualifierProperty ENDS WITH "P2241" AND
                                    row.qualifierValue IS NOT NULL THEN [1] ELSE [] END |
                SET r.reason_for_deprecated_rank = row.qualifierValue
                )
            
            FOREACH (_ IN CASE WHEN row.qualifierProperty ENDS WITH "P4241" AND
                                    row.qualifierValue IS NOT NULL THEN [1] ELSE [] END |
                SET r.refine_date = row.qualifierValue
                )
                  
            FOREACH (_ IN CASE WHEN row.qualifierProperty ENDS WITH "P6949" AND
                                    row.qualifierValue IS NOT NULL THEN [1] ELSE [] END |
                SET r.announcement_date = row.qualifierValue)         
                
            FOREACH (_ IN CASE WHEN row.qualifierProperty ENDS WITH "P7452" AND
                                    row.qualifierValue IS NOT NULL THEN [1] ELSE [] END |
                SET r.reason_for_preferred_rank = row.qualifierValue)        

            FOREACH (_ IN CASE WHEN row.qualifierProperty ENDS WITH "P8327" AND
                                    row.qualifierValue IS NOT NULL THEN [1] ELSE [] END |
                SET r.intended_subject_of_deprecated_statement = row.qualifierValue)                 
                
                
            FOREACH (_ IN CASE WHEN row.qualifierProperty ENDS WITH "P8554" AND
                                    row.qualifierValue IS NOT NULL THEN [1] ELSE [] END |
                SET r.earliest_end_date = row.qualifierValue)       
                           
        """
        with driver.session(database="finkg") as session:
            result = session.run(
                neo4j_command,
                property_rows=property_rows
            )
            summary = result.consume()
            total_nodes_created += summary.counters.nodes_created
            total_relationships_created += summary.counters.relationships_created
            total_properties_set += summary.counters.properties_set
    print("Total nodes created:", total_nodes_created)
    print("Total relationships created:", total_relationships_created)
    print("Total properties set:", total_properties_set)


def add_fin_reflect_data ():
    print("Adding FinReflect data to Neo4j")
    df = pd.read_parquet(FINREFLECTDATA_PARQUET)
    reconciliation = (
        pd.concat([
            pd.read_csv(COMPANY_TO_WIKIDATA_RECONCILIATION),
            pd.read_csv(PERSON_TO_WIKIDATA_RECONCILIATION)
        ], ignore_index=True).dropna(subset=["wikidata_id"])
        .set_index("name")
        .to_dict("index")
    )
    total_nodes_created = 0
    total_relationships_created = 0
    total_properties_set = 0
    for _, row in df.iterrows():
        if row["entity"] in reconciliation and row["target"] in reconciliation:
            if row.entity_type == "ORG":
                source_type = "Company"
            elif row.entity_type == "PERSON":
                source_type = "Person"
            else:
                print("Erroneous Source Type")
                continue
            source_wikidata_id = reconciliation[row["entity"]]["wikidata_id"]
            target_wikidata_id = reconciliation[row["target"]]["wikidata_id"]
            if source_wikidata_id == target_wikidata_id:
                continue
            relationship_type = row["relationship"].upper()
            start_time = row["start_date"]
            end_time = row["end_date"]
            neo4j_command = f"""
                    MERGE (source:Entity:{source_type} {{wikidata_id: $source_wikidata_id }})
                    MERGE (target:Entity:Company {{wikidata_id: $target_wikidata_id }})
                    MERGE (source)-[r:{relationship_type}]->(target)
                    SET r.start_time = $start_time
                    SET r.end_time = $end_time
            """
            with driver.session(database="finkg") as session:
                result = session.run(
                    neo4j_command,
                    source_wikidata_id=source_wikidata_id,
                    target_wikidata_id=target_wikidata_id,
                    start_time=start_time,
                    end_time=end_time,
                )
                summary = result.consume()
                total_nodes_created += summary.counters.nodes_created
                total_relationships_created += summary.counters.relationships_created
                total_properties_set += summary.counters.properties_set
    print("Total nodes created:", total_nodes_created)
    print("Total relationships created:", total_relationships_created)
    print("Total properties set:", total_properties_set)










if __name__ == "__main__":
    clear_neo4j_database()
    create_snp_100_company_nodes()
    create_snp_100_edges()
    add_fin_reflect_data()





