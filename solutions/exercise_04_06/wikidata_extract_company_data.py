import csv
from py_compile import main
from snp import snp100_tickers, properties_to_query
from send_sparql_to_wikidata import run_sparql
from pathlib import Path
from pprint import pprint

WORKING_DIR = Path(__file__).parent 
OUTPUT_FILE = WORKING_DIR / "wikidata_company_data.csv"
COLUMNS =  [
        "company",
        "companyLabel",
        "property",
        "propertyLabel",
        "statement",
        "mainValue",
        "mainValueLabel",
        "qualifierProperty",
        "qualifierPropertyLabel",
        "qualifierValue"
    ]
TICKER_MAPPING_FILE = WORKING_DIR / "ticker_to_wid_mapping_enhanced.csv"


def get_wikidata_property_query (company_id, property_to_query):
    select_clause = " ".join(f"?{column}" for column in COLUMNS)
    return (
    f"""
   SELECT
        {select_clause}
    WHERE 
        {{  VALUES ?company {{{company_id}}} 
            VALUES ?property    {{{property_to_query}}}

        # Lookup the namespace-specific properties corresponding to
        # the selected Wikidata property.
        ?property wikibase:claim ?claimProperty .
        ?property wikibase:statementProperty ?statementProperty .

        # p: Follow the claim property from the company to its statement.
        ?company ?claimProperty ?statement .

        # ps: Retrieve the statement's main value.
        ?statement ?statementProperty ?mainValue .

        # pq: Retrieve the statement's qualifiers.
        OPTIONAL {{
            # Match any qualifier property attached to the statement.
            ?statement ?qualifierStatementProperty ?qualifierValue .

            FILTER(
                STRSTARTS(
                    STR(?qualifierStatementProperty),
                    STR(pq:)
                )
            )

            # Recover the corresponding Wikidata property (for example,
            # pq:P414 -> wd:P414).
            ?qualifierProperty
                wikibase:qualifier
                ?qualifierStatementProperty .
        }}
        SERVICE wikibase:label {{
            bd:serviceParam wikibase:language "en".
        }}
    }}
""" )


def get_value(binding, variable):
    value = binding.get(variable, {}).get("value", "")   
    if value.startswith("http://www.wikidata.org/entity/"):
        return value.rsplit("/", 1)[-1]
    return value


def extract_property(ticker, wikidata_property):
    sparql_query = get_wikidata_property_query(ticker, wikidata_property)
    result = run_sparql(sparql_query)
    rows = []
    for binding in result["results"]["bindings"]:
        rows.append(
            [
                get_value(binding, column) for column in COLUMNS
            ]
        )
    return rows


def extract_and_write_properties ():
    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(COLUMNS)
        with TICKER_MAPPING_FILE.open(newline="", encoding="utf-8") as mapping_file:
            reader = csv.DictReader(mapping_file)
            for company in reader:
                print(f"Extracting properties for {company['companyLabel']} ({company['wikidata_id']})")
                company_rows = []
                for wikidata_property_info in properties_to_query:
                    wikidata_property = "wd:" + wikidata_property_info["id"]
                    extracted_rows = extract_property(F"wd:{company['wikidata_id']}", wikidata_property)
                    company_rows.extend(extracted_rows)
                writer.writerows(company_rows)
                csvfile.flush()
                print(f"Wrote {len(company_rows)} rows")
                


if __name__ == "__main__":
    extract_and_write_properties()
