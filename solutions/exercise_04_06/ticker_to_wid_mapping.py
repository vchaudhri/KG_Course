from pathlib import Path
from snp import snp100_tickers as tickers
from send_sparql_to_wikidata import run_sparql
import csv

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"

OUTPUT_FILE = Path("ticker_company_mapping.csv")


# ----------------------------------------------------------------------
# Build SPARQL query
# ----------------------------------------------------------------------

values = " ".join(f'"{t}"' for t in tickers)

query = f"""
    SELECT DISTINCT ?ticker ?company ?companyLabel
        WHERE {{
        # Input ticker symbols to be matched.
        VALUES ?ticker {{ {values} }}

        # Restrict the search to listings on NASDAQ and NYSE.
        VALUES ?exchange {{ wd:Q82059 wd:Q13677 }}

        # p: Follow the stock exchange claim from the company
        # to the corresponding listing statement.
        ?company p:P414 ?listingStmt .

        # ps: Retrieve the main value of the statement,
        # namely the stock exchange.
        ?listingStmt ps:P414 ?exchange .

        # pq: Retrieve the ticker symbol qualifier associated
        # with the stock exchange listing.
        ?listingStmt pq:P249 ?ticker .

        # Ignore historical listings that have an end date.
        FILTER NOT EXISTS {{ ?listingStmt pq:P582 ?endDate .}}

        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
ORDER BY ?ticker
"""

results = run_sparql(query)

mapping = {}

for binding in results["results"]["bindings"]:
    ticker = binding["ticker"]["value"]

    # Extract the Wikidata identifier (e.g., Q312)
    company = binding["company"]["value"].rsplit("/", 1)[-1]
    
    company_label = binding["companyLabel"]["value"]

    mapping[ticker] = (company, company_label)


with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["ticker", "company", "companyLabel"])
    for ticker in tickers:
        if ticker not in mapping:
            print(f"Missing: {ticker}")
            continue
        company, label = mapping[ticker]
        writer.writerow([ticker, company, label])

print(f"Mapping of {len(mapping)} tickers to Wikidata IDs written to {OUTPUT_FILE}")