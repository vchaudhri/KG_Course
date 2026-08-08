import requests
import time

WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"

def run_sparql(query: str):
    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": "KG Course/1.0 (vchaudhri@acm.org)"
    }
    for attempt in range(5):
        try:
            response = requests.post(
                WIKIDATA_SPARQL_URL,
                data={"query": query},
                headers=headers,
                timeout=120
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code
            if status not in [429, 502, 503, 504]:
                raise
            wait = 2 ** attempt
            print(
                f"HTTP {status}. "
                f"Retry {attempt+1}/5 in {wait} seconds"
            )
            time.sleep(wait)
    raise RuntimeError("SPARQL query failed after 5 retries")


if __name__ == "__main__":
    test_query = """
    SELECT ?company ?companyLabel
    WHERE {
      VALUES ?company { wd:Q312 wd:Q3884 }
      SERVICE wikibase:label {
        bd:serviceParam wikibase:language "en".
      }
    }
    """
    result = run_sparql(test_query)
    for row in result["results"]["bindings"]:
        print(
            row["company"]["value"],
            row["companyLabel"]["value"]
        )