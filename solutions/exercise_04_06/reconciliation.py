import json
from pathlib import Path
import requests
import pandas as pd
import csv

RECONCILIATION_ENDPOINT = "https://wikidata-reconciliation.wmcloud.org/en/api"
WORKING_DIR = Path(__file__).parent 
FINREFLECTDATA_PARQUET = WORKING_DIR / "snp100_finreflectkg.parquet"
PERSON_TO_WIKIDATA_RECONCILIATION = WORKING_DIR / "person_to_wikidata_reconciliation.csv"
COMPANY_TO_WIKIDATA_RECONCILIATION = WORKING_DIR / "company_to_wikidata_reconciliation.csv"

def reconcile(
    name,
    type_qid=None,
    limit=5,
    type_strict="should"
):
    """
    Reconcile a name against Wikidata.

    Parameters
    ----------
    name : str
        Entity name.

    type_qid : str | None
        Wikidata type (e.g. Q5 for human).

    limit : int
        Maximum number of candidates.

    type_strict : str
        "should", "all", or "any".

    Returns
    -------
    list
        Candidate matches.
    """

    query = {
        "q0": {
            "query": name,
            "limit": limit,
            "type_strict": type_strict
        }
    }

    if type_qid:
        query["q0"]["type"] = type_qid

    response = requests.post(
        RECONCILIATION_ENDPOINT,
        data={
            "queries": json.dumps(query)
        },
        timeout=30
    )

    response.raise_for_status()

    result = response.json()

    return result["q0"]["result"]


def best_match(name, type_qid=None):
    results = reconcile(name, type_qid, limit=1)

    if not results:
        return None

    return {
        "wikidata_id": results[0]["id"],
        "label": results[0]["name"],
        "score": results[0]["score"],
        "matched": results[0]["match"]
    }


def reconcile_people ():
    df = pd.read_parquet(FINREFLECTDATA_PARQUET)
    people = sorted(
        set(df.loc[df["entity_type"] == "PERSON", "entity"].dropna())
        |
        set(df.loc[df["target_type"] == "PERSON", "target"].dropna())
    )

    with open(PERSON_TO_WIKIDATA_RECONCILIATION, "w", newline="",
              encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "wikidata_id",
                "label",
                "score"
            ]
        )
        writer.writeheader()
        for person in people:
            print(f"Looking up {person}")
            try:
                result = best_match(person, type_qid="Q5")
            except Exception as e:
                print(f"Failed on {person}: {e}")
                result = None
            if result is None:
                writer.writerow({
                    "name": person,
                    "wikidata_id": "",
                    "label": ""
                })
            elif result["matched"] == True or result["score"] > 90.0:
                print("matched with score = " + str(result["score"]))
                writer.writerow({
                    "name": person,
                    "wikidata_id": result["wikidata_id"],
                    "label": result["label"],
                })


def reconcile_companies ():
    df = pd.read_parquet(FINREFLECTDATA_PARQUET)
    companies = sorted(
        set(df.loc[df["entity_type"] == "ORG", "entity"].dropna())
        |
        set(df.loc[df["target_type"] == "ORG", "target"].dropna())
    )
    with open(COMPANY_TO_WIKIDATA_RECONCILIATION, "w", newline="",
              encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "wikidata_id",
                "label",
                "score"
            ]
        )
        writer.writeheader()
        for company in companies:
            print(f"Looking up {company}")
            try:
                result = best_match(company, type_qid="Q783794")
            except Exception as e:
                print(f"Failed on {company}: {e}")
                result = None
            if result is None:
                writer.writerow({
                    "name": company,
                    "wikidata_id": "",
                    "label": ""
                })
            elif result["matched"] == True or result["score"] > 90.0:
                print("matched with score = " + str(result["score"]))
                writer.writerow({
                    "name": company,
                    "wikidata_id": result["wikidata_id"],
                    "label": result["label"],
                    "score": result["score"]
                })

reconcile_companies()
reconcile_people()
