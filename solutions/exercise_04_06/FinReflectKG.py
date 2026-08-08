from huggingface_hub import snapshot_download
from snp import snp100_tickers
from datasets import Dataset
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from pathlib import Path

WORKING_DIR = Path(__file__).parent 
LOCAL_DIR = WORKING_DIR / "huggingface_hub"
INPUT_DIR = LOCAL_DIR / "data"
OUTPUT_FILE = LOCAL_DIR / "snp100_finreflectkg.parquet"
CSV_FILE = LOCAL_DIR / "snp100_finreflectkg.csv"


CORE_RELATIONS = {
    "parent_of",
    "subsidiary_of",
    "produces",
    "supply",
    "invests_in",
    "has_stake_in"
}


def download_data ():
    snapshot_download(
    repo_id="domyn/FinReflectKG",
    repo_type="dataset",
    local_dir=LOCAL_DIR,
    )

import pyarrow.dataset as ds

def examine_data():
    dataset = ds.dataset(INPUT_DIR, format="parquet")
    print(dataset.schema)
    table = dataset.to_table(
        columns=[
            "relationship",
            "ticker",
            "entity",
            "entity_type",
            "target",
            "target_type"
        ]
    )
    print(f"Rows: {table.num_rows:,}")
    df = table.to_pandas()
    relationship_counts = df["relationship"].value_counts()
    print(f"Distinct relationships: {len(relationship_counts)}")

    ticker_counts = df["ticker"].value_counts()

    print(f"Distinct tickers: {len(ticker_counts)}")

    print("\nEntity type counts")
    print(df["entity_type"].value_counts())

    for core_relation in CORE_RELATIONS:
        print(f"\n\nExamining {core_relation} relationships")
        relationship_df = df[df["relationship"] == core_relation]
        print(f"\nNumber of {core_relation} rows: {len(relationship_df)}")
        print(f"\nRepresentative {core_relation} rows")
        print(
            relationship_df[
                [
                    "ticker",
                    "entity",
                    "entity_type",
                    "target",
                    "target_type"
                ]
            ].sample(min(20, len(relationship_df)), random_state=42)
        )
        print("\nDistinct entity types")
        print(relationship_df["entity_type"].value_counts())
        print("\nDistinct target types")
        print(relationship_df["target_type"].value_counts())
        print("\nRepresentative entities")
        print(sorted(relationship_df["entity"].dropna().unique())[:50])
        print("\nRepresentative targets")
        print(sorted(relationship_df["target"].dropna().unique())[:50])


# Allowed type constraints
RELATION_CONSTRAINTS = {
    "parent_of": {
        "entity_type": {"ORG"},
        "target_type": {"ORG"},
    },
    "subsidiary_of": {
        "entity_type": {"ORG"},
        "target_type": {"ORG"},
    },
    "produces": {
        "entity_type": {"ORG"},
        "target_type": {"PRODUCT"},
    },
    "invests_in": {
        "entity_type": {"ORG", "PERSON"},
        "target_type": {"ORG"},
    },
    "has_stake_in": {
        "entity_type": {"ORG", "PERSON"},
        "target_type": {"ORG"},
    },

    "supply": {
        "entity_type": {"ORG"},
        "target_type": {"ORG"},
    }
}


def extract_snp100_data():
    # Read all parquet files in the directory as one dataset
    dataset = ds.dataset(INPUT_DIR, format="parquet")
    expr = None
    for relation, c in RELATION_CONSTRAINTS.items():
        e = (
                (ds.field("relationship") == relation)
                &
                ds.field("entity_type").isin(c["entity_type"])
                &
                ds.field("target_type").isin(c["target_type"])
        )
        expr = e if expr is None else (expr | e)

    filter_expr = (
            ds.field("ticker").isin(list(snp100_tickers))
            &
            expr
    )

    # Scan only matching rows
    table = dataset.to_table(filter=filter_expr, use_threads=True)
    print(f"Selected {table.num_rows:,} rows")
    # Write to parquet
    pq.write_table(table, OUTPUT_FILE)
    # Write CSV
    df = table.to_pandas()
    df.to_csv(CSV_FILE, index=False)
    print(f"Wrote {CSV_FILE}")
    print("Done.")




if __name__ == "__main__":
    examine_data()
    extract_snp100_data()
