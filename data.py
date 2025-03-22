import gzip
import pandas as pd
import numpy as np
from utils import log


def load_top_100_file(top100_file):
    """
    Load the top 100 file into a pandas DataFrame
    """
    columns = ["qid", "Q0", "docid", "rank", "score", "runstring"]
    top100_df = pd.read_csv(top100_file, sep="\s+", names=columns, header=None)

    log(f"[load_top_100_file] Loaded '{top100_file}' with {len(top100_df)} rows.")
    log(f"[load_top_100_file] Columns: {list(top100_df.columns)}")

    return top100_df


def load_queries_file(queries_file):
    """
    Load the queries file into a pandas DataFrame
    """
    # Create qrels DF
    with gzip.open(queries_file, 'rt') as file_in:  # Open gzipped file
        queries_data = []
        for line in file_in:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                queries_data.append({"qid": parts[0], "query": parts[1]})
        queries_df = pd.DataFrame(queries_data)

    log(f"[load_queries_file] Loaded '{queries_file}' with {len(queries_df)} queries.")
    log(f"[load_queries_file] Columns: {list(queries_df.columns)}")

    return queries_df


def load_qrels_file(qrels_file):
    """
    Load the qrels file into a pandas DataFrame
    """
    # Initialize an empty list to store parsed data
    qrels_data = []

    # Read and parse the file line by line
    with open(qrels_file, 'r', encoding='utf-8') as file:
        for line in file:
            # Split the line into parts by whitespace
            parts = line.strip().split()
            # Ensure the line has the expected number of columns
            if len(parts) == 4:
                qid, q0, docid, rating = parts
                qrels_data.append({
                    "qid": qid,
                    "Q0": q0,
                    "docid": docid,
                    "rating": int(rating)  # Convert rating to integer
                })

    log(f"[load_qrels_file] Loaded '{qrels_file}' with {len(qrels_df)} rows.")
    log(f"[load_qrels_file] Columns: {list(qrels_df.columns)}")

    return pd.DataFrame(qrels_data)


def load_corpus_file(corpus_file):
    """
    Load the corpus file into a pandas DataFrame
    """
    corpus_df = pd.read_csv(
        corpus_file,
        sep="\t",
        on_bad_lines="skip",  # Skip problematic lines
        encoding="utf-8",
        names=["docid", "url", "title", "body"],  # Column names
        header=0,  # Use the first row as the actual data
        usecols=["docid", "title", "body"]  # Columns of interest
    )

    log(f"[load_corpus_file] Loaded '{corpus_file}' with {len(corpus_df)} documents.")
    log(f"[load_corpus_file] Columns: {list(corpus_df.columns)}")

    return corpus_df


def combine_and_sort_data(top100_df, queries_df, qrels_df, corpus_df):
    """
    Combine the data into a single DataFrame
    """
    # Ensure both columns are strings (objects) before merging
    top100_df["qid"] = top100_df["qid"].astype(str)
    top100_df["docid"] = top100_df["docid"].astype(str)
    qrels_df["qid"] = qrels_df["qid"].astype(str)
    qrels_df["docid"] = qrels_df["docid"].astype(str)
    corpus_df["docid"] = corpus_df["docid"].astype(str)

    # Keep only qid values that exist in qrels_df
    top100_df = top100_df[top100_df["qid"].isin(qrels_df["qid"])]

    # Join top100_df with queries_df on 'qid'
    top100_with_queries = top100_df.merge(queries_df, on="qid", how="inner")

    # Join the resulting DataFrame with corpus_df on 'docid'
    df_sorted = top100_with_queries.merge(corpus_df, on="docid", how="inner")

    # Perform a left join on qid and docid
    df_sorted = df_sorted.merge(qrels_df, on=["qid", "docid"], how="left")

    # Fill missing rating values with 0
    df_sorted["rating"] = df_sorted["rating"].fillna(0).astype(int)

    # Sort by query ID and rank
    df_sorted = df_sorted.sort_values(by=["qid", "rank"]).reset_index(drop=True)

    # Drop unnecessary columns
    df_sorted = df_sorted.drop(columns=['Q0_x', 'Q0_y', 'runstring'])

    # get rid of all records where query id doesnt have any relevant documents
    df_sorted = df_sorted[df_sorted.groupby('qid')['rating'].transform('max') > 0]

    log("[combine_and_sort_data] Combined data summary:")
    log(f"    Total rows: {len(df_sorted)}")
    log(f"    Columns: {list(df_sorted.columns)}")
    log(f"    Unique QIDs retained: {df_sorted['qid'].nunique()}")

    return df_sorted


def load_generated_documents(generated_documents_file):
    """
    Load the generated documents file into a pandas DataFrame
    """
    # load generated documents
    generated_documents_df = pd.read_csv("2020_all_generated_documents.csv")

    # convert qid to str
    generated_documents_df['qid'] = generated_documents_df['qid'].astype(str)

    log(f"[load_generated_documents] Loaded '{generated_documents_file}' with {len(generated_documents_df)} rows.")
    log(f"[load_generated_documents] Columns: {list(generated_documents_df.columns)}")

    return generated_documents_df
