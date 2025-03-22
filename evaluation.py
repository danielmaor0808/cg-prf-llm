import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score, average_precision_score
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize

from utils import preprocess_text, concat_top_docs, binarize_ratings, log
# If grf_query_expansion is in another module:
# from your_grf_module import grf_query_expansion

class Evaluator:
    def __init__(self, df_sorted: pd.DataFrame, generated_documents_df: pd.DataFrame):
        """
        df_sorted: DataFrame with columns at least:
            ['qid', 'rating', 'score', 'body', 'query', ...]
        generated_documents_df: DataFrame with columns at least:
            ['qid', 'generated_document', 'query', ...]
        """
        self.df_sorted = df_sorted.copy()
        self.generated_documents_df = generated_documents_df.copy()

        # Log basic info about the dataset
        unique_qids_df = self.df_sorted['qid'].nunique()
        unique_qids_gen = self.generated_documents_df['qid'].nunique()
        log(f"[Evaluator] Initializing with {unique_qids_df} QIDs in df_sorted, {unique_qids_gen} QIDs in generated_docs.")

        # Compute and store initial metrics
        (self.initial_mean_ndcg,
         self.initial_mean_ndcg_10,
         self.initial_map) = self._compute_initial_metrics()

        log("[Evaluator] Initial metrics:")
        log(f"   - Mean NDCG:     {self.initial_mean_ndcg:.4f}")
        log(f"   - Mean NDCG@10:  {self.initial_mean_ndcg_10:.4f}")
        log(f"   - Mean MAP:      {self.initial_map:.4f}")

        # Optional: dictionary storing final metrics keyed by parameter combos
        self.results_by_param = {}

    def _compute_initial_metrics(self):
        """
        Compute the initial (no re-ranking) mean NDCG, NDCG@10, and MAP
        across all qids based on the existing 'score' column in self.df_sorted.
        """
        ndcg_values = []
        ndcg_10_values = []
        ap_values = []

        for qid, group in self.df_sorted.groupby('qid'):
            ratings = np.array([group['rating'].tolist()])   # shape (1, n_docs)
            scores  = np.array([group['score'].tolist()])    # shape (1, n_docs)

            ndcg_values.append(ndcg_score(ratings, scores))
            ndcg_10_values.append(ndcg_score(ratings, scores, k=10))

            # Convert to binary for AP
            ratings_ap = binarize_ratings(ratings[0], threshold=1.0)
            ap_values.append(average_precision_score(ratings_ap, scores[0]))

        mean_ndcg  = float(np.mean(ndcg_values))
        mean_ndcg10= float(np.mean(ndcg_10_values))
        mean_map   = float(np.mean(ap_values))

        return (mean_ndcg, mean_ndcg10, mean_map)

    def evaluate_params(self, params: dict) -> dict:
        """
        Run the entire expansion + BM25 re-score + metrics for one set of parameters.

        Expected params keys: 'num_top_docs', 'num_llm_docs', 'beta', 'theta', 'b1', 'k'
        (Adapt as needed.)

        Returns a dict with:
            {
              'initial_mean_ndcg': float,
              'initial_mean_ndcg@10': float,
              'initial_mean_ap': float,
              'mean_ndcg': float,
              'mean_ndcg@10': float,
              'mean_ap': float
            }
        """
        log("[evaluate_params] Evaluating param set:")
        log(f"   {params}")

        # 1) Expand queries
        self._expand_queries_for_all_qids(params)

        # 2) BM25 re-scoring
        self._bm25_rescore(params)

        # 3) Compute final metrics
        final_metrics = self._compute_final_metrics(params)

        # Store results under a param key
        key = (f"beta={params['beta']},theta={params['theta']},"
               f"b1={params['b1']},k={params['k']},"
               f"top_docs={params['num_top_docs']},"
               f"llm_docs={params['num_llm_docs']}")
        self.results_by_param[key] = final_metrics

        log("[evaluate_params] Final metrics:")
        log(f"   - mean_ndcg:    {final_metrics['mean_ndcg']:.4f}")
        log(f"   - mean_ndcg@10: {final_metrics['mean_ndcg@10']:.4f}")
        log(f"   - mean_ap:      {final_metrics['mean_ap']:.4f}")
        return final_metrics

    def _expand_queries_for_all_qids(self, params: dict):
        """
        For each qid, gather top docs, gather top LLM docs,
        then call 'grf_query_expansion' to get the expanded query.
        Store that expanded query in 'df_sorted' for each row of that qid.
        """
        # Example placeholder import
        from your_grf_module import grf_query_expansion

        num_top_docs = int(params['num_top_docs'])
        num_llm_docs = int(params['num_llm_docs'])
        beta_iter     = params['beta']
        theta_iter    = params['theta']

        log("[_expand_queries_for_all_qids] Expanding queries for each QID:")
        log(f"   - num_top_docs={num_top_docs}, num_llm_docs={num_llm_docs}, beta={beta_iter}, theta={theta_iter}")

        # Ensure 'expanded_query' column exists
        if 'expanded_query' not in self.df_sorted.columns:
            self.df_sorted['expanded_query'] = None

        unique_qids = self.generated_documents_df['qid'].unique()
        for i, qid in enumerate(unique_qids, start=1):
            # 1) Gather top docs from df_sorted
            qid_group = self.df_sorted[self.df_sorted['qid'] == qid]
            top_docs_group = qid_group.sort_values(by='score', ascending=False).head(num_top_docs)
            top_docs = top_docs_group['body'].tolist()

            # Preprocess & concat
            top_docs_str = preprocess_text(concat_top_docs(top_docs))

            # 2) Gather top LLM docs
            top_llm_docs_group = self.generated_documents_df[self.generated_documents_df['qid'] == qid].head(num_llm_docs)
            top_llm_docs = top_llm_docs_group['generated_document'].tolist()
            llm_docs_str = preprocess_text(concat_top_docs(top_llm_docs))

            # 3) Original query
            query = self.generated_documents_df.loc[self.generated_documents_df['qid'] == qid, 'query'].iloc[0]
            query_tokens = word_tokenize(query.lower())

            # 4) Expand
            expanded_query = grf_query_expansion(
                query_tokens=query_tokens,
                llm_generated_doc=llm_docs_str,
                top_documents=top_docs_str,
                beta=beta_iter,
                theta=theta_iter
            )

            # 5) Store
            self.df_sorted.loc[self.df_sorted['qid'] == qid, 'expanded_query'] = [expanded_query]*len(qid_group)

            # Optionally log every query or every Nth query
            if i % 10 == 0:
                log(f"   -> Processed {i}/{len(unique_qids)} queries so far. (Current QID: {qid})")

    def _bm25_rescore(self, params: dict):
        """
        Build a BM25Okapi per-qid using the expanded query, then produce a new 'score2'.
        """
        b1_iter = params['b1']
        k_iter  = params['k']

        log(f"[_bm25_rescore] Re-scoring with b1={b1_iter}, k={k_iter}")

        # Ensure 'score2' exists
        self.df_sorted['score2'] = 0.0

        for qid, group in self.df_sorted.groupby('qid', group_keys=False):
            docs = group['body'].tolist()
            row_indices = group.index.tolist()

            expanded_query = group['expanded_query'].iloc[0]
            # expanded_query is something like: [(term, weight), (term2, weight2), ...]

            # Tokenize each doc
            tokenized_docs = [preprocess_text(doc).split() for doc in docs]

            # Build BM25
            bm25 = BM25Okapi(tokenized_docs, k1=k_iter, b=b1_iter)

            # Score with weighted terms
            scores = np.zeros(len(docs))
            for (term, weight) in expanded_query:
                term_tokens = word_tokenize(term)
                partial_scores = bm25.get_scores(term_tokens)
                scores += weight * partial_scores

            # Update 'score2'
            for idx, s in zip(row_indices, scores):
                self.df_sorted.at[idx, 'score2'] = s

    def _compute_final_metrics(self, params: dict) -> dict:
        """
        After the re-scoring with 'score2', compute NDCG, NDCG@10, and MAP
        for each qid, then average them.
        """
        log("[_compute_final_metrics] Computing final metrics after BM25 re-score...")
        ndcg_values2 = []
        ndcg_10_values2 = []
        ap_values2 = []

        for qid, group in self.df_sorted.groupby('qid'):
            ratings = np.array([group['rating'].tolist()])
            scores2 = np.array([group['score2'].tolist()])

            ndcg_values2.append(ndcg_score(ratings, scores2))
            ndcg_10_values2.append(ndcg_score(ratings, scores2, k=10))

            ratings_ap = binarize_ratings(ratings[0], threshold=1.0)
            ap_values2.append(average_precision_score(ratings_ap, scores2[0]))

        mean_ndcg2 = float(np.mean(ndcg_values2))
        mean_ndcg_10_2 = float(np.mean(ndcg_10_values2))
        mean_ap2 = float(np.mean(ap_values2))

        result = {
            "initial_mean_ndcg":    self.initial_mean_ndcg,
            "initial_mean_ndcg@10": self.initial_mean_ndcg_10,
            "initial_mean_ap":      self.initial_map,
            "mean_ndcg":            mean_ndcg2,
            "mean_ndcg@10":         mean_ndcg_10_2,
            "mean_ap":              mean_ap2
        }

        return result
