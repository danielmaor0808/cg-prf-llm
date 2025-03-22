from utils import *
from generation import *
from data import *
import argparse
from utils import log
from config import (
    API_KEY,
    MODEL_NAME,
    ROLE,
    PROMPT_TEMPLATE,
    QUERIES_FILE_DL20,
    QRELS_FILE_DL20,
    CORPUS_FILE_DL20,
    TOP100_FILE_DL20,
    QUERIES_FILE_DL19,
    QRELS_FILE_DL19,
    CORPUS_FILE_DL19,
    TOP100_FILE_DL19
)

def get_dataset(name):
    if name == "DL19":
        return {
            "queries": QUERIES_FILE_DL19,
            "qrels": QRELS_FILE_DL19,
            "corpus": CORPUS_FILE_DL19,
            "top100": TOP100_FILE_DL19
        }
    elif name == "DL20":
        return {
            "queries": QUERIES_FILE_DL20,
            "qrels": QRELS_FILE_DL20,
            "corpus": CORPUS_FILE_DL20,
            "top100": TOP100_FILE_DL20
        }
    else:
        raise ValueError(f"Unknown dataset: {name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", choices=["DL19", "DL20"], required=True)
    parser.add_argument("--test", choices=["DL19", "DL20"], required=True)
    args = parser.parse_args()

    train_data = get_dataset(args.train)
    test_data = get_dataset(args.test)

    # Setup necessary resources
    setup_nltk_resources()
    stop_words = get_stopwords()
    stemmer = get_stemmer()

    # Create a client with your OpenAI API key
    client = create_client(API_KEY)

    # Define the parameters for the document generation
    top_k =10

    # Load the dataframes for the training data
    top100_df = load_top100(train_data["top100"])
    queries_df = load_queries(train_data["queries"])
    qrels_df = load_qrels(train_data["qrels"])
    corpus_df = load_corpus(train_data["corpus"])

    # Create the combined and sorted dataframe
    df_sorted = combine_and_sort_data(top100_df, queries_df, qrels_df, corpus_df)

    if GENERATE_DOCS_DL20:
        # Generate documents
        generated_documents_df = generate_documents_for_all_queries(df_sorted, top_k, client)
    else:
        generated_documents_df = load_generated_documents(GENERATED_DOCS_FILE_DL20)
        generated_documents_df = generated_documents_df[generated_documents_df['qid'].isin(df_sorted['qid'])]

    # Create evaluator
    evaluator = Evaluator(df_sorted, generated_documents_df)

    # Prepare a DataFrame for collecting final parameter results
    columns = ['beta', 'theta', 'b1', 'k', 'num_top_docs', 'num_llm_docs',
               'initial_mean_ndcg', 'initial_mean_ndcg@10', 'initial_mean_ap',
               'mean_ndcg', 'mean_ndcg@10', 'mean_ap']
    final_results_df = pd.DataFrame(columns=columns)

    # Sweep over parameters
    for params in parameter_grid():
        metrics = evaluator.evaluate_params(params)

        row = {
            'beta': params['beta'],
            'theta': params['theta'],
            'b1': params['b1'],
            'k': params['k'],
            'num_top_docs': params['num_top_docs'],
            'num_llm_docs': params['num_llm_docs'],
            'initial_mean_ndcg': metrics['initial_mean_ndcg'],
            'initial_mean_ndcg@10': metrics['initial_mean_ndcg@10'],
            'initial_mean_ap': metrics['initial_mean_ap'],
            'mean_ndcg': metrics['mean_ndcg'],
            'mean_ndcg@10': metrics['mean_ndcg@10'],
            'mean_ap': metrics['mean_ap']
        }
        final_results_df = pd.concat([final_results_df, pd.DataFrame([row])],
                                     ignore_index=True)

        # print progress
        log(f"[RUN] ✅ Params = {params} → mean_ndcg@10 = {metrics['mean_ndcg@10']:.4f}")

    # Save the final results
    final_results_df.to_csv("final_results.csv", index=False)
    print("Final results saved to final_results.csv")

    #Save the final results
    best_row = final_results_df.sort_values("mean_ndcg@10", ascending=False).iloc[0]
    best_params = {
        "beta": best_row["beta"],
        "theta": best_row["theta"],
        "b1": best_row["b1"],
        "k": best_row["k"],
        "num_top_docs": best_row["num_top_docs"],
        "num_llm_docs": best_row["num_llm_docs"]
    }

    # Load the dataframes for the test data
    top100_df_test = load_top100(train_data["top100"])
    queries_df_test = load_queries(train_data["queries"])
    qrels_df_test = load_qrels(train_data["qrels"])
    corpus_df_test = load_corpus(train_data["corpus"])

    # Create the combined and sorted test-dataframe
    test_df_sorted = combine_and_sort_data(top100_df_test, queries_df_test, qrels_df_test, corpus_df_test)

    if GENERATE_DOCS_DL19:
        # Generate testset documents
        test_generated_documents_df = generate_documents_for_all_queries(test_df_sorted, top_k, client)
    else:
        test_generated_documents_df = load_generated_documents(GENERATED_DOCS_FILE_DL19)
        test_generated_documents_df = generated_documents_df[generated_documents_df['qid'].isin(test_df_sorted['qid'])]

    # Create evaluator
    evaluator = Evaluator(test_df_sorted, test_generated_documents_df)

    # Run evaluation with best training params
    test_metrics = test_evaluator.evaluate_params(best_params)

    print("🚀 Final test results on DL2019 with best training params:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()





