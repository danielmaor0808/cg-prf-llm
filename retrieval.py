

# Main secondary retrieval function of the project
def grf_query_expansion(query_tokens, llm_generated_doc, top_documents, beta=0.5, theta=10):
    """
    Implements GRF query expansion, incorporating LLM-generated terms.

    Args:
        query_tokens (list): Tokens of the original query.
        bm25 (BM25Okapi): BM25 object initialized with tokenized documents.
        top_documents (list): List of top-ranked tokenized documents from BM25.
        llm_generated_doc (str): LLM-generated text assumed to be relevant.
        beta (float): Weight for the original query terms (0 <= beta <= 1).
        theta (int): Number of top LLM-generated terms to include.

    Returns:
        list: Expanded query tokens.
    """
    # Step 1: Compute P(w|Q) (Probability of term given original query)
    query_term_counts = Counter(query_tokens)
    total_query_terms = sum(query_term_counts.values())
    p_w_given_q = {term: count / total_query_terms for term, count in query_term_counts.items()}

    # Step 2: Compute P(w|R) (Probability of term in top documents)
    top_doc_tokens = word_tokenize(top_documents.lower())
    top_doc_term_counts = Counter(top_doc_tokens)
    total_top_doc_terms = sum(top_doc_term_counts.values())
    p_w_given_r = {term: top_doc_term_counts[term] / total_top_doc_terms for term in top_doc_term_counts}

    # Step 3: Compute P(w|D_LLM) (Probability of term in LLM-generated document)
    llm_tokens = word_tokenize(llm_generated_doc.lower())
    llm_term_counts = Counter(llm_tokens)
    total_llm_terms = sum(llm_term_counts.values())
    p_w_given_dllm = {term: llm_term_counts[term] / total_llm_terms for term in llm_term_counts}

    # Step 4: Select top-θ terms from LLM-generated terms
    w_theta = sorted({word: max(p_w_given_dllm.get(word, 0), p_w_given_r.get(word, 0))
               for word in set(p_w_given_dllm) | set(p_w_given_r)}.items(), key=lambda x: x[1], reverse=True)[:theta]
    w_theta_terms = set([term for term, _ in w_theta])

    # Normalize probabilities for W_theta
    total_w_theta_terms = sum(prob for _, prob in w_theta)
    p_w_theta = {term: prob / total_w_theta_terms for term, prob in w_theta}

    # Combine P(w|Q) and P(w|D_LLM) for GRF
    p_grf = {}
    for term in set(list(p_w_given_q.keys()) + list(p_w_theta.keys())):
        p_grf[term] = beta * p_w_given_q.get(term, 0) + (1 - beta) * p_w_given_dllm.get(term, 0)

    sum_probs = sum(p_grf.values())
    p_grf = {term: prob / sum_probs for term, prob in p_grf.items()}

    # # Step 5: Select top-θ terms from LLM-generated terms
    # w_theta = sorted(p_w_given_dllm.items(), key=lambda x: x[1], reverse=True)[:theta]
    # w_theta_terms = set([term for term, _ in w_theta])

    # Step 6: Build the expanded query
    # expanded_query = query_tokens[:]
    expanded_query = []
    # for term in query_tokens:
    #   if term in p_grf:
    #     expanded_query.append((term, p_grf[term]))

    for term in w_theta_terms:
        if term in p_grf:  # Add terms only if in the GRF model
            expanded_query.append((term, p_grf[term]))

    return expanded_query