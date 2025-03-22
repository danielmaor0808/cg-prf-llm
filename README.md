# Prompting the Future of Search
**Corpus-Guided Generative PRF (CG-PRF) with GPT-4 for Enhanced Information Retrieval**

Daniel Maor & Ravid Gersh

---

## Table of Contents
1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Methodology](#methodology)
6. [Experimental Setup](#experimental-setup)
7. [Results](#results)
8. [Future Work](#future-work)
9. [References](#references)

---

## Introduction
This project explores a **Corpus-Guided Generative Pseudo-Relevance Feedback (CG-PRF)** approach that leverages GPT-4 to produce new, corpus-aligned documents for query expansion. We build upon the **Generative Relevance Feedback (GRF)** framework proposed by Mackie et al. [1], aiming to address limitations of classical Pseudo-Relevance Feedback (PRF) and purely generative methods.

- **Key Goal:** Combine LLM-based generation with initial retrieval to produce context-aware documents that align with the corpus’s structure, vocabulary, and time period.  
- **Datasets:** Primarily evaluated on **TREC Deep Learning (DL) 2019/2020**, with potential extension to other corpora.

---

## Project Structure

```
Prompting-Future-of-Search/
├── data_loading.py              # Functions to load queries, qrels, corpus, top-100 docs
├── evaluation.py                # Evaluator class to compute NDCG, MAP, etc.
├── generation.py                # (Optional) Logic for GPT-4 document generation
├── run.py                       # Main script to orchestrate training and testing
├── config.py                    # Loads environment variables (API_KEY, file paths, etc.)
├── utils.py                     # Helper functions: logging, text preprocessing, etc.
├── parameter_sweep.py           # (Optional) Parameter grid for BM25 re-ranking, expansions
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── .env.example                 # Template for environment variables
└── ...
```

## Installation

Follow these steps to set up the environment and install dependencies:

```bash
git clone https://github.com/YourUsername/Prompting-Future-Search.git
cd Prompting-Future-Search
pip install -r requirements.txt```
```

### ⚙️ Environment Setup

This project uses a `.env` file to manage configuration.

1. Create a `.env` file in the root directory.
2. You can start from the provided template:

   ```bash
   cp .env.example .env
   ```
3. Open `.env` and fill in the required values, such as:
   - `API_KEY` — your OpenAI API key
   - `MODEL_NAME` — e.g., `gpt-4o`
   - File paths for:
     - `QUERIES_FILE_DL20`
     - `QRELS_FILE_DL20`
     - `CORPUS_FILE_DL20`
     - `TOP100_FILE_DL20`
     - (And corresponding DL19 paths)
   - Generation flags:
     - `GENERATE_DOCS_DL20=true` if you want to generate documents using GPT-4
     - `GENERATED_DOCS_FILE_DL20=/path/to/generated_docs_dl20.csv` if you already have them
     - (Same options for `DL19`)
   - Logging:
     - `VERBOSE_LOGGING=true` to show progress messages


## Usage

### 1. Prepare Data
- Download the **TREC DL 2019/2020** corpora, queries, qrels, and top-100 documents.
- Update file paths in your `.env` file. For example:
```bash
QUERIES_FILE_DL20=/your/path/to/msmarco-test2020-queries.tsv.gz
QRELS_FILE_DL20=/your/path/to/2020qrels-docs.txt
CORPUS_FILE_DL20=/your/path/to/msmarco-docs-2020.tsv.gz
TOP100_FILE_DL20=/your/path/to/msmarco-doctest2020-top100.gz
```
*(Repeat similarly for the DL19 paths.)*

---

### 2. (Optional) Generate Documents
- If `GENERATE_DOCS_DL20=true` or `GENERATE_DOCS_DL19=true` in `.env`, the script will call GPT-4 to produce corpus-aware documents for each query.

> **Note**: GPT-4 calls can be expensive. If you already have generated documents, simply set:
```env
GENERATE_DOCS_DL20=false
GENERATED_DOCS_FILE_DL20=/your/path/to/generated_docs_dl20.csv
```
*(Likewise for DL19 if relevant.)*

Then you can run:
```bash
python run_generation.py
```

---

### 3. Run the Main Pipeline
Execute the full training + testing evaluation by running:
```bash
python run.py --train=DL20 --test=DL19
```
This will:
- Load the top-100 docs, corpus, queries, and qrels.
- Expand queries using the **CG-PRF** approach (LLM + BM25).
- Re-rank the documents based on the expanded query.
- Evaluate using **nDCG@10**, **MAP**, and other metrics.

---

### 4. (Optional) Parameter Tuning
If you want to perform a full sweep over hyperparameters (e.g., `β`, `θ`, BM25 `k1`, `b`, etc.):

```bash
python run_parameter_sweep.py
```

This will:
- Cross-validate different values for each parameter.
- Save the results in a CSV for later analysis.


## Methodology

To address the limitations of Pseudo-Relevance Feedback (PRF) and Generative Relevance Feedback (GRF), we propose a **Corpus-Guided Generative PRF (CG-PRF)** approach that leverages GPT-4 to condition text generation on retrieved documents. 

Unlike GRF (which used GPT-3.5), we employ GPT-4 primarily for its enhanced text comprehension and reasoning capabilities—*not* for additional knowledge. Since the underlying corpus and relevance labels have not changed, GPT-4’s newer knowledge has no direct advantage for these datasets. We also follow the original GRF paper’s query expansion formulation and evaluation protocols, ensuring a **fair comparison** to both GRF and traditional PRF-based methods.

---

### Our Approach

1. **Initial Retrieval**  
   We start with a **fixed list of the top 100 documents** for each query (as provided by TREC). This list was originally retrieved via BM25 (with default parameters).

2. **Corpus-Aware Generation**  
   - **Context**: Unlike GRF (which simply prompted GPT-3.5 with a query + subtask instructions), we provide GPT-4 with the corpus name, publication years, *and* the top-3 retrieved documents.  
   - **Goal**: Generate 10 new documents that *expand* the query while staying aligned with the corpus’s style, structure, and time period.  
   - **Example Prompt** (for TREC DL 2019/2020):
     ```text
     "Generate {n} different full-length documents relevant to this query: '{query}' 
      for the TREC DL 2019-2020 corpus and relevant to these years for query expansion. 
      Use these top-scored documents to do so:
      first document: {top_k_documents[0]}.
      second document: {top_k_documents[1]}.
      third document: {top_k_documents[2]}.
      Make sure the generated documents are about the same length as these, 
      and the entire output is solely the documents without extra words.
      Separate the documents with ' &&& '"
     ```
     Here, `{top_k_documents}` is the list of the top-10 BM25-retrieved documents.

3. **Query Expansion (Probability Computation)**  
   We use a **probability-based formulation** inspired by RM3 [13] and employed by GRF, ensuring consistent normalization and direct comparability. Let:

   - **w** be a term.
   - **D<sub>LLM</sub>** be the concatenation of the 10 generated documents.
   - **P(w | Q)** be the probability of *w* in the original query.
   - **P(w | D<sub>LLM</sub>)** be the probability of *w* in the generated documents.
   - **β** be a hyperparameter balancing the original query terms and generated terms.
   - **W<sub>θ</sub>** be the set of top-θ terms selected for expansion.

   **Expansion Formula:**

![image](https://github.com/user-attachments/assets/4076cf4b-23c8-4bea-a71d-6224a20ab66d)

4. **Re-Ranking**  
   Rather than performing an end-to-end retrieval (as in GRF), we **re-rank only the top 100** initially retrieved documents by computing a new relevance score based on the expanded query. This final scoring step follows the same BM25 (or alternative) re-weighting approach, but now includes terms from the **generated** documents.

---

> **Why GPT-4?**  
> GPT-4 provides stronger text comprehension and reasoning skills for producing corpus-aligned expansions. Its newer knowledge is not required—our focus is purely on leveraging its generative capabilities to produce high-quality, contextually accurate expansions.

This setup allows us to **directly assess the benefits** of integrating corpus-aware generation with pseudo-relevance feedback, relative to both **GRF** and other PRF baselines.

## Experimental Setup

### 4.1 Datasets

#### 4.1.1 Retrieval Corpora
**TREC Deep Learning (DL) 2019/2020** builds upon the MS MARCO web queries and documents.  
- DL-19 contains 43 queries, and DL-20 contains 45 queries.
- Queries are predominantly factoid-based.
- Judgments are provided by NIST with deeper pooling than MS MARCO.

#### 4.1.2 Preprocessing and Evaluation
- We preprocess the data using the **NLTK** library for:
  - Stopword removal
  - Porter stemming
- Unlike the GRF experiment (which used full end-to-end retrieval and optimized R@1k), we **re-rank a fixed list of 100 retrieved documents per query**.
- As a result, **R@1k is not applicable**, and we focus on:
  - **nDCG@10** — our primary optimization target
  - **MAP** — for binary relevance effectiveness

- Cross-validation setup:
  - **Train on DL-20, test on DL-19**, and vice versa.
  - Use averaged best parameters from the validation set.

- **Relevance label binarization** for MAP:
  - Original labels: {0, 1, 2, 3}
  - Labels 2 & 3 → *relevant*
  - Labels 0 & 1 → *non-relevant*

---

### 4.2 CG-PRF Implementation

#### 4.2.1 LLM Generation
- We use the **GPT-4 API**, specifically the `gpt-4o` model.
- **Generation parameters**:
  - `temperature = 0.7`
  - All other parameters set to default.

#### 4.2.2 Retrieval and Expansion
To avoid **query drift**, we re-rank only the top-100 BM25-retrieved documents, using:

- **BM25 Retrieval Settings**:
  - `k1 = 1.2`
  - `b = 0.75`

- **CG-PRF Hyperparameters** (tuned on training set):
  - **θ**: Number of feedback terms  
    `range = [5, 10, ..., 70] (step 5)`
  - **β**: Interpolation weight between original query and LLM terms  
    `range = [0.2, 0.3, ..., 0.8] (step 0.1)`
  - **k1** (BM25 for re-ranking):  
    `range = [0.9, 1.1, ..., 2.3] (step 0.2)`
  - **b** (BM25 for re-ranking):  
    `range = [0.5, 0.6, ..., 0.9] (step 0.1)`

> These hyperparameter ranges match or are contained within those used in the original GRF paper to ensure fair comparison.

---

### 4.3 Comparison Methods

We compare CG-PRF against the following baselines from the original GRF paper:

- **BM25 [18]**  
  - Sparse retrieval baseline  
  - Tuned `k1 ∈ [0.1–5.0]`, `b ∈ [0.1–1.0]`

- **BM25 + RM3 [19]**  
  - Relevance model expansion  
  - Tuned:
    - `fb_terms ∈ [5–95]` (step 5)
    - `fb_docs ∈ [5–50]` (step 5)
    - `original_query_weight ∈ [0.2–0.8]` (step 0.1)

- **CEQE [20]**  
  - Contextualized query expansion using vector pooling  
  - Used CEQE-MaxPool runs provided by the authors

- **SPLADE + RM3**  
  - Combines sparse expansion with dense semantic retrieval  
  - Uses the `splade-cocondenser-ensembledistil` checkpoint  
  - Tuned:
    - `fb_docs ∈ {5, 10, 15, 20, 25, 30}`
    - `fb_terms ∈ {20, 40, 60, 80, 100}`
    - `original_query_weight ∈ [0.1–0.9]` (step 0.1)

- **TCT + PRF [21]**  
  - Rocchio-style feedback with TCT-ColBERT-v2-HNP  
  - Tuned:
    - `depth ∈ {2, 3, 5, 7, 10, 17}`
    - `α, β ∈ [0.1–0.9]` (step 0.1)

- **ColBERT + PRF [41]**  
  - Based on runs provided by Wang et al. using the PyTerrier framework

- **GRF [1]**  
  - Uses the same RM3-inspired expansion formula as CG-PRF  
  - Tuned:
    - `θ ∈ [5–95]` (step 5)
    - `β ∈ [0.2–0.8]` (step 0.1)
    - BM25: `k1 ∈ [0.1–5.0]`, `b ∈ [0.1–1.0]`

> 📝 *This comparison section has been taken directly from the GRF paper for accuracy and clarity. No modifications were made.*

---
## References

[1] Mackie, Iain. *Generative relevance feedback with large language models.* Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2023.

[2] Bubeck, S. *Sparks of artificial general intelligence: Early experiments with GPT-4*, 2023.

[3] Manning, C. D., Raghavan, P., & Schütze, H. *Introduction to Information Retrieval.* Cambridge University Press, 2008.

[4] Rocchio, J. J. *Relevance feedback in information retrieval*, 1971.

[5] Zhai, C., & Lafferty, J. *Model-based feedback in the language modeling approach to information retrieval.* Proceedings of the 10th International Conference on Information and Knowledge Management, 2001.

[6] Lavrenko, V., & Croft, W. B. *Relevance models for topic detection and tracking.* Proceedings of the Human Language Technology Conference (HLT), 2002.

[7] Khattab, O., & Zaharia, M. *ColBERT: Efficient and effective passage search via contextualized late interaction over BERT.* Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval, 2020.

[8] Xiong, C., et al. *Improving query representations for dense retrieval with pseudo relevance feedback.* Proceedings of the 30th ACM International Conference on Information & Knowledge Management, 2021.

[9] Formal, T., et al. *A white box analysis of ColBERT.* Advances in Information Retrieval: 43rd European Conference on IR Research, ECIR 2021.

[10] Bonifacio, L., et al. *InPars: Data augmentation for information retrieval using large language models*, 2022.

[11] Vakulenko, S., et al. *Question rewriting for conversational question answering.* Proceedings of the 14th ACM International Conference on Web Search and Data Mining, 2021.

[12] Rahaman, M. *From ChatGPT-3 to GPT-4: A significant advancement in AI-driven NLP tools.* Journal of Engineering and Emerging Technologies, 2023.

[13] Lavrenko, V., & Croft, W. B. *Relevance-Based Language Models.* Center for Intelligent Information Retrieval, 2001.

[14] Craswell, N., et al. *Overview of the TREC 2020 Deep Learning Track.* In Text REtrieval Conference (TREC), 2020.

[15] Nguyen, T., et al. *MS MARCO: A human-generated machine reading comprehension dataset*, 2016.

[16] Mackie, I., et al. *How deep is your learning: The DL-HARD annotated deep learning dataset.* Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval.

[17] OpenAI. *GPT-4 API documentation*. Available: [https://platform.openai.com/docs/](https://platform.openai.com/docs/) (Accessed: Mar. 10, 2025).

[18] Robertson, S. E., & Walker, S. *Some simple effective approximations to the 2-Poisson model for probabilistic weighted retrieval.* SIGIR 1994.

[19] Jaleel, N. A., et al. *UMass at TREC 2004.* Computer Science Department Faculty Publication Series, 2004.

[20] Shahrzad, N., et al. *CEQE: Contextualized embeddings for query expansion.* Advances in Information Retrieval, ECIR 2021.

[21] Khattab, O., & Zaharia, M. *ColBERT: Efficient and effective passage retrieval.* In SIGIR, 2020.

[22] Wang, X., Macdonald, C., Tonellotto, N., & Ounis, I. *ColBERT-PRF: Semantic pseudo-relevance feedback for dense passage and document retrieval.* ACM Transactions on the Web, 2022.

[23] Macdonald, C., Tonellotto, N., MacAvaney, S., & Ounis, I. *PyTerrier: Declarative experimentation in Python from BM25 to dense retrieval.* In ACM International Conference on Information & Knowledge Management, 2021.
