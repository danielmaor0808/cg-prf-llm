from openai import OpenAI
from config import MODEL_NAME, ROLE, PROMPT_TEMPLATE
from utils import log


def create_client(api_key):
    return OpenAI(api_key=key)


def count_tokens(text):
    """Returns the number of tokens in a given text."""
    encoding = tiktoken.get_encoding("cl100k_base")  # Adjust this based on your model
    return len(encoding.encode(text))


def truncate_text(text, max_tokens=10000):
    """Truncates text from the end to fit within max_tokens."""
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    if len(tokens) > max_tokens:
        truncated_tokens = tokens[:max_tokens]  # Keep only the first max_tokens tokens
        return encoding.decode(truncated_tokens)

    return text  # Return the original text if it's already within the limit


# Define a function to build a prompt for the GPT-4o model
def build_prompt(n, query, top_k_documents):
    return PROMPT_TEMPLATE.format(
        n=n,
        query=query,
        doc1=top_k_documents[0],
        doc2=top_k_documents[1],
        doc3=top_k_documents[2]
    )

# Define a function to generate text using the GPT-4o model
def chat_gpt(query, top_k_documents, max_tokens, n, client):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": ROLE,
                "content": build_prompt(n, query, top_k_documents)
            }
        ],

        max_tokens=max_tokens,
        temperature=0.7,  # Controls the randomness
        n=1,  # Generate one response at a time
        # stop=["\n\n"]       # Optional: Add a stopping condition if needed
    )

    return response.choices[0].message.content.strip()


# Define a function to generate documents for a given query
def generate_documents(query, init_k, n, token_counter, client):
    for i in range(len(init_k)):
        if i > 2:
            continue
        if count_tokens(init_k[i]) > 7000: # Truncate the document if it exceeds the token limit
            init_k[i] = truncate_text(init_k[i], max_tokens=7000)

    return chat_gpt(query=query, top_k_documents=init_k, max_tokens=8500, n=n, client)


def clean_generated_documents(generated_docs):
    generated_docs = generated_docs.replace('#', '').replace('-', ' ').replace('*', '')
    generated_docs = re.sub(r"\*\*Document \d+: ", " ", generated_docs)
    generated_docs = re.sub(r"Document \d+: ", " ", generated_docs)
    generated_docs = re.sub(r"\*\*Document\d+: ", " ", generated_docs)
    generated_docs = re.sub(r"Document\d+: ", " ", generated_docs)
    generated_docs = generated_docs.split('&&&')
    return generated_docs


def generate_documents_for_all_queries(df_sorted, top_k, client):
    generated_documents = []

    log(f"🔄 Starting document generation for {df_sorted['qid'].nunique()} unique queries...")
    log(f"🔢 Top-k source documents per query: {top_k}")

    for qid, group in df_sorted.groupby('qid'):
        # Get the query text
        query = group["query"].iloc[0]  # All rows in the group have the same query

        # Take the top-k documents' bodies
        top_k_documents = group.head(top_k)["body"].tolist()

        log(f"\n🧠 [{i}/{df_sorted['qid'].nunique()}] Generating docs for QID: {qid}")
        log(f"   - Query: {query}")
        log(f"   - Using {len(top_k_documents)} top documents as context.")

        # Generate r documents for the query using the black-box function
        generated_docs = generate_documents(query=query, init_k=top_k_documents, n=10, token_counter=token_counter)

        # Clean the generated documents
        generated_docs = clean_generated_documents(generated_docs)

        log(f"   ✅ Generated {len(generated_docs)} documents.")

        # Append the generated documents to the results list
        for doc in generated_docs:
            generated_documents.append({
                "qid": qid,
                "query": query,
                "generated_document": doc
            })

        # Sleep for 60 seconds to avoid exceeding token limit
        log("   💤 Sleeping 60 seconds to avoid token limit...")
        time.sleep(60)

    # Convert the results to a single DataFrame (if needed for further analysis)
    generated_documents_df = pd.DataFrame(generated_documents)

    generated_documents_df = generated_documents_df[
        generated_documents_df['generated_document'].notna() &
        generated_documents_df['generated_document'].str.strip().astype(bool)
        ]

    post_filter_count = len(generated_documents_df)
    log(f"\n📊 Generation complete.")
    log(f"   - Total raw generated documents: {pre_filter_count}")
    log(f"   - After filtering invalid/empty docs: {post_filter_count}")
    log(f"   - Unique QIDs with at least one document: {generated_documents_df['qid'].nunique()}")

    return generated_documents_df





