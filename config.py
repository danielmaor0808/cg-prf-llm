import os
from dotenv import load_dotenv

load_dotenv()

# 🔐 Model
API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
ROLE = os.getenv("ROLE")

# 🤖 Prompt
PROMPT_TEMPLATE = os.getenv("PROMPT_TEMPLATE")

# ⚙️ Document generation flags
GENERATE_DOCS_DL20 = os.getenv("GENERATE_DOCS_DL20", "false").lower() == "true"
GENERATE_DOCS_DL19 = os.getenv("GENERATE_DOCS_DL19", "false").lower() == "true"

# 📄 Generated document file paths
GENERATED_DOCS_FILE_DL20 = os.getenv("GENERATED_DOCS_FILE_DL20")
GENERATED_DOCS_FILE_DL19 = os.getenv("GENERATED_DOCS_FILE_DL19")

# 📂 DL2020 file paths
QUERIES_FILE_DL20 = os.getenv("QUERIES_FILE_DL20")
QRELS_FILE_DL20 = os.getenv("QRELS_FILE_DL20")
CORPUS_FILE_DL20 = os.getenv("CORPUS_FILE_DL20")
TOP100_FILE_DL20 = os.getenv("TOP100_FILE_DL20")

# 📂 DL2019 file paths
QUERIES_FILE_DL19 = os.getenv("QUERIES_FILE_DL19")
QRELS_FILE_DL19 = os.getenv("QRELS_FILE_DL19")
CORPUS_FILE_DL19 = os.getenv("CORPUS_FILE_DL19")
TOP100_FILE_DL19 = os.getenv("TOP100_FILE_DL19")

# logging
VERBOSE_LOGGING = os.getenv("VERBOSE_LOGGING", "false").lower() == "true"