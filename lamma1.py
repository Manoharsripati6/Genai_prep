import os
from dotenv import load_dotenv
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader

# Load environment variables
load_dotenv()

# Check for API Key
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("Error: GROQ_API_KEY not found in .env file.")
    exit(1)

# Configure LLM
# Groq's OpenAILike base defaults context_window to 3900 — must override
# for llama-3.3-70b-versatile which supports 128k tokens
Settings.llm = Groq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=api_key,
    context_window=131072,
    max_tokens=4096,
)
Settings.context_window = 131072
Settings.num_output = 4096

# Configure Embeddings
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

print("Loading documents from 'data' directory...")
try:
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    query_engine = index.as_query_engine(similarity_top_k=5)
    print("Index created successfully.\n")
except Exception as e:
    print(f"Error initializing index: {e}")
    exit(1)

while True:
    try:
        text = input("Ask Question (type 'quit' to exit): ")
        if text.lower() == "quit":
            break
        
        print("Querying...")
        response = query_engine.query(text)
        print(f"Response: {response.response}")
    except EOFError:
        break
    except Exception as e:
        print(f"Error during query: {e}")
    print("\n")
