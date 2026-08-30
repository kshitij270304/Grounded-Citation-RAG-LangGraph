import os
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables (like GOOGLE_API_KEY)
load_dotenv()

PDF_URL = "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32024R1689"
PDF_PATH = "data/EU_AI_Act.pdf"
FAISS_INDEX_PATH = "faiss_index"

def download_pdf(url, save_path):
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists(save_path):
        print(f"Downloading PDF from {url}...")
        response = requests.get(url)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(response.content)
        print("Download complete.")
    else:
        print("PDF already exists.")

def main():
    if not os.environ.get("GOOGLE_API_KEY"):
        print("WARNING: GOOGLE_API_KEY environment variable not set.")
        print("Please set it in a .env file or export it to run the embedding.")
        return

    # 1. Download data
    download_pdf(PDF_URL, PDF_PATH)

    # 2. Load the document
    print("Loading document...")
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
    
    # Keep it to 5 pages to avoid free-tier API rate limits
    docs = docs[:5]
    print(f"Loaded {len(docs)} pages.")

    # 3. Chunk the document
    print("Chunking document...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks.")

    # 4. Embed and store
    print("Embedding and storing in FAISS...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # 5. Save the FAISS index to disk
    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"FAISS index saved to {FAISS_INDEX_PATH}")

if __name__ == "__main__":
    main()
