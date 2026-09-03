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
    NIST_URL = "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf"
    NIST_PATH = "data/NIST_AI_RMF.pdf"
    download_pdf(NIST_URL, NIST_PATH)

    # 2. Load the document
    print("Loading documents...")
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()[:20] # Increased back to 20 pages
    
    nist_loader = PyPDFLoader(NIST_PATH)
    docs.extend(nist_loader.load()[:20]) # Increased back to 20 pages
    
    print(f"Loaded {len(docs)} pages total.")

    # 3. Chunk the document
    print("Chunking document...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks.")

    # 4. Embed and store in batches to avoid API limits
    print("Embedding and storing in FAISS (with rate limit handling)...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
    
    import time
    print("Waiting 65 seconds before starting to ensure Gemini Free Tier quota is fully reset...")
    time.sleep(65)
    
    vectorstore = None
    batch_size = 20  # Ultra-safe batch size
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}...")
        
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)
            
        if i + batch_size < len(chunks):
            print("Sleeping for 65 seconds to respect Gemini API Free Tier limits...")
            time.sleep(65)

    # 5. Save the FAISS index to disk
    if vectorstore:
        vectorstore.save_local(FAISS_INDEX_PATH)
        print(f"FAISS index saved to {FAISS_INDEX_PATH}")

if __name__ == "__main__":
    main()
