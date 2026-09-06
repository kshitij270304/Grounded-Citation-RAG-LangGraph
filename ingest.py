import os
import requests
import time
from bs4 import BeautifulSoup
from langchain_text_splitters import HTMLHeaderTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

FAISS_INDEX_PATH = "faiss_index"

def scrape_html(url, save_path):
    if not os.path.exists("data"):
        os.makedirs("data")
        
    if not os.path.exists(save_path):
        print(f"Scraping HTML from {url}...")
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'lxml')
        
        # Filtering out junk (nav, footer, script, styles)
        for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
            element.extract()
            
        # Get the cleaned HTML string
        clean_html = str(soup)
        
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(clean_html)
        print("Scraping and cleaning complete.")
        return clean_html
    else:
        print("HTML already scraped.")
        with open(save_path, "r", encoding="utf-8") as f:
            return f.read()

def main():
    if not os.environ.get("GOOGLE_API_KEY"):
        print("WARNING: GOOGLE_API_KEY environment variable not set.")
        return

    # 1. Scrape Live HTML (Task 1)
    EU_URL = "https://artificialintelligenceact.eu/high-level-summary/" # Using summary page so scraping doesn't take 5 hours for free-tier embedding
    EU_PATH = "data/EU_AI_Act.html"
    eu_html = scrape_html(EU_URL, EU_PATH)

    # 2. Smart, Section-Aware Chunking (Task 2)
    print("Performing Section-Aware Chunking...")
    # Define the headers we want to split on
    headers_to_split_on = [
        ("h1", "Title"),
        ("h2", "Chapter"),
        ("h3", "Section"),
        ("h4", "Article")
    ]
    
    html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    # The splitter directly parses the raw HTML and maintains header hierarchies
    chunks = html_splitter.split_text(eu_html)
    
    # We will slice to first 60 chunks to keep free tier embeddings reasonable
    chunks = chunks[:60]
    
    print(f"Created {len(chunks)} intelligently grouped chunks.")
    
    for i, c in enumerate(chunks[:2]):
        print(f"Sample Chunk {i} Metadata: {c.metadata}")

    # 3. Embed and store in batches to avoid API limits
    print("Embedding and storing in FAISS (with rate limit handling)...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
    
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

    # 4. Save the FAISS index to disk
    if vectorstore:
        vectorstore.save_local(FAISS_INDEX_PATH)
        print(f"FAISS index saved to {FAISS_INDEX_PATH}")

if __name__ == "__main__":
    main()
