from flask import Flask, render_template, request, jsonify
import requests
from typing import List, Tuple
import sys
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
sys.path.append("..")

from transformers import AutoTokenizer, AutoModel
from chunker import chunk_text, truncate_text
from qdrant import QdrantStore
from groq_client import GroqAPI
from loader import load_urls

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    # Add your chatbot logic here
    response = f"You said: {user_message}"  # Placeholder response
    return jsonify({'response': response})

# Initialize Hugging Face model
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

vectorizer = TfidfVectorizer(max_features=1024)

def get_embeddings(texts: List[str]) -> List[List[float]]:
    return vectorizer.fit_transform(texts).toarray().tolist()

def extract_content(url: str) -> str:
    """Extract text content from a URL"""
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        # Get text content and clean it up
        text = soup.get_text(separator=" ", strip=True)
        return truncate_text(text)  # Truncate to avoid token limits
    except Exception as e:
        print(f"Error fetching URL: {str(e)}")
        return ""

def process_url(url: str) -> Tuple[List[str], List[List[float]]]:
    """Process URL content into chunks and embeddings"""
    # Get content and chunk it
    content = extract_content(url)
    chunks = chunk_text(content)
    
    # Get embeddings for all chunks
    embeddings = get_embeddings(chunks)
    
    return chunks, embeddings

def store_in_qdrant(chunks: List[str], embeddings: List[List[float]], url: str):
    """Store chunks and embeddings in Qdrant"""
    qdrant = QdrantStore(collection_name="url_content", vector_size=1024)  # Hugging Face embed dimension
    metadata = {"source_url": url}
    qdrant.store_text_with_embedding(chunks, embeddings, metadata)

def query_and_respond(query: str, qdrant: QdrantStore, groq: GroqAPI, top_k: int = 3):
    """Query Qdrant for relevant chunks and get response from Groq"""
    # Get embedding for the query
    query_embedding = get_embeddings([query])[0]
    
    # Search Qdrant for similar chunks
    results = qdrant.search_similar(query_embedding, top_k=top_k)
    
    # Prepare context from relevant chunks
    context = "\n".join([result["text"] for result in results])
    
    # Get response from Groq
    response = groq.ask_question(query, context)
    return response

def main():
    # Initialize components
    qdrant = QdrantStore(collection_name="url_content", vector_size=1024)
    groq = GroqAPI()
    
    # List of URLs to extract data from
    urls = ["https://brainlox.com/courses/category/technical"]

    # Load and extract content
    docs = load_urls(urls)

    # Print extracted content
    for doc in docs:
        print(doc.page_content)

    while True:
        print("\n1. Process new URL")
        print("2. Ask a question")
        print("3. Exit")
        choice = input("Choose an option (1-3): ")
        
        if choice == "1":
            url = input("Enter URL to process: ")
            print("Processing URL...")
            chunks, embeddings = process_url(url)
            print(f"Found {len(chunks)} chunks")
            store_in_qdrant(chunks, embeddings, url)
            print("Content stored successfully!")
            
        elif choice == "2":
            query = input("Enter your question: ")
            print("Getting answer...")
            response = query_and_respond(query, qdrant, groq)
            print("\nResponse:", response)
            
        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    app.run(debug=True)
    # main()
