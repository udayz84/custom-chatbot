import cohere
from dotenv import load_dotenv
import os

load_dotenv()


COHERE_API_KEY  = os.getenv("COHERE_API_KEY")



def getEmbeddings(text_chunk):
     co = cohere.Client(COHERE_API_KEY)
     response = co.embed(texts=text_chunk, model="embed-english-v3.0", input_type="search_document")
     return response.embeddings