import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import FireCrawlLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from firecrawl import FirecrawlApp
from langchain_core.documents import Document

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_firecrawl")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def create_vector_store():
    #FIRECRAWL

    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        raise ValueError("FIRECRAWL_API_KEY environment variable not set")
    
    app = FirecrawlApp(api_key=api_key)
    result = app.crawl_url("https://www.doh.gov.ph/")
    
    
    
    print("Begin crawling on this website.")
    docs = []

    for page in result.data:
        content = page["markdown"]  # or "html" if preferred
        metadata = page.get("metadata", {})
    
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "title": metadata.get("title", ""),
                    "source": metadata.get("sourceURL", "")
            }
        )
    )
    print("Finished crawling the website.")

   
    for doc in docs:
        for key, value in doc.metadata.items():
            if isinstance(value, list):
                doc.metadata[key] = ",".join(map(str, value))

    text_splitter = CharacterTextSplitter(chunk_size = 1200, chunk_overlap = 200)
    split_docs = text_splitter.split_documents(docs)

    print("------DOCUMENT CHUCK INFORMATION-------")
    print(f"Number of chunks: {len(split_docs)}")

   

    print("Creating vector store")
    db = Chroma.from_documents(split_docs, embeddings, persist_directory=persistent_directory)


if not os.path.exists(persistent_directory):
    create_vector_store()
else:
    print(f"Vector store already exists in {persistent_directory}. No need to initialize. ")
    

db = Chroma(embedding_function=embeddings, persist_directory=persistent_directory)

def query_vector_store(query):
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    relevant_docs = retriever.invoke(query)

    print("\n--- Relevant Documents ---")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"Document {i}:\n{doc.page_content}\n")
        if doc.metadata:
            print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")



query = "Is it true that duterte did not fight torres and flew to singapore?"


query_vector_store(query)


