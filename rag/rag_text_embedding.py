import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
file_dir = os.path.join(current_dir, "books", "odyssey.txt")
db_dir = os.path.join(current_dir, "db")


if not os.path.exists(file_dir):
    raise FileNotFoundError(
        print(f"{file_dir} doesnt exist.")
    )

loader = TextLoader(file_dir, encoding="utf-8")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

def create_vector_store(docs, embeddings, store_name):
    persistent_directory = os.path.join(db_dir, store_name)
    if not os.path.exists(persistent_directory):
        print("Creating vector store.")
        Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory )
        print("Finished creating vector store.")
    else:
        print("Vector store already exists.")



print("Using OpenAI Embeddings")
openai_embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
create_vector_store(docs, openai_embeddings, "chroma_db_openai")

print("\n--- Using Hugging Face Transformers ---")
huggingface_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
create_vector_store(docs, huggingface_embeddings, "chroma_db_huggingface")


def query_vector_store(store_name, query, embedding_function):
    persistent_directory = os.path.join(db_dir, store_name )
    if  os.path.exists(persistent_directory):
        print("Querying the vector store")
        db = Chroma(
            persist_directory=persistent_directory,
            embedding_function = embedding_function
        )

        retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 1},
        )

        relevant_docs = retriever.invoke(query)

        print("Relevant Documents")
        for i, doc in enumerate(relevant_docs, 1):
            print(f"Document {i}\n {doc.page_content}")
            if doc.metadata:
                print(f"Source: {doc.metadata.get('source', 'Unknown')}")
    
    else:
        print(f"Vector store {store_name} already exists")
    

query = "Who is Odysseus' wife?"


query_vector_store("chroma_db_openai", query, openai_embeddings)
query_vector_store("chroma_db_huggingface", query, huggingface_embeddings)

print("Querying demonstrations completed.")

