import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter, TextSplitter, TokenTextSplitter


load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "romeo_and_juliet.txt")
db_dir = os.path.join(current_dir, "db")


if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"The file {file_path} does not exist."
    )


loader = TextLoader(file_path, encoding="utf-8")
documents = loader.load()


embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def create_vector_store(docs, store_name):
    persistent_directory = os.path.join(db_dir, store_name)
    if not os.path.exists(persistent_directory):
        print("Initalizing Vector Store")
        db = Chroma.from_documents(
            docs, embeddings, persist_directory=persistent_directory
        )
    else:
        print("Vector already exists")


print("\n--- Using Character-based Splitting ---")
char_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
char_docs = char_splitter.split_documents(documents)
create_vector_store(char_docs, "chroma_db_char")

print("\n--- Using Sentence-based Splitting ---")
sent_splitter = SentenceTransformersTokenTextSplitter(chunk_size = 1000)
sent_docs = sent_splitter.split_documents(documents)
create_vector_store(sent_docs, "chroma_db_sent")

print("\n--- Using Token-based Splitting ---")
token_splitter = TokenTextSplitter(chunk_overlap=0, chunk_size=512)
token_docs = token_splitter.split_documents(documents)
create_vector_store(token_docs, "chroma_db_token")

print("\n--- Using Recursive Character-based Splitting ---")
rec_char_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
rec_char_docs = rec_char_splitter.split_documents(documents)
create_vector_store(rec_char_docs, "chroma_db_rec_char")

# Function to query a vector store

def query_vector_store(store_name, query):
    persistent_directory = os.path.join(db_dir, store_name)

    if os.path.exists(persistent_directory):
        print("Querying the vector store.")

        db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 1, "score_threshold": 0.1}
        )

        relevant_docs = retriever.invoke(query)

        for i, doc in enumerate(relevant_docs, 1):
            print(f"Document {i}:\n {doc.page_content}")
            if doc.metadata:
                print(f"Source: {doc.metadata.get('source', 'Unknown')}")
    
    else:
        print(f"Vector store do not exists.")


query="How did juliet die?"

query_vector_store("chroma_db_char", query)
query_vector_store("chroma_db_sent", query)
query_vector_store("chroma_db_token", query)
query_vector_store("chroma_db_rec_char", query)
