import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter



load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
books_dir = os.path.join(current_dir, "books")
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

if not os.path.exists(persistent_directory):
    print("Initializing vector store.")
    
    if not os.path.exists(books_dir):
        raise FileNotFoundError(
            f"The directory {books_dir} does not exist. Please check the path."
        )
    
    documents = []
    book_files = [f for f in os.listdir(books_dir) if f.endswith(".txt")]
    
    for book_file in book_files:
        file_path = os.path.join(books_dir, book_file)
        loader = TextLoader(file_path, encoding="utf-8")
        book_docs = loader.load()
        for doc in book_docs:
            doc.metadata = {"source": book_file}
            documents.append(doc)

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap = 0 )
    docs = text_splitter.split_documents(documents)

    print("Creating embeddings")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    print("Finished creating embeddings.")

    print("Creating vector store")
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_directory
    )
    print("Finished creating vector store")
else:
    print("Vector store already exists.")


        