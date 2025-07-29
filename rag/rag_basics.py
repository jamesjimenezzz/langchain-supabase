import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()


current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "odyssey.txt")
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

if not os.path.exists(persistent_directory):
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exist. Please check the path. "
        )
    
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    
    print(f"Number of document chunks: {len(docs)}") 
    print(f"Sample chunk:\n {docs[0].page_content}")

    #SAMPLE PAG NACHUNK IS Document(page_content="1000characters"), Document(page_content="1000characters"), 


    print("Creating embeddings")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    print("Finished creating embeddings")

    print("Creating vector store")
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_directory
    )
    print("Finished creating vector store")

else:
    print("Vector store already exists. No need to initalize")


    

