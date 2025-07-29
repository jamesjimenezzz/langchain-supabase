import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db_with_metadata")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

query = "How did juliet die?"

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.1},
)

relevant_docs = retriever.invoke(query)

print("Relevant Documents: ")

for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n {doc.page_content}\n")
    print(f"Source: {doc.metadata['source']}\n")


