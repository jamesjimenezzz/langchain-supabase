import os
import json
from dotenv import load_dotenv
from langchain_community.document_loaders import PlaywrightURLLoader
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

# Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_rappler_real")

json_path="rappler_fax_checks.json"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

documents = []

for entry in data:
    content = entry.get("content", "").strip()
    if content:
        doc = Document(page_content=content, metadata={"source": entry.get("url"), "title": entry.get("title")})
        documents.append(doc)



# 2. Split documents
text_splitter = CharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

# 3. Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 4. Store in Chroma vector DB
if not os.path.exists(persistent_directory):
    print("Creating vector store...")
    db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
    print(f"Vector store created at {persistent_directory}")
else:
    print("Vector store already exists.")
    db = Chroma(embedding_function=embeddings, persist_directory=persistent_directory)

# 5. Create retriever
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# 6. Query
query = "Photo of bedridden Duterte"
relevant_docs = retriever.invoke(query)

if not relevant_docs:
    print("No relevant documents found.")
    exit()

# 7. Display matched chunks
print("===== RELEVANT DOCUMENTS =====")
for i, doc in enumerate(relevant_docs, 1):
    print(f"\nDocument {i}:\n{doc.page_content[:500]}...")
    print(f"Source: {doc.metadata.get('source', 'Unknown')}")

# 8. Prepare context
context = "\n\n".join([doc.page_content for doc in relevant_docs])
sources = [doc.metadata.get("source", "Unknown") for doc in relevant_docs]

# 9. Ask GPT
llm = ChatOpenAI(model="gpt-4o")

messages = [
    SystemMessage(content="You are a fact-checking assistant. Only answer based on the provided context. Identify if the claim is true, false, or unknown."),
    HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}")
]

response = llm.invoke(messages)

# 10. Output
print("\n===== VERDICT =====")
print(response.content)

print("\n===== SOURCES =====")
for i, src in enumerate(sources, 1):
    print(f"{i}. {src}")
