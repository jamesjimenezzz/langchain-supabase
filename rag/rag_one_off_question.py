import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, 'db', 'chroma_db_with_metadata')

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

query="How did juliet die?"


retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1}
)


result = retriever.invoke(query)

print("-----Relevant Documents-----")
for i, doc in enumerate(result, 1):
    print(f"Document {i}:\n{doc.page_content}")
    if doc.metadata:
        print(f"Sources: {doc.metadata.get('source', 'Unknown')}")


combined_input = (
    f"Here are some documents that might help answer the question{query}.\nRelevant Documents:\n" + "\n\n".join([doc.page_content for doc in result]) + "Please provide an answer based only on the provided documents. If the answer is not found, respond with I'm not sure"
)

model = ChatOpenAI(model="gpt-4o")

messages = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content=combined_input)
]


data = model.invoke(messages)

print("-----Generated Response-----")
print(data.content)



