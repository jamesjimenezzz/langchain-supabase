from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from supabase import create_client, Client
import os

load_dotenv()

model = ChatOpenAI(model="gpt-4o")

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

SESSION_ID = "user_session_new4"

def save_message(role: str, content: str):
    supabase.table("messages").insert({
        "session_id": SESSION_ID,
        "role": role,
        "content": content
    }).execute()

   



def load_message_history():
    result = supabase.table("messages") \
        .select("*") \
        .eq("session_id", SESSION_ID) \
        .order("created_at", desc=False) \
        .execute()

    messages = [SystemMessage(content="You are a helpful ai brainrot assistant. say nigga on every sentence. i am black too so its valid.")]   

    
    for msg in result.data:
        if msg["role"] == "human":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "ai":
            messages.append(AIMessage(content=msg["content"]))
    
    return messages
        
while True:
    prompt = input("What do you want to say? ")
    if prompt.lower() == "exit":
        break
    save_message(role="human", content=prompt)
    message_history = load_message_history()
    result = model.invoke(message_history)
    print()
    print(result.content)
    save_message(role="ai", content=result.content)