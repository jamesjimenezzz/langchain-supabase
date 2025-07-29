from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
import os

load_dotenv()

#template = "Tell me a joke about {topic}."
#prompt_template = ChatPromptTemplate.from_template(template)

#prompt = prompt_template.invoke({"topic": "cats"})
#print(prompt)


#template_multiple = """You are a helpful assistant.
#Human: Tell me a {adjective} story about {animal}
#Assistant:"""

#prompt_multiple = ChatPromptTemplate.from_template(template_multiple)
#prompt = prompt_multiple.invoke({"adjective": "funny", "animal": "panda"})
#print(prompt)

messages = [
    (
        "system", "You are a comedian who tell jokes about {topic}"
    ),
    (
        "human", "Tell me {count} jokes"
    )
]


prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": "lawyers", "count": "5"})
print(prompt)
