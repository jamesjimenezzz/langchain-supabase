from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os


load_dotenv()

model = ChatOpenAI(model="gpt-4o")

#PART1 Chat prompt template using a template string.

#template = "Tell me a joke about {topic}"

#prompt_template = ChatPromptTemplate.from_template(template)
#prompt =  prompt_template.invoke({"topic": "cats"})
#result  = model.invoke(prompt)
#print(result.content)

#PART2 Prompt with multiple placeholders.

#template_multiple = """You are a helpful assistant
#Human: Tell me a {adjective} short story about a {animal}
#Assistant:""
#"""

#prompt_template = ChatPromptTemplate.from_template(template_multiple)
#prompt = prompt_template.invoke({"adjective": "funny", "animal": "panda"})
#result = model.invoke(prompt)
#print(result.content)

#PART3 Prompt with system and human message.

topic_data = input("What topic do you want? ")
joke_count_data = input("How many jokes do you want? ")

messages =[
    ("system", "You are a comedian who tells jokes about {topic}"),
    ("human", "Tell me a {joke_count} jokes ")
]

prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": topic_data, "joke_count": joke_count_data})
result = model.invoke(prompt)
print(result.content)