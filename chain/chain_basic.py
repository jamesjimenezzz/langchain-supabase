from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser


load_dotenv()

model = ChatOpenAI(model="gpt-4o")

joke_topic_data = input("What topic of joke do you want? ")
joke_count_data = input("How many jokes do you want? ")

messages = [
    (
        "system", "You are a comedian who tells jokes about {joke_topic}"
    ), (
        "human", "Give me {joke_count} jokes"
    )
]

uppercase_output = RunnableLambda(lambda x: x.upper())
count_words = RunnableLambda(lambda x: f"Word count: {len(x.split())}")

prompt_template =  ChatPromptTemplate.from_messages(messages)
chain = prompt_template | model | StrOutputParser() | uppercase_output | count_words



result = chain.invoke({"joke_topic": joke_topic_data, "joke_count": joke_count_data})

print(result)