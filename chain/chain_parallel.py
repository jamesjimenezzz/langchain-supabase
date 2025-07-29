from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_openai import ChatOpenAI


load_dotenv()

model = ChatOpenAI(model="gpt-4o")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert product reviewer"),
        ("human", "List the main features of the product {product_name}" )
    ]
)

def analyze_pros(features):

    pros_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert product reviewer"),
            ("human", "Given these features: {features}, list the pros of these features." )
        ]
    )

    return pros_template.format_prompt(features=features)

def analyze_cons(features):
    cons_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert product reviewer"),
            ("human", "Given these features: {features}, list the cons of these features. ")
        ]
    )

    return cons_template.format_prompt(features=features)


def combine_pros_cons(pros, cons):
    return f"Pros: {pros}\n Cons: {cons}"

pros_branch_chain = (
    RunnableLambda(lambda x: analyze_pros(x)) | model | StrOutputParser()
)

cons_branch_chain = (
    RunnableLambda(lambda x: analyze_cons(x)) | model | StrOutputParser()
)

chain = prompt_template | model | StrOutputParser() | RunnableParallel(branches={"pros": pros_branch_chain, "cons": cons_branch_chain}) | RunnableLambda(lambda x: combine_pros_cons(x["branches"]["pros"], x["branches"]["cons"]))

result = chain.invoke({"product_name": "macbook"})

print(result)