from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch
from langchain_openai import ChatOpenAI
 

load_dotenv()

model = ChatOpenAI(model="gpt-4o")

prompt_template  = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful asssistant"),
        ("human", "Classify this sentiment of this feedback as positive negative, neutral, or escalate: {feedback}.")
    ]
)

positive_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "Generate a thank you note for this positive feedback: {feedback}")
    ]
)

negative_feedback_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ( "Generate a response addressing this negative feedback: {feedback}.")
])

neutral_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        (
            "human",
            "Generate a request for more details for this neutral feedback: {feedback}.",
        ),
    ]
)

escalate_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        (
            "human",
            "Generate a message to escalate this feedback to a human agent: {feedback}.",
        ),
    ]
)

branches = RunnableBranch(
    (
        lambda x: "positive" in x,
        positive_feedback_template | model  |StrOutputParser()
    ),
    (
        lambda x: "negative" in x,
        negative_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x: "neutral" in x,
        neutral_feedback_template | model | StrOutputParser()
    ),
    (
        escalate_feedback_template | model | StrOutputParser()
    )



)

classification_chain = prompt_template | model | StrOutputParser()

chain = classification_chain | branches


review = "The product is terrible. It broke after just one use and the quality is very poor."
result = chain.invoke({"feedback": review})

# Output the result
print(result)
