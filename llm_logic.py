import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
import chromadb
from chromadb.config import Settings

# Import prompt functions
from prompt_engineering import (
    create_qa_system_prompt,
    create_qa_prompt_template,
    create_intent_detection_prompt,
    create_query_categorization_prompt,
    create_relevance_check_prompt,
)

# Load environment variables from .env
load_dotenv()

# Load API keys
def get_env_var(var_name: str, required: bool = True) -> str:
    value = os.getenv(var_name)
    if required and not value:
        raise ValueError(f"{var_name} is not set in the environment.")
    return value

groq_api_key = get_env_var("GROQ_API_KEY")

# For OpenAIEmbeddings
embeddings = OpenAIEmbeddings()
openai_api_key = get_env_var("OPENAI_API_KEY")
# For ChromaDBVector Data Path
persist_directory = "./CC_Chat_db"
# Check if the directory and database exist
if not os.path.exists(persist_directory) or not os.listdir(persist_directory):
    raise FileNotFoundError(f"No embedding database found in '{persist_directory}'.")
db3 = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

# LLM setup
llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")

# Metadata configuration
metadata_field_info = [
    AttributeInfo(name="unit", description="The unit of the content", type="string"),
    AttributeInfo(name="topic", description="The topic of the content", type="string"),
    AttributeInfo(name="topic_title", description="The topic title", type="string"),
    AttributeInfo(name="source", description="The source file name", type="string"),
]

document_content_description = "contains transcription of the video topics of Information Technology Operation, MGMT8680 - Spring 2020 course."

# Initialize the SelfQueryRetriever
retriever = SelfQueryRetriever.from_llm(
    llm,
    db3,
    document_content_description,
    metadata_field_info,
    search_kwargs={"k": 5},
)

# Generate QA chain
qa_system_prompt = create_qa_system_prompt()
qa_prompt = create_qa_prompt_template(qa_system_prompt)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


# Intent detection
def detect_intent(prompt: str) -> str:
    # Detects the user's intent from the query using the LLM.
    intent_prompt = create_intent_detection_prompt(prompt)
    response = llm.invoke(intent_prompt).content.strip().lower()
    return response

def handle_intent(intent):
    if intent == "greeting":
        return {
            "title":"Greeting",
            "answer":"Hello! I'm CCchat 2025, here to assist you with questions about the Information Technology Operation course."
        }
    elif intent == "introduction":
        return {
            "title":"Introduction",
            "answer":"I am CCchat 2025, your course assistant. My role is to help you with queries related to the Information Technology Operation course (MGMT8680) taught by Prof. Sean Yo"
        }
    return None  # Return None if the intent doesn't match predefined cases

# Query categorization
def categorize_query(prompt: str) -> str:
    #  Categorizes the user's query into predefined categories using the LLM.
    """
    Categorizes the user's query into predefined categories using the LLM.

    Args:
        prompt (str): The user query to be categorized.

    Returns:
        str: The category of the query. Valid categories are "unit query", "topic query", and "content query".
            If the query does not match any of the predefined categories, returns "content query" by default.
    """
    categorization_prompt = create_query_categorization_prompt(prompt)
    response = llm.invoke(categorization_prompt).content.strip().lower()
    # Validate response against expected categories
    valid_categories = ["unit query", "topic query", "content query"]
    for category in valid_categories:
        if category in response:
            return category
    return "content query"  # Default category if none match


# Unit extraction
def get_unit(query: str) -> str:
    import re
    unit_number = re.search(r"Unit\s*\d+(\.\d+)?", query, re.IGNORECASE)
    return unit_number.group(0) if unit_number else None


# Topic extraction
def get_topic(query: str) -> str:
    import re
    topic_number = re.search(r"Topic\s*\d+(\.\d+)?", query, re.IGNORECASE)
    if topic_number:
        return topic_number.group(0)
    topic_title = re.search(r"'([^']+)'", query)
    if topic_title:
        return topic_title.group(1)
    return query.strip()


# Main processing function
def process_query_and_filter_documents(request_prompt):
    query_category = categorize_query(request_prompt)
    unit_query = get_unit(request_prompt)
    topic_query = get_topic(request_prompt)
    retrieved_docs = retriever.invoke({"query": request_prompt})
    relevant_docs = filter_retrieved_docs(
        query_category=query_category,
        retrieved_docs=retrieved_docs,
        unit_query=unit_query,
        topic_query=topic_query,
    )

    print(f"Query Category: {query_category}")
    print(f"Unit Query: {unit_query}")
    print(f"Topic Query: {topic_query}")
    print(f"Retrieved Documents: {len(retrieved_docs)}")
    print(f"Relevant Documents: {len(relevant_docs)}")

    print("### Debugging Metadata ###")
    for doc in relevant_docs:
        print(doc.metadata)

    return relevant_docs


# Filter documents
def filter_retrieved_docs(query_category, retrieved_docs, unit_query=None, topic_query=None):
    if query_category == "unit query" and unit_query:
        return [
            doc for doc in retrieved_docs
            if doc.metadata.get("unit", "").lower() == unit_query.lower()
        ]
    elif query_category == "topic query" and topic_query:
        return [
            doc for doc in retrieved_docs
            if topic_query.lower() in doc.metadata.get("topic", "").lower()
        ]
    return retrieved_docs


def query_llm_relevance(request_prompt: str, response: dict) -> str:
    print(f"Query LLM Relevance: {request_prompt}")
    # Generate the prompt using the utility function
    relevance_check_prompt = create_relevance_check_prompt(request_prompt, response['answer'])
    
    # Debugging output
    print(f"\n\n######### Relevance Evaluation Prompt############: {relevance_check_prompt}\n\n")
    # Invoke the LLM to evaluate relevance
    response_relevant = llm.invoke(relevance_check_prompt).content.strip().lower()
    
    # Debugging output
    print(f"\n\n######### Relevance Evaluation Result############: {response_relevant}\n\n")
    print(f"Response Content: {response['answer']}")
    
    return response_relevant

def generate_response_with_rag(user_input: str, context: str) -> dict:
    """
    Generates a response using the RAG chain by invoking it with the given input and context.

    Args:
        rag_chain: The RAG (Retrieval-Augmented Generation) chain object.
        user_input (str): The user's query or input.
        context (str): The context to be used for generating the response.

    Returns:
        dict: The response generated by the RAG chain.
    """
    # Prepare the input payload
    input_payload = {"input": user_input, "context": context}
    
    # Invoke the RAG chain
    response = rag_chain.invoke(input_payload)
    
    # Debugging output
    print(f"Generated Response: {response['answer']}")
    return response