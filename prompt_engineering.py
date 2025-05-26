from langchain_core.prompts import ChatPromptTemplate

def create_qa_system_prompt() -> str:
    """
    Generates the system-level prompt for the QA assistant.

    Returns:
        str: The system prompt for guiding the assistant's behavior.
    """
    return """You are an assistant for question-answering tasks named CCChat.
Guidelines:
- Only answer questions based on the provided context. If the answer is in the context, provide a brief summary directly from the context.
- Do not attempt to generate an answer or explanation from external knowledge if the context is unrelated or missing.
- If the answer cannot be found in the context or the question is irrelevant, reply: "The question is not relevant as this content is not covered in the class" and provide no further details.
<context>
{context}
<context/>
"""

def create_qa_prompt_template(system_prompt: str) -> ChatPromptTemplate:
    """
    Creates a ChatPromptTemplate using the system-level prompt and human input.

    Args:
        system_prompt (str): The system prompt for guiding the assistant's behavior.

    Returns:
        ChatPromptTemplate: The combined chat prompt template.
    """
    return ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{input}")]
    )

def create_intent_detection_prompt(prompt: str) -> str:
    """
    Generates the prompt for intent detection.

    Args:
        prompt (str): The user query for which the intent needs to be detected.

    Returns:
        str: The intent detection prompt.
    """
    return f"""
    Classify the user's intent for the given query into one of the following categories:
    - Greeting (e.g., "hello", "hi there", "hey")
    - Introduction (e.g., "hello, who are you", "who are you", "what is your purpose")
    
    Query: "{prompt}"
    Intent:
    """

def create_query_categorization_prompt(prompt: str) -> str:
    """
    Generates the prompt for query categorization.

    Args:
        prompt (str): The user query to be categorized.

    Returns:
        str: The query categorization prompt.
    """
    return f"""
    Classify the user's query into one of these categories:
    - Unit Query: Asking about a specific unit (e.g., "Summarize Unit 2.1").
    - Topic Query: Asking about a specific topic or topic title (e.g., "What is Unit 2.1 Topic 1 about?").
    - Content Query: Asking about specific content (e.g., "What is the difference between ITIL and ITSM?").
    
    Query: "{prompt}"
    Category (only return one of the following: 'unit query', 'topic query', 'content query'):
    """
def create_relevance_check_prompt(request_prompt: str, response_answer: str) -> str:
    """
    Creates a prompt to evaluate the relevance of a response using an LLM.

    Args:
        request_prompt (str): The original user query.
        response_answer (str): The response answer to evaluate.

    Returns:
        str: The relevance evaluation prompt.
    """
    return f"""
    Determine if the following response indicates that the query is relevant to the provided context. 
    If the response mentions that the information is not mentioned in context or doesn't exist in the context, then it is irrelevant.
    Respond with "no" if the response indicates irrelevance, otherwise respond with "yes".

    Query: "{request_prompt}"
    Response: "{response_answer}"
    """