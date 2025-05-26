from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llm_logic import (
    detect_intent,
    handle_intent,
    process_query_and_filter_documents,
    generate_response_with_rag,
    query_llm_relevance,
)

app = FastAPI(
    title="CCchat LLM Server",
    version="1.0",
    description="API Server for CCchat",
)

origins = [
    "https://cvri-ai.site",
    "https://ai.cvri.ca",
    "http://127.0.0.1:8086",
    "http://localhost:8086",
    "http://127.0.0.1:7000",
    "http://localhost:7000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    prompt: str
    session_id: str


class QueryResponse(BaseModel):
    answer: str
    source: list
    title: str


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:

        print(f"Received query: {request.prompt}")
         # Detect the user's intent and Handle predefined intents
        intent = detect_intent(request.prompt)
        intent_result = handle_intent(intent)
        print(f"Detected intent: {intent_result}")
        if intent_result:
        # Convert the result to a QueryResponse object
            response = QueryResponse(
                title=intent_result["title"],
                answer=intent_result["answer"],
                source=[],  # Default empty list for sources
            )
            print(f"Detected response: {response}")
            return response


        # Retrieve documents before processing (can be reused across logic)
        relevant_docs = process_query_and_filter_documents(request.prompt)
        context = "\n\n".join(
            doc.page_content for doc in relevant_docs if doc.page_content.strip()
        )
        print("Context:")
        # print(len(context))

        response = generate_response_with_rag(request.prompt,  context)
        print("RAG response:")
        # print(response)
        response_relevant = query_llm_relevance(request.prompt, response)

        print(f"Relevance Evaluation Result: {response_relevant}")
        
        if response_relevant.strip().lower() == "no":
            return QueryResponse(
                answer="WARNING:\nThe question is not relevant as this content is not covered in the class.",
                source=[],  
                title="Irrelevant Question",
            )

        #collect sources from documents retrieved
        source_set = {doc.metadata.get("source") for doc in relevant_docs}
        final_sources = list(source_set)[:3]

        # Generate a title
        try:
            title_prompt = f"Generate a concise title (maximum 5 words) for the following question: '{request.prompt}'."
            title = llm.invoke(title_prompt).content.strip()
            if len(title.split()) > 5:
                title = " ".join(title.split()[:5])
        except Exception as e:
            print(f"Error generating title: {e}")
            title = "Untitled Response"

        return QueryResponse(
            answer=response["answer"],
            source=final_sources,
            title=title,
        )
        # return QueryResponse(
        #     answer=response["answer"],
        #     source=[],
        #     title="Query Response",
        # )
    except Exception as e:
        return QueryResponse(
            answer=f"Error: {str(e)}",
            source=[],
            title="Error",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7000)