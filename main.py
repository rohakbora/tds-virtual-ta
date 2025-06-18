"""
Clean TDS Virtual Teaching Assistant API
Built for IIT Madras Tools in Data Science course
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import requests
import base64
import json
import re
from VectorDB import SemanticSearchDB
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="TDS Virtual TA",
    description="AI Teaching Assistant for Tools in Data Science course",
    version="3.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize knowledge base
print("🔧 Initializing knowledge base...")
try:
    db = SemanticSearchDB()
    print("✅ Knowledge base ready")
except Exception as e:
    print(f"❌ Knowledge base error: {e}")
    db = None

# API Configuration
API_TOKEN = os.getenv("AIPIPE_TOKEN") or os.getenv("OPENAI_API_KEY")
API_URL = "https://aipipe.org/openai/v1/responses"  # Change to OpenAI URL if using OpenAI directly
MODEL_NAME = "gpt-4o-mini"
MODEL_NAME_VISION = "gpt-4o"

if not API_TOKEN:
    print("⚠️ Warning: No API token found. Set AIPIPE_TOKEN or OPENAI_API_KEY")

# Request/Response Models
class QuestionRequest(BaseModel):
    question: str
    image: Optional[str] = None

class Link(BaseModel):
    url: str
    text: str

class AnswerResponse(BaseModel):
    answer: str
    links: List[Link]

def is_valid_base64_image(image_data: str) -> bool:
    """Validate base64 image data"""
    if not image_data:
        return False
    try:
        # Remove data URL prefix if present
        if image_data.startswith('data:image/'):
            image_data = image_data.split(',')[1]
        
        # Decode and check if it's substantial
        decoded = base64.b64decode(image_data)
        return len(decoded) > 500  # Basic size check
    except Exception:
        return False

def search_knowledge_base(query: str) -> tuple:
    """Search the knowledge base for relevant information"""
    if not db:
        return [], [], []

    try:
        # Use hybrid search for best results
        results = db.hybrid_search(query, n_results=5)
        scores = results['scores']
        documents = results['documents']
        metadatas = results['metadatas']

        # Filter out very short or irrelevant documents
        filtered_docs = []
        filtered_metas = []
        filtered_scores = []

        for doc, meta, score in zip(documents, metadatas, scores):
            if len(doc.strip()) > 100:  # Only substantial content
                filtered_docs.append(doc)
                filtered_metas.append(meta)
                filtered_scores.append(score)

        return filtered_docs[:5], filtered_metas[:5], filtered_scores[:5]  # Top 3 results

    except Exception as e:
        print(f"Knowledge base search error: {e}")
        return [], [], []


def generate_answer(question: str, context_docs: List[str], context_metas: List[dict],
                     image_data: str = "") -> str:
    """Generate answer using AI API with context and optional image"""

    model_to_use = MODEL_NAME_VISION if image_data else MODEL_NAME

    # Determine which endpoint to use
    endpoint = API_URL
    if model_to_use == "gpt-4o":
        endpoint = "https://aipipe.org/openai/v1/chat/completions"

    # Prepare context from knowledge base
    context_text = ""
    if context_docs:
        context_parts = []
        for i, (doc, meta) in enumerate(zip(context_docs, context_metas)):
            author = meta.get('username', 'Course Material')
            category = meta.get('category', 'general')
            preview = doc[:600] + ("…" if len(doc) > 600 else "")
            context_parts.append(f"Source {i+1} [{category}] (by {author}): {preview}")

        context_text = "\n\n".join(context_parts)
    else:
        context_text = "No specific course context found."

    # Build the messages for the chat
    messages = [
        {
            "role": "user",
            "content": []
        }
    ]

    # First add text
    messages[0]["content"].append(
        {"type": "text" if model_to_use == "gpt-4o" else "input_text",
         "text": f"""You are a helpful Teaching Assistant for the "Tools in Data Scientist" course at IIT Madras.

Your role is to answer student questions based on course materials and forum discussions.

Guidelines:
- Provide clear, accurate answers based on the context provided
- If you don't have specific information, say so and provide general guidance
- Be encouraging and supportive
- Give practical, step-by-step advice when appropriate
- Reference specific course concepts when relevant
- Always end with "Feel free to ask if you need clarification!"

{f"- If an image is provided, analyze it carefully and incorporate relevant visual information into your response"}

Available Context:
{context_text}

Student Question: {question}

Please provide a helpful answer:"""}
    )

    # If there is an image, add it to messages
    if image_data:
        if not image_data.startswith("data:image/"):
            image_data = f"data:image/jpeg;base64,{image_data}"

        messages[0]["content"].append(
            {
                "type": "image_url",
                "image_url": {"url": image_data}
            }
        )

    try:
        # Prepare request payload
        request_payload = {
            "model": model_to_use,
            "temperature": 0.3,
        }

        if model_to_use == "gpt-4o":
            request_payload["messages"] = messages
        else:
            request_payload["input"] = messages

        response = requests.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {API_TOKEN}",
                "Content-Type": "application/json"
            },
            json=request_payload,
            timeout=25
        )

        if response.status_code == 200:
            result = response.json()
            if "output" in result and isinstance(result["output"], list):
                content = result["output"][0].get("content", [{}])[0].get("text", "")
            elif "choices" in result:
                content = result["choices"][0]["message"]["content"]
            else:
                content = result.get("output", "")

            return content.strip() if content else "I'm having trouble generating a response. Please try rephrasing your question."
        else:
            print(f"API error: {response.status_code} - {response.text}")
            return "I'm experiencing technical difficulties. Please try again in a moment."

    except Exception as e:
        print(f"Error during API call: {e}")
        return "An error occurred while generating a response."



def extract_links(metadatas: List[dict], scores: List[float], documents: List[str]) -> List[Link]:
    """Extract the most relevant links from metadata, scoring by semantic score and document length."""
    links = []
    seen_urls = set()

    # Combine metadata, scores, and documents
    combined = []
    for meta, score, doc in zip(metadatas, scores, documents):
        url = meta.get('url', '') or '' 
        title = meta.get('title', '') or '' 
        username = meta.get('username', 'User')
        category = meta.get('category', 'Discussion')
        if not title:
            title = f"{category.title()} by {username}"

        combined.append((score, len(doc), url, title, username, category, doc))

    # Sort by semantic score first, then by document length
    combined.sort(key=lambda x: (x[0], x[1]), reverse=True)

    # Deduplicate while preserving order
    for item in combined:
        score, doc_length, url, title, username, category, doc = item
        if url and url not in seen_urls:
            links.append(Link(url=url, text=title))
            seen_urls.add(url)
    
    return links


@app.post("/api/", response_model=AnswerResponse)
async def answer_question(request: QuestionRequest):
    """
    Main API endpoint to answer student questions
    
    Accepts:
    - question: The student's question (required)
    - image: Optional base64-encoded image
    
    Returns:
    - answer: AI-generated answer
    - links: Relevant course discussion links
    """

    print(request)  #redundant
    
    # Validate input
    if not request.question or len(request.question.strip()) < 1:
        raise HTTPException(status_code=400, detail="Question must be at least 3 characters long")
    
    if not db:
        raise HTTPException(status_code=503, detail="Knowledge base unavailable")
    
    if not API_TOKEN:
        raise HTTPException(status_code=503, detail="AI service unavailable")
    
    try:
        print(f"📝 Processing question: {request.question[:100]}...")
        
        # Validate image if provided
        valid_image = ""
        if request.image:
            if is_valid_base64_image(request.image):
                valid_image = request.image
                print("🖼️ Valid image detected - using GPT-4o")
            else:
                print("⚠️ Invalid image format - ignoring image")
        
        # Search knowledge base
        print("🔍 Searching knowledge base...")
        context_docs, context_metas, context_scores = search_knowledge_base(request.question)
        print(f"📚 Found {len(context_docs)} relevant documents")
        
        # Generate answer
        model_used = MODEL_NAME_VISION if valid_image else MODEL_NAME
        print(f"🤖 Generating answer using {model_used}...")
        answer = generate_answer(request.question, context_docs, context_metas, valid_image)
        
        # Extract links
        links = extract_links(context_metas, context_scores, context_docs)
        
        print(f"✅ Response ready: {len(answer)} chars, {len(links)} links")
        
        return AnswerResponse(
            answer=answer,
            links=links
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if db and API_TOKEN else "degraded",
        "knowledge_base": "loaded" if db else "unavailable", 
        "ai_service": "configured" if API_TOKEN else "missing_token",
        "documents": db.collection.count() if db else 0
    }

@app.get("/stats")
async def get_stats():
    """Get knowledge base statistics"""
    if not db:
        return {"error": "Knowledge base unavailable"}
    
    try:
        return db.get_stats()
    except Exception as e:
        return {"error": f"Unable to get stats: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting TDS Virtual TA API...")
    print(f"Knowledge Base: {'✅ Ready' if db else '❌ Failed'}")
    print(f"AI Service: {'✅ Configured' if API_TOKEN else '❌ Missing Token'}")
    
    if db:
        try:
            count = db.collection.count()
            print(f"📚 Documents loaded: {count}")
        except:
            print("📚 Documents: Unknown")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)