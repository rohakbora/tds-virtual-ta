from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import requests
from dotenv import load_dotenv
from vector_db import TDSKnowledgeBase
from typing import List, Optional
import re
import base64
import json
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI(
    title="TDS Virtual TA",
    description="Virtual Teaching Assistant for Tools in Data Science",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Initialize knowledge base
try:
    kb = TDSKnowledgeBase()
    print("✅ Knowledge base initialized successfully")
except Exception as e:
    print(f"❌ Error initializing knowledge base: {e}")
    kb = None

# AIPipe configuration
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")
AIPIPE_CHAT_URL = "https://aipipe.org/openai/v1/responses"
AIPIPE_VISION_MODEL = "gpt-4o-mini"  # Updated to a more reliable model

if not AIPIPE_TOKEN:
    print("⚠️ Warning: AIPIPE_TOKEN not found in environment variables")

HEADERS = {
    "Authorization": f"Bearer {AIPIPE_TOKEN}" if AIPIPE_TOKEN else "",
    "Content-Type": "application/json"
}

class QuestionRequest(BaseModel):
    question: str
    image: Optional[str] = None

class Link(BaseModel):
    url: str
    text: str

class AnswerResponse(BaseModel):
    answer: str
    links: List[Link]

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

def is_base64_image(image_data: str) -> bool:
    """Check if the provided string is a valid base64 encoded image"""
    try:
        if not image_data:
            return False
        # Remove data URL prefix if present
        if image_data.startswith('data:image/'):
            image_data = image_data.split(',')[1]
        
        # Try to decode base64
        decoded = base64.b64decode(image_data)
        return len(decoded) > 100  # Basic size check
    except Exception:
        return False

def extract_text_from_filename(filename: str) -> str:
    """Extract meaningful text from filename for context"""
    if not filename:
        return ""
    
    # Remove file extensions
    clean_name = re.sub(r'\.(jpg|jpeg|png|gif|webp|bmp)$', '', filename.lower())
    # Replace separators with spaces
    clean_name = re.sub(r'[_\-\.]+', ' ', clean_name)
    # Remove numbers and special characters but keep meaningful words
    clean_name = re.sub(r'\b\d+\b', '', clean_name)
    clean_name = re.sub(r'[^\w\s]', ' ', clean_name)
    # Clean up multiple spaces
    clean_name = ' '.join(clean_name.split())
    
    return clean_name.strip()

def analyze_image(image_data: str, question: str = "") -> str:
    """Analyze image using AIPipe proxy"""
    try:
        if not image_data:
            return ""
        
        # Check if it's base64 image data
        if is_base64_image(image_data):
            # Prepare image for vision model
            if image_data.startswith('data:image/'):
                image_base64 = image_data
            else:
                image_base64 = f"data:image/jpeg;base64,{image_data}"
            
            # Use vision-capable prompt
            vision_prompt = f"""
            Analyze this image in the context of a Tools in Data Science course question: "{question}"
            
            Please describe:
            1. Any code, commands, or technical content visible
            2. Error messages or output shown
            3. UI elements, screenshots, or diagrams
            4. Any text that might be relevant to the question
            
            Focus on technical details that would help answer the student's question.
            """
            
            try:
                response = requests.post(
                    AIPIPE_CHAT_URL,
                    headers=HEADERS,
                    json={
                        "model": AIPIPE_VISION_MODEL,
                        "input": vision_prompt,
                        "image": image_base64
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result.get("output", [{}])[0].get("content", [{}])[0].get("text", "")
                    return content.strip() if content else "Image analysis completed but no details extracted."
                else:
                    print(f"Vision API error: {response.status_code} - {response.text}")
                    return "Unable to analyze the image with vision model."
                    
            except requests.exceptions.Timeout:
                return "Image analysis timed out."
            except Exception as e:
                print(f"Vision analysis error: {e}")
                return "Error occurred during image analysis."
        
        else:
            # Treat as filename and extract context
            context = extract_text_from_filename(image_data)
            return f"Image filename context: {context}" if context else "Image provided without description."
            
    except Exception as e:
        print(f"Error in analyze_image: {e}")
        return "Unable to process the provided image."

def clean_and_validate_context(context_parts: List[str]) -> str:
    """Clean and validate context to ensure quality"""
    if not context_parts:
        return "No relevant context found in course materials."
    
    # Filter out very short or low-quality context
    filtered_parts = []
    for part in context_parts:
        if len(part.strip()) > 100:  # Only include substantial content
            filtered_parts.append(part)
    
    if not filtered_parts:
        return "Limited relevant context found in course materials."
    
    return "\n\n".join(filtered_parts)

def generate_answer(question: str, context: str, image_description: str = "") -> str:
    """Generate answer using AIPipe GPT proxy with improved error handling"""
    system_prompt = """You are a virtual Teaching Assistant for the "Tools in Data Science" course at IIT Madras.

Your role is to help students with their questions based on:
1. Course content and materials
2. Previous discussions from the course Discourse forum

Guidelines:
- Provide clear, helpful answers based on the provided context
- If the information is not available in the context, say "I don't have specific information about that in the course materials, but I can provide general guidance"
- Be encouraging and supportive like a good TA
- Reference specific course concepts when relevant
- For technical questions, provide practical step-by-step guidance
- When referencing forum discussions, mention the username if available
- If the question involves code or commands, provide specific examples when possible
- Always end with "Feel free to ask follow-up questions!"

Context information:
{context}

{image_context}

Question: {question}"""

    image_context = f"\nAdditional context from image: {image_description}" if image_description else ""
    
    input_text = system_prompt.format(
        context=context, 
        image_context=image_context,
        question=question
    )

    # Add after line ~175 (before requests.post):
    print(f"🔍 DEBUG: Making request to {AIPIPE_CHAT_URL}")
    print(f"🔍 DEBUG: Headers: {bool(HEADERS.get('Authorization'))}")
    print(f"🔍 DEBUG: Token present: {bool(AIPIPE_TOKEN)}")

    try:
        response = requests.post(
            AIPIPE_CHAT_URL,
            headers=HEADERS,
            json={
                "model": AIPIPE_VISION_MODEL,
                "input": input_text,
                "temperature": 0.3
            },
            timeout=30
        )
        
        print(f"🔍 DEBUG: Response status: {response.status_code}")
        print(f"🔍 DEBUG: Response body: {response.text[:200]}...")
        
        response.raise_for_status()
        result = response.json()
        # Add this line before the response parsing:
        print(f"🔍 DEBUG: text field type: {type(result.get('text'))}")
        print(f"🔍 DEBUG: text field content: {result.get('text')}")
        # Add this debug line:
        print(f"🔍 DEBUG: output field: {result.get('output')}")

        # Try parsing the output field first
        # Handle AIPipe response format
        if "output" in result and isinstance(result["output"], list):
            output_list = result["output"]
            if output_list and "content" in output_list[0]:
                content_list = output_list[0]["content"]
                if content_list and "text" in content_list[0]:
                    return content_list[0]["text"].strip()
                # NEW: Handle the 'output_text' type
                elif content_list and content_list[0].get("type") == "output_text":
                    return content_list[0]["text"].strip()

        # Fallback for other formats
        elif "output" in result and isinstance(result["output"], str):
            return result["output"].strip()
        elif "choices" in result:
            return result["choices"][0]["message"]["content"].strip()
        else:
            print(f"🔍 DEBUG: Unexpected response format: {list(result.keys())}")
            return "I received an unexpected response format. Please try again."
        
    except requests.exceptions.Timeout:
        return "Request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        print(f"❌ REQUEST ERROR: {e}")
        print(f"❌ Response status: {getattr(e.response, 'status_code', 'No response')}")
        print(f"❌ Response text: {getattr(e.response, 'text', 'No response text')}")
        return f"API request failed: {str(e)}"
    except Exception as e:
        print(f"❌ GENERAL ERROR: {e}")
        return f"Error generating answer: {str(e)}"

@app.post("/api/", response_model=AnswerResponse)
async def answer_question(request: QuestionRequest):
    """Main API endpoint to answer student questions"""
    
        # ADD THESE LINES HERE:
    print(f"🔥 RECEIVED REQUEST: {request.question[:100]}...")
    print(f"🔥 Has image: {bool(request.image)}")
    print(f"🔥 Request timestamp: {__import__('datetime').datetime.now()}")

    # Validate knowledge base
    if kb is None:
        raise HTTPException(
            status_code=503, 
            detail="Knowledge base is not available. Please contact the administrator."
        )
    
    # Validate input
    if not request.question or len(request.question.strip()) < 3:
        raise HTTPException(
            status_code=400, 
            detail="Question must be at least 3 characters long."
        )
    
    try:
        # Analyze image if provided
        image_description = ""
        if request.image:
            image_description = analyze_image(request.image, request.question)
            print(f"Image analysis result: {image_description}")

        # Form search query
        search_query = request.question
        if image_description and not image_description.startswith("Unable to"):
            search_query += f" {image_description}"

        # Search vector database
        search_results = kb.search(search_query, n_results=5)
        
        if not search_results or not search_results.get('documents'):
            # Fallback search with just the question
            search_results = kb.search(request.question, n_results=3)

        # Prepare context and links
        context_parts = []
        links = []
        seen_urls = set()

        for i, (doc, metadata) in enumerate(zip(
            search_results.get('documents', []), 
            search_results.get('metadatas', [])
        )):
            if not doc or len(doc.strip()) < 50:
                continue
                
            doc_preview = doc[:800] + ("..." if len(doc) > 800 else "")
            username = metadata.get('username', 'Course Material')
            context_parts.append(f"Context {i+1} (from {username}): {doc_preview}")

            # Add unique links
            url = metadata.get('url')
            if url and url not in seen_urls and url.startswith('http'):
                seen_urls.add(url)
                title = metadata.get('title', 'Relevant discussion')
                # Clean title
                clean_title = title[:100].strip()
                if not clean_title:
                    clean_title = 'Course Discussion'
                    
                links.append(Link(url=url, text=clean_title))

        # Clean and validate context
        context = clean_and_validate_context(context_parts)

        # Generate answer
        answer = generate_answer(request.question, context, image_description)
        
        # Ensure we have a valid answer
        if not answer or len(answer.strip()) < 10:
            answer = "I apologize, but I'm having trouble generating a comprehensive response. Could you please rephrase your question or provide more details?"

        # Limit links to top 3 most relevant
        links = links[:3]

        return AnswerResponse(answer=answer, links=links)

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing request: {e}")
        raise HTTPException(
            status_code=500, 
            detail="An unexpected error occurred while processing your question."
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    kb_status = "loaded" if kb is not None else "failed"
    api_status = "configured" if AIPIPE_TOKEN else "missing_token"
    
    return {
        "status": "healthy" if kb_status == "loaded" and api_status == "configured" else "degraded",
        "knowledge_base": kb_status,
        "aipipe_api": api_status
    }

@app.get("/stats")
async def get_stats():
    """Get knowledge base statistics"""
    if kb is None:
        return {"error": "Knowledge base not available"}
    
    try:
        count = kb.collection.count()
        return {
            "total_documents": count,
            "status": "operational"
        }
    except Exception as e:
        return {"error": f"Unable to retrieve stats: {str(e)}"}

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return {
        "error": exc.detail,
        "status_code": exc.status_code
    }

if __name__ == "__main__":
    import uvicorn
    
    # Check environment setup
    print("🔧 Starting TDS Virtual TA...")
    print(f"Knowledge Base: {'✅ Ready' if kb else '❌ Failed'}")
    print(f"AIPipe Token: {'✅ Configured' if AIPIPE_TOKEN else '❌ Missing'}")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )