"""
FastAPI Server for Hotel Customer Support RAG Chatbot
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_system import get_rag_system
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Hotel Customer Support Chatbot API",
    description="RAG-powered hotel support chatbot with intent routing using MongoDB Atlas Vector Search",
    version="2.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "*"  # For development - restrict in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === REQUEST/RESPONSE MODELS ===
class ChatRequest(BaseModel):
    question: str

    class Config:
        json_schema_extra = {
            "example": {
                "question": "Do you have rooms available for this weekend?"
            }
        }


class ChatResponse(BaseModel):
    answer: str
    intent: Optional[str] = None
    sources: Optional[list[str]] = None
    requires_action: bool = False

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "I'd be happy to help you check availability! We have Standard, Deluxe, and Suite rooms available.",
                "intent": "check_availability",
                "sources": ["Room Availability"],
                "requires_action": True
            }
        }


# === STARTUP/SHUTDOWN EVENTS ===
@app.on_event("startup")
async def startup_event():
    """Initialize RAG system when server starts"""
    logger.info("Starting up Hotel Support API server...")
    try:
        get_rag_system()
        logger.info("✅ Hotel RAG system initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG system: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    logger.info("Shutting down Hotel Support API server...")
    rag = get_rag_system()
    rag.close()
    logger.info("✅ Resources cleaned up")


# === ENDPOINTS ===
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "message": "Hotel Customer Support Chatbot API is running",
        "version": "2.0.0",
        "features": [
            "Intent-aware routing",
            "Room availability checks",
            "Reservation requests",
            "Cancellation handling",
            "Hotel information (RAG)",
            "Human escalation"
        ]
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        rag = get_rag_system()
        return {
            "status": "healthy",
            "rag_system": "connected",
            "intents_supported": [
                "check_availability",
                "make_reservation",
                "cancel_reservation",
                "hotel_information",
                "talk_to_human",
                "unknown"
            ]
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process a guest message and return a response with intent classification.
    
    The system will:
    1. Classify the intent of the message
    2. Route to the appropriate handler
    3. Return a structured response with the answer and metadata
    """
    try:
        logger.info(f"Received question: {request.question}")

        # Validate question
        if not request.question or len(request.question.strip()) == 0:
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        if len(request.question) > 500:
            raise HTTPException(status_code=400, detail="Question too long (max 500 characters)")

        # Process through RAG system with intent routing
        rag = get_rag_system()
        result = rag.ask(request.question)

        logger.info(f"Intent: {result.get('intent', 'N/A')} | Sources: {len(result.get('sources', []))}")

        return ChatResponse(
            answer=result["answer"],
            intent=result.get("intent"),
            sources=result.get("sources") if result.get("sources") else None,
            requires_action=result.get("requires_action", False)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your question. Please try again."
        )


# === DASHBOARD API ENDPOINTS ===
@app.get("/api/catalog")
async def get_catalog():
    """Get all catalog items from MongoDB"""
    try:
        rag = get_rag_system()
        db = rag.client[rag.client.list_database_names()[0] if "RAG-assistant" not in rag.client.list_database_names() else "RAG-assistant"]
        
        if "catalog" in db.list_collection_names():
            items = list(db["catalog"].find({}, {"_id": 0}))
            return items
        else:
            return []
    except Exception as e:
        logger.error(f"Error fetching catalog: {e}")
        return []


@app.get("/api/reservations")
async def get_reservations():
    """Get all reservations from MongoDB"""
    try:
        rag = get_rag_system()
        db = rag.client[rag.client.list_database_names()[0] if "RAG-assistant" not in rag.client.list_database_names() else "RAG-assistant"]
        
        # Try to get reservations collection
        if "reservations" in db.list_collection_names():
            reservations = list(db["reservations"].find({}, {"_id": 0}))
            return reservations
        else:
            return []
    except Exception as e:
        logger.error(f"Error fetching reservations: {e}")
        return []


@app.get("/api/rooms")
async def get_rooms():
    """Get all rooms from MongoDB"""
    try:
        rag = get_rag_system()
        db = rag.client[rag.client.list_database_names()[0] if "RAG-assistant" not in rag.client.list_database_names() else "RAG-assistant"]
        
        if "rooms" in db.list_collection_names():
            rooms = list(db["rooms"].find({}, {"_id": 0}))
            return rooms
        else:
            return []
    except Exception as e:
        logger.error(f"Error fetching rooms: {e}")
        return []


# === RESERVATION CREATION ===
class ReservationRequest(BaseModel):
    guest_name: str
    email: str
    phone: str
    check_in: str
    check_out: str
    room_type: str
    guests: int
    special_requests: Optional[str] = ""


@app.post("/api/reservations")
async def create_reservation(request: ReservationRequest):
    """Create a new reservation request"""
    try:
        from datetime import datetime
        
        rag = get_rag_system()
        db = rag.client["RAG-assistant"]
        
        # Generate reservation ID
        date_part = datetime.now().strftime("%Y%m%d")
        
        # Get count of today's reservations
        existing = db["reservations"].count_documents({"reservation_id": {"$regex": f"SC-{date_part}"}})
        reservation_id = f"SC-{date_part}-{str(existing + 1).zfill(3)}"
        
        # Calculate nights and total
        from datetime import datetime as dt
        check_in_date = dt.strptime(request.check_in, "%Y-%m-%d")
        check_out_date = dt.strptime(request.check_out, "%Y-%m-%d")
        nights = (check_out_date - check_in_date).days
        
        # Room rates
        rates = {"standard": 89, "superior": 115, "family": 145}
        base_rate = rates.get(request.room_type, 100)
        total_amount = nights * base_rate
        
        # Create reservation document
        reservation = {
            "reservation_id": reservation_id,
            "guest_name": request.guest_name,
            "email": request.email,
            "phone": request.phone,
            "room_id": None,  # Will be assigned by staff
            "room_type": request.room_type,
            "check_in": request.check_in,
            "check_out": request.check_out,
            "guests": request.guests,
            "status": "pending",
            "total_amount": total_amount,
            "payment_status": "pending",
            "special_requests": request.special_requests,
            "add_ons": [],
            "created_at": datetime.now().isoformat(),
            "source": "chatbot"
        }
        
        # Insert into MongoDB
        db["reservations"].insert_one(reservation)
        
        logger.info(f"Created reservation: {reservation_id}")
        
        return {
            "success": True,
            "reservation_id": reservation_id,
            "total_amount": total_amount,
            "nights": nights,
            "message": "Demande de réservation créée avec succès"
        }
        
    except Exception as e:
        logger.error(f"Error creating reservation: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la création de la réservation")





if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Hotel Support server...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
