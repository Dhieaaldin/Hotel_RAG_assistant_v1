# HotelBot - RAG Assistant Backend v1

A sophisticated **Retrieval-Augmented Generation (RAG)** powered hotel customer support chatbot with intent routing and sales optimization. Built for **Hotels** .

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [File Descriptions](#file-descriptions)
- [Development](#development)

---

## 🎯 Project Overview

HotelBot is an AI-powered customer support system designed specifically for luxury hotel operations. It uses:

- **RAG (Retrieval-Augmented Generation)** to provide accurate, context-aware responses
- **MongoDB Atlas Vector Search** for semantic document retrieval
- **FastAPI** backend with REST API endpoints
- **Interactive HTML dashboards** for chat and analytics
- **Intent-based routing** to categorize and handle customer requests intelligently

The system is configured to operate in **French** with a sales-focused conversational strategy, emphasizing premium services like signature breakfast, spa experiences, and private tours.

---

## ✨ Features

✅ **Intent-Aware Routing** - Automatically classifies customer queries into:
- `check_availability` - Room availability and pricing inquiries
- `make_reservation` - Booking requests
- `cancel_reservation` - Cancellation handling
- `hotel_information` - General hotel inquiries
- `talk_to_human` - Escalation requests
- `unknown` - Unclassified queries

✅ **Sales-Optimized Responses** - Proactively suggests upsells (breakfast, spa, tours)

✅ **Vector Search** - Semantic similarity-based document retrieval via MongoDB Atlas

✅ **Security** - Environment-based configuration, blocked keyword filtering for sensitive data

✅ **CORS Support** - Cross-origin requests enabled for frontend integration

✅ **Dual Frontend Interfaces**:
- Chat widget (`index.html`) - Customer-facing chatbot
- Analytics dashboard (`dashboard.html`) - Admin monitoring

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (HTML)                       │
│            Chat Widget | Dashboard                       │
└─────────────────┬───────────────────────────────────────┘
                  │ HTTP/REST
┌─────────────────▼───────────────────────────────────────┐
│              FastAPI Backend (main.py)                   │
│         • Chat endpoints                                  │
│         • Health checks                                   │
│         • CORS middleware                                 │
└─────────────────┬───────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────┐
│            RAG System (rag_system.py)                    │
│    • Intent classification                               │
│    • Query processing                                    │
│    • LangChain orchestration                             │
└─────────────────┬───────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────┐
│         LangChain + MongoDB Vector Search                │
│    • Document retrieval                                  │
│    • Semantic similarity search                          │
│    • OpenAI embeddings                                   │
└─────────────────┬───────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────┐
│         MongoDB Atlas Vector Index                       │
│    • hotel_knowledge collection                          │
│    • Vector embeddings storage                           │
│    • Metadata indexing                                   │
└─────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | HTML5, CSS3, JavaScript |
| **Backend** | Python, FastAPI, Uvicorn |
| **LLM** | OpenAI (ChatOpenAI embeddings) |
| **Vector DB** | MongoDB Atlas Vector Search |
| **Database** | MongoDB |
| **RAG Framework** | LangChain, LangChain-MongoDB |
| **Server** | Gunicorn (production), Uvicorn (dev) |
| **Environment** | Python-dotenv |

---

## 📁 Project Structure

```
HotelBot/RAG_assistant_backend_v1/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── inspect_mongo.py                   # MongoDB inspection utility
│
├── backend/
│   ├── main.py                        # FastAPI application server
│   ├── rag_system.py                  # RAG engine with intent routing
│   ├── ingest_hotel_data.py           # Data ingestion pipeline
│   └── __pycache__/                   # Python cache
│
├── data/
│   ├── hotel_knowledge.json           # Hotel knowledge base
│   └── mock_operations.json           # Mock operation data
│
└── frontend/
    ├── index.html                     # Chat widget interface
    └── dashboard.html                 # Admin dashboard
```

---

## 🚀 Setup & Installation

### Prerequisites

- Python 3.8+
- MongoDB Atlas account (with Vector Search enabled)
- OpenAI API key (or OpenRouter API key)
- npm/Node.js (optional, for frontend development)

### Step 1: Clone & Install Dependencies

```bash
cd RAG_assistant_backend_v1
pip install -r requirements.txt
```

### Step 2: Environment Configuration

Create a `.env` file in the project root:

```env
# OpenAI/OpenRouter Configuration
OPENROUTER_API_KEY=your_openrouter_api_key_here

# MongoDB Configuration
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority
DATABASE_NAME=RAG-assistant
COLLECTION_NAME=hotel_knowledge
VECTOR_INDEX_NAME=vector_index
```

### Step 3: Ingest Hotel Data

Populate MongoDB with hotel knowledge:

```bash
python backend/ingest_hotel_data.py
```

### Step 4: Verify MongoDB Connection

Inspect the MongoDB collection:

```bash
python inspect_mongo.py
```

Expected output shows collection statistics and document structure.

---

## ⚙️ Configuration

### MongoDB Vector Index Setup

Create a vector search index in MongoDB Atlas:

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 1536,
      "similarity": "cosine"
    },
    {
      "type": "filter",
      "path": "metadata"
    }
  ]
}
```

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `OPENROUTER_API_KEY` | LLM API key | **Required** |
| `MONGODB_URI` | Database connection | **Required** |
| `DATABASE_NAME` | MongoDB database | `RAG-assistant` |
| `COLLECTION_NAME` | MongoDB collection | `hotel_knowledge` |
| `VECTOR_INDEX_NAME` | Vector search index | `vector_index` |

---

## 💻 Usage

### Start the Backend Server

```bash
# Development
cd backend
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app
```

Server runs on `http://localhost:8000`

### Access Frontend

- **Chat Widget**: `http://localhost:8000/` (served from static files)
- **Dashboard**: Open `frontend/dashboard.html` in browser

### Test Chat Endpoint

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "Do you have rooms available this weekend?"}'
```

---

## 🔌 API Endpoints

### `GET /`
**Health Check**
- Returns API status and feature list
- **Response**: JSON with `status`, `version`, `features`

### `POST /chat`
**Main Chat Endpoint**
- **Request Body**:
  ```json
  {
    "question": "string"
  }
  ```
- **Response**:
  ```json
  {
    "answer": "string",
    "intent": "string (optional)",
    "sources": ["string (optional)"],
    "requires_action": "boolean"
  }
  ```

### `GET /docs`
**Interactive API Documentation** (Swagger UI)

### `GET /redoc`
**Alternative API Documentation** (ReDoc)

---

## 📄 File Descriptions

### Backend Files

#### [backend/main.py](backend/main.py)
FastAPI application server
- REST API endpoint definitions
- CORS middleware configuration
- Startup/shutdown event handlers
- Request/response models (Pydantic)
- Health check and chat endpoints

#### [backend/rag_system.py](backend/rag_system.py)
RAG engine with intent routing
- `RAGSystem` class for managing RAG pipeline
- Intent classification logic
- Document retrieval from MongoDB Vector Search
- LangChain chain orchestration
- French system prompt for luxury hotel support
- Security: blocked keyword filtering
- Singleton pattern via `get_rag_system()` function

#### [backend/ingest_hotel_data.py](backend/ingest_hotel_data.py)
Data ingestion pipeline
- Loads hotel knowledge from JSON
- Generates OpenAI embeddings
- Chunks documents with RecursiveCharacterTextSplitter
- Stores in MongoDB Atlas Vector Search
- Supports multiple content sources

### Utility Files

#### [inspect_mongo.py](inspect_mongo.py)
MongoDB inspection and debugging utility
- Connects to MongoDB Atlas
- Lists all documents in collection
- Displays document structure and metadata
- Useful for verifying data ingestion

### Data Files

#### [data/hotel_knowledge.json](data/hotel_knowledge.json)
Hotel knowledge base
- Contains hotel information, amenities, policies
- Structured with id, text, and metadata fields
- Used for RAG retrieval

#### [data/mock_operations.json](data/mock_operations.json)
Mock operational data
- Sample reservation, cancellation, and inquiry data
- For testing and development

### Frontend Files

#### [frontend/index.html](frontend/index.html)
Customer-facing chat widget
- Modern glassmorphism UI design
- Real-time chat interface
- Responsive design for mobile/desktop
- Color scheme: Primary #0052a5, Secondary #d4af37
- WebSocket-ready chat messaging

#### [frontend/dashboard.html](frontend/dashboard.html)
Admin analytics dashboard
- Sidebar navigation
- Chat metrics and statistics
- Intent distribution visualization
- User engagement analytics
- Dark theme for admin interface

---

## 👨‍💻 Development

### Project Goals
✅ Transform customer inquiries into sales opportunities
✅ Provide accurate, context-aware hotel information
✅ Route complex requests to human agents intelligently
✅ Upsell premium services (breakfast, spa, tours)

### Key Design Decisions

1. **French-Only Responses**: System configured for Hôtel So'Co's French-speaking guests
2. **Sales-First Approach**: System prompt emphasizes conversion and upselling
3. **Vector Search**: Semantic similarity for better context matching vs keyword search
4. **Intent Routing**: Enables custom handling per query type
5. **MongoDB Atlas**: Cloud-native, scalable vector database

### Extending the System

**Add New Intent**:
1. Update `INTENTS` list in `rag_system.py`
2. Add intent-specific prompt template
3. Implement intent handler in RAG chain

**Add Hotel Data**:
1. Create JSON file in `data/content/`
2. Update `ingest_hotel_data.py` transformation logic
3. Run ingestion script

**Customize Behavior**:
- Edit `SYSTEM_PROMPT` in `rag_system.py` for response style
- Modify `INTENT_CLASSIFICATION_PROMPT` for intent detection
- Update `BLOCKED_KEYWORDS` for security

---

## 📦 Dependencies

See [requirements.txt](requirements.txt):

- **fastapi**: Modern web framework
- **uvicorn[standard]**: ASGI server
- **langchain**: LLM orchestration framework
- **langchain-openai**: OpenAI integration
- **langchain-mongodb**: MongoDB Vector Search integration
- **pymongo**: MongoDB driver
- **python-dotenv**: Environment configuration
- **pydantic**: Data validation
- **langchain-text-splitters**: Document chunking
- **gunicorn**: Production server

---

## 🔒 Security Notes

- **API Keys**: Stored in `.env`, never committed to git
- **Blocked Keywords**: Sensitive terms filtered to prevent information leakage
- **CORS**: Configured for development; restrict `allow_origins` in production
- **No Real Payments**: System confirms intent but never processes actual payments

---

## 🎓 Example Workflow

```
User: "Do you have a room available for 2 people this weekend?"
        ↓
    Intent Classification: check_availability
        ↓
    MongoDB Vector Search: Retrieves room availability info
        ↓
    LangChain RAG Chain: Processes with context
        ↓
    System Response (French):
    "Bien sûr! Nous avons plusieurs options disponibles. 
     Préférez-vous une Deluxe ou une Suite? 
     Et souhaitez-vous ajouter notre Petit-Déjeuner Signature?"
        ↓
    API Response:
    {
      "answer": "...",
      "intent": "check_availability",
      "sources": ["Room Availability"],
      "requires_action": true
    }
```


---

**Version**: 2.0.0  
**Last Updated**: January 2026  
**Status**: Active Development
