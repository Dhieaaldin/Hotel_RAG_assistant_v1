"""
Hotel RAG Data Ingestion Script
Chunks hotel knowledge documents, generates embeddings, stores in MongoDB Atlas Vector Search
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pymongo import MongoClient

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME", "RAG-assistant")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "hotel_knowledge")
VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME", "vector_index")

DATA_DIR = Path(__file__).parent.parent / "data"


def load_hotel_knowledge() -> list[Document]:
    """Load hotel knowledge documents from JSON"""
    knowledge_file = DATA_DIR / "hotel_knowledge.json"
    
    with open(knowledge_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    documents = []
    for item in data:
        doc = Document(
            page_content=item["text"],
            metadata={
                "id": item["id"],
                "type": item["metadata"]["type"],
                "category": item["metadata"]["category"],
                "source": "hotel_knowledge"
            }
        )
        documents.append(doc)
    
    print(f"Loaded {len(documents)} hotel knowledge documents")
    return documents


def load_content_json_files() -> list[Document]:
    """Load documents from data/content/*.json files"""
    content_dir = DATA_DIR / "content"
    documents = []
    
    if not content_dir.exists():
        print("No content directory found, skipping...")
        return documents
    
    for json_file in content_dir.glob("*.json"):
        print(f"Loading: {json_file.name}")
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        source_name = json_file.stem
        
        if isinstance(data, list):
            for item in data:
                text = transform_item(item, source_name)
                metadata = {
                    "source": source_name,
                    "type": source_name
                }
                if "id" in item:
                    metadata["id"] = item["id"]
                if "title" in item:
                    metadata["title"] = item["title"]
                if "category" in item:
                    metadata["category"] = item["category"]
                
                documents.append(Document(page_content=text, metadata=metadata))
        else:
            text = transform_item(data, source_name)
            documents.append(Document(
                page_content=text,
                metadata={"source": source_name, "type": source_name}
            ))
    
    print(f"Loaded {len(documents)} documents from content directory")
    return documents


def transform_item(data: dict, doc_type: str) -> str:
    """Transform data item to narrative text"""
    if doc_type == "hotel_info":
        category = data.get("category", "general")
        title = data.get("title", "")
        description = data.get("description", "")
        return f"Hotel {category.title()} - {title}: {description}"
    
    title = data.get("title", "")
    description = data.get("description", "")
    category = data.get("category", "")
    
    if title and description:
        return f"{title}: {description}"
    elif description:
        return description
    else:
        return str(data)


def chunk_documents(documents: list[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> list[Document]:
    """Chunk documents for better retrieval"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunked_docs = []
    for doc in documents:
        if len(doc.page_content) > chunk_size:
            chunks = text_splitter.split_documents([doc])
            for i, chunk in enumerate(chunks):
                chunk.metadata["chunk_index"] = i
                chunk.metadata["total_chunks"] = len(chunks)
                chunked_docs.append(chunk)
        else:
            doc.metadata["chunk_index"] = 0
            doc.metadata["total_chunks"] = 1
            chunked_docs.append(doc)
    
    print(f"Created {len(chunked_docs)} chunks from {len(documents)} documents")
    return chunked_docs


def ingest_to_mongodb(documents: list[Document]):
    """Ingest documents with embeddings into MongoDB Atlas Vector Search"""
    print("\nConnecting to MongoDB Atlas...")
    
    client = MongoClient(MONGODB_URI)
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]
    
    print(f"Clearing existing data from {COLLECTION_NAME}...")
    collection.delete_many({})
    
    print("Initializing OpenAI embeddings via OpenRouter...")
    embeddings = OpenAIEmbeddings(
        model="openai/text-embedding-ada-002",
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1"
    )
    
    print(f"Generating embeddings and storing {len(documents)} documents...")
    
    vector_store = MongoDBAtlasVectorSearch.from_documents(
        documents=documents,
        embedding=embeddings,
        collection=collection,
        index_name=VECTOR_INDEX_NAME
    )
    
    print(f"\n✅ Successfully ingested {len(documents)} documents!")
    print(f"   Database: {DATABASE_NAME}")
    print(f"   Collection: {COLLECTION_NAME}")
    print(f"   Vector Index: {VECTOR_INDEX_NAME}")
    
    doc_count = collection.count_documents({})
    print(f"\n📊 Total documents in collection: {doc_count}")
    
    sample = collection.find_one()
    if sample:
        print(f"\n📄 Sample document:")
        print(f"   Text: {sample.get('text', 'N/A')[:80]}...")
        print(f"   Embedding dims: {len(sample.get('embedding', []))}")
        print(f"   Metadata: {sample.get('metadata', {})}")
    
    client.close()


def load_mock_operations():
    """Load mock operational data into separate collections"""
    mock_file = DATA_DIR / "mock_operations.json"
    
    if not mock_file.exists():
        print("No mock operations file found, skipping...")
        return
    
    with open(mock_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    client = MongoClient(MONGODB_URI)
    db = client[DATABASE_NAME]
    
    if "rooms" in data:
        rooms_collection = db["rooms"]
        rooms_collection.delete_many({})
        rooms_collection.insert_many(data["rooms"])
        print(f"✅ Inserted {len(data['rooms'])} rooms")
    
    if "reservations" in data:
        reservations_collection = db["reservations"]
        reservations_collection.delete_many({})
        reservations_collection.insert_many(data["reservations"])
        print(f"✅ Inserted {len(data['reservations'])} reservations")
    
    if "catalog" in data:
        catalog_collection = db["catalog"]
        catalog_collection.delete_many({})
        catalog_collection.insert_many(data["catalog"])
        print(f"✅ Inserted {len(data['catalog'])} catalog items")
    
    client.close()


def main():
    """Main ingestion pipeline"""
    print("=" * 60)
    print("Hotel RAG - Data Ingestion")
    print("=" * 60)
    
    print("\n📁 Step 1: Loading hotel knowledge...")
    hotel_docs = load_hotel_knowledge()
    
    print("\n📁 Step 2: Loading content JSON files...")
    content_docs = load_content_json_files()
    
    all_docs = hotel_docs + content_docs
    print(f"\n📊 Total documents: {len(all_docs)}")
    
    print("\n✂️ Step 3: Chunking documents...")
    chunked_docs = chunk_documents(all_docs)
    
    print("\n🚀 Step 4: Ingesting to MongoDB with embeddings...")
    ingest_to_mongodb(chunked_docs)
    
    print("\n📦 Step 5: Loading mock operational data...")
    load_mock_operations()
    
    print("\n" + "=" * 60)
    print("✅ Data ingestion complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
