import os
from pymongo import MongoClient
from dotenv import load_dotenv
load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME", "RAG-assistant")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "portfolio_content")

client = MongoClient(MONGODB_URI)
collection = client[DATABASE_NAME][COLLECTION_NAME]

print(f"Checking collection: {DATABASE_NAME}.{COLLECTION_NAME}")
docs = list(collection.find({}))
print(f"Total docs found: {len(docs)}")

for i, doc in enumerate(docs):
    print(f"\n--- Document {i} ---")
    print(f"Keys: {list(doc.keys())}")
    print(f"Text (first 50 chars): {doc.get('text', 'MISSING')[:50]}...")
    if 'metadata' in doc:
        print(f"Metadata Field: {doc['metadata']}")
    else:
        # Check for inferred metadata fields
        print(f"Source: {doc.get('source', 'N/A')}")
        print(f"Type: {doc.get('type', 'N/A')}")

client.close()
