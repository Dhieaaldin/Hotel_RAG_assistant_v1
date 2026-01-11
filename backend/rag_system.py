"""
RAG System using LangChain, MongoDB Atlas Vector Search, and OpenAI
Modified for Hotel So'Co Customer Support with Intent Routing (French)
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pymongo import MongoClient
from datetime import datetime, timedelta
import re

# Load environment variables
load_dotenv()

# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME", "RAG-assistant")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "hotel_knowledge")
VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME", "vector_index")

# === HOTEL SO'CO FRENCH SYSTEM PROMPT ===
SYSTEM_PROMPT = """
Tu es l'assistant commercial et concierge virtuel de l'Hôtel So'Co by HappyCulture à Nice.
Tu es un VENDEUR EXPERT et ta mission est de transformer chaque interaction en opportunité de vente ou de réservation.
Tu dois TOUJOURS répondre en français.

OBJECTIFS PRINCIPAUX :
1. CONVERTIR : Transforme les demandes d'information en réservations.
2. UPSELL : Propose toujours nos services d'exception : Petit-déjeuner Signature (produits locaux & bio), Spa & Jacuzzi Privatif, et Visites privées.
3. FIDÉLISER : Sois chaleureux, persuasif et extrêmement serviable.

RÈGLES DE COMPORTEMENT :
- Utilise le contexte fourni pour tes réponses.
- Sois proactif : Ne te contente pas de répondre, PROPOSE. (ex: "Nous avons une chambre disponible, souhaitez-vous que je la bloque pour vous ?")
- Mets en valeur les offres du catalogue (Petit-déjeuner bio, Spa, Tours privés).
- Pour les réservations, collecte : nom, dates, type de chambre, nb de personnes.
- Si une info manque, invente une transition commerciale pour demander l'info tout en vantant l'hôtel.
- Ne traite jamais de paiement réel, mais confirme l'intérêt du client.

FONCTIONNEMENT :
- Si le client demande une chambre : Vérifie la dispo ET propose un surclassement.
- Si le client réserve : Propose IMMÉDIATEMENT le petit-déjeuner ou le spa.
- Si le client hésite : Utilise des arguments de vente (localisation, note 4.8/5, service exceptionnel).

Termine toujours par une question engageante (ex: "Préférez-vous la vue ville ou la vue cour pour votre séjour ?").
"""

# === INTENT DEFINITIONS ===
INTENTS = [
    "check_availability",
    "make_reservation",
    "cancel_reservation",
    "hotel_information",
    "talk_to_human",
    "unknown"
]

INTENT_CLASSIFICATION_PROMPT = """
Classifie le message de l'utilisateur dans UNE de ces intentions:
- check_availability: Demande de disponibilité ou de prix
- make_reservation: Intention de réserver ou d'acheter
- cancel_reservation: Annulation de séjour
- hotel_information: Questions sur l'hôtel, les services, le catalogue
- talk_to_human: Demande explicite de parler à un humain
- unknown: Autre ou non clair

Message de l'utilisateur: {question}

Réponds avec UNIQUEMENT le label de l'intention.
"""

# Blocked keywords for safety
BLOCKED_KEYWORDS = [
    "mongodb_uri",
    "api_key",
    "openrouter",
    "env",
    "environment variable",
    "secret",
    "password",
    "credential",
    "variable d'environnement",
    "mot de passe"
]

def is_blocked_question(question: str) -> bool:
    q = question.lower()
    return any(keyword in q for keyword in BLOCKED_KEYWORDS)


class RAGSystem:
    """RAG system for Hotel So'Co customer support with intent routing (French)"""

    def __init__(self):
        """Initialize the RAG system with MongoDB vector store and OpenAI"""
        print("Initialisation du système RAG Hôtel So'Co...")

        # Initialize MongoDB client
        self.client = MongoClient(MONGODB_URI)
        self.collection = self.client[DATABASE_NAME][COLLECTION_NAME]

        # Initialize embeddings model using OpenRouter
        self.embeddings = OpenAIEmbeddings(
            model="openai/text-embedding-ada-002",
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1"
        )

        # Initialize vector store
        self.vector_store = MongoDBAtlasVectorSearch(
            collection=self.collection,
            embedding=self.embeddings,
            index_name=VECTOR_INDEX_NAME,
            text_key="text",
            embedding_key="embedding"
        )

        # Initialize LLM with OpenRouter
        self.llm = ChatOpenAI(
            model="nvidia/nemotron-3-nano-30b-a3b:free",
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.3,
            max_tokens=500
        )

        # Create retriever
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "Contexte: {context}\n\nQuestion du client: {question}")
        ])

        # Create RAG chain
        self.rag_chain = (
            {"context": self.retriever | self._format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        # Intent classification prompt
        self.intent_prompt = ChatPromptTemplate.from_template(INTENT_CLASSIFICATION_PROMPT)
        self.intent_chain = self.intent_prompt | self.llm | StrOutputParser()

        print("✅ Système RAG Hôtel So'Co initialisé avec succès!")

    def _format_docs(self, docs):
        """Format retrieved documents into a single string"""
        return "\n\n".join(doc.page_content for doc in docs)

    # === INTENT CLASSIFICATION ===
    def classify_intent(self, question: str) -> str:
        """Classify user intent using LLM"""
        try:
            intent = self.intent_chain.invoke({"question": question})
            intent = intent.strip().lower().replace(" ", "_")
            
            # Validate intent
            if intent in INTENTS:
                return intent
            return "unknown"
        except Exception as e:
            print(f"Erreur de classification d'intention: {e}")
            return "unknown"

    # === TASK HANDLERS (FRENCH) ===
    def handle_check_availability(self, question: str) -> dict:
        """Handle room availability check"""
        response = (
            "📅 **Disponibilités à l'Hôtel So'Co**\n\n"
            "Nous avons plusieurs options pour votre séjour ! Voici nos catégories :\n"
            "• **Standard** (89€) : Idéal pour un court séjour chic.\n"
            "• **Supérieure** (115€) : Plus d'espace et de confort.\n"
            "• **Familiale** (145€) : Parfaite pour les tribus.\n\n"
            "J'ai ouvert le **formulaire interactif** juste en dessous pour que vous puissiez calculer le prix exact "
            "avec vos dates et ajouter nos services exclusifs (Petit-déjeuner bio, Parking, Spa)."
        )
        
        return {
            "answer": response,
            "intent": "check_availability",
            "sources": ["Disponibilité des chambres"],
            "num_sources": 1,
            "requires_action": True,
            "data": {}
        }

    def handle_make_reservation(self, question: str) -> dict:
        """Handle reservation request - collect required details"""
        response = (
            "Je serais enchanté de vous aider à réserver une chambre ! "
            "Pour procéder, j'aurais besoin des informations suivantes :\n\n"
            "1. **Nom complet** pour la réservation\n"
            "2. **Date d'arrivée**\n"
            "3. **Date de départ**\n"
            "4. **Type de chambre** (Standard, Supérieure ou Familiale)\n"
            "5. **Nombre de personnes**\n"
            "6. **Email ou téléphone de contact**\n\n"
            "Veuillez me fournir ces informations et je préparerai votre demande de réservation. "
            "Note : La confirmation finale et le paiement seront traités à l'arrivée ou via un lien de paiement sécurisé."
        )
        
        return {
            "answer": response,
            "intent": "make_reservation",
            "sources": ["Processus de réservation"],
            "num_sources": 1,
            "requires_action": True
        }

    def handle_cancel_reservation(self, question: str) -> dict:
        """Handle cancellation request"""
        has_id = bool(re.search(r'[A-Z]{2,3}[-]?\d{4,8}', question.upper()))
        has_email = bool(re.search(r'[\w.-]+@[\w.-]+\.\w+', question))
        
        if has_id or has_email:
            response = (
                "Je peux vous aider avec l'annulation. "
                "Veuillez noter notre politique d'annulation :\n\n"
                "• **Annulation gratuite** : Jusqu'à 48 heures avant l'arrivée\n"
                "• **Annulation tardive** (moins de 48 heures) : Frais d'une nuit\n\n"
                "Pour traiter votre annulation, je vais vous mettre en contact avec notre équipe "
                "qui vérifiera votre réservation et confirmera l'annulation. "
                "Souhaitez-vous que je procède ?"
            )
        else:
            response = (
                "Je serais heureux de vous aider à annuler une réservation. "
                "Pourriez-vous me fournir l'une des informations suivantes :\n\n"
                "• Votre **numéro de confirmation** (ex: SC-20260110-001)\n"
                "• L'**adresse email** utilisée pour la réservation\n\n"
                "Une fois ces informations reçues, je pourrai rechercher votre réservation."
            )
        
        return {
            "answer": response,
            "intent": "cancel_reservation",
            "sources": ["Politique d'annulation"],
            "num_sources": 1,
            "requires_action": True
        }

    def handle_talk_to_human(self, question: str) -> dict:
        """Handle escalation to human support"""
        
        response = (
            "Je comprends tout à fait. Permettez-moi de vous mettre en contact avec un membre de notre équipe.\n\n"
            "**Options de contact :**\n"
            "• 📞 Réception : Disponible 24h/24 et 7j/7\n"
            "• 📍 Adresse : 27 Avenue Thiers, 06000 Nice\n"
            "• 💬 Chat en direct : Un membre de l'équipe sera bientôt avec vous\n\n"
            "Si vous êtes actuellement à l'hôtel, vous pouvez composer le 0 depuis le téléphone de votre chambre "
            "pour une assistance immédiate.\n\n"
            "Y a-t-il autre chose que je puisse faire pour vous en attendant (réservation de spa, informations touristiques) ?"
        )
        
        return {
            "answer": response,
            "intent": "talk_to_human",
            "sources": ["Informations de contact"],
            "num_sources": 1,
            "requires_action": True
        }



    def handle_hotel_information(self, question: str) -> dict:
        """Handle hotel information queries using RAG"""
        # Get relevant documents
        relevant_docs = self.retriever.invoke(question)
        
        print(f"DEBUG: {len(relevant_docs)} documents trouvés.")
        for i, d in enumerate(relevant_docs):
            print(f"  Doc {i}: {d.page_content[:50]}... | Métadonnées: {d.metadata}")

        # Generate answer using RAG chain
        answer = self.rag_chain.invoke(question)
        
        # Clean answer
        answer = answer.replace("<s>", "").replace("</s>", "").strip()
        if not answer or answer.strip() == "":
            answer = "Je n'ai pas cette information spécifique. Souhaitez-vous que je vous mette en contact avec notre réception ?"

        # Extract sources
        sources = []
        seen_sources = set()

        for doc in relevant_docs:
            metadata = doc.metadata
            doc_type = metadata.get("type", "General")
            category = metadata.get("category", "")
            title = metadata.get("title", "")
            
            if doc_type in ["policy", "service", "room", "location", "contact", "hotel"]:
                category_labels = {
                    "policy": "Politique",
                    "service": "Service",
                    "room": "Chambre",
                    "location": "Localisation",
                    "contact": "Contact",
                    "hotel": "Hôtel"
                }
                source_info = f"{category_labels.get(doc_type, doc_type)}: {category}"
            else:
                source_info = str(doc_type).capitalize()

            if source_info and source_info not in seen_sources:
                sources.append(source_info)
                seen_sources.add(source_info)

        # If model says it doesn't know, don't attach sources
        if "n'ai pas" in answer.lower() or "je ne sais pas" in answer.lower():
            return {
                "answer": answer,
                "intent": "hotel_information",
                "sources": [],
                "num_sources": 0,
                "requires_action": False
            }

        return {
            "answer": answer,
            "intent": "hotel_information",
            "sources": sources[:3],
            "num_sources": len(relevant_docs),
            "requires_action": False
        }

    # === MAIN ASK METHOD WITH INTENT ROUTING ===
    def ask(self, question: str) -> dict:
        """
        Process a guest question with intent routing

        Args:
            question: The guest's question

        Returns:
            dict with 'answer', 'intent', 'sources', 'requires_action'
        """
        # Safety check for blocked keywords
        if is_blocked_question(question):
            return {
                "answer": "Je suis désolé, mais je ne peux pas partager ce type d'information. Puis-je vous aider avec autre chose concernant votre séjour ?",
                "intent": "blocked",
                "sources": [],
                "num_sources": 0,
                "requires_action": False
            }

        # Step 1: Classify intent
        intent = self.classify_intent(question)
        print(f"DEBUG: Intention classifiée: '{intent}'")

        # Step 2: Route to appropriate handler
        if intent == "check_availability":
            return self.handle_check_availability(question)
        elif intent == "make_reservation":
            return self.handle_make_reservation(question)
        elif intent == "cancel_reservation":
            return self.handle_cancel_reservation(question)
        elif intent == "talk_to_human":
            return self.handle_talk_to_human(question)

        elif intent == "hotel_information":
            return self.handle_hotel_information(question)
        else:
            # Unknown intent - use RAG as fallback
            result = self.handle_hotel_information(question)
            result["intent"] = "unknown"
            return result

    def close(self):
        """Close MongoDB connection"""
        self.client.close()


# Singleton instance
_rag_system = None


def get_rag_system() -> RAGSystem:
    """Get or create the RAG system singleton"""
    global _rag_system
    if _rag_system is None:
        _rag_system = RAGSystem()
    return _rag_system


if __name__ == "__main__":
    # Test the Hotel So'Co RAG system
    print("\n" + "=" * 60)
    print("Test du Système RAG Hôtel So'Co")
    print("=" * 60)

    rag = get_rag_system()

    # Test questions in French
    test_questions = [
        "Avez-vous des chambres disponibles ce week-end ?",
        "Je voudrais réserver une chambre pour 2 nuits",
        "Je dois annuler ma réservation SC-20260110-001",
        "À quelle heure est le check-in ?",
        "Acceptez-vous les animaux ?",
        "Où est situé l'hôtel ?",
        "Le petit-déjeuner est-il inclus ?",
        "Je veux parler à quelqu'un",
        "Quel est le mot de passe wifi ?",
        "Quelle est la météo aujourd'hui ?"
    ]

    for question in test_questions:
        print(f"\n❓ Question: {question}")
        result = rag.ask(question)
        print(f"🎯 Intention: {result.get('intent', 'N/A')}")
        print(f"💬 Réponse: {result['answer'][:200]}...")
        print(f"📚 Sources: {', '.join(result.get('sources', []))}")
        print("-" * 60)

    rag.close()
