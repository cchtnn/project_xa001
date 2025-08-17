from typing import Dict, List, Any, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain.schema import BaseMessage, HumanMessage, AIMessage
import json
import re
from datetime import datetime
from query_classifier import classify_user_query, QueryType
from student_transcript_csv_handler import process_transcript_query
import logic
import logging

class ConversationState(TypedDict):
    """State schema for the conversation graph"""
    current_query: str
    chat_history: List[Dict[str, Any]]
    user_context: Dict[str, Any]
    query_type: str
    confidence_score: float
    retrieved_chunks: List[str]
    contextual_query: str
    response: str
    requires_history: bool
    entities: Dict[str, Any]

class ConversationGraph:
    """LangGraph implementation for context-aware conversations"""
    
    def __init__(self, collection, tab_data):
        self.collection = collection
        self.tab_data = tab_data
        self.graph = self._build_graph()
        
    def _build_graph(self) -> StateGraph:
        """Build the conversation graph with all nodes and edges"""
        
        # Create the graph
        workflow = StateGraph(ConversationState)
        
        # Add nodes
        workflow.add_node("classify_query", self._classify_query_node)
        workflow.add_node("analyze_history", self._analyze_history_node)
        workflow.add_node("enhance_context", self._enhance_context_node)
        workflow.add_node("retrieve_documents", self._retrieve_documents_node)
        workflow.add_node("generate_response", self._generate_response_node)
        workflow.add_node("handle_transcript", self._handle_transcript_node)
        
        # Add edges
        workflow.set_entry_point("classify_query")
        
        # Conditional routing based on query type
        workflow.add_conditional_edges(
            "classify_query",
            self._route_by_query_type,
            {
                "transcript": "handle_transcript",
                "policy": "analyze_history"
            }
        )
        
        # Policy query flow
        workflow.add_edge("analyze_history", "enhance_context")
        workflow.add_edge("enhance_context", "retrieve_documents")
        workflow.add_edge("retrieve_documents", "generate_response")
        
        # Both paths end at response generation
        workflow.add_edge("handle_transcript", END)
        workflow.add_edge("generate_response", END)
        
        return workflow.compile()
    
    def _classify_query_node(self, state: ConversationState) -> ConversationState:
        """Node to classify the incoming query"""
        try:
            query_type, confidence_score = classify_user_query(state["current_query"])
            
            state["query_type"] = query_type.value if hasattr(query_type, 'value') else str(query_type)
            state["confidence_score"] = confidence_score
            
            # Enhanced check for queries that need historical context
            follow_up_indicators = [
                "it", "that", "this", "the previous", "last time", "before", 
                "earlier", "what about", "and also", "additionally",
                "last question", "previous question", "from above", "mentioned earlier",
                "key points", "details", "explain the", "elaborate on",
                "from the last", "in the previous", "you mentioned", "you said",
                "the answer", "your response", "what you", "as stated"
            ]

            # Also check for reference patterns
            reference_patterns = [
                r"explain.*(?:key points?|details?|information)",
                r"(?:more|further).*(?:detail|information|explanation)",
                r"(?:last|previous|above).*(?:question|answer|response)",
                r"(?:expand|elaborate).*on",
                r"tell me more about"
            ]

            query_lower = state["current_query"].lower()

            # Check indicators
            has_indicators = any(indicator in query_lower for indicator in follow_up_indicators)

            # Check patterns
            has_patterns = any(re.search(pattern, query_lower) for pattern in reference_patterns)

            state["requires_history"] = has_indicators or has_patterns or len(state["chat_history"]) > 0
            
            print(f"🎯 Query classified as: {state['query_type']} (confidence: {confidence_score:.2f})")
            print(f"🔄 Requires history: {state['requires_history']}")
            
        except Exception as e:
            print(f"❌ Error in query classification: {e}")
            state["query_type"] = "POLICY"
            state["confidence_score"] = 0.5
            state["requires_history"] = False
            
        return state
    
    def _route_by_query_type(self, state: ConversationState) -> str:
        """Route to appropriate handler based on query type"""
        if state["query_type"] == "STUDENT_TRANSCRIPT":
            return "transcript"
        else:
            return "policy"
    
    def _analyze_history_node(self, state: ConversationState) -> ConversationState:
        """Node to analyze chat history and extract relevant context"""
        try:
            chat_history = state.get("chat_history", [])
            
            if not chat_history or not state["requires_history"]:
                state["entities"] = {}
                state["contextual_query"] = state["current_query"]
                return state
            
            # Get last 5 conversations for context
            recent_history = chat_history[-5:] if len(chat_history) > 5 else chat_history
            
            # Extract entities and topics from recent conversations
            entities = self._extract_entities_from_history(recent_history)
            state["entities"] = entities
            
            # Create contextual query by combining current query with relevant history
            contextual_query = self._create_contextual_query(
                state["current_query"], 
                recent_history, 
                entities
            )
            state["contextual_query"] = contextual_query
            
            print(f"🧠 Extracted entities: {entities}")
            print(f"📝 Contextual query: {contextual_query}")
            
        except Exception as e:
            print(f"❌ Error in history analysis: {e}")
            state["entities"] = {}
            state["contextual_query"] = state["current_query"]
            
        return state
    
    def _enhance_context_node(self, state: ConversationState) -> ConversationState:
        """Node to enhance context based on conversation flow"""
        try:
            # If we have entities from history, enhance the search context
            if state["entities"] and state["requires_history"]:
                enhanced_context = self._build_enhanced_context(
                    state["contextual_query"], 
                    state["entities"],
                    state["chat_history"]
                )
                state["contextual_query"] = enhanced_context
                print(f"🚀 Enhanced context: {enhanced_context}")
            
        except Exception as e:
            print(f"❌ Error in context enhancement: {e}")
            
        return state
    
    def _retrieve_documents_node(self, state: ConversationState) -> ConversationState:
        """Node to retrieve relevant documents from ChromaDB"""
        try:
            query_to_search = state["contextual_query"]
            
            # Use existing search logic
            retrieved_titles, retrieved_chunks, distances = logic.search_query(
                query_to_search, 
                self.collection,
                top_k=5  # Get more chunks for better context
            )
            
            state["retrieved_chunks"] = retrieved_chunks
            
            # Print token counts for each chunk
            print("📊 Token counts for retrieved chunks:")
            for i, chunk in enumerate(retrieved_chunks):
                print(f"  Chunk {i+1}: {logic.count_tokens(chunk)} tokens")
                
        except Exception as e:
            print(f"❌ Error in document retrieval: {e}")
            state["retrieved_chunks"] = []
            
        return state
    
    def _generate_response_node(self, state: ConversationState) -> ConversationState:
        """Node to generate the final response with context"""
        try:
            # Create enhanced prompt with conversation context
            enhanced_prompt = self._create_enhanced_prompt(
                state["current_query"],
                state["contextual_query"],
                state["retrieved_chunks"],
                state["chat_history"][-3:] if state["chat_history"] else [],  # Last 3 exchanges
                state["entities"]
            )
            
            # Use existing answer generation with enhanced context
            answer = logic.generate_answer(
                enhanced_prompt,
                state["retrieved_chunks"],
                self.tab_data,
                state["user_context"].get("language", "English")
            )
            
            state["response"] = answer.content if hasattr(answer, 'content') else str(answer)
            
            # Count response tokens
            response_tokens = logic.count_tokens(state["response"])
            print(f"📊 Response contains {response_tokens} tokens")
            
        except Exception as e:
            print(f"❌ Error in response generation: {e}")
            state["response"] = "I apologize, but I encountered an error while processing your request. Please try again."
            
        return state
    
    def _handle_transcript_node(self, state: ConversationState) -> ConversationState:
        """Node to handle student transcript queries"""
        try:
            csv_path = state["user_context"].get("active_transcript_csv_path")
            language = state["user_context"].get("language", "English")
            
            # For transcript queries, we might still want some context
            query_to_use = state["contextual_query"] if state["requires_history"] else state["current_query"]
            
            response = process_transcript_query(query_to_use, language, csv_path=csv_path)
            state["response"] = response
            
            print("📊 Processed transcript query with context")
            
        except Exception as e:
            print(f"❌ Error in transcript handling: {e}")
            error_messages = {
                "English": "I encountered an error while processing your transcript query. Please try again.",
                "Spanish": "Encontré un error al procesar tu consulta del expediente. Por favor, inténtalo de nuevo.",
                "French": "J'ai rencontré une erreur lors du traitement de votre requête de relevé. Veuillez réessayer."
            }
            language = state["user_context"].get("language", "English")
            state["response"] = error_messages.get(language, error_messages["English"])
            
        return state
    
    def _extract_entities_from_history(self, history: List[Dict]) -> Dict[str, Any]:
        """Extract entities and topics from conversation history"""
        entities = {
            "topics": set(),
            "names": set(),
            "courses": set(),
            "policies": set(),
            "references": []
        }
        
        for exchange in history:
            question = exchange.get("question", "").lower()
            answer = exchange.get("answer", "").lower()
            
            # Extract course-related terms
            course_pattern = r'\b[A-Z]{2,4}\s*\d{3,4}\b'
            entities["courses"].update(re.findall(course_pattern, question + " " + answer, re.IGNORECASE))
            
            # Extract policy-related terms
            policy_keywords = ["policy", "regulation", "requirement", "guideline", "procedure"]
            for keyword in policy_keywords:
                if keyword in question or keyword in answer:
                    entities["policies"].add(keyword)
            
            # Store recent references for pronoun resolution
            entities["references"].append({
                "question": question,
                "answer": answer,
                "timestamp": exchange.get("timestamp")
            })
        
        # Convert sets to lists for JSON serialization
        entities["topics"] = list(entities["topics"])
        entities["names"] = list(entities["names"])
        entities["courses"] = list(entities["courses"])
        entities["policies"] = list(entities["policies"])
        
        return entities
    
    def _create_contextual_query(self, current_query: str, history: List[Dict], entities: Dict) -> str:
        """Enhanced contextual query creation with better reference resolution"""
        
        # First resolve pronouns and direct references
        contextual_query = self._resolve_pronouns(current_query, history)
        
        # If the query is asking for elaboration and we have recent history
        if history and any(phrase in current_query.lower() for phrase in ["explain", "detail", "more about", "elaborate"]):
            # Get the most recent relevant context
            last_exchange = history[-1]
            
            # If asking about "key points" specifically
            if "key points" in current_query.lower():
                # Try to find and extract the key points from the last answer
                last_answer = last_exchange.get("answer", "")
                contextual_query = self._extract_key_points_context(current_query, last_answer)
        
        # Add entity context if relevant
        if entities.get("courses"):
            contextual_query += f" (Related courses: {', '.join(entities['courses'][:3])})"
        
        if entities.get("policies"):
            contextual_query += f" (Policy context: {', '.join(entities['policies'][:2])})"
        
        return contextual_query

    def _extract_key_points_context(self, query: str, last_answer: str) -> str:
        """Extract key points from previous answer for detailed explanation"""
        
        # Try to find key points section
        key_points_patterns = [
            r"Key Points[:\s]+(.*?)(?=\n\n|\n[A-Z][a-z]+:|\n\*|$)",
            r"(?:Key Points?|Main Points?)[:\s]+(.*?)(?=\n\n|\n[A-Z]|$)"
        ]
        
        key_points_content = ""
        for pattern in key_points_patterns:
            match = re.search(pattern, last_answer, re.DOTALL | re.IGNORECASE)
            if match:
                key_points_content = match.group(1).strip()
                break
        
        if key_points_content:
            # Clean up the key points text
            key_points_content = re.sub(r'\n+', ' ', key_points_content)
            key_points_content = key_points_content[:500]  # Limit length
            return f"Provide detailed explanation of these key points: {key_points_content}"
        else:
            # Fallback: use the entire previous answer as context
            return f"{query} [Context: Previous answer contained: {last_answer[:400]}...]"
    
    def _resolve_pronouns(self, query: str, history: List[Dict]) -> str:
        """Enhanced resolution of pronouns and references in the query"""
        if not history:
            return query
        
        last_exchange = history[-1]
        query_lower = query.lower()
        resolved_query = query
        
        # Handle specific reference patterns
        if "last question" in query_lower or "previous question" in query_lower:
            last_question = last_exchange.get("question", "")
            resolved_query = f"{query} [Context: Previous question was about '{last_question}']"
        
        elif "key points" in query_lower and ("last" in query_lower or "previous" in query_lower):
            last_answer = last_exchange.get("answer", "")
            # Extract key points section from last answer
            key_points_match = re.search(r"Key Points[:\s]+(.*?)(?:\n\n|\n[A-Z]|$)", last_answer, re.DOTALL | re.IGNORECASE)
            if key_points_match:
                key_points_text = key_points_match.group(1).strip()
                resolved_query = f"Explain in detail: {key_points_text}"
            else:
                resolved_query = f"{query} [Context: From previous answer: {last_answer[:200]}...]"
        
        elif any(phrase in query_lower for phrase in ["explain the", "more detail", "elaborate on"]):
            # Check if it's asking for more details about something from previous answer
            last_answer = last_exchange.get("answer", "")
            resolved_query = f"{query} [Context: Previous response included: {last_answer[:300]}...]"
        
        elif any(pronoun in query_lower for pronoun in ["it", "that", "this", "them"]):
            last_answer = last_exchange.get("answer", "")
            resolved_query = f"{query} [Referring to information from previous response: {last_answer[:200]}...]"
        
        return resolved_query
    
    def _build_enhanced_context(self, query: str, entities: Dict, history: List[Dict]) -> str:
        """Build enhanced context for better retrieval"""
        context_parts = [query]
        
        # Add conversation context
        if history:
            recent_topics = []
            for exchange in history[-2:]:  # Last 2 exchanges
                question = exchange.get("question", "")
                # Extract key topics (simplified)
                words = question.split()
                important_words = [w for w in words if len(w) > 4 and w.lower() not in ["what", "how", "when", "where", "why"]]
                recent_topics.extend(important_words[:3])
            
            if recent_topics:
                context_parts.append(f"Context: {' '.join(recent_topics[:5])}")
        
        return " ".join(context_parts)
    
    def _create_enhanced_prompt(self, original_query: str, contextual_query: str, 
                               chunks: List[str], recent_history: List[Dict], 
                               entities: Dict) -> str:
        """Create an enhanced prompt with conversation context"""
        
        prompt_parts = [original_query]
        
        # Add conversation context if available
        if recent_history and entities:
            context_info = []
            
            if entities.get("courses"):
                context_info.append(f"Previously discussed courses: {', '.join(entities['courses'][:3])}")
            
            if recent_history:
                last_topic = recent_history[-1].get("question", "")[:50]
                context_info.append(f"Previous topic: {last_topic}")
            
            if context_info:
                prompt_parts.append(f"[Conversation context: {'; '.join(context_info)}]")
        
        return " ".join(prompt_parts)
    
    def process_conversation(self, query: str, chat_history: List[Dict], user_context: Dict) -> Dict[str, Any]:
        """Main method to process a conversation with context"""
        
        # Initialize state
        initial_state = ConversationState(
            current_query=query,
            chat_history=chat_history,
            user_context=user_context,
            query_type="",
            confidence_score=0.0,
            retrieved_chunks=[],
            contextual_query="",
            response="",
            requires_history=False,
            entities={}
        )
        
        # Run the graph
        try:
            final_state = self.graph.invoke(initial_state)
            
            return {
                "response": final_state["response"],
                "query_type": final_state["query_type"],
                "confidence_score": final_state["confidence_score"],
                "contextual_query": final_state["contextual_query"],
                "entities": final_state["entities"]
            }
            
        except Exception as e:
            print(f"❌ Error in conversation processing: {e}")
            return {
                "response": "I apologize, but I encountered an error processing your request. Please try again.",
                "query_type": "UNKNOWN",
                "confidence_score": 0.0,
                "contextual_query": query,
                "entities": {}
            }

def create_conversation_graph(collection, tab_data):
    """Factory function to create a ConversationGraph instance"""
    return ConversationGraph(collection, tab_data)