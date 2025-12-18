import os
import logging
from typing import Dict, Any, Optional, List
from catalog_query_reformulator import reformulate_catalog_query, get_catalog_reformulator
import warnings
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import re
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv

logging.getLogger("watchdog").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
load_dotenv()


class HybridSearchSystem:
    """
    A hybrid search system combining semantic search (embeddings) 
    with metadata filtering and query intent classification.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the search system with a SentenceTransformer model."""
        print(f"📚 Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        self.chunks = []
        self.embeddings = None
        self.index = None
        
        print(f"✅ Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def load_chunks(self, chunks_file: str) -> None:
        """Load processed chunks from JSON file"""
        print(f"📂 Loading chunks from {chunks_file}...")
        
        with open(chunks_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.chunks = data.get('chunks', [])
        print(f"✅ Loaded {len(self.chunks)} chunks")
        
        if 'statistics' in data:
            print("\n📊 Dataset Statistics:")
            stats = data['statistics']
            for key, value in stats.items():
                if key != 'pages_processed':
                    print(f"  - {key}: {value}")
    
    def create_embeddings(self, batch_size: int = 32, show_progress: bool = True) -> None:
        """Create embeddings for all chunks using SentenceTransformer."""
        print("\n🔄 Creating embeddings...")
        
        texts = [chunk['text'] for chunk in self.chunks]
        
        self.embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        print(f"✅ Created embeddings with shape: {self.embeddings.shape}")
    
    def build_index(self, index_type: str = 'flat') -> None:
        """Build FAISS index for fast similarity search."""
        if self.embeddings is None:
            raise ValueError("Embeddings not created. Call create_embeddings() first.")
        
        print(f"\n🗂️ Building FAISS index (type: {index_type})...")
        
        if index_type == 'flat':
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.index.add(self.embeddings)
        elif index_type == 'ivf':
            nlist = min(100, len(self.chunks) // 10)
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
            self.index.train(self.embeddings)
            self.index.add(self.embeddings)
            self.index.nprobe = 10
        
        print(f"✅ Index built with {self.index.ntotal} vectors")
    
    def save_system(self, save_dir: str = 'search_system') -> None:
        """Save the search system to disk"""
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n💾 Saving search system to {save_dir}...")
        
        np.save(os.path.join(save_dir, 'embeddings.npy'), self.embeddings)
        faiss.write_index(self.index, os.path.join(save_dir, 'faiss_index.bin'))
        
        with open(os.path.join(save_dir, 'chunks.pkl'), 'wb') as f:
            pickle.dump(self.chunks, f)
        
        from datetime import datetime
        metadata = {
            'embedding_dim': self.embedding_dim,
            'num_chunks': len(self.chunks),
            'created_at': datetime.now().isoformat()
        }
        with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("✅ System saved successfully")
    
    def load_system(self, save_dir: str = 'search_system') -> None:
        """Load the search system from disk"""
        print(f"\n📂 Loading search system from {save_dir}...")
        
        self.embeddings = np.load(os.path.join(save_dir, 'embeddings.npy'))
        self.index = faiss.read_index(os.path.join(save_dir, 'faiss_index.bin'))
        
        with open(os.path.join(save_dir, 'chunks.pkl'), 'rb') as f:
            self.chunks = pickle.load(f)
        
        print(f"✅ System loaded with {len(self.chunks)} chunks")
    
    def classify_query_intent(self, query: str) -> Dict[str, Any]:
        """Classify the intent of the user's query."""
        query_lower = query.lower()
        
        intent = {
            'type': 'specific',
            'requires_all': False,
            'entities': {}
        }
        
        # Pattern 1: List/All queries (EXPANDED)
        list_patterns = [
            r'\ball\b.*\bcourse',
            r'\blist\b.*\bcourse',
            r'\blist\b.*\bmember',  # NEW: for Board of Regents
            r'\blist\b.*\bnames',   # NEW: for listing names
            r'\bshow\b.*\ball',
            r'\bgive\b.*\ball',
            r'\bname\b.*\ball',
            r'\bwhat\b.*\ball',
            r'\bhow many\b',
            r'\btotal\b.*\bcourse',
            r'\bmember\b.*\bof\b',  # NEW: "members of X"
            r'\bnames?\b.*\bof\b',  # NEW: "names of X"
        ]
        
        for pattern in list_patterns:
            if re.search(pattern, query_lower):
                intent['type'] = 'list_all'
                intent['requires_all'] = True
                break
        
        # Pattern 1b: Detect section/category filters (NEW)
        section_patterns = [
            (r'college\s+board\s+of\s+regents', 'board_of_regents'),
            (r'board\s+of\s+regents', 'board_of_regents'),
            (r'college\s+board', 'board_of_regents'),
            (r'administration', 'administration'),
            (r'faculty\s+association', 'faculty'),
            (r'staff\s+association', 'staff'),
        ]

        for pattern, section_id in section_patterns:
            if re.search(pattern, query_lower):
                intent['entities']['section'] = section_id
                print(f"   Detected section pattern: '{pattern}' -> '{section_id}'")
                break
        
        # Pattern 2: Extract department/prefix from query
        dept_patterns = [
            # Match department names in parentheses: "ENVIRONMENTAL SCIENCE (ENV)"
            (r'\(([A-Z]{2,4})\)', 'prefix_in_parens'),
            # Match standalone course prefix at start: "ENV courses"
            (r'^([A-Z]{2,4})\s+(?:course|class)', 'prefix_standalone'),
            # Match "in [DEPT NAME]" or "under [DEPT NAME]"
            (r'(?:in|under)\s+([A-Z][A-Z\s&]+?)(?:\s+category|\s+department|\s*$)', 'full_dept_name'),
        ]

        for pattern, pattern_type in dept_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches:
                extracted = matches[0].strip()
                
                if pattern_type == 'full_dept_name':
                    # Store the full department name for matching
                    intent['entities']['department_name'] = extracted.upper()
                    print(f"   Extracted department name: '{extracted}'")
                else:
                    # Store the course prefix (2-4 letter code)
                    intent['entities']['department'] = extracted.upper()
                    print(f"   Extracted course prefix: '{extracted}'")
                break
        
        # Pattern 3: Specific course code
        course_code_match = re.search(r'\b([A-Z]{2,4})\s*(\d{3})\b', query)
        if course_code_match:
            intent['type'] = 'specific_course'
            intent['entities']['course_code'] = f"{course_code_match.group(1)} {course_code_match.group(2)}"
        
        return intent
    
    def execute_list_all_query(self, intent: Dict[str, Any]) -> list:
        """Execute queries that require ALL matching results."""
        
        # Handle section-based queries (Board of Regents, Administration, etc.)
        if 'section' in intent['entities']:
            section_name = intent['entities']['section'].replace('_', ' ')
            
            print(f"🔍 Searching for section: '{section_name}'")
            
            # Extract key terms for matching
            stop_words = {'of', 'the', 'a', 'an', 'and', 'or', 'in', 'on', 'at'}
            key_terms = [word for word in section_name.split() if word not in stop_words]
            
            print(f"   Key terms: {key_terms}")
            
            # DEBUG: Show actual metadata structure
            print(f"\n🔍 DEBUG: Checking metadata structure...")
            for i in range(min(5, len(self.chunks))):
                chunk = self.chunks[i]
                metadata = chunk.get('metadata', {})
                print(f"\nChunk {i}:")
                print(f"  Available metadata keys: {list(metadata.keys())}")
                print(f"  Text preview: {chunk.get('text', '')[:100]}")
                if metadata:
                    for key, value in list(metadata.items())[:5]:
                        print(f"  - {key}: {value}")
                        
            # DEBUG: Show what sections actually exist in chunks
            print(f"\n🔍 DEBUG: Sampling chunk sections...")
            unique_sections = set()
            for i, chunk in enumerate(self.chunks[:50]):  # Check first 50 chunks
                section = chunk.get('metadata', {}).get('section', '')
                if section:
                    unique_sections.add(section)
                    if i < 10:  # Show first 10
                        chunk_type = chunk.get('metadata', {}).get('chunk_type', 'N/A')
                        title = chunk.get('metadata', {}).get('title', 'N/A')
                        print(f"   Chunk {i}: section='{section}', type={chunk_type}, title={title[:40]}")
            
            print(f"\n   Unique sections found: {sorted(unique_sections)}")
            print(f"   Total chunks to search: {len(self.chunks)}")
            
            matching_chunks = []
            for chunk in self.chunks:
                chunk_section = chunk.get('metadata', {}).get('section', '').lower()
                
                # Match if all key terms are in the section name
                if chunk_section and all(term in chunk_section for term in key_terms):
                    matching_chunks.append(chunk)
                    title = chunk.get('metadata', {}).get('title', 'N/A')
                    print(f"  ✓ Match: {title[:60]}")
            
            # If no matches found with key terms, try more flexible matching
            if not matching_chunks:
                print(f"\n⚠️ No matches with key terms. Trying flexible search...")
                
                for chunk in self.chunks:
                    chunk_section = chunk.get('metadata', {}).get('section', '').lower()
                    
                    # Try matching ANY key term
                    if chunk_section and any(term in chunk_section for term in key_terms):
                        matching_chunks.append(chunk)
                        title = chunk.get('metadata', {}).get('title', 'N/A')
                        print(f"  ✓ Partial match: section='{chunk_section}', title={title[:40]}")
            
            # Sort by title
            matching_chunks.sort(
                key=lambda x: x.get('metadata', {}).get('title', '')
            )
            
            print(f"\n✅ Total matches: {len(matching_chunks)}")
            return matching_chunks
        
        # Handle department-based course queries
        if 'department_name' in intent['entities']:
            # User specified full department name like "ENVIRONMENTAL SCIENCE AND TECHNOLOGY"
            dept_name = intent['entities']['department_name']
            print(f"🔍 Searching by department name: '{dept_name}'")
            
            matching_chunks = []
            for chunk in self.chunks:
                chunk_dept = chunk.get('metadata', {}).get('department', '').upper()
                chunk_type = chunk.get('metadata', {}).get('chunk_type', '')
                
                # Match if department name is contained in chunk's department field
                if chunk_type == 'course_description' and dept_name in chunk_dept:
                    matching_chunks.append(chunk)
                    course_code = chunk.get('metadata', {}).get('course_code', 'N/A')
                    print(f"  ✓ Matched: {course_code} (dept: {chunk_dept})")
            
        elif 'department' in intent['entities']:
            # User specified course prefix like "ENV"
            dept_prefix = intent['entities']['department']
            print(f"🔍 Searching by course prefix: '{dept_prefix}'")
            
            matching_chunks = [
                chunk for chunk in self.chunks
                if chunk.get('metadata', {}).get('chunk_type') == 'course_description' and
                chunk.get('metadata', {}).get('course_prefix', '') == dept_prefix
            ]
            
        else:
            # Default: all course descriptions
            print(f"🔍 Listing all courses (no filter)")
            matching_chunks = [
                chunk for chunk in self.chunks
                if chunk.get('metadata', {}).get('chunk_type') == 'course_description'
            ]
        
        matching_chunks.sort(
            key=lambda x: x.get('metadata', {}).get('course_code', '')
        )
        
        return matching_chunks
    
    def semantic_search(self, query: str, top_k: int = 5) -> list:
        """Perform semantic search using embeddings."""
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        scores, indices = self.index.search(query_embedding, top_k)
        results = [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0])]
        return results
    
    def keyword_filter(self, query: str, chunks: list) -> list:
        """Filter chunks based on keyword matching for hybrid search."""
        course_codes = re.findall(r'\b[A-Z]{2,4}\s*\d{3}\b', query.upper())
        
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                     'to', 'for', 'of', 'with', 'by', 'from', 'is', 'are',
                     'what', 'which', 'who', 'when', 'where', 'how', 'tell',
                     'me', 'about', 'show', 'find', 'list', 'all', 'name'}
        
        query_words = set(re.findall(r'\b\w{3,}\b', query.lower()))
        query_keywords = query_words - stop_words
        
        filtered_chunks = []
        
        for chunk in chunks:
            keyword_score = 0
            metadata = chunk.get('metadata', {})
            
            if course_codes:
                chunk_code = metadata.get('course_code', '')
                for code in course_codes:
                    if code.replace(' ', '') in chunk_code.replace(' ', ''):
                        keyword_score += 10
            
            chunk_keywords = set(metadata.get('keywords', []))
            matching_keywords = query_keywords & chunk_keywords
            keyword_score += len(matching_keywords) * 2
            
            chunk['keyword_score'] = keyword_score
            filtered_chunks.append(chunk)
        
        return filtered_chunks
    
    def hybrid_search(self, query: str, top_k: int = 10,
                     semantic_weight: float = 0.7,
                     keyword_weight: float = 0.3) -> list:
        """Perform hybrid search combining semantic and keyword search."""
        semantic_results = self.semantic_search(query, top_k * 3)
        
        candidate_chunks = []
        for idx, semantic_score in semantic_results:
            chunk = self.chunks[idx].copy()
            chunk['index'] = idx
            chunk['semantic_score'] = semantic_score
            candidate_chunks.append(chunk)
        
        candidate_chunks = self.keyword_filter(query, candidate_chunks)
        
        max_keyword_score = max([c.get('keyword_score', 0) for c in candidate_chunks]) or 1
        
        for chunk in candidate_chunks:
            norm_keyword_score = chunk.get('keyword_score', 0) / max_keyword_score
            chunk['hybrid_score'] = (
                semantic_weight * chunk['semantic_score'] + 
                keyword_weight * norm_keyword_score
            )
        
        candidate_chunks.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        return candidate_chunks[:top_k]
    
    def search(self, query: str, top_k: int = 5) -> list:
        """
        Main search interface with automatic query intent detection.
        Automatically determines whether to use 'auto' or 'hybrid' search.
        """
        print(f"\n{'='*80}")
        print(f"🔍 QUERY: {query}")
        print(f"{'='*80}")
        
        # Classify query intent
        intent = self.classify_query_intent(query)
        
        print(f"📋 Detected Intent: {intent['type']}")
        if intent['entities']:
            print(f"🏷️ Extracted Entities: {intent['entities']}")
        print(f"{'-'*80}")
        
        # Determine search type automatically
        if intent['requires_all'] or intent['type'] == 'list_all':
            # Use AUTO search for listing all courses
            print("🔄 Using AUTO search (list all matching courses)")
            results = self.execute_list_all_query(intent)
            print(f"✅ Found {len(results)} matching courses")
            
        elif intent['type'] == 'specific_course':
            # Direct lookup for specific course
            print("🎯 Using DIRECT lookup for specific course")
            course_code = intent['entities'].get('course_code', '')
            results = [
                chunk for chunk in self.chunks
                if chunk.get('metadata', {}).get('course_code', '').replace(' ', '') == 
                   course_code.replace(' ', '')
            ]
            
            if not results:
                print("⚠️ Direct lookup failed, falling back to HYBRID search")
                results = self.hybrid_search(query, top_k=3)
            else:
                print(f"✅ Found exact match for {course_code}")
        
        else:
            # Use HYBRID search for general queries
            print("🔀 Using HYBRID search (semantic + keyword)")
            results = self.hybrid_search(query, top_k=top_k)
            print(f"✅ Found {len(results)} relevant results")
        
        return results


class CatalogAnswerGenerator:
    """Generate natural language answers from retrieved catalog information using LLM"""
    
    def __init__(self, model_name: str = "llama-3.1-8b-instant"):
        """Initialize the answer generator with ChatGroq LLM"""
        self.model_name = model_name
        self.llm = None
        self._setup_llm()
    
    def _setup_llm(self):
        """Setup the ChatGroq LLM for answer generation"""
        try:
            self.llm = ChatGroq(
                model=self.model_name,
                api_key=os.getenv("GROQ_API_KEY"),
                temperature=0,
                max_tokens=4192,
                timeout=60,
                max_retries=2,
            )
            print("✅ Answer Generator LLM initialized successfully")
        except Exception as e:
            print(f"⚠️ Error initializing Answer Generator LLM: {str(e)}")
            self.llm = None
    
    def _create_system_prompt(self, query_type: str) -> str:
        """Create system prompt based on query type"""
        
        base_prompt = """You are an academic advisor assistant for Diné College, specializing in course catalog information. Your role is to provide clear, accurate, and helpful responses to student queries about courses.

**Core Guidelines:**
1. Provide accurate information based solely on the retrieved course data
2. Use a professional yet approachable tone suitable for academic advising
3. Structure your response logically with clear sections when appropriate
4. If asked about multiple courses, organize information clearly for easy comparison
5. Include relevant details like course codes, credits, prerequisites, and descriptions
6. If information is incomplete or unavailable, acknowledge this professionally
7. For list queries, present information in an organized, scannable format

**Response Quality Standards:**
- Be concise but comprehensive - avoid unnecessary verbosity
- Use bullet points or numbered lists only when it enhances clarity
- Highlight key information that directly answers the user's question
- Maintain consistency in formatting throughout the response
- Use section headers (###) only when dealing with multiple distinct topics"""

        query_specific_additions = {
            'list_all': """

**For Course Listing Queries:**
- Begin with a summary statement (e.g., "Here are the X courses in [Department]:")
- Present courses in a clean, organized format
- Include: Course Code, Title, and Credits for each course
- Use consistent formatting for easy scanning
- If there are many courses (10+), consider grouping by course number or level""",
            
            'specific_course': """

**For Specific Course Queries:**
- Start with the course code and full title
- Present key details prominently: credits, level, prerequisites
- Provide the course description in clear, readable prose
- Highlight any special requirements or important notes
- If asked about specific attributes (like prerequisites), prioritize that information""",
            
            'specific': """

**For Detailed Course Information Queries:**
- Address the specific aspect being asked about (description, prerequisites, etc.)
- Provide relevant context from the course information
- If comparing courses, use clear organizational structure
- Include all pertinent details without overwhelming the reader"""
        }
        
        return base_prompt + query_specific_additions.get(query_type, query_specific_additions['specific'])
    
    def _format_context_from_results(self, results: List[Dict], query_type: str) -> str:
        """Format retrieved results into context for the LLM"""
        
        if not results:
            return "No relevant course information found."
        
        context_parts = []
        # Check if this is Board of Regents or other informational content
        first_chunk_type = results[0].get('metadata', {}).get('chunk_type', '')
        
        if first_chunk_type == 'informational_content':
            # Handle Board of Regents and similar content
            context_parts.append(f"Total entries found: {len(results)}\n")
            
            for i, result in enumerate(results, 1):
                metadata = result.get('metadata', {})
                title = metadata.get('title', 'N/A')
                section = metadata.get('section', 'N/A')
                text = result.get('text', '')
                
                context_parts.append(f"\n{i}. {title}")
                context_parts.append(f"   Section: {section}")
                context_parts.append(f"   Details: {text}")
            
            return "\n".join(context_parts)
        
        # For list_all queries, provide structured data (threshold lowered to 5)
        if query_type == 'list_all' and len(results) >= 5:
            context_parts.append(f"Total courses found: {len(results)}\n")
            context_parts.append("Course Listing:\n")
            
            for i, result in enumerate(results, 1):
                metadata = result.get('metadata', {})
                code = metadata.get('course_code', 'N/A')
                title = metadata.get('course_title', 'N/A')
                credits = metadata.get('credits', 'N/A')
                dept = metadata.get('department', 'N/A')
                
                context_parts.append(f"{i}. {code} - {title}")
                context_parts.append(f"   Credits: {credits}")
                context_parts.append(f"   Department: {dept}")
        
        # For detailed queries, provide full course information
        else:
            for i, result in enumerate(results, 1):
                metadata = result.get('metadata', {})
                chunk_type = metadata.get('chunk_type', 'unknown')
                
                if chunk_type == 'course_description':
                    course_code = metadata.get('course_code', 'N/A')
                    course_title = metadata.get('course_title', 'N/A')
                    credits = metadata.get('credits', 'N/A')
                    level = metadata.get('course_level', 'N/A')
                    prereqs = metadata.get('prerequisites', [])
                    
                    context_parts.append(f"\n--- Course {i} ---")
                    context_parts.append(f"Code: {course_code}")
                    context_parts.append(f"Title: {course_title}")
                    context_parts.append(f"Credits: {credits}")
                    context_parts.append(f"Level: {level}")
                    context_parts.append(f"Prerequisites: {', '.join(prereqs) if prereqs else 'None'}")
                    context_parts.append(f"Description: {result.get('text', '')}")
        
        return "\n".join(context_parts)
    
    def generate_answer(self, user_query: str, retrieved_results: List[Dict], 
                       query_type: str = 'specific') -> str:
        """
        Generate a natural language answer from retrieved results
        
        Args:
            user_query: Original user query
            retrieved_results: List of retrieved course information chunks
            query_type: Type of query (list_all, specific_course, specific)
            
        Returns:
            str: Natural language answer
        """
        
        if self.llm is None:
            print("⚠️ LLM not available, returning formatted results")
            return self._fallback_formatting(retrieved_results, query_type)
        
        try:
            print(f"\n{'='*60}")
            print(f"🤖 GENERATING LLM ANSWER")
            print(f"{'='*60}")
            print(f"Query Type: {query_type}")
            print(f"Retrieved Results: {len(retrieved_results)}")
            
            # Create context from retrieved results
            context = self._format_context_from_results(retrieved_results, query_type)
            
            # Create system prompt
            system_prompt = self._create_system_prompt(query_type)
            
            # Create user prompt
            user_prompt = f"""Based on the following course catalog information, please answer the user's question.

**User Question:** {user_query}

**Retrieved Course Information:**
{context}

**Instructions:**
- Answer the question directly and comprehensively
- Use only the information provided in the retrieved course data
- If the information doesn't fully answer the question, state what is available
- Format your response professionally and clearly
- Do not add information not present in the retrieved data"""
            
            # Generate answer
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            print("📤 Sending request to LLM...")
            response = self.llm.invoke(messages)
            answer = response.content.strip()
            
            print(f"✅ LLM answer generated ({len(answer)} characters)")
            print(f"{'='*60}\n")
            
            return answer
            
        except Exception as e:
            print(f"❌ Error generating LLM answer: {str(e)}")
            import traceback
            traceback.print_exc()
            print("⚠️ Falling back to formatted results")
            return self._fallback_formatting(retrieved_results, query_type)
    
    def _fallback_formatting(self, results: List[Dict], query_type: str) -> str:
        """Fallback formatting if LLM is unavailable"""
        if not results:
            return "No matching courses found for your query."
        
        # Use simple formatting as fallback
        if query_type == 'list_all' and len(results) > 10:
            response = f"📚 **Found {len(results)} courses:**\n\n"
            
            for i, result in enumerate(results, 1):
                metadata = result.get('metadata', {})
                code = metadata.get('course_code', 'N/A')
                title = metadata.get('course_title', 'N/A')
                credits = metadata.get('credits', 'N/A')
                
                response += f"{i}. **{code}** - {title} ({credits} credits)\n"
            
            return response
        
        # Detailed fallback format
        response = ""
        
        for i, result in enumerate(results, 1):
            metadata = result.get('metadata', {})
            chunk_type = metadata.get('chunk_type', 'unknown')
            
            if chunk_type == 'course_description':
                course_code = metadata.get('course_code', 'N/A')
                course_title = metadata.get('course_title', 'N/A')
                credits = metadata.get('credits', 'N/A')
                level = metadata.get('course_level', 'N/A')
                prereqs = metadata.get('prerequisites', [])
                
                response += f"\n### {i}. {course_code} - {course_title}\n\n"
                response += f"**Credits:** {credits}\n"
                response += f"**Level:** {level}\n"
                
                if prereqs:
                    response += f"**Prerequisites:** {', '.join(prereqs)}\n"
                else:
                    response += "**Prerequisites:** None\n"
                
                response += f"\n{result.get('text', '')}\n"
                response += f"\n---\n"
        
        return response


class CatalogQueryProcessor:
    """Processes course catalog queries with RAG implementation"""
    
    def __init__(self, catalog_path: str = None, 
                 chunks_file: str = None,
                 search_system_dir: str = 'search_system'):
        """
        Initialize the catalog query processor
        
        Args:
            catalog_path: Path to catalog PDF or data source
            chunks_file: Path to embedding chunks JSON file
            search_system_dir: Directory to save/load search system
        """
        self.catalog_path = catalog_path
        self.chunks_file = chunks_file
        self.search_system_dir = search_system_dir
        self.search_system = None
        self.answer_generator = None
        self.is_initialized = False
        
    def initialize(self) -> bool:
        """Initialize the catalog retrieval and generation system"""
        try:
            print("📚 Initializing Catalog Query Processor...")
            
            # Initialize search system
            if os.path.exists(self.search_system_dir):
                print(f"📂 Loading pre-built search system from {self.search_system_dir}...")
                self.search_system = HybridSearchSystem()
                self.search_system.load_system(self.search_system_dir)
            elif self.chunks_file and os.path.exists(self.chunks_file):
                print(f"🔨 Building new search system from {self.chunks_file}...")
                self.search_system = HybridSearchSystem()
                self.search_system.load_chunks(self.chunks_file)
                self.search_system.create_embeddings(batch_size=32, show_progress=True)
                self.search_system.build_index(index_type='flat')
                self.search_system.save_system(self.search_system_dir)
            else:
                print("⚠️ No search system or chunks file found")
                return False
            
            # Initialize answer generator
            print("🤖 Initializing Answer Generator...")
            self.answer_generator = CatalogAnswerGenerator()
            
            self.is_initialized = True
            print("✅ Catalog Query Processor initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing catalog processor: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def query(self, user_query: str) -> str:
        """
        Process a catalog query and return the answer
        
        Args:
            user_query: Original user query about course catalog
            
        Returns:
            str: Natural language answer to the query
        """
        print(f"\n{'='*60}")
        print(f"📚 CATALOG QUERY PROCESSING")
        print(f"{'='*60}")
        print(f"❓ Original Query: {user_query}")
        
        try:
            if not self.is_initialized:
                return "Catalog system is not initialized. Please contact support."
            
            # Step 1: Reformulate the query
            print(f"\n🔄 Step 1: Query Reformulation")
            print(f"{'-'*60}")
            
            reformulator = get_catalog_reformulator()
            reformulation_result = reformulator.process_catalog_query(user_query)
            
            reformulated_query = reformulation_result["reformulated_query"]
            query_type = reformulation_result.get("query_type", "UNKNOWN")
            
            print(f"✅ Reformulation completed:")
            print(f"   📥 Original: {user_query}")
            print(f"   📤 Reformulated: {reformulated_query}")
            print(f"   📋 Query Type: {query_type}")
            
            # Step 2: Search with automatic type detection
            print(f"\n🔍 Step 2: Intelligent Search")
            print(f"{'-'*60}")
            
            results = self.search_system.search(
                query=reformulated_query,
                top_k=5
            )
            
            # ADD THIS DEBUG OUTPUT
            print(f"\n🔍 RETRIEVAL RESULTS:")
            print(f"{'='*60}")
            for i, result in enumerate(results[:3], 1):  # Show first 3
                metadata = result.get('metadata', {})
                print(f"\nResult {i}:")
                print(f"  Title: {metadata.get('title', 'N/A')}")
                print(f"  Section: {metadata.get('section', 'N/A')}")
                print(f"  Chunk Type: {metadata.get('chunk_type', 'N/A')}")
                print(f"  Text Preview: {result.get('text', '')[:100]}...")
            print(f"{'='*60}\n")

            # Step 3: Generate answer using LLM
            print(f"\n💬 Step 3: LLM Answer Generation")
            print(f"{'-'*60}")
            
            # Determine query intent for answer generation
            intent = self.search_system.classify_query_intent(user_query)
            
            answer = self.answer_generator.generate_answer(
                user_query=user_query,
                retrieved_results=results,
                query_type=intent['type']
            )
            
            print(f"✅ Query processing completed")
            print(f"{'='*60}\n")
            
            return answer
            
        except Exception as e:
            print(f"❌ Error processing catalog query: {str(e)}")
            import traceback
            traceback.print_exc()
            return "An error occurred while processing your catalog query. Please try again or contact support."


def process_catalog_query(user_query: str, catalog_path: str = None,
                          chunks_file: str = None,
                          search_system_dir: str = 'search_system') -> str:
    """
    Convenience function to process catalog queries
    
    Args:
        user_query: User's catalog query
        catalog_path: Path to catalog data
        chunks_file: Path to embedding chunks JSON file
        search_system_dir: Directory for search system
        
    Returns:
        str: Natural language answer to the query
    """
    processor = CatalogQueryProcessor(
        catalog_path=catalog_path,
        chunks_file=chunks_file,
        search_system_dir=search_system_dir
    )
    
    # Initialize if not already done
    if not processor.is_initialized:
        if not processor.initialize():
            return "Failed to initialize catalog system. Please ensure the search system or chunks file is available."
    
    return processor.query(user_query)