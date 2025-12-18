"""
Query Classification Module for Diné College Assistant
Classifies user queries as STUDENT TRANSCRIPT, PAYROLL_CALENDAR, BOR_MEETING, CATALOG, or POLICY
"""

from sentence_transformers import SentenceTransformer, util
import streamlit as st
import logging

logging.getLogger("watchdog").setLevel(logging.ERROR)


class QueryClassifier:
    """Classifies user queries into different types"""

    def __init__(self, similarity_threshold=0.6):
        self.similarity_threshold = similarity_threshold
        self.model = None
        self.transcript_embeddings = None
        self.payroll_embeddings = None
        self.bor_embeddings = None
        self.catalog_embeddings = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the SentenceTransformer model and precompute embeddings"""
        try:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self._precompute_transcript_embeddings()
            self._precompute_payroll_embeddings()
            self._precompute_bor_embeddings()
            self._precompute_catalog_embeddings()
        except Exception as e:
            print(f"❌ Error initializing query classifier: {e}")
            self.model = None

    def _precompute_catalog_embeddings(self):
        """Precompute embeddings for catalog example queries"""
        catalog_examples = [
            # Course details queries
            "Give me details about AGR 323 Mushroom and Molds",
            "give me details about ENV 105 Climate Change for Tribal Peoples",
            "i want name of all the course code that are coming under ENVIRONMENTAL SCIENCE AND TECHNOLOGY.",
            "tell me about College Board of Regents from Academic Catalog.",
            "list all the courses under 'GEOLOGY (GLG)' in catalog document."
            
            # Department/category listing queries
            "I want name of all the course code that are coming under AGRICULTURE (AGR)",
            "List all courses in ENVIRONMENTAL SCIENCE AND TECHNOLOGY",
            
            # Course search queries
            "Find courses related to climate change",
            "Search for courses about mushrooms",

            # Course code queries
            "What is course code AGR 323?",
            "Tell me about course ENV 105",
            "What does course code AGR 323 mean?",
            "Explain course code ENV 105",
            "What is the full name of AGR 323?",
            
            # General catalog queries
            "Show me the course catalog",
            "What courses are available?",
            
            # Specific course attribute queries
            "How many credits is AGR 323?",
            "What are the prerequisites for ENV 105?",
        ]

        if self.model:
            self.catalog_embeddings = self.model.encode(
                catalog_examples,
                convert_to_tensor=True,
            )

    def _precompute_payroll_embeddings(self):
        """Precompute embeddings for payroll calendar example queries"""
        payroll_examples = [
            "Tell me the check date where optional withholdings changes is 2/27/2026?",
            "When is the check date for payroll period between 01/03/2026 to 01/16/2026?",
            "What is the payroll period where check date is 2/6/2026?",
            "What is the check date for payroll period?",
            "When will I get paid?",
            "Show me the payroll calendar",
            "What are the pay dates?",
            "When is the next pay period?",
            "What is the pay period start date?",
            "What is the pay period end date?",
            "When can I change my withholdings?",
            "What is the deadline for withholding changes?",
            "Tell me the payroll schedule",
            "What is payroll number 5?",
            "Show me check dates",
            "When does pay period start?",
            "When does pay period end?",
            "What is the optional withholdings deadline?",
            "How many payroll periods do we have in 2026?",
            "How many payroll periods in current year?",
            "Count the payroll periods",
            "Total number of pay periods this year",
        ]

        if self.model:
            self.payroll_embeddings = self.model.encode(
                payroll_examples,
                convert_to_tensor=True,
            )

    def _precompute_transcript_embeddings(self):
        """Precompute embeddings for transcript example queries"""
        transcript_examples = [
            "What is the student's GPA?",
            "Tell me the academic information of the student",
            "Which term and sub term is the course taken?",
            "Give me the types of courses the student is pursuing",
            "Show me the career total",
            "What is the student's credit hour total?",
            "How has the student performed in each semester?",
            "What courses did the student take?",
            "Show me the student's grades",
            "What is the student's academic progress?",
            "Display the student's transcript",
            "What are the student's course completions?",
            "Show me the student's enrollment history",
            "What is the student's academic standing?",
            "Tell me about the student's degree progress",
            "Sort students in descending order of GPA",
            "How many students have GPA >= 4.2",
            "How many Students have A grade in Fall 2024-2025 and their details?",
            "List of students from Murray State College",
            "Give me the GPA details of Trista Barrett.",
            "Tell me the courses which Joshua Don Gaitan has enrolled?",
            "Tell me the course name which Leslie Nichole Bright has enrolled?",
            "How many Students have A grade in 2024-2025 Fall and their details?",
            "Name of student name where organization is NEWMAN UNIVERSITY",
            "Calculate average GPA of students and sort that in descending order.",
            "Tell me the courses which Trista Denay Barrett has enrolled?",
            "Give me all student name whose advisor is Laura Lyndsey",
            "Tell me the name of advisor of student Blen Tadesse Bezuwork.",
            "Tell me the course number and term information in which student Trista Denay Barrett has got A grade?",
        ]

        if self.model:
            self.transcript_embeddings = self.model.encode(
                transcript_examples,
                convert_to_tensor=True,
            )

    def _precompute_bor_embeddings(self):
        """Precompute embeddings for Board of Regents (BOR) example queries"""
        bor_examples = [

            "When is the next BOR meeting?",

            "When is the next Board of Regents meeting?",

            "Give me the BOR meeting schedule for this year",

            "BOR meeting date in March 2026",

            "When is the BOR meeting in May?",

            "What are the BOR meeting dates?",

            "When are BOR reports due?",

            "What is the report due date before the September BOR?",

            "When do we submit bi-monthly reports to the Board of Regents?",

            "When is the Finance/Audit/Investment Committee meeting?",

            "What time does the Governance Committee meet?",

            "When are committee meetings for Academic and Student Success?",

            "List all Board of Regents committee meetings in August",

            "What are the confirmed BOR-related key events?",

            "When is the DC Winter Graduation as per BOR planner?",

            "When is DC Spring Graduation as per the Board of Regents schedule?",

            "Show me all BOR-related events in 2026",

            "What is the Board of Regents meeting planner?",

            "when is ACCT NLS ‘26 event scheduled",

            "AIHEC SPRING BOARD",

            # BOR meeting timing

            "When is the next BOR meeting?",

            "When is the next Board of Regents meeting?",

            "What are the Board of Regents meeting dates for 2025-2026?",

            "When is the BOR meeting in November 2025?",

            "When is the BOR meeting in January 2026?",

            "When is the BOR meeting in March 2026?",

            "When is the BOR meeting in May 2026?",

            "When is the BOR meeting in July 2026?",

            "When is the BOR meeting in September 2026?",

            "What is the regular BOR meeting schedule?",

            "On which day of the week are BOR meetings held?",

            "Are BOR meetings bi-monthly?",

            "Are BOR meetings generally on the 2nd Friday?",



            # BOR report due dates

            "When is the BOR report due?",

            "When are BOR reports due?",

            "What are the report due dates before each BOR meeting?",

            "When is the report due for the November 2025 BOR meeting?",

            "When is the report due for the January 2026 BOR meeting?",

            "When is the report due for the March 2026 BOR meeting?",

            "When is the report due for the May 2026 BOR meeting?",

            "When is the report due for the July 2026 BOR meeting?",

            "When is the report due for the September 2026 BOR meeting?",

            "Are BOR reports due on Wednesday prior to the meeting?",



            # Bi-monthly written reports content

            "What must be included in BOR reports?",

            "What are the components of the bi-monthly written reports?",

            "What is required in the BOR bi-monthly written report?",

            "What should the BOR dashboard of key metrics include?",

            "What are the strategic goals report requirements for BOR?",

            "What are the department goals reporting requirements for BOR?",

            "What are other activities in the BOR written report?",



            # Association reporting (Faculty & Staff)

            "What is the association reporting schedule for faculty and staff?",

            "When do the Faculty and Staff Associations report to the Board of Regents?",

            "Do Faculty and Staff Associations provide written and oral reports?",

            "In which months do faculty and staff give BOR reports?",

            "What report format must Faculty and Staff Associations use for BOR?",



            # Committee schedules and times

            "When do the committee meetings occur?",

            "What is the standing committee meeting schedule?",

            "When does the Finance/Audit/Investment Committee meet?",

            "What time is the Finance/Audit/Investment Committee meeting?",

            "When does the Governance Committee meet?",

            "What time is the Governance Committee meeting?",

            "When does the Academic & Student Success Committee meet?",

            "What time is the Academic & Student Success Committee meeting?",

            "Are committee meetings on the 2nd Friday of alternating months?",

            "In which months do committees meet (October, December, February, April, June, August)?",



            # Key events and graduations

            "When is AIHEC Fall 2025 event scheduled?",

            "When is ACCT Leadership Congress scheduled?",

            "When is ACCT GLI scheduled?",

            "When is the DC Winter Graduation?",

            "When is the DC Spring Graduation?",

            "What are the confirmed BOR-related key events?",

            "What AIHEC events are planned for 2025-2026?",

            "What ACCT events are listed in the BOR planner?",



            # ACCT NLS and TBA events

            "When is ACCT NLS 26 event scheduled?",

            "When does ACCT NLS 2026 start and end?",

            "What is the schedule for AIHEC Spring Board Meeting 2026?",

            "What is the schedule for AIHEC Student Conference 2026?",

            "What is the schedule for AIHEC Summer 2026?",

            "Which BOR-related events have dates TBA?",



            # High-level planner questions

            "What is the Board of Regents meeting planner?",

            "What does the BOR planner cover for 2025-2026?",

            "What is the resolution number and approval date for the BOR planner?",

            "What is the academic year for the current BOR planner?",

            "Give me the full BOR meeting and reporting schedule for 2025-2026.",

            ]

        if self.model:
            self.bor_embeddings = self.model.encode(
                bor_examples,
                convert_to_tensor=True,
            )

    def classify_query(self, user_query):
        """
        Classify user query as STUDENT_TRANSCRIPT, PAYROLL_CALENDAR, BOR_MEETING, CATALOG, or POLICY

        Args:
            user_query (str): The user's question

        Returns:
            tuple: (query_type, confidence_score)
                query_type: 'STUDENT_TRANSCRIPT', 'PAYROLL_CALENDAR', 'BOR_MEETING', 'CATALOG', or 'POLICY'
                confidence_score: float between 0 and 1
        """
        if (
            not self.model
            or self.transcript_embeddings is None
            or self.payroll_embeddings is None
            or self.bor_embeddings is None
            or self.catalog_embeddings is None
        ):
            # Fallback to POLICY type if model is not available
            print("⚠️ Query classifier not available, defaulting to POLICY type")
            return "POLICY", 0.0

        try:
            # Embed the user question
            user_embedding = self.model.encode(user_query, convert_to_tensor=True)

            # Compute cosine similarities with all reference types
            transcript_scores = util.cos_sim(user_embedding, self.transcript_embeddings)
            payroll_scores = util.cos_sim(user_embedding, self.payroll_embeddings)
            bor_scores = util.cos_sim(user_embedding, self.bor_embeddings)
            catalog_scores = util.cos_sim(user_embedding, self.catalog_embeddings)

            # Get the highest similarity score for each type
            max_transcript_score = transcript_scores.max().item()
            max_payroll_score = payroll_scores.max().item()
            max_bor_score = bor_scores.max().item()
            max_catalog_score = catalog_scores.max().item()

            # Classify based on highest score above threshold
            scores = {
                "STUDENT_TRANSCRIPT": max_transcript_score,
                "PAYROLL_CALENDAR": max_payroll_score,
                "BOR_MEETING": max_bor_score,
                "CATALOG": max_catalog_score,
            }

            # Find the type with highest score
            max_type = max(scores, key=scores.get)
            max_score = scores[max_type]

            # If highest score is below threshold, classify as POLICY
            if max_score >= self.similarity_threshold:
                query_type = max_type
            else:
                query_type = "POLICY"
                max_score = 0.0  # No strong match found

            print("🔍 Query Classification:")
            print(f"   Query: {user_query}")
            print(f"   Type: {query_type}")
            print(f"   Confidence: {max_score:.3f}")
            print(
                "   Scores - Transcript: "
                f"{max_transcript_score:.3f}, "
                f"Payroll: {max_payroll_score:.3f}, "
                f"BOR: {max_bor_score:.3f}, "
                f"Catalog: {max_catalog_score:.3f}"
            )
            print(f"   Threshold: {self.similarity_threshold}")

            return query_type, max_score

        except Exception as e:
            print(f"❌ Error during query classification: {e}")
            # Fallback to POLICY type on error
            return "POLICY", 0.0

    def update_threshold(self, new_threshold):
        """Update the similarity threshold"""
        self.similarity_threshold = new_threshold
        print(f"📊 Updated similarity threshold to: {new_threshold}")


# Global classifier instance with caching
@st.cache_resource(show_spinner=False)
def get_query_classifier():
    """Get cached query classifier instance"""
    return QueryClassifier()


def classify_user_query(user_query, threshold=0.6):
    """
    Convenience function to classify a user query

    Args:
        user_query (str): The user's question
        threshold (float): Similarity threshold for classification

    Returns:
        tuple: (query_type, confidence_score)
    """
    classifier = get_query_classifier()
    if classifier.similarity_threshold != threshold:
        classifier.update_threshold(threshold)

    return classifier.classify_query(user_query)


# Constants for query types
class QueryType:
    STUDENT_TRANSCRIPT = "STUDENT_TRANSCRIPT"
    POLICY = "POLICY"
    PAYROLL_CALENDAR = "PAYROLL_CALENDAR"
    BOR_MEETING = "BOR_MEETING"
    CATALOG = "CATALOG"