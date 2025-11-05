import os
from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
import json

logger = logging.getLogger(__name__)

@dataclass
class RAGResponse:
    """Response from RAG system"""
    answer: str
    relevant_jobs: List[Dict[str, Any]]
    sources: List[str]
    confidence_score: float

class JobRAGGenerator:
    """Generates responses using retrieved job information"""
    
    def __init__(self, 
                 model_name: str = "gpt-3.5-turbo",
                 temperature: float = 0.7,
                 max_tokens: int = 1000,
                 openai_api_key: Optional[str] = None):
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Set up OpenAI API key
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        elif not os.getenv("OPENAI_API_KEY"):
            logger.warning("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        
        # Initialize LLM
        try:
            self.llm = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )
            logger.info(f"Initialized LLM: {model_name}")
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            raise
    
    def create_system_prompt(self) -> str:
        """Create system prompt for job-related queries"""
        return """You are an expert government job counselor with deep knowledge of Indian government job opportunities, exams, and recruitment processes. 

Your role is to:
1. Provide accurate, helpful information about government jobs
2. Help users find suitable job opportunities based on their qualifications and interests
3. Explain job requirements, selection processes, and application procedures
4. Offer career guidance and exam preparation advice

Guidelines:
- Always base your responses on the provided job information
- Be specific about eligibility criteria, age limits, and selection processes
- Mention official websites and application cycles
- Provide practical advice for exam preparation
- If you don't have specific information, say so clearly
- Be encouraging and supportive in your responses
- Use clear, easy-to-understand language"""
    
    def create_user_prompt_template(self) -> PromptTemplate:
        """Create user prompt template"""
        template = """Based on the following government job information, please answer the user's question: {question}

Relevant Job Information:
{job_context}

User Question: {question}

Please provide a comprehensive answer that:
1. Directly addresses the user's question
2. References specific jobs and their details
3. Provides actionable advice
4. Includes relevant eligibility criteria and application information
5. Mentions official websites for further information

Answer:"""
        
        return PromptTemplate(
            input_variables=["question", "job_context"],
            template=template
        )
    
    def format_job_context(self, jobs: List[Dict[str, Any]]) -> str:
        """Format job information for the prompt"""
        if not jobs:
            return "No relevant job information found."
        
        context_parts = []
        for i, job in enumerate(jobs, 1):
            metadata = job.get('metadata', {})
            context_parts.append(f"""
Job {i}: {metadata.get('exam_name', 'N/A')}
Organization: {metadata.get('organization', 'N/A')}
Category: {metadata.get('category', 'N/A')}
Description: {metadata.get('description', 'N/A')}
Eligibility: {metadata.get('eligibility', 'N/A')}
Age Limit: {metadata.get('age_limit', 'N/A')}
Selection Process: {metadata.get('selection_process', 'N/A')}
Job Roles: {metadata.get('job_roles', 'N/A')}
Salary Range: {metadata.get('salary_range', 'N/A')}
Application Cycle: {metadata.get('application_cycle', 'N/A')}
Official Website: {metadata.get('official_website', 'N/A')}
Tags: {metadata.get('tags', 'N/A')}
""")
        
        return "\n".join(context_parts)
    
    def generate_response(self, 
                         question: str, 
                         relevant_jobs: List[Dict[str, Any]],
                         include_sources: bool = True) -> RAGResponse:
        """Generate response using retrieved job information"""
        
        try:
            # Format job context
            job_context = self.format_job_context(relevant_jobs)
            
            # Create prompt
            prompt_template = self.create_user_prompt_template()
            
            # Generate response
            messages = [
                SystemMessage(content=self.create_system_prompt()),
                HumanMessage(content=prompt_template.format(
                    question=question,
                    job_context=job_context
                ))
            ]
            
            response = self.llm(messages)
            answer = response.content
            
            # Extract sources
            sources = []
            if include_sources and relevant_jobs:
                for job in relevant_jobs:
                    metadata = job.get('metadata', {})
                    source = f"{metadata.get('exam_name', 'Unknown')} - {metadata.get('organization', 'Unknown')}"
                    if metadata.get('official_website'):
                        source += f" ({metadata['official_website']})"
                    sources.append(source)
            
            # Calculate confidence score (simplified)
            confidence_score = min(1.0, len(relevant_jobs) / 5.0) if relevant_jobs else 0.0
            
            return RAGResponse(
                answer=answer,
                relevant_jobs=relevant_jobs,
                sources=sources,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return RAGResponse(
                answer=f"I apologize, but I encountered an error while generating a response: {str(e)}",
                relevant_jobs=relevant_jobs,
                sources=[],
                confidence_score=0.0
            )
    
    def generate_job_recommendations(self, 
                                   user_profile: Dict[str, Any],
                                   relevant_jobs: List[Dict[str, Any]]) -> RAGResponse:
        """Generate personalized job recommendations"""
        
        question = f"""Based on my profile, please recommend suitable government jobs for me.

My Profile:
- Education Level: {user_profile.get('education_level', 'Not specified')}
- Interests: {', '.join(user_profile.get('interests', []))}
- Experience Level: {user_profile.get('experience_level', 'Not specified')}
- Preferred Location: {user_profile.get('location', 'Not specified')}
- Salary Expectations: {user_profile.get('salary_expectations', 'Not specified')}

Please provide personalized recommendations with detailed explanations."""
        
        return self.generate_response(question, relevant_jobs)
    
    def generate_exam_preparation_advice(self, 
                                       exam_name: str,
                                       relevant_jobs: List[Dict[str, Any]]) -> RAGResponse:
        """Generate exam preparation advice"""
        
        question = f"""I want to prepare for {exam_name}. Please provide comprehensive preparation advice including:
1. Syllabus and exam pattern
2. Study materials and resources
3. Preparation timeline
4. Important topics to focus on
5. Tips for success
6. Previous year question patterns"""
        
        return self.generate_response(question, relevant_jobs)
    
    def generate_career_guidance(self, 
                                career_field: str,
                                relevant_jobs: List[Dict[str, Any]]) -> RAGResponse:
        """Generate career guidance for a specific field"""
        
        question = f"""I'm interested in pursuing a career in {career_field} through government jobs. Please provide:
1. Available job opportunities in this field
2. Career progression paths
3. Required qualifications and skills
4. Salary prospects
5. Job responsibilities
6. Future growth opportunities"""
        
        return self.generate_response(question, relevant_jobs)

class SimpleRAGGenerator:
    """Simple RAG generator without external LLM dependencies"""
    
    def __init__(self):
        logger.info("Initialized Simple RAG Generator (no external LLM)")
    
    def generate_response(self, 
                         question: str, 
                         relevant_jobs: List[Dict[str, Any]]) -> RAGResponse:
        """Generate response using template-based approach"""
        
        if not relevant_jobs:
            return RAGResponse(
                answer="I couldn't find any relevant government jobs matching your query. Please try rephrasing your question or check if the job category exists in our database.",
                relevant_jobs=[],
                sources=[],
                confidence_score=0.0
            )
        
        # Analyze question type
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["eligibility", "qualification", "requirement"]):
            answer = self._generate_eligibility_response(relevant_jobs)
        elif any(word in question_lower for word in ["salary", "pay", "compensation"]):
            answer = self._generate_salary_response(relevant_jobs)
        elif any(word in question_lower for word in ["apply", "application", "how to"]):
            answer = self._generate_application_response(relevant_jobs)
        elif any(word in question_lower for word in ["exam", "selection", "process"]):
            answer = self._generate_exam_response(relevant_jobs)
        else:
            answer = self._generate_general_response(relevant_jobs)
        
        # Extract sources
        sources = []
        for job in relevant_jobs:
            metadata = job.get('metadata', {})
            source = f"{metadata.get('exam_name', 'Unknown')} - {metadata.get('organization', 'Unknown')}"
            if metadata.get('official_website'):
                source += f" ({metadata['official_website']})"
            sources.append(source)
        
        confidence_score = min(1.0, len(relevant_jobs) / 5.0)
        
        return RAGResponse(
            answer=answer,
            relevant_jobs=relevant_jobs,
            sources=sources,
            confidence_score=confidence_score
        )
    
    def _generate_eligibility_response(self, jobs: List[Dict[str, Any]]) -> str:
        """Generate eligibility-focused response"""
        response_parts = ["Based on the available government jobs, here are the eligibility requirements:\n"]
        
        for i, job in enumerate(jobs[:3], 1):  # Limit to top 3 jobs
            metadata = job.get('metadata', {})
            response_parts.append(f"{i}. **{metadata.get('exam_name', 'N/A')}**")
            response_parts.append(f"   - Organization: {metadata.get('organization', 'N/A')}")
            response_parts.append(f"   - Eligibility: {metadata.get('eligibility', 'N/A')}")
            response_parts.append(f"   - Age Limit: {metadata.get('age_limit', 'N/A')}")
            response_parts.append("")
        
        return "\n".join(response_parts)
    
    def _generate_salary_response(self, jobs: List[Dict[str, Any]]) -> str:
        """Generate salary-focused response"""
        response_parts = ["Here are the salary ranges for the relevant government jobs:\n"]
        
        for i, job in enumerate(jobs[:3], 1):
            metadata = job.get('metadata', {})
            response_parts.append(f"{i}. **{metadata.get('exam_name', 'N/A')}**")
            response_parts.append(f"   - Organization: {metadata.get('organization', 'N/A')}")
            response_parts.append(f"   - Salary Range: {metadata.get('salary_range', 'N/A')}")
            response_parts.append(f"   - Job Roles: {metadata.get('job_roles', 'N/A')}")
            response_parts.append("")
        
        return "\n".join(response_parts)
    
    def _generate_application_response(self, jobs: List[Dict[str, Any]]) -> str:
        """Generate application-focused response"""
        response_parts = ["Here's how to apply for these government jobs:\n"]
        
        for i, job in enumerate(jobs[:3], 1):
            metadata = job.get('metadata', {})
            response_parts.append(f"{i}. **{metadata.get('exam_name', 'N/A')}**")
            response_parts.append(f"   - Organization: {metadata.get('organization', 'N/A')}")
            response_parts.append(f"   - Application Cycle: {metadata.get('application_cycle', 'N/A')}")
            response_parts.append(f"   - Official Website: {metadata.get('official_website', 'N/A')}")
            response_parts.append(f"   - Selection Process: {metadata.get('selection_process', 'N/A')}")
            response_parts.append("")
        
        return "\n".join(response_parts)
    
    def _generate_exam_response(self, jobs: List[Dict[str, Any]]) -> str:
        """Generate exam-focused response"""
        response_parts = ["Here are the exam details for the relevant government jobs:\n"]
        
        for i, job in enumerate(jobs[:3], 1):
            metadata = job.get('metadata', {})
            response_parts.append(f"{i}. **{metadata.get('exam_name', 'N/A')}**")
            response_parts.append(f"   - Organization: {metadata.get('organization', 'N/A')}")
            response_parts.append(f"   - Selection Process: {metadata.get('selection_process', 'N/A')}")
            response_parts.append(f"   - Frequency: {metadata.get('frequency', 'N/A')}")
            response_parts.append("")
        
        return "\n".join(response_parts)
    
    def _generate_general_response(self, jobs: List[Dict[str, Any]]) -> str:
        """Generate general response"""
        response_parts = ["Here are the relevant government job opportunities:\n"]
        
        for i, job in enumerate(jobs[:5], 1):  # Show more jobs for general queries
            metadata = job.get('metadata', {})
            response_parts.append(f"{i}. **{metadata.get('exam_name', 'N/A')}**")
            response_parts.append(f"   - Organization: {metadata.get('organization', 'N/A')}")
            response_parts.append(f"   - Category: {metadata.get('category', 'N/A')}")
            response_parts.append(f"   - Description: {metadata.get('description', 'N/A')[:100]}...")
            response_parts.append(f"   - Salary: {metadata.get('salary_range', 'N/A')}")
            response_parts.append(f"   - Website: {metadata.get('official_website', 'N/A')}")
            response_parts.append("")
        
        return "\n".join(response_parts)

if __name__ == "__main__":
    # Test the RAG generator
    logger.info("Testing RAG Generator...")
    
    # Sample job data for testing
    sample_jobs = [
        {
            'metadata': {
                'exam_name': 'IBPS PO',
                'organization': 'Institute of Banking Personnel Selection',
                'category': 'Banking',
                'eligibility': 'Graduate in any discipline',
                'age_limit': '20-30 years',
                'salary_range': '₹36,000 – ₹50,000',
                'official_website': 'https://www.ibps.in/'
            }
        }
    ]
    
    # Test simple generator
    simple_generator = SimpleRAGGenerator()
    response = simple_generator.generate_response("What are the eligibility requirements for banking jobs?", sample_jobs)
    
    print("Simple RAG Response:")
    print(response.answer)
    print(f"Confidence: {response.confidence_score}")
    print(f"Sources: {response.sources}")
