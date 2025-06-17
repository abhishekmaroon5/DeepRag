import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    GenerationConfig, BitsAndBytesConfig
)
from typing import List, Dict, Optional, Tuple
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from retriever import Document, WikipediaKnowledgeBase

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for LLM models"""
    model_name: str = "microsoft/DialoGPT-medium"
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    use_cache: bool = True
    load_in_4bit: bool = False
    load_in_8bit: bool = False

class BaseLLM(ABC):
    """Abstract base class for LLM implementations"""
    
    @abstractmethod
    def generate_subquery(self, question: str, context: str) -> str:
        pass
    
    @abstractmethod
    def should_terminate(self, question: str, context: str, subquery: str) -> bool:
        pass
    
    @abstractmethod
    def answer_parametric(self, question: str, context: str, subquery: str) -> str:
        pass
    
    @abstractmethod
    def answer_with_retrieval(self, question: str, context: str, subquery: str, retrieved_docs: List[Document]) -> str:
        pass
    
    @abstractmethod
    def generate_final_answer(self, question: str, context: str) -> str:
        pass

class DeepRAGLLM(BaseLLM):
    """LLM implementation for DeepRAG using HuggingFace transformers"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup quantization if requested
        quantization_config = None
        if config.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif config.load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            quantization_config=quantization_config,
            device_map="auto" if quantization_config else None,
            torch_dtype=torch.float16 if quantization_config else None,
        )
        
        if not quantization_config:
            self.model.to(self.device)
        
        self.generation_config = GenerationConfig(
            max_length=config.max_length,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            do_sample=config.do_sample,
            use_cache=config.use_cache,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        logger.info(f"Loaded model {config.model_name} on {self.device}")
    
    def _generate_text(self, prompt: str, max_new_tokens: int = 100) -> str:
        """Generate text using the model"""
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=self.config.max_length-max_new_tokens)
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                generation_config=self.generation_config,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        # Decode only the new tokens
        generated_text = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        return generated_text.strip()
    
    def generate_subquery(self, question: str, context: str) -> str:
        """Generate the next subquery based on question and current context"""
        prompt = self._format_subquery_prompt(question, context)
        subquery = self._generate_text(prompt, max_new_tokens=50)
        
        # Extract the actual subquery from the response
        subquery = self._extract_subquery(subquery)
        return subquery
    
    def _format_subquery_prompt(self, question: str, context: str) -> str:
        """Format prompt for subquery generation"""
        if not context:
            return f"""Question: {question}

To answer this question step by step, I need to first ask:"""
        else:
            return f"""Question: {question}

Progress so far:
{context}

To continue answering this question, my next subquery should be:"""
    
    def _extract_subquery(self, response: str) -> str:
        """Extract clean subquery from model response"""
        # Remove common prefixes and clean up
        response = response.strip()
        
        # Remove common starting phrases
        prefixes_to_remove = [
            "The next subquery should be:",
            "I need to ask:",
            "Let me ask:",
            "Next question:",
            "Subquery:",
        ]
        
        for prefix in prefixes_to_remove:
            if response.lower().startswith(prefix.lower()):
                response = response[len(prefix):].strip()
        
        # Take only the first sentence/question
        sentences = re.split(r'[.!?]', response)
        if sentences:
            subquery = sentences[0].strip()
            if not subquery.endswith('?'):
                subquery += '?'
            return subquery
        
        return response
    
    def should_terminate(self, question: str, context: str, subquery: str) -> bool:
        """Decide whether to terminate the reasoning process"""
        # Simple heuristics for termination
        termination_indicators = [
            "final answer",
            "conclude",
            "therefore",
            "in summary",
            "so the answer is"
        ]
        
        subquery_lower = subquery.lower()
        for indicator in termination_indicators:
            if indicator in subquery_lower:
                return True
        
        # Terminate if context is getting too long
        if len(context.split()) > 200:
            return True
            
        return False
    
    def answer_parametric(self, question: str, context: str, subquery: str) -> str:
        """Answer using parametric knowledge only"""
        prompt = self._format_parametric_prompt(question, context, subquery)
        answer = self._generate_text(prompt, max_new_tokens=80)
        return self._clean_answer(answer)
    
    def _format_parametric_prompt(self, question: str, context: str, subquery: str) -> str:
        """Format prompt for parametric answering"""
        return f"""Question: {question}

{context}

Subquery: {subquery}

Answer this subquery using only my internal knowledge:"""
    
    def answer_with_retrieval(self, question: str, context: str, subquery: str, retrieved_docs: List[Document]) -> str:
        """Answer using retrieved documents"""
        prompt = self._format_retrieval_prompt(question, context, subquery, retrieved_docs)
        answer = self._generate_text(prompt, max_new_tokens=80)
        return self._clean_answer(answer)
    
    def _format_retrieval_prompt(self, question: str, context: str, subquery: str, retrieved_docs: List[Document]) -> str:
        """Format prompt for retrieval-based answering"""
        # Format retrieved documents
        doc_context = ""
        for i, doc in enumerate(retrieved_docs[:3], 1):  # Use top 3 docs
            doc_context += f"Document {i}: {doc.title}\n{doc.text}\n\n"
        
        return f"""Question: {question}

{context}

Subquery: {subquery}

Retrieved Information:
{doc_context}

Based on the retrieved information, answer the subquery:"""
    
    def _clean_answer(self, answer: str) -> str:
        """Clean and format the answer"""
        answer = answer.strip()
        
        # Remove common prefixes
        prefixes_to_remove = [
            "Answer:",
            "The answer is:",
            "Based on the information:",
            "According to the documents:",
        ]
        
        for prefix in prefixes_to_remove:
            if answer.lower().startswith(prefix.lower()):
                answer = answer[len(prefix):].strip()
        
        # Take only the first sentence if multiple sentences
        sentences = re.split(r'[.!?]', answer)
        if sentences and len(sentences) > 1:
            answer = sentences[0].strip() + '.'
        
        return answer
    
    def generate_final_answer(self, question: str, context: str) -> str:
        """Generate final answer based on accumulated context"""
        prompt = self._format_final_answer_prompt(question, context)
        answer = self._generate_text(prompt, max_new_tokens=100)
        return self._clean_final_answer(answer)
    
    def _format_final_answer_prompt(self, question: str, context: str) -> str:
        """Format prompt for final answer generation"""
        return f"""Question: {question}

Information gathered:
{context}

Based on all the information above, the final answer to the question is:"""
    
    def _clean_final_answer(self, answer: str) -> str:
        """Clean the final answer"""
        answer = answer.strip()
        
        # Remove common prefixes
        prefixes_to_remove = [
            "The final answer is:",
            "Therefore:",
            "In conclusion:",
            "So:",
        ]
        
        for prefix in prefixes_to_remove:
            if answer.lower().startswith(prefix.lower()):
                answer = answer[len(prefix):].strip()
        
        return answer

class MockLLM(BaseLLM):
    """Improved mock LLM for testing and demonstration"""
    
    def __init__(self, model_name: str = "mock"):
        self.model_name = model_name
        
        # Knowledge patterns for parametric responses
        self.parametric_knowledge = {
            "lord of the rings trilogy": "The Lord of the Rings trilogy consists of The Fellowship of the Ring, The Two Towers, and The Return of the King.",
            "lord of the rings movies": "The Lord of the Rings trilogy consists of The Fellowship of the Ring, The Two Towers, and The Return of the King.",
            "fellowship runtime": "The Fellowship of the Ring has a runtime of 178 minutes.",
            "two towers runtime": "The Two Towers has a runtime of 179 minutes.", 
            "return king runtime": "The Return of the King has a runtime of 201 minutes.",
            "total runtime calculation": "178 + 179 + 201 = 558 minutes total.",
        }
    
    def generate_subquery(self, question: str, context: str) -> str:
        """Generate next subquery based on current state"""
        context_lower = context.lower()
        question_lower = question.lower()
        
        if not context:
            # First subquery
            if "director" in question_lower and "birth" in question_lower:
                return "Who is the director of the film mentioned?"
            elif "runtime" in question_lower and "total" in question_lower:
                return "What are the individual movies in the trilogy?"
        else:
            # Subsequent subqueries
            if "director" in context_lower and "who is the director" in context_lower:
                return "What is the birthplace of this director?"
            elif "movies" in context_lower or "trilogy" in context_lower:
                return "What is the runtime of each movie?"
            elif "runtime" in context_lower and "minutes" in context_lower:
                return "What is the total runtime of all movies?"
        
        return "What is the final answer?"
    
    def should_terminate(self, question: str, context: str, subquery: str) -> bool:
        """Decide whether to terminate reasoning"""
        termination_indicators = ["final answer", "total runtime", "birthplace"]
        subquery_lower = subquery.lower()
        
        for indicator in termination_indicators:
            if indicator in subquery_lower:
                return True
        
        # Terminate if we have enough context
        if len(context.split('\n')) >= 4:
            return True
            
        return False
    
    def answer_parametric(self, question: str, context: str, subquery: str) -> str:
        """Answer using parametric knowledge"""
        subquery_lower = subquery.lower()
        
        # Check parametric knowledge patterns
        for pattern, answer in self.parametric_knowledge.items():
            if pattern in subquery_lower:
                return answer
        
        # General patterns
        if "lord of the rings" in subquery_lower and ("movies" in subquery_lower or "trilogy" in subquery_lower):
            return "The Fellowship of the Ring, The Two Towers, The Return of the King"
        elif "total" in subquery_lower and "runtime" in subquery_lower:
            if "178" in context and "179" in context and "201" in context:
                return "178 + 179 + 201 = 558 minutes"
        
        return "I don't have this information in my parametric knowledge."
    
    def answer_with_retrieval(self, question: str, context: str, subquery: str, retrieved_docs: List[Document]) -> str:
        """Answer using retrieved documents"""
        if not retrieved_docs:
            return "No relevant information found in retrieval."
        
        # Use the most relevant document
        doc = retrieved_docs[0]
        subquery_lower = subquery.lower()
        
        if "director" in subquery_lower:
            if "directed by" in doc.text or "director" in doc.text:
                # Extract director name
                if "Kenneth Branagh" in doc.text:
                    return "Kenneth Branagh"
                elif "Peter Jackson" in doc.text:
                    return "Peter Jackson"
        
        elif "birthplace" in subquery_lower or "birth" in subquery_lower:
            if "Belfast" in doc.text:
                return "Belfast, Northern Ireland"
        
        elif "runtime" in subquery_lower:
            # Extract runtime information
            import re
            runtime_match = re.search(r'(\d+)\s*minutes?', doc.text)
            if runtime_match:
                return f"{runtime_match.group(1)} minutes"
        
        # Fallback: return excerpt from most relevant document
        return f"According to {doc.title}: {doc.text[:100]}..."
    
    def generate_final_answer(self, question: str, context: str) -> str:
        """Generate final answer based on accumulated information"""
        question_lower = question.lower()
        context_lower = context.lower()
        
        if "birthplace" in question_lower or "birth" in question_lower:
            if "belfast" in context_lower:
                return "Belfast, Northern Ireland"
        
        elif "total runtime" in question_lower:
            if "558" in context:
                return "558 minutes"
            elif "178" in context and "179" in context and "201" in context:
                return "The total runtime is 178 + 179 + 201 = 558 minutes"
        
        # Extract the last meaningful statement from context
        lines = [line.strip() for line in context.split('\n') if line.strip()]
        if lines:
            return lines[-1]
        
        return "Unable to determine the final answer from the available information."

# Factory function to create LLM instances
def create_llm(model_name: str = "mock", **kwargs) -> BaseLLM:
    """Factory function to create LLM instances"""
    if model_name == "mock":
        return MockLLM()
    else:
        config = ModelConfig(model_name=model_name, **kwargs)
        return DeepRAGLLM(config)

# Example usage
if __name__ == "__main__":
    # Test mock LLM
    llm = create_llm("mock")
    
    question = "What is the place of birth of the director of film Peter's Friends?"
    context = ""
    
    for step in range(3):
        subquery = llm.generate_subquery(question, context)
        print(f"Step {step+1} - Subquery: {subquery}")
        
        if llm.should_terminate(question, context, subquery):
            final_answer = llm.generate_final_answer(question, context)
            print(f"Final Answer: {final_answer}")
            break
        
        # Simulate answering (would use retrieval or parametric in real implementation)
        if step == 0:
            answer = "Kenneth Branagh"
        elif step == 1:
            answer = "Belfast, Northern Ireland"
        else:
            answer = "Final answer determined"
        
        context += f"Q: {subquery}\nA: {answer}\n"
        print(f"Answer: {answer}\n") 