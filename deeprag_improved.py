"""
DeepRAG: Thinking to Retrieve Step by Step for Large Language Models

This module implements the complete DeepRAG framework with three main stages:
1. Binary Tree Search - for optimal trajectory discovery
2. Imitation Learning - for teaching effective query decomposition
3. Chain of Calibration - for knowledge boundary awareness

Based on the paper: https://arxiv.org/pdf/2502.01142
"""

import json
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import heapq
import random
from collections import defaultdict
import torch
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import pickle
import os

from retriever import WikipediaKnowledgeBase, Document
from models import create_llm, BaseLLM, ModelConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ActionType(Enum):
    CONTINUE = "continue"
    TERMINATE = "terminate"

class KnowledgeSource(Enum):
    PARAMETRIC = "parametric"
    RETRIEVE = "retrieve"

@dataclass
class State:
    """Represents the state in the MDP formulation"""
    question: str
    subqueries: List[str]
    responses: List[str]
    retrieval_count: int = 0
    
    def __hash__(self):
        return hash((self.question, tuple(self.subqueries), tuple(self.responses)))
    
    def __lt__(self, other):
        """Less than comparison for heap operations"""
        if not isinstance(other, State):
            return NotImplemented
        return len(self.subqueries) < len(other.subqueries)
    
    def __eq__(self, other):
        """Equality comparison"""
        if not isinstance(other, State):
            return NotImplemented
        return (self.question == other.question and 
                self.subqueries == other.subqueries and 
                self.responses == other.responses)
    
    def get_context(self) -> str:
        """Get formatted context string"""
        context = ""
        for sq, resp in zip(self.subqueries, self.responses):
            context += f"Q: {sq}\nA: {resp}\n"
        return context

@dataclass
class Action:
    """Represents an action in the MDP"""
    termination: ActionType
    knowledge_source: Optional[KnowledgeSource] = None

@dataclass
class TrainingExample:
    """Training example for imitation learning"""
    question: str
    optimal_trajectory: List[Tuple[str, str, KnowledgeSource]]  # (subquery, response, source)
    final_answer: str
    retrieval_count: int

@dataclass
class PreferencePair:
    """Preference pair for calibration training"""
    state: State
    subquery: str
    preferred_source: KnowledgeSource
    preferred_response: str
    dispreferred_response: str

class DeepRAGDataset(Dataset):
    """Dataset class for training DeepRAG models"""
    
    def __init__(self, examples: List[str], tokenizer, max_length: int = 512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        encoding = self.tokenizer(
            example,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # For causal LM, labels are the same as input_ids
        encoding['labels'] = encoding['input_ids'].clone()
        
        return {key: val.squeeze() for key, val in encoding.items()}

class BinaryTreeSearch:
    """
    Stage I: Binary Tree Search for optimal trajectory finding
    
    Implements Algorithm 1 from the paper:
    Uses priority queue to find trajectory with minimal retrieval cost
    """
    
    def __init__(self, llm: BaseLLM, knowledge_base: WikipediaKnowledgeBase):
        self.llm = llm
        self.knowledge_base = knowledge_base
        
    def construct_optimal_trajectory(self, question: str, target_answer: str, max_depth: int = 5) -> Optional[TrainingExample]:
        """
        Algorithm 1: Data Construction for Stage I
        Uses priority queue to find trajectory with minimal retrieval cost
        """
        # Priority queue: (retrieval_count, counter, state, trajectory_info)
        # Using counter to ensure unique comparison values
        initial_state = State(question=question, subqueries=[], responses=[])
        pq = [(0, 0, initial_state, [])]
        visited = set()
        counter = 1  # For unique heap entries
        
        logger.info(f"Starting binary tree search for: {question}")
        
        while pq:
            retrieval_count, _, state, trajectory_info = heapq.heappop(pq)
            
            state_key = (state.question, tuple(state.subqueries), tuple(state.responses))
            if state_key in visited or len(state.subqueries) > max_depth:
                continue
            visited.add(state_key)
            
            # Generate next subquery
            context = state.get_context()
            subquery = self.llm.generate_subquery(question, context)
            
            # Check if should terminate
            if self.llm.should_terminate(question, context, subquery):
                final_answer = self.llm.generate_final_answer(question, context)
                if self._is_correct_answer(final_answer, target_answer):
                    logger.info(f"Found optimal trajectory with {retrieval_count} retrievals")
                    return self._create_training_example(question, state, final_answer, retrieval_count, trajectory_info)
                continue
            
            # Explore parametric knowledge path
            try:
                parametric_answer = self.llm.answer_parametric(question, context, subquery)
                new_state_param = State(
                    question=question,
                    subqueries=state.subqueries + [subquery],
                    responses=state.responses + [parametric_answer],
                    retrieval_count=retrieval_count
                )
                new_trajectory_param = trajectory_info + [(subquery, parametric_answer, KnowledgeSource.PARAMETRIC)]
                heapq.heappush(pq, (retrieval_count, counter, new_state_param, new_trajectory_param))
                counter += 1
            except Exception as e:
                logger.warning(f"Error in parametric answering: {e}")
            
            # Explore retrieval path
            try:
                retrieved_docs = self.knowledge_base.retrieve(subquery, top_k=3)
                retrieval_answer = self.llm.answer_with_retrieval(question, context, subquery, retrieved_docs)
                new_state_retr = State(
                    question=question,
                    subqueries=state.subqueries + [subquery],
                    responses=state.responses + [retrieval_answer],
                    retrieval_count=retrieval_count + 1
                )
                new_trajectory_retr = trajectory_info + [(subquery, retrieval_answer, KnowledgeSource.RETRIEVE)]
                heapq.heappush(pq, (retrieval_count + 1, counter, new_state_retr, new_trajectory_retr))
                counter += 1
            except Exception as e:
                logger.warning(f"Error in retrieval answering: {e}")
        
        logger.warning(f"No valid trajectory found for: {question}")
        return None
    
    def _is_correct_answer(self, generated_answer: str, target_answer: str) -> bool:
        """Simple answer correctness check - can be made more sophisticated"""
        generated_lower = generated_answer.lower().strip()
        target_lower = target_answer.lower().strip()
        
        # Check if target is substring of generated or vice versa
        return (target_lower in generated_lower or 
                generated_lower in target_lower or
                self._fuzzy_match(generated_lower, target_lower))
    
    def _fuzzy_match(self, answer1: str, answer2: str, threshold: float = 0.8) -> bool:
        """Simple fuzzy matching based on word overlap"""
        words1 = set(answer1.split())
        words2 = set(answer2.split())
        
        if not words1 or not words2:
            return False
        
        intersection = words1 & words2
        union = words1 | words2
        
        jaccard_similarity = len(intersection) / len(union)
        return jaccard_similarity >= threshold
    
    def _create_training_example(self, question: str, state: State, final_answer: str, 
                                retrieval_count: int, trajectory_info: List) -> TrainingExample:
        """Create training example from optimal trajectory"""
        return TrainingExample(
            question=question,
            optimal_trajectory=trajectory_info,
            final_answer=final_answer,
            retrieval_count=retrieval_count
        )

class ImitationLearning:
    """
    Stage II: Imitation Learning
    
    Teaches the model effective query decomposition and response generation
    based on optimal trajectories from Stage I
    """
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        
    def create_training_data(self, binary_tree_search: BinaryTreeSearch, 
                           questions_answers: List[Tuple[str, str]]) -> List[TrainingExample]:
        """Create training data using binary tree search"""
        training_examples = []
        
        logger.info(f"Creating training data for {len(questions_answers)} examples")
        
        for question, answer in tqdm(questions_answers, desc="Processing Q&A pairs"):
            example = binary_tree_search.construct_optimal_trajectory(question, answer)
            if example:
                training_examples.append(example)
                logger.debug(f"Created training example for: {question}")
            else:
                logger.warning(f"Failed to create training example for: {question}")
        
        logger.info(f"Successfully created {len(training_examples)} training examples")
        return training_examples
    
    def format_training_example(self, example: TrainingExample) -> str:
        """Format training example according to paper's answer format"""
        formatted = f"Question: {example.question}\n\n"
        
        for i, (subquery, response, source) in enumerate(example.optimal_trajectory):
            formatted += f"Follow up {i+1}: {subquery}\n"
            if source == KnowledgeSource.RETRIEVE:
                formatted += "Let's search the question in Wikipedia.\n"
                formatted += f"Context: {response}\n"
            formatted += f"Intermediate answer: {response}\n\n"
        
        formatted += f"So the final answer is: {example.final_answer}"
        return formatted
    
    def prepare_training_data(self, training_examples: List[TrainingExample]) -> List[str]:
        """Prepare formatted training data for model training"""
        formatted_examples = []
        for example in training_examples:
            formatted = self.format_training_example(example)
            formatted_examples.append(formatted)
        return formatted_examples

class ChainOfCalibration:
    """
    Stage III: Chain of Calibration
    
    Creates preference pairs and trains the model to better understand
    its knowledge boundaries using DPO-style objectives
    """
    
    def __init__(self):
        self.preference_pairs = []
    
    def generate_preference_data(self, binary_tree_search: BinaryTreeSearch, 
                               training_examples: List[TrainingExample]) -> List[PreferencePair]:
        """Generate preference pairs for calibration training"""
        preference_pairs = []
        
        logger.info(f"Generating preference data for {len(training_examples)} examples")
        
        for example in tqdm(training_examples, desc="Creating preference pairs"):
            try:
                pairs = self._create_preference_pairs_for_example(binary_tree_search, example)
                preference_pairs.extend(pairs)
            except Exception as e:
                logger.warning(f"Error creating preference pairs for example: {e}")
        
        logger.info(f"Generated {len(preference_pairs)} preference pairs")
        return preference_pairs
    
    def _create_preference_pairs_for_example(self, binary_tree_search: BinaryTreeSearch,
                                           example: TrainingExample) -> List[PreferencePair]:
        """Create preference pairs for a single training example"""
        pairs = []
        
        for i, (subquery, preferred_response, preferred_source) in enumerate(example.optimal_trajectory):
            # Create state up to this point
            state = State(
                question=example.question,
                subqueries=[sq for sq, _, _ in example.optimal_trajectory[:i]],
                responses=[resp for _, resp, _ in example.optimal_trajectory[:i]]
            )
            
            # Generate alternative response with different source
            dispreferred_source = (KnowledgeSource.PARAMETRIC if preferred_source == KnowledgeSource.RETRIEVE 
                                 else KnowledgeSource.RETRIEVE)
            
            try:
                if dispreferred_source == KnowledgeSource.RETRIEVE:
                    retrieved_docs = binary_tree_search.knowledge_base.retrieve(subquery, top_k=3)
                    dispreferred_response = binary_tree_search.llm.answer_with_retrieval(
                        example.question, state.get_context(), subquery, retrieved_docs
                    )
                else:
                    dispreferred_response = binary_tree_search.llm.answer_parametric(
                        example.question, state.get_context(), subquery
                    )
                
                pair = PreferencePair(
                    state=state,
                    subquery=subquery,
                    preferred_source=preferred_source,
                    preferred_response=preferred_response,
                    dispreferred_response=dispreferred_response
                )
                pairs.append(pair)
                
            except Exception as e:
                logger.warning(f"Error generating alternative response: {e}")
        
        return pairs
    
    def compute_calibration_loss(self, preferred_logits: torch.Tensor, 
                               dispreferred_logits: torch.Tensor, beta: float = 0.1) -> torch.Tensor:
        """
        Compute Chain of Calibration loss (DPO-style objective)
        Based on Appendix B.2 equation from the paper
        """
        # Simplified version - in practice you'd need reference model logits
        log_ratio = (torch.log_softmax(preferred_logits, dim=-1).mean() - 
                    torch.log_softmax(dispreferred_logits, dim=-1).mean())
        loss = -torch.log(torch.sigmoid(beta * log_ratio))
        return loss

class DeepRAGTrainer:
    """
    Main trainer that orchestrates all three stages of DeepRAG training
    """
    
    def __init__(self, model_name: str = "mock", retriever_type: str = "hybrid"):
        self.model_name = model_name
        self.llm = create_llm(model_name)
        
        # Initialize knowledge base
        self.knowledge_base = WikipediaKnowledgeBase(retriever_type)
        self.knowledge_base.create_sample_knowledge_base()
        
        # Initialize training components
        self.binary_tree_search = BinaryTreeSearch(self.llm, self.knowledge_base)
        self.imitation_learning = ImitationLearning(model_name)
        self.calibration = ChainOfCalibration()
        
        logger.info(f"Initialized DeepRAG trainer with {model_name} model and {retriever_type} retriever")
    
    def load_custom_knowledge_base(self, file_path: str):
        """Load custom knowledge base from file"""
        self.knowledge_base.load_documents_from_file(file_path)
        
    def train_stage1_and_2(self, train_data: List[Tuple[str, str]]) -> List[TrainingExample]:
        """Stage I: Binary Tree Search + Stage II: Imitation Learning"""
        logger.info("Starting Stage I & II: Binary Tree Search + Imitation Learning")
        
        # Generate training examples using binary tree search
        training_examples = self.imitation_learning.create_training_data(
            self.binary_tree_search, train_data
        )
        
        # Format training data
        formatted_examples = self.imitation_learning.prepare_training_data(training_examples)
        
        logger.info(f"Stage I & II completed with {len(training_examples)} examples")
        
        # Save training examples for later use
        self._save_training_examples(training_examples, "stage2_training_examples.json")
        
        return training_examples
    
    def train_stage3(self, training_examples: List[TrainingExample]) -> List[PreferencePair]:
        """Stage III: Chain of Calibration"""
        logger.info("Starting Stage III: Chain of Calibration")
        
        # Generate preference pairs
        preference_pairs = self.calibration.generate_preference_data(
            self.binary_tree_search, training_examples
        )
        
        logger.info(f"Stage III completed with {len(preference_pairs)} preference pairs")
        
        # Save preference pairs
        self._save_preference_pairs(preference_pairs, "stage3_preference_pairs.json")
        
        return preference_pairs
    
    def inference(self, question: str, max_steps: int = 5) -> Dict[str, Any]:
        """Run inference using trained DeepRAG model"""
        state = State(question=question, subqueries=[], responses=[])
        reasoning_trace = []
        final_answer = ""
        
        logger.info(f"Starting inference for: {question}")
        
        for step in range(max_steps):
            context = state.get_context()
            
            # Generate subquery
            subquery = self.llm.generate_subquery(question, context)
            logger.info(f"Step {step+1} - Subquery: {subquery}")
            
            # Check termination
            if self.llm.should_terminate(question, context, subquery):
                final_answer = self.llm.generate_final_answer(question, context)
                reasoning_trace.append({
                    "step": step + 1,
                    "subquery": subquery,
                    "action": "terminate",
                    "final_answer": final_answer
                })
                logger.info(f"Terminated at step {step+1}. Final answer: {final_answer}")
                break
            
            # Make atomic decision (simplified - in practice this would use trained model)
            should_retrieve = self._should_retrieve(question, context, subquery)
            
            if should_retrieve:
                # Retrieve and answer
                retrieved_docs = self.knowledge_base.retrieve(subquery, top_k=3)
                response = self.llm.answer_with_retrieval(question, context, subquery, retrieved_docs)
                source = "retrieve"
                state.retrieval_count += 1
                logger.info(f"Retrieved: {response}")
            else:
                # Use parametric knowledge
                response = self.llm.answer_parametric(question, context, subquery)
                source = "parametric"
                logger.info(f"Parametric: {response}")
            
            # Update state
            state.subqueries.append(subquery)
            state.responses.append(response)
            
            reasoning_trace.append({
                "step": step + 1,
                "subquery": subquery,
                "response": response,
                "source": source,
                "retrieval_count": state.retrieval_count
            })
        
        # Generate final answer if not already terminated
        if not final_answer and step == max_steps - 1:
            final_answer = self.llm.generate_final_answer(question, state.get_context())
            reasoning_trace.append({
                "step": max_steps + 1,
                "action": "max_steps_reached",
                "final_answer": final_answer
            })
        
        return {
            "question": question,
            "final_answer": final_answer,
            "total_retrievals": state.retrieval_count,
            "reasoning_trace": reasoning_trace
        }
    
    def _should_retrieve(self, question: str, context: str, subquery: str) -> bool:
        """
        Atomic decision for retrieval
        In practice, this would use the trained model's decision mechanism
        """
        # Simple heuristic for demonstration
        retrieve_keywords = ["director", "birthplace", "runtime", "when", "where", "who", "what year"]
        parametric_keywords = ["total", "sum", "calculate", "add"]
        
        subquery_lower = subquery.lower()
        
        # Strong indicators for parametric knowledge
        for keyword in parametric_keywords:
            if keyword in subquery_lower:
                return False
        
        # Strong indicators for retrieval
        for keyword in retrieve_keywords:
            if keyword in subquery_lower:
                return True
        
        # Default to retrieval for factual questions
        return True
    
    def _save_training_examples(self, examples: List[TrainingExample], filename: str):
        """Save training examples to file"""
        serializable_examples = []
        for example in examples:
            serializable_examples.append(asdict(example))
        
        with open(filename, 'w') as f:
            json.dump(serializable_examples, f, indent=2)
        logger.info(f"Saved {len(examples)} training examples to {filename}")
    
    def _save_preference_pairs(self, pairs: List[PreferencePair], filename: str):
        """Save preference pairs to file"""
        serializable_pairs = []
        for pair in pairs:
            pair_dict = asdict(pair)
            # Convert State to dict manually since it has custom __hash__
            pair_dict['state'] = {
                'question': pair.state.question,
                'subqueries': pair.state.subqueries,
                'responses': pair.state.responses,
                'retrieval_count': pair.state.retrieval_count
            }
            serializable_pairs.append(pair_dict)
        
        with open(filename, 'w') as f:
            json.dump(serializable_pairs, f, indent=2)
        logger.info(f"Saved {len(pairs)} preference pairs to {filename}")

def create_sample_training_data() -> List[Tuple[str, str]]:
    """Create sample training data for demonstration"""
    return [
        ("What is the place of birth of the director of film Peter's Friends?", "Belfast, Northern Ireland"),
        ("What is the total runtime of all movies in The Lord of the Rings trilogy?", "558 minutes"),
        ("Who directed the 1992 film Peter's Friends?", "Kenneth Branagh"),
        ("What is the runtime of The Fellowship of the Ring?", "178 minutes"),
        ("What is the runtime of The Two Towers?", "179 minutes"),
        ("What is the runtime of The Return of the King?", "201 minutes"),
    ]

def demo_deeprag():
    """Demonstrate the complete DeepRAG pipeline"""
    
    print("=== DeepRAG Implementation Demo ===\n")
    
    # Initialize trainer
    trainer = DeepRAGTrainer(model_name="mock", retriever_type="hybrid")
    
    # Create sample training data
    train_data = create_sample_training_data()
    
    print(f"Training data: {len(train_data)} question-answer pairs\n")
    
    # Stage I & II: Binary Tree Search + Imitation Learning
    print("--- Stage I & II: Binary Tree Search + Imitation Learning ---")
    training_examples = trainer.train_stage1_and_2(train_data)
    
    # Stage III: Chain of Calibration  
    print("\n--- Stage III: Chain of Calibration ---")
    preference_pairs = trainer.train_stage3(training_examples)
    
    print(f"\n=== Training Results ===")
    print(f"Generated {len(training_examples)} training examples")
    print(f"Generated {len(preference_pairs)} preference pairs")
    
    # Demonstrate inference
    print(f"\n=== Inference Demo ===")
    test_questions = [
        "What is the place of birth of the director of film Peter's Friends?",
        "What is the total runtime of all movies in The Lord of the Rings trilogy?"
    ]
    
    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print(f"{'='*60}")
        
        result = trainer.inference(question)
        
        print(f"Final Answer: {result['final_answer']}")
        print(f"Total Retrievals: {result['total_retrievals']}")
        print("\nReasoning Trace:")
        
        for trace in result['reasoning_trace']:
            if 'subquery' in trace:
                print(f"  Step {trace['step']}: {trace['subquery']}")
                if 'response' in trace:
                    print(f"    → [{trace['source']}] {trace['response']}")
            if 'final_answer' in trace:
                print(f"  Final: {trace['final_answer']}")

if __name__ == "__main__":
    demo_deeprag() 