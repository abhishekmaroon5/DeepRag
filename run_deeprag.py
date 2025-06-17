#!/usr/bin/env python3
"""
Command-line interface for DeepRAG

Usage:
    python run_deeprag.py train --data data.json --model mock --retriever hybrid
    python run_deeprag.py inference --question "What is the place of birth of the director of film Peter's Friends?"
    python run_deeprag.py demo
"""

import argparse
import json
import sys
import logging
from typing import List, Tuple

from deeprag_improved import DeepRAGTrainer, create_sample_training_data, demo_deeprag

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_training_data(file_path: str) -> List[Tuple[str, str]]:
    """Load training data from JSON or JSONL file"""
    try:
        training_data = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            # Try to load as JSONL first (one JSON object per line)
            content = f.read().strip()
            if content.startswith('['):
                # JSON array format
                data = json.loads(content)
                for item in data:
                    if isinstance(item, dict) and 'question' in item and 'answer' in item:
                        training_data.append((item['question'], item['answer']))
                    elif isinstance(item, list) and len(item) == 2:
                        training_data.append((item[0], item[1]))
            else:
                # JSONL format (one JSON object per line)
                for line in content.split('\n'):
                    line = line.strip()
                    if line:
                        item = json.loads(line)
                        if isinstance(item, dict) and 'question' in item and 'answer' in item:
                            training_data.append((item['question'], item['answer']))
                        elif isinstance(item, list) and len(item) == 2:
                            training_data.append((item[0], item[1]))
        
        return training_data
    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        return []

def save_results(results, output_file: str):
    """Save results to JSON file"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")

def train_command(args):
    """Run training pipeline"""
    logger.info("Starting DeepRAG training pipeline...")
    
    # Load training data
    if args.data:
        train_data = load_training_data(args.data)
    else:
        logger.info("No training data provided, using sample data")
        train_data = create_sample_training_data()
    
    if not train_data:
        logger.error("No valid training data found")
        return
    
    logger.info(f"Loaded {len(train_data)} training examples")
    
    # Initialize trainer
    trainer = DeepRAGTrainer(model_name=args.model, retriever_type=args.retriever)
    
    # Load custom knowledge base if provided
    if args.knowledge_base:
        trainer.load_custom_knowledge_base(args.knowledge_base)
    
    # Stage I & II: Binary Tree Search + Imitation Learning
    logger.info("Running Stage I & II: Binary Tree Search + Imitation Learning")
    training_examples = trainer.train_stage1_and_2(train_data)
    
    # Stage III: Chain of Calibration
    logger.info("Running Stage III: Chain of Calibration")
    preference_pairs = trainer.train_stage3(training_examples)
    
    # Save results
    results = {
        "training_examples_count": len(training_examples),
        "preference_pairs_count": len(preference_pairs),
        "model_name": args.model,
        "retriever_type": args.retriever
    }
    
    if args.output:
        save_results(results, args.output)
    
    logger.info(f"Training completed successfully!")
    logger.info(f"Generated {len(training_examples)} training examples")
    logger.info(f"Generated {len(preference_pairs)} preference pairs")

def inference_command(args):
    """Run inference"""
    logger.info("Starting DeepRAG inference...")
    
    # Initialize trainer
    trainer = DeepRAGTrainer(model_name=args.model, retriever_type=args.retriever)
    
    # Load custom knowledge base if provided
    if args.knowledge_base:
        trainer.load_custom_knowledge_base(args.knowledge_base)
    
    # Run inference
    result = trainer.inference(args.question, max_steps=args.max_steps)
    
    # Display results
    print(f"\n{'='*60}")
    print(f"Question: {result['question']}")
    print(f"{'='*60}")
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
    
    # Save results if output file specified
    if args.output:
        save_results(result, args.output)

def demo_command(args):
    """Run demo"""
    demo_deeprag()

def evaluate_command(args):
    """Run evaluation on test set"""
    logger.info("Starting DeepRAG evaluation...")
    
    # Load test data
    if not args.test_data:
        logger.error("Test data file is required for evaluation")
        return
    
    test_data = load_training_data(args.test_data)
    if not test_data:
        logger.error("No valid test data found")
        return
    
    # Initialize trainer
    trainer = DeepRAGTrainer(model_name=args.model, retriever_type=args.retriever)
    
    # Load custom knowledge base if provided
    if args.knowledge_base:
        trainer.load_custom_knowledge_base(args.knowledge_base)
    
    # Run evaluation
    results = []
    correct_answers = 0
    total_retrievals = 0
    
    for i, (question, expected_answer) in enumerate(test_data):
        logger.info(f"Evaluating question {i+1}/{len(test_data)}")
        
        result = trainer.inference(question, max_steps=args.max_steps)
        
        # Simple answer matching (can be improved)
        is_correct = expected_answer.lower() in result['final_answer'].lower()
        if is_correct:
            correct_answers += 1
        
        total_retrievals += result['total_retrievals']
        
        results.append({
            "question": question,
            "expected_answer": expected_answer,
            "predicted_answer": result['final_answer'],
            "is_correct": is_correct,
            "retrieval_count": result['total_retrievals'],
            "reasoning_trace": result['reasoning_trace']
        })
    
    # Calculate metrics
    accuracy = correct_answers / len(test_data)
    avg_retrievals = total_retrievals / len(test_data)
    
    evaluation_summary = {
        "accuracy": accuracy,
        "correct_answers": correct_answers,
        "total_questions": len(test_data),
        "average_retrievals": avg_retrievals,
        "total_retrievals": total_retrievals,
        "results": results
    }
    
    # Display summary
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy:.2%} ({correct_answers}/{len(test_data)})")
    print(f"Average Retrievals: {avg_retrievals:.2f}")
    print(f"Total Retrievals: {total_retrievals}")
    
    # Save results
    if args.output:
        save_results(evaluation_summary, args.output)
    
    logger.info("Evaluation completed successfully!")

def main():
    parser = argparse.ArgumentParser(description="DeepRAG: Thinking to Retrieve Step by Step")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train DeepRAG model')
    train_parser.add_argument('--data', type=str, help='Training data JSON file')
    train_parser.add_argument('--model', type=str, default='mock', help='Model name (default: mock)')
    train_parser.add_argument('--retriever', type=str, default='hybrid', choices=['bm25', 'dense', 'hybrid'], help='Retriever type')
    train_parser.add_argument('--knowledge-base', type=str, help='Custom knowledge base JSON file')
    train_parser.add_argument('--output', type=str, help='Output file for training results')
    
    # Inference command
    inference_parser = subparsers.add_parser('inference', help='Run inference')
    inference_parser.add_argument('--question', type=str, required=True, help='Question to answer')
    inference_parser.add_argument('--model', type=str, default='mock', help='Model name (default: mock)')
    inference_parser.add_argument('--retriever', type=str, default='hybrid', choices=['bm25', 'dense', 'hybrid'], help='Retriever type')
    inference_parser.add_argument('--knowledge-base', type=str, help='Custom knowledge base JSON file')
    inference_parser.add_argument('--max-steps', type=int, default=5, help='Maximum reasoning steps')
    inference_parser.add_argument('--output', type=str, help='Output file for results')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run demo')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate on test set')
    eval_parser.add_argument('--test-data', type=str, required=True, help='Test data JSON file')
    eval_parser.add_argument('--model', type=str, default='mock', help='Model name (default: mock)')
    eval_parser.add_argument('--retriever', type=str, default='hybrid', choices=['bm25', 'dense', 'hybrid'], help='Retriever type')
    eval_parser.add_argument('--knowledge-base', type=str, help='Custom knowledge base JSON file')
    eval_parser.add_argument('--max-steps', type=int, default=5, help='Maximum reasoning steps')
    eval_parser.add_argument('--output', type=str, help='Output file for evaluation results')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_command(args)
    elif args.command == 'inference':
        inference_command(args)
    elif args.command == 'demo':
        demo_command(args)
    elif args.command == 'evaluate':
        evaluate_command(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 