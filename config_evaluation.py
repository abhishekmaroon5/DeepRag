import json
import yaml
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import asdict
import pandas as pd

class DeepRAGConfig:
    """Configuration class for DeepRAG pipeline"""
    
    def __init__(self, config_path: str = None):
        if config_path:
            self.load_config(config_path)
        else:
            self.set_default_config()
    
    def set_default_config(self):
        """Set default configuration parameters"""
        self.config = {
            "model": {
                "name": "microsoft/DialoGPT-medium",  # Can be changed to any HF model
                "max_length": 512,
                "temperature": 0.7,
                "top_p": 0.9
            },
            "retriever": {
                "type": "bm25",  # or "dense", "hybrid"
                "corpus_path": "wikipedia_passages.jsonl",
                "top_k": 5
            },
            "training": {
                "stage1": {
                    "max_depth": 5,
                    "beam_size": 3,
                    "samples_per_question": 4000
                },
                "stage2": {
                    "learning_rate": 5e-5,
                    "batch_size": 8,
                    "epochs": 3,
                    "warmup_steps": 100
                },
                "stage3": {
                    "learning_rate": 1e-5,
                    "batch_size": 4,
                    "epochs": 1,
                    "beta": 0.1  # DPO beta parameter
                }
            },
            "inference": {
                "max_steps": 5,
                "retrieval_threshold": 0.5,
                "confidence_threshold": 0.8
            },
            "datasets": {
                "train": ["hotpotqa", "2wikimultihopqa"],
                "test": ["cag", "popqa", "webquestions", "musique"],
                "data_dir": "./data/"
            }
        }
    
    def load_config(self, config_path: str):
        """Load configuration from file"""
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                self.config = yaml.safe_load(f)
            else:
                self.config = json.load(f)
    
    def save_config(self, save_path: str):
        """Save configuration to file"""
        with open(save_path, 'w') as f:
            if save_path.endswith('.yaml') or save_path.endswith('.yml'):
                yaml.dump(self.config, f, default_flow_style=False)
            else:
                json.dump(self.config, f, indent=2)
    
    def get(self, key_path: str, default=None):
        """Get nested configuration value using dot notation"""
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

class DataLoader:
    """Data loader for various QA datasets"""
    
    def __init__(self, config: DeepRAGConfig):
        self.config = config
        self.data_dir = config.get("datasets.data_dir", "./data/")
    
    def load_hotpotqa(self) -> List[Dict]:
        """Load HotpotQA dataset"""
        # Mock data - replace with actual dataset loading
        return [
            {
                "id": "1",
                "question": "What is the place of birth of the director of film Peter's Friends?",
                "answer": "Belfast, Northern Ireland",
                "supporting_facts": [
                    ["Peter's Friends", 0],
                    ["Kenneth Branagh", 1]
                ]
            },
            {
                "id": "2", 
                "question": "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?",
                "answer": "Chief of Protocol",
                "supporting_facts": [
                    ["Kiss and Tell (1945 film)", 0],
                    ["Shirley Temple", 3]
                ]
            }
        ]
    
    def load_2wikimultihopqa(self) -> List[Dict]:
        """Load 2WikiMultihopQA dataset"""
        return [
            {
                "id": "1",
                "question": "What is the total runtime of all movies in The Lord of the Rings trilogy?",
                "answer": "558 minutes",
                "decomposition": [
                    "What are the movies in The Lord of the Rings trilogy?",
                    "What is the runtime of each movie?"
                ]
            }
        ]
    
    def load_cag(self) -> List[Dict]:
        """Load CAG (time-sensitive) dataset"""
        return [
            {
                "id": "1",
                "question": "Who is the current president of the United States?",
                "answer": "Joe Biden",
                "timestamp": "2024-01-01"
            }
        ]
    
    def load_dataset(self, dataset_name: str) -> List[Dict]:
        """Load specified dataset"""
        loaders = {
            "hotpotqa": self.load_hotpotqa,
            "2wikimultihopqa": self.load_2wikimultihopqa,
            "cag": self.load_cag
        }
        
        if dataset_name in loaders:
            return loaders[dataset_name]()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

class DeepRAGEvaluator:
    """Evaluation module for DeepRAG"""
    
    def __init__(self, config: DeepRAGConfig):
        self.config = config
        self.metrics = {}
    
    def exact_match(self, prediction: str, gold: str) -> bool:
        """Compute exact match score"""
        return prediction.strip().lower() == gold.strip().lower()
    
    def f1_score(self, prediction: str, gold: str) -> float:
        """Compute F1 score at token level"""
        pred_tokens = set(prediction.lower().split())
        gold_tokens = set(gold.lower().split())
        
        if not gold_tokens:
            return 1.0 if not pred_tokens else 0.0
        
        if not pred_tokens:
            return 0.0
        
        common = pred_tokens & gold_tokens
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gold_tokens)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)
    
    def evaluate_dataset(self, predictions: List[str], gold_answers: List[str], 
                        retrieval_counts: List[int]) -> Dict[str, float]:
        """Evaluate on a dataset"""
        assert len(predictions) == len(gold_answers) == len(retrieval_counts)
        
        em_scores = [self.exact_match(pred, gold) for pred, gold in zip(predictions, gold_answers)]
        f1_scores = [self.f1_score(pred, gold) for pred, gold in zip(predictions, gold_answers)]
        
        results = {
            "exact_match": sum(em_scores) / len(em_scores),
            "f1_score": sum(f1_scores) / len(f1_scores),
            "avg_retrievals": sum(retrieval_counts) / len(retrieval_counts),
            "total_retrievals": sum(retrieval_counts)
        }
        
        return results
    
    def evaluate_retrieval_efficiency(self, correct_predictions: List[bool], 
                                    retrieval_counts: List[int]) -> Dict[str, float]:
        """Evaluate retrieval efficiency"""
        correct_indices = [i for i, correct in enumerate(correct_predictions) if correct]
        incorrect_indices = [i for i, correct in enumerate(correct_predictions) if not correct]
        
        correct_retrievals = [retrieval_counts[i] for i in correct_indices]
        incorrect_retrievals = [retrieval_counts[i] for i in incorrect_indices]
        
        return {
            "avg_retrievals_correct": sum(correct_retrievals) / len(correct_retrievals) if correct_retrievals else 0,
            "avg_retrievals_incorrect": sum(incorrect_retrievals) / len(incorrect_retrievals) if incorrect_retrievals else 0,
            "efficiency_ratio": (sum(correct_retrievals) / len(correct_retrievals)) / (sum(retrieval_counts) / len(retrieval_counts)) if retrieval_counts else 0
        }

class DeepRAGVisualizer:
    """Visualization module for DeepRAG results"""
    
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    def plot_performance_comparison(self, results: Dict[str, Dict[str, float]], 
                                  save_path: str = None):
        """Plot performance comparison across methods"""
        methods = list(results.keys())
        metrics = ['exact_match', 'f1_score']
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        for i, metric in enumerate(metrics):
            values = [results[method].get(metric, 0) for method in methods]
            bars = axes[i].bar(methods, values, color=self.colors[:len(methods)])
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_ylabel('Score')
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_retrieval_efficiency(self, retrieval_data: Dict[str, List[int]], 
                                save_path: str = None):
        """Plot retrieval efficiency analysis"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Retrieval count distribution
        methods = list(retrieval_data.keys())
        for i, method in enumerate(methods):
            counts = retrieval_data[method]
            ax1.hist(counts, alpha=0.7, label=method, color=self.colors[i], bins=10)
        
        ax1.set_xlabel('Number of Retrievals')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Retrieval Count Distribution')
        ax1.legend()
        
        # Average retrievals comparison
        avg_retrievals = [sum(retrieval_data[method]) / len(retrieval_data[method]) 
                         for method in methods]
        bars = ax2.bar(methods, avg_retrievals, color=self.colors[:len(methods)])
        ax2.set_ylabel('Average Retrievals')
        ax2.set_title('Average Retrieval Count by Method')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars, avg_retrievals):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_training_curves(self, training_logs: Dict[str, List[float]], 
                           save_path: str = None):
        """Plot training curves for different stages"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        # Stage II: Imitation Learning
        if 'stage2_loss' in training_logs:
            axes[0].plot(training_logs['stage2_loss'], color=self.colors[0], linewidth=2)
            axes[0].set_title('Stage II: Imitation Learning Loss')
            axes[0].set_xlabel('Steps')
            axes[0].set_ylabel('Loss')
            axes[0].grid(True, alpha=0.3)
        
        # Stage III: Calibration
        if 'stage3_loss' in training_logs:
            axes[1].plot(training_logs['stage3_loss'], color=self.colors[1], linewidth=2)
            axes[1].set_title('Stage III: Chain of Calibration Loss')
            axes[1].set_xlabel('Steps')
            axes[1].set_ylabel('Loss')
            axes[1].grid(True, alpha=0.3)
        
        # Accuracy progression
        if 'accuracy' in training_logs:
            axes[2].plot(training_logs['accuracy'], color=self.colors[2], linewidth=2)
            axes[2].set_title('Validation Accuracy')
            axes[2].set_xlabel('Epochs')
            axes[2].set_ylabel('Accuracy')
            axes[2].grid(True, alpha=0.3)
        
        # Retrieval efficiency
        if 'avg_retrievals' in training_logs:
            axes[3].plot(training_logs['avg_retrievals'], color=self.colors[3], linewidth=2)
            axes[3].set_title('Average Retrievals per Question')
            axes[3].set_xlabel('Epochs')
            axes[3].set_ylabel('Retrievals')
            axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

class ExperimentRunner:
    """Orchestrates full DeepRAG experiments"""
    
    def __init__(self, config_path: str = None):
        self.config = DeepRAGConfig(config_path)
        self.data_loader = DataLoader(self.config)
        self.evaluator = DeepRAGEvaluator(self.config)
        self.visualizer = DeepRAGVisualizer()
        
    def run_full_experiment(self):
        """Run complete DeepRAG experiment pipeline"""
        print("🚀 Starting DeepRAG Full Experiment Pipeline")
        print("=" * 50)
        
        # Initialize trainer
        from deeprag_pipeline import DeepRAGTrainer
        trainer = DeepRAGTrainer(self.config.get("model.name", "gpt2"))
        
        # Load training data
        print("\n📂 Loading Training Data...")
        train_datasets = self.config.get("datasets.train", ["hotpotqa", "2wikimultihopqa"])
        train_data = []
        
        for dataset_name in train_datasets:
            dataset = self.data_loader.load_dataset(dataset_name)
            for item in dataset:
                train_data.append((item["question"], item["answer"]))
            print(f"   ✓ Loaded {len(dataset)} examples from {dataset_name}")
        
        # Stage I & II: Training
        print(f"\n🏗️  Stage I & II: Binary Tree Search + Imitation Learning")
        training_examples = trainer.train_stage1_and_2(train_data)
        print(f"   ✓ Generated {len(training_examples)} training trajectories")
        
        # Stage III: Calibration
        print(f"\n⚖️  Stage III: Chain of Calibration")
        preference_pairs = trainer.train_stage3(training_examples)
        print(f"   ✓ Generated {len(preference_pairs)} preference pairs")
        
        # Evaluation on test sets
        print(f"\n📊 Evaluation Phase")
        test_datasets = self.config.get("datasets.test", ["cag", "popqa"])
        results = {}
        
        for dataset_name in test_datasets:
            print(f"\n   Testing on {dataset_name}...")
            try:
                test_data = self.data_loader.load_dataset(dataset_name)
                predictions = []
                gold_answers = []
                retrieval_counts = []
                
                for item in test_data[:5]:  # Limit for demo
                    question = item["question"]
                    gold_answer = item["answer"]
                    
                    # Get prediction and retrieval count
                    prediction = trainer.inference(question)
                    # Mock retrieval count - in practice, track from trainer
                    retrieval_count = trainer.llm.model_name.count("retrieve") if hasattr(trainer.llm, 'model_name') else 1
                    
                    predictions.append(prediction)
                    gold_answers.append(gold_answer)
                    retrieval_counts.append(retrieval_count)
                
                # Evaluate
                dataset_results = self.evaluator.evaluate_dataset(
                    predictions, gold_answers, retrieval_counts
                )
                results[dataset_name] = dataset_results
                
                print(f"      EM: {dataset_results['exact_match']:.3f}")
                print(f"      F1: {dataset_results['f1_score']:.3f}")
                print(f"      Avg Retrievals: {dataset_results['avg_retrievals']:.2f}")
                
            except Exception as e:
                print(f"      ❌ Error evaluating {dataset_name}: {e}")
        
        # Generate visualizations
        print(f"\n📈 Generating Visualizations...")
        if results:
            self.visualizer.plot_performance_comparison(results, "performance_comparison.png")
            print("   ✓ Performance comparison plot saved")
        
        # Save results
        self.save_experiment_results(results, training_examples, preference_pairs)
        
        print(f"\n✅ Experiment completed successfully!")
        return results
    
    def save_experiment_results(self, results: Dict, training_examples: List, preference_pairs: List):
        """Save experiment results to files"""
        # Save evaluation results
        with open("experiment_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Save training statistics
        training_stats = {
            "num_training_examples": len(training_examples),
            "num_preference_pairs": len(preference_pairs),
            "avg_trajectory_length": sum(len(ex.optimal_trajectory) for ex in training_examples) / len(training_examples) if training_examples else 0,
            "avg_retrieval_count": sum(ex.retrieval_count for ex in training_examples) / len(training_examples) if training_examples else 0
        }
        
        with open("training_statistics.json", "w") as f:
            json.dump(training_stats, f, indent=2)
        
        print("   ✓ Results saved to experiment_results.json")
        print("   ✓ Training stats saved to training_statistics.json")

def create_sample_config():
    """Create a sample configuration file"""
    config = DeepRAGConfig()
    config.save_config("deeprag_config.yaml")
    print("Sample configuration saved to deeprag_config.yaml")

def run_ablation_study():
    """Run ablation study comparing different components"""
    print("🔬 Running Ablation Study")
    print("=" * 30)
    
    # Test different configurations
    configs = {
        "DeepRAG_Full": {
            "use_binary_tree": True,
            "use_calibration": True,
            "max_retrievals": 3
        },
        "No_Calibration": {
            "use_binary_tree": True,
            "use_calibration": False,
            "max_retrievals": 3
        },
        "No_Binary_Tree": {
            "use_binary_tree": False,
            "use_calibration": True,
            "max_retrievals": 3
        },
        "Baseline_RAG": {
            "use_binary_tree": False,
            "use_calibration": False,
            "max_retrievals": 5
        }
    }
    
    results = {}
    for config_name, config_params in configs.items():
        print(f"\n   Testing {config_name}...")
        # Mock results for demonstration
        results[config_name] = {
            "exact_match": 0.4 + 0.1 * config_params.get("use_binary_tree", 0) + 0.05 * config_params.get("use_calibration", 0),
            "f1_score": 0.5 + 0.1 * config_params.get("use_binary_tree", 0) + 0.05 * config_params.get("use_calibration", 0),
            "avg_retrievals": config_params.get("max_retrievals", 3) * (0.8 if config_params.get("use_calibration", False) else 1.0)
        }
        print(f"      EM: {results[config_name]['exact_match']:.3f}")
        print(f"      F1: {results[config_name]['f1_score']:.3f}")
        print(f"      Avg Retrievals: {results[config_name]['avg_retrievals']:.2f}")
    
    # Visualize ablation results
    visualizer = DeepRAGVisualizer()
    visualizer.plot_performance_comparison(results, "ablation_study.png")
    print("\n   ✓ Ablation study visualization saved")
    
    return results

# Demo and main execution
def main():
    """Main execution function"""
    print("🎯 DeepRAG Implementation Demo")
    print("=" * 40)
    
    # Create sample config
    print("\n1. Creating sample configuration...")
    create_sample_config()
    
    # Run ablation study
    print("\n2. Running ablation study...")
    ablation_results = run_ablation_study()
    
    # Run full experiment
    print("\n3. Running full experiment...")
    runner = ExperimentRunner("deeprag_config.yaml")
    experiment_results = runner.run_full_experiment()
    
    print("\n" + "=" * 40)
    print("🎉 All experiments completed!")
    print(f"📁 Check the following files for results:")
    print(f"   - deeprag_config.yaml (configuration)")
    print(f"   - experiment_results.json (evaluation results)")
    print(f"   - training_statistics.json (training stats)")
    print(f"   - performance_comparison.png (visualization)")
    print(f"   - ablation_study.png (ablation results)")

if __name__ == "__main__":
    main()