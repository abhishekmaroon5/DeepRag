# DeepRAG Implementation

This repository contains a complete implementation of **DeepRAG: Thinking to Retrieve Step by Step for Large Language Models** based on the research paper ([arXiv:2502.01142](https://arxiv.org/pdf/2502.01142)).

## Overview

DeepRAG presents a sophisticated framework for improving Retrieval-Augmented Generation (RAG) systems by modeling the retrieval process as a strategic decision-making problem. The key innovation is teaching models to make intelligent retrieval decisions at each reasoning step.

## Architecture

### Three-Stage Framework

1. **Stage I: Binary Tree Search**
   - Creates a decision tree for each question where each node represents a choice: use internal knowledge (parametric) or retrieve external information
   - Systematically explores all possible retrieval strategies to find the optimal path
   - Generates training data that shows the model effective retrieval patterns

2. **Stage II: Imitation Learning**
   - Uses optimal paths from Stage I to train the model on effective query decomposition
   - Teaches the model to break complex questions into atomic subqueries
   - Shows the model how to generate faithful intermediate answers

3. **Stage III: Chain of Calibration**
   - Creates preference pairs comparing good vs. poor retrieval decisions
   - Fine-tunes the model to better understand its own knowledge boundaries
   - Calibrates when the model should trust its internal knowledge vs. seek external information

## Implementation Components

### Core Modules

- **`deeprag_improved.py`** - Main implementation with all three stages
- **`retriever.py`** - Real retriever implementations (BM25, Dense, Hybrid)
- **`models.py`** - LLM wrapper with actual transformer models
- **`run_deeprag.py`** - Command-line interface
- **`config_evaluation.py`** - Configuration and evaluation utilities
- **`pipeline.py`** - Original pipeline implementation
- **`requirements.txt`** - Dependencies

### Key Features

✅ **Complete Three-Stage Pipeline**
- Binary Tree Search with priority queue optimization
- Imitation Learning with trajectory formatting
- Chain of Calibration with preference pair generation

✅ **Multiple Retriever Types**
- BM25 (sparse retrieval)
- Dense retrieval with sentence transformers
- Hybrid retrieval combining both approaches

✅ **Flexible LLM Support**
- Mock LLM for testing and demonstration
- Real transformer models with HuggingFace integration
- Support for quantization (4-bit, 8-bit)

✅ **Comprehensive Evaluation**
- Accuracy metrics
- Retrieval efficiency analysis
- Reasoning trace visualization

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd DeepRag

# Install dependencies
pip install -r requirements.txt

# For development with real models, also install:
pip install bitsandbytes  # For quantization
```

## Quick Start

### 1. Run Demo

```bash
python run_deeprag.py demo
```

This runs a complete demonstration of the DeepRAG pipeline with sample data.

### 2. Single Question Inference

```bash
python run_deeprag.py inference --question "What is the place of birth of the director of film Peter's Friends?"
```

### 3. Train on Custom Data

```bash
python run_deeprag.py train --data training_data.json --model mock --retriever hybrid
```

Training data format:
```json
[
  {
    "question": "What is the place of birth of the director of film Peter's Friends?",
    "answer": "Belfast, Northern Ireland"
  },
  {
    "question": "What is the total runtime of all movies in The Lord of the Rings trilogy?",
    "answer": "558 minutes"
  }
]
```

### 4. Evaluate on Test Set

```bash
python run_deeprag.py evaluate --test-data test_data.json --output evaluation_results.json
```

## Usage Examples

### Python API

```python
from deeprag_improved import DeepRAGTrainer

# Initialize trainer
trainer = DeepRAGTrainer(model_name="mock", retriever_type="hybrid")

# Train the model
train_data = [
    ("What is the place of birth of the director of film Peter's Friends?", "Belfast, Northern Ireland"),
    ("What is the total runtime of all movies in The Lord of the Rings trilogy?", "558 minutes")
]

training_examples = trainer.train_stage1_and_2(train_data)
preference_pairs = trainer.train_stage3(training_examples)

# Run inference
result = trainer.inference("What is the place of birth of the director of film Peter's Friends?")
print(f"Answer: {result['final_answer']}")
print(f"Retrievals: {result['total_retrievals']}")
```

### Command Line Interface

```bash
# Train with different configurations
python run_deeprag.py train --model mock --retriever bm25
python run_deeprag.py train --model mock --retriever dense
python run_deeprag.py train --model mock --retriever hybrid

# Inference with different models
python run_deeprag.py inference --question "Your question here" --model mock
python run_deeprag.py inference --question "Your question here" --model gpt2 --max-steps 3

# Evaluation
python run_deeprag.py evaluate --test-data test.json --output results.json
```

## Configuration

### Model Configuration

```python
from models import ModelConfig, create_llm

# Configure a real model
config = ModelConfig(
    model_name="microsoft/DialoGPT-medium",
    max_length=512,
    temperature=0.7,
    load_in_4bit=True  # For memory efficiency
)

llm = create_llm(config.model_name, **config.__dict__)
```

### Retriever Configuration

```python
from retriever import WikipediaKnowledgeBase

# Different retriever types
kb_bm25 = WikipediaKnowledgeBase("bm25")
kb_dense = WikipediaKnowledgeBase("dense") 
kb_hybrid = WikipediaKnowledgeBase("hybrid")

# Load custom knowledge base
kb.load_documents_from_file("custom_knowledge.json")
```

## Results and Analysis

The implementation provides detailed analysis including:

- **Accuracy Metrics**: Exact match and F1 scores
- **Retrieval Efficiency**: Average retrievals per question
- **Knowledge Boundary Analysis**: Correlation between retrieval decisions and parametric knowledge
- **Reasoning Traces**: Step-by-step decision visualization

### Sample Output

```
============================================================
Question: What is the place of birth of the director of film Peter's Friends?
============================================================
Final Answer: Belfast, Northern Ireland
Total Retrievals: 2

Reasoning Trace:
  Step 1: Who is the director of the film mentioned?
    → [retrieve] Kenneth Branagh
  Step 2: What is the birthplace of this director?
    → [retrieve] Belfast, Northern Ireland
  Final: Belfast, Northern Ireland
```

## Key Results from Paper

According to the original paper, DeepRAG achieves:

- **26.4% improvement** in answer accuracy over baseline methods
- **21.99% increase** in accuracy while enhancing retrieval efficiency
- **Superior correlation** between retrieval decisions and parametric knowledge
- **Consistent improvements** across in-distribution, out-of-distribution, and time-sensitive datasets

## Technical Details

### Markov Decision Process Formulation

The framework models retrieval-augmented reasoning as an MDP where:

- **States**: Current partial solution to the question
- **Actions**: (1) Continue/terminate reasoning, (2) Retrieve/use parametric knowledge  
- **Rewards**: Prioritize correctness while minimizing retrieval costs

### Training Stages

1. **Binary Tree Search**: Explores all possible paths and finds optimal trajectory with minimal retrieval cost
2. **Imitation Learning**: Trains model to replicate optimal reasoning patterns
3. **Chain of Calibration**: Uses preference learning to calibrate knowledge boundaries

### Evaluation Metrics

- **Accuracy**: Exact match and F1 scores
- **Efficiency**: Average number of retrievals per question
- **Correlation**: Matthews Correlation Coefficient between retrieval decisions and parametric knowledge

## Extending the Implementation

### Adding New Retrievers

```python
from retriever import BaseRetriever

class CustomRetriever(BaseRetriever):
    def retrieve(self, query: str, top_k: int = 5):
        # Implement custom retrieval logic
        pass
    
    def index_documents(self, documents):
        # Implement indexing logic
        pass
```

### Custom LLM Models

```python
from models import BaseLLM

class CustomLLM(BaseLLM):
    def generate_subquery(self, question: str, context: str) -> str:
        # Implement subquery generation
        pass
    
    # Implement other required methods...
```

### Custom Evaluation Metrics

```python
def custom_evaluation_metric(predicted: str, expected: str) -> float:
    # Implement custom evaluation logic
    pass
```

## Troubleshooting

### Common Issues

1. **Memory Issues with Large Models**
   - Use quantization: `load_in_4bit=True` or `load_in_8bit=True`
   - Reduce batch size and sequence length

2. **Slow Retrieval**
   - Use BM25 instead of dense retrieval for speed
   - Reduce knowledge base size
   - Index documents in advance

3. **Poor Answer Quality**
   - Increase max_steps for more reasoning
   - Use better base models
   - Improve knowledge base quality

### Performance Optimization

- Use GPU acceleration for dense retrieval
- Cache embeddings and indices
- Use quantized models for inference
- Batch process multiple questions

## Future Work

- [ ] Integration with larger foundation models (GPT-4, Claude, etc.)
- [ ] Real-time knowledge base updates
- [ ] Multi-modal retrieval support
- [ ] Advanced evaluation metrics
- [ ] Distributed training support

## References

- [DeepRAG Paper](https://arxiv.org/pdf/2502.01142)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [Sentence Transformers](https://www.sbert.net/)
- [FAISS](https://github.com/facebookresearch/faiss)

## License

This implementation is provided for research and educational purposes. Please refer to the original paper for citation requirements.

## Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@article{guan2025deeprag,
  title={DeepRAG: Thinking to Retrieve Step by Step for Large Language Models},
  author={Guan, Xinyan and Zeng, Jiali and Meng, Fandong and Xin, Chunlei and Lu, Yaojie and Lin, Hongyu and Han, Xianpei and Sun, Le and Zhou, Jie},
  journal={arXiv preprint arXiv:2502.01142},
  year={2025}
}
``` 