# DeepRAG Instructions

## Quick Start Guide

This guide will help you set up and run the DeepRAG framework on your local machine.

## Prerequisites

- Python 3.9 or higher
- Conda (Miniconda or Anaconda)
- Git
- At least 8GB RAM (16GB recommended for larger models)

## Installation

### Option 1: Using Conda Environment (Recommended)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/abhishekmaroon5/DeepRag.git
   cd DeepRag
   ```

2. **Create and activate the conda environment:**
   ```bash
   # Create environment from environment.yml
   conda env create -f environment.yml
   
   # Activate the environment
   conda activate DeepRag
   ```

### Option 2: Using pip

1. **Clone the repository:**
   ```bash
   git clone https://github.com/abhishekmaroon5/DeepRag.git
   cd DeepRag
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Command Line Interface

The main entry point is `run_deeprag.py` which provides four main commands:

```bash
python run_deeprag.py --help
```

### 2. Running a Demo

To see DeepRAG in action with sample data:

```bash
python run_deeprag.py demo
```

This will:
- Initialize the knowledge base with sample documents
- Run the three-stage DeepRAG training pipeline
- Demonstrate inference with example questions

### 3. Single Question Inference

To ask a specific question:

```bash
python run_deeprag.py inference --question "Who directed Peter's Friends?"
```

Advanced options:
```bash
python run_deeprag.py inference \
    --question "What is the runtime of The Lord of the Rings trilogy?" \
    --max-steps 5 \
    --retrieval-limit 3
```

### 4. Training on Custom Data

To train DeepRAG on your own data:

```bash
python run_deeprag.py train \
    --train-file path/to/your/training_data.jsonl \
    --num-epochs 3 \
    --learning-rate 1e-5
```

Expected training data format:
```json
{"question": "What is the capital of France?", "answer": "Paris", "context": "France is a country..."}
{"question": "Who wrote Romeo and Juliet?", "answer": "William Shakespeare", "context": "Shakespeare was..."}
```

### 5. Evaluation

To evaluate on a test set:

```bash
python run_deeprag.py evaluate --test-file path/to/test_data.jsonl
```

## Configuration

### Model Configuration

Edit the model settings in your script or use command-line arguments:

```python
from models import ModelConfig

config = ModelConfig(
    model_name="microsoft/DialoGPT-medium",  # Change model
    quantization="4bit",                     # Use quantization
    max_length=512,                          # Max sequence length
    temperature=0.7                          # Generation temperature
)
```

### Retriever Configuration

Choose between different retrieval strategies:

- **BM25**: Fast sparse retrieval
- **Dense**: Semantic similarity using embeddings
- **Hybrid**: Combination of BM25 and dense retrieval

```python
from retriever import BM25Retriever, DenseRetriever, HybridRetriever

# Use specific retriever
retriever = HybridRetriever(bm25_weight=0.3, dense_weight=0.7)
```

## Project Structure

```
DeepRag/
├── deeprag_improved.py      # Main DeepRAG implementation
├── models.py               # LLM wrapper classes
├── retriever.py           # Retrieval system implementations
├── config_evaluation.py   # Configuration and evaluation utilities
├── run_deeprag.py         # Command-line interface
├── requirements.txt       # Python dependencies
├── environment.yml        # Conda environment file
├── instructions.md        # This file
├── README_IMPLEMENTATION.md # Technical documentation
└── .gitignore            # Git ignore rules
```

## Key Features

### 1. Three-Stage Framework

1. **Stage I: Binary Tree Search** - Strategic retrieval planning
2. **Stage II: Imitation Learning** - Learning from successful trajectories
3. **Stage III: Chain of Calibration** - Preference learning for boundary detection

### 2. Multiple Retrieval Methods

- **BM25**: Traditional keyword-based retrieval
- **Dense**: Neural embedding-based retrieval
- **Hybrid**: Best of both worlds

### 3. Flexible LLM Support

- Mock models for testing
- HuggingFace transformer models
- Quantization support (4-bit, 8-bit)

## Common Use Cases

### Research and Experimentation

```bash
# Run demo to understand the framework
python run_deeprag.py demo

# Test different questions
python run_deeprag.py inference --question "Your research question here"
```

### Production Deployment

```bash
# Train on your domain-specific data
python run_deeprag.py train --train-file domain_data.jsonl --num-epochs 5

# Evaluate performance
python run_deeprag.py evaluate --test-file test_set.jsonl
```

### Development and Testing

```bash
# Quick inference test
python run_deeprag.py inference --question "Test question" --max-steps 2

# Check model performance
python -c "from models import DeepRAGLLM; model = DeepRAGLLM(); print('Model loaded successfully')"
```

## Troubleshooting

### Common Issues

1. **NLTK Download Error:**
   ```bash
   python -c "import nltk; nltk.download('punkt_tab')"
   ```

2. **Memory Issues:**
   - Use quantization: `--quantization 4bit`
   - Reduce batch size in training
   - Use smaller models

3. **CUDA Out of Memory:**
   ```bash
   export CUDA_VISIBLE_DEVICES=""  # Force CPU usage
   ```

4. **Import Errors:**
   ```bash
   # Ensure environment is activated
   conda activate DeepRag
   
   # Reinstall dependencies
   pip install -r requirements.txt --force-reinstall
   ```

### Performance Optimization

1. **For faster inference:**
   - Use quantized models (`4bit` or `8bit`)
   - Reduce `max_steps` parameter
   - Limit retrieval results with `--retrieval-limit`

2. **For better accuracy:**
   - Increase `max_steps` for more reasoning
   - Use larger models
   - Fine-tune on domain-specific data

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8GB | 16GB+ |
| Storage | 5GB | 20GB+ |
| GPU | None (CPU works) | 8GB+ VRAM |

## Environment Variables

Set these for optimal performance:

```bash
# For CPU-only usage
export CUDA_VISIBLE_DEVICES=""

# For limiting GPU memory
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# For debugging
export TRANSFORMERS_VERBOSITY=info
```

## Advanced Configuration

### Custom Knowledge Base

Replace the sample knowledge base with your own:

```python
from retriever import WikipediaKnowledgeBase, Document

# Create custom documents
documents = [
    Document(
        id="doc1",
        title="Custom Document",
        content="Your custom content here...",
        metadata={"source": "custom"}
    )
]

# Initialize knowledge base
kb = WikipediaKnowledgeBase()
kb.documents = documents
```

### Model Customization

```python
from models import DeepRAGLLM, ModelConfig

config = ModelConfig(
    model_name="your-custom-model",
    quantization="4bit",
    device="cuda",  # or "cpu"
    max_length=1024,
    temperature=0.8
)

model = DeepRAGLLM(config)
```

## Support

For issues and questions:

1. Check this instructions file
2. Review `README_IMPLEMENTATION.md` for technical details
3. Check GitHub issues
4. Create a new issue with detailed error information

## Next Steps

1. **Start with the demo**: `python run_deeprag.py demo`
2. **Try your own questions**: `python run_deeprag.py inference --question "Your question"`
3. **Explore the code**: Read through `deeprag_improved.py` to understand the framework
4. **Customize for your use case**: Modify retrievers, models, or knowledge base

Happy experimenting with DeepRAG! 🚀 