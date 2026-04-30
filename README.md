# Multi-Modal Few-shot Learning

A comprehensive implementation of multi-modal few-shot learning using CLIP and advanced meta-learning techniques. This project demonstrates how to leverage vision-language models for few-shot classification tasks with minimal labeled data.

## Overview

This project implements several approaches to multi-modal few-shot learning:

- **CLIP-based Few-shot Learning**: Using pre-trained CLIP models for zero-shot and few-shot classification
- **Prototypical Networks**: Learning class prototypes from few examples
- **Meta-Learning**: MAML (Model-Agnostic Meta-Learning) for rapid adaptation
- **Adapter-based Fine-tuning**: Parameter-efficient adaptation for large models

## Features

- Clean, modular codebase with type hints and comprehensive documentation
- Support for multiple few-shot learning algorithms
- Comprehensive evaluation metrics and visualization tools
- Interactive demo with Streamlit/Gradio
- Reproducible experiments with deterministic seeding
- Device-agnostic implementation (CUDA/MPS/CPU)

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Multi-Modal-Few-shot-Learning.git
cd Multi-Modal-Few-shot-Learning

# Install dependencies
pip install -r requirements.txt
# or
pip install -e .
```

### Basic Usage

```python
from src.models.clip_fewshot import CLIPFewShotLearner
from src.data.datasets import ToyDataset

# Initialize the model
model = CLIPFewShotLearner(model_name="openai/clip-vit-base-patch32")

# Load toy dataset
dataset = ToyDataset()

# Train on few-shot examples
model.fit(dataset.get_support_set(n_way=5, k_shot=1))

# Evaluate on query set
accuracy = model.evaluate(dataset.get_query_set())
print(f"Few-shot accuracy: {accuracy:.2%}")
```

### Running the Demo

```bash
# Streamlit demo
streamlit run demo/streamlit_app.py

# Gradio demo
python demo/gradio_app.py
```

## Project Structure

```
├── src/                    # Source code
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # Model implementations
│   ├── losses/            # Loss functions
│   ├── eval/              # Evaluation metrics
│   ├── viz/               # Visualization tools
│   └── utils/             # Utility functions
├── configs/               # Configuration files
│   ├── model/             # Model configurations
│   ├── train/             # Training configurations
│   ├── eval/              # Evaluation configurations
│   └── demo/              # Demo configurations
├── data/                  # Data directory
│   ├── images/            # Image data
│   ├── audio/             # Audio data
│   ├── video/             # Video data
│   └── text/              # Text data
├── scripts/               # Training and evaluation scripts
├── notebooks/             # Jupyter notebooks
├── tests/                 # Unit tests
├── assets/                # Generated assets and results
└── demo/                  # Interactive demos
```

## Models

### CLIP Few-shot Learner
- Pre-trained CLIP model for vision-language understanding
- Support for zero-shot and few-shot classification
- Cosine similarity-based matching

### Prototypical Networks
- Learn class prototypes from support examples
- Euclidean distance-based classification
- Support for multi-modal prototypes

### Meta-Learning (MAML)
- Model-Agnostic Meta-Learning implementation
- Rapid adaptation to new tasks
- Gradient-based meta-optimization

### Adapter-based Fine-tuning
- Parameter-efficient adaptation
- LoRA (Low-Rank Adaptation) support
- Task-specific adapters

## Evaluation Metrics

- **Accuracy**: Standard classification accuracy
- **F1-Score**: Harmonic mean of precision and recall
- **Confidence Intervals**: Bootstrap confidence intervals
- **Ablation Studies**: Component-wise analysis
- **Cross-modal Retrieval**: Image-to-text and text-to-image retrieval

## Configuration

The project uses YAML configuration files for easy experimentation:

```yaml
# configs/model/clip.yaml
model:
  name: "openai/clip-vit-base-patch32"
  device: "auto"
  precision: "fp16"

training:
  batch_size: 32
  learning_rate: 1e-4
  num_epochs: 100
  warmup_steps: 1000

few_shot:
  n_way: 5
  k_shot: 1
  query_size: 15
```

## Training

```bash
# Train CLIP few-shot learner
python scripts/train.py --config configs/train/clip_fewshot.yaml

# Train prototypical networks
python scripts/train.py --config configs/train/prototypical.yaml

# Meta-learning training
python scripts/train.py --config configs/train/maml.yaml
```

## Evaluation

```bash
# Evaluate on test set
python scripts/evaluate.py --config configs/eval/clip_fewshot.yaml

# Run ablation studies
python scripts/ablation.py --config configs/eval/ablation.yaml
```

## Safety and Limitations

**IMPORTANT DISCLAIMERS:**

- This project is for research and educational purposes only
- Models may exhibit biases present in training data
- Results should not be used for critical decision-making without validation
- Generated content should be reviewed for appropriateness
- Privacy considerations apply when processing personal data

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite: `pytest`
6. Format code: `black . && ruff check .`
7. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{multimodal_fewshot_learning,
  title={Multi-Modal Few-shot Learning},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Multi-Modal-Few-shot-Learning}
}
```

## Acknowledgments

- OpenAI for the CLIP model
- Hugging Face for the transformers library
- The broader open-source ML community
# Multi-Modal-Few-shot-Learning
