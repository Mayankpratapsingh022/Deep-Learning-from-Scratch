# Gemma 3 270M: Small Language Model Implementation from Scratch

A complete PyTorch implementation of Google's Gemma 3 270M small language model, trained from scratch on the TinyStories dataset.

<img width="1920" height="1920" alt="Gemma_3_arc" src="https://github.com/user-attachments/assets/74ef4bcb-83fa-4921-a996-1e7f0d946f61" />


## Overview

This repository contains  implementation of the Gemma 3 270M architecture, featuring modern transformer optimizations including sliding window attention, rotary position embedding (RoPE), and root mean square normalization (RMSNorm). The model achieves efficient performance while maintaining the core capabilities of larger language models.

## Model Architecture

### Core Specifications
- **Parameters**: 270M total (170M embedding + 100M transformer)
- **Layers**: 18 transformer blocks
- **Attention Heads**: 4 query heads, 1 key-value group
- **Hidden Dimension**: 2048
- **Embedding Dimension**: 640
- **Head Dimension**: 256
- **Vocabulary Size**: 50,257 (GPT-2 tokenizer)
- **Context Length**: 32,768 tokens (trained with 128 block size)
- **Sliding Window**: 512 tokens
- **Epochs** : 60,000

[Trained Model Weights on Huggingface](https://huggingface.co/Mayank022/Gemma_3_270M_SLM_from_scratch)

[Gemma_3_270M_from_scratch.pdf](https://github.com/user-attachments/files/21978285/Gemma_3_270M_from_scratch.pdf)



### Key Architectural Features

#### Hybrid Attention Pattern
- **15 Sliding Window Attention layers**: Efficient local attention with 512-token window
- **3 Full Attention layers**: Global context at positions 6, 12, and 18
- Reduces computational complexity from O(n²) to O(n×W) for most layers

#### Advanced Components

- **RoPE (Rotary Position Embedding)**: Preserves token semantics while encoding positional information
- **RMSNorm**: Stable normalization with zero-centered weights and (1+w) scaling
- **Grouped Query Attention**: Memory-efficient attention mechanism
- **Query-Key Normalization**: Optional normalization for improved training stability

## Training Configuration

### Dataset
- **Source**: TinyStories by Roneneldan
- **Tokenizer**: GPT-2 BPE encoding
- **Preprocessing**: Binary memory-mapped files for efficient loading

### Training Hyperparameters
```python
TRAINING_CONFIG = {
    "max_iterations": 60000,
    "batch_size": 32,
    "block_size": 128,
    "learning_rate": 1e-4,
    "warmup_steps": 1000,
    "min_lr": 5e-4,
    "weight_decay": 0.1,
    "gradient_accumulation_steps": 32,
    "gradient_clip_norm": 0.5,
    "eval_interval": 500,
    "dtype": "bfloat16"
}
```

### Optimization Strategy
- **Optimizer**: AdamW with β₁=0.9, β₂=0.95, ε=1e-9
- **Scheduler**: Linear warmup (1000 steps) + Cosine annealing decay
- **Mixed Precision**: bfloat16 training with automatic scaling
- **Regularization**: Weight decay and gradient clipping

## Model Configuration

```python
GEMMA3_CONFIG_270M = {
    "vocab_size": 50257,
    "context_length": 32768,
    "emb_dim": 640,
    "n_heads": 4,
    "n_layers": 18,
    "hidden_dim": 2048,
    "head_dim": 256,
    "qk_norm": True,
    "n_kv_groups": 1,
    "rope_local_base": 10000.0,
    "rope_base": 1000000.0,
    "sliding_window": 512,
    "layer_types": [
        # 15 sliding attention layers + 3 full attention layers
        "sliding_attention", "sliding_attention", "sliding_attention",
        "sliding_attention", "sliding_attention", "full_attention",
        "sliding_attention", "sliding_attention", "sliding_attention",
        "sliding_attention", "sliding_attention", "full_attention",
        "sliding_attention", "sliding_attention", "sliding_attention",
        "sliding_attention", "sliding_attention", "full_attention"
    ],
    "dtype": torch.bfloat16,
    "query_pre_attn_scalar": 256
}
```

## Installation & Usage

### Requirements
```bash
pip install torch datasets tiktoken numpy tqdm matplotlib
```

### Training from Scratch
```python
# Load and tokenize dataset
from datasets import load_dataset
ds = load_dataset("roneneldan/TinyStories")

# Initialize model
model = Gemma3Model(GEMMA3_CONFIG_270M)

# Train for 60,000 iterations
# (See full training loop in the notebook)
```

### Inference
```python
# Load trained model
model = Gemma3Model(GEMMA3_CONFIG_270M)
model.load_state_dict(torch.load("best_model_params.pt"))

# Generate text
sentence = "Once upon a time there was a pumpkin."
context = torch.tensor(enc.encode_ordinary(sentence)).unsqueeze(0)
output = model.generate(context, max_new_tokens=200)
generated_text = enc.decode(output.squeeze().tolist())
```

## Implementation Details

### RoPE Implementation
- Dual frequency bases: 10,000 (local) and 1,000,000 (global)
- Preserves semantic meaning while encoding positional relationships
- Applied to both query and key vectors before attention computation

### Sliding Window Attention
- Causal mask combined with sliding window constraint
- Tokens attend to recent W=512 tokens plus all previous tokens for full attention layers
- Significant memory and computation savings for long sequences

### RMSNorm Features
- Zero-centered weight initialization
- (1 + weight) scaling during forward pass
- Float32 computation with dtype preservation
- Optional bias parameters

## Training Results

The model was successfully trained for 60,000 iterations with:
- Converging training and validation loss curves
- Stable gradient norms with clipping
- Effective learning rate scheduling
- Generated coherent stories in the TinyStories domain

## File Structure

```
├── gemma_3_270_m_slm_from_scratch.py  # Main implementation
├── train.bin                          # Processed training data
├── validation.bin                     # Processed validation data  
├── best_model_params.pt              # Best model checkpoint
└── README.md                         # This file
```

## Key Innovations Implemented

1. **Efficient Architecture**: Hybrid attention pattern balances performance and computational cost
2. **Modern Optimizations**: RoPE, RMSNorm, and grouped query attention
3. **Training Stability**: Careful hyperparameter tuning and gradient management
4. **Memory Efficiency**: Quantization-aware training ready and memory-mapped data loading

## Performance Characteristics

- **Memory Usage**: ~550MB RAM for inference
- **Training Time**: Approximately 60,000 iterations on GPU
- **Generation Speed**: Fast inference suitable for edge deployment
- **Specialization Ready**: Architecture optimized for task-specific fine-tuning
