# PyTorch Baseline Engine

## Overview

**Unoptimized PyTorch baseline** representing typical business inference deployments without optimization. This serves as the performance baseline for comparing all Rust-based inference engines.

## What We're Testing

### Realistic Business Baseline
- **Python Subprocess**: Full Python overhead including GIL and interpreter startup
- **Standard PyTorch**: No optimizations, no kernel fusion, no mixed precision
- **Individual Operations**: Separate operations without batching optimization
- **Cold GPU Context**: Fresh CUDA initialization for each inference

### Performance Characteristics
- **Expected Latency**: 200-400ms per inference
- **Expected Throughput**: 30-50 tokens/second
- **Memory Usage**: Standard PyTorch memory patterns
- **GPU Utilization**: Suboptimal due to Python overhead

## Architecture

```mermaid
graph LR
    A[pytorch-baseline binary] --> B[Python subprocess]
    B --> C[Standard PyTorch Model]
    C --> D[Unoptimized CUDA kernels]
    D --> E[JSON Results]
```

## Implementation

### Unoptimized PyTorch Model
```python
class UnoptimizedTransformer(torch.nn.Module):
    def __init__(self):
        # Standard layers without optimization
        self.embedding = torch.nn.Embedding(32000, 512)
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(512, 512) for _ in range(8)
        ])
        self.output = torch.nn.Linear(512, 32000)
    
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = torch.relu(layer(x))  # Unoptimized activation
            torch.cuda.synchronize()  # Force GPU sync (inefficient)
        return self.output(x)
```

### Key Inefficiencies (Intentional)
- **No Kernel Fusion**: Each operation runs separately
- **Python GIL**: Global Interpreter Lock overhead
- **No KV Caching**: Recompute attention for each token
- **No Batching**: Process one request at a time
- **Cold Starts**: No model warmup or optimization
- **Standard Activation**: ReLU instead of optimized GELU/SiLU
- **Frequent Synchronization**: GPU sync points for "safety"

## Why This Baseline Matters

### Business Reality
Most businesses run inference with:
- Standard PyTorch installations
- No specialized optimization
- Python-based serving infrastructure
- Generic CUDA operations

### Fair Comparison
This baseline shows the **actual performance improvement** that businesses can expect when migrating from typical PyTorch setups to optimized Rust inference engines.

## Expected Results vs Optimized Engines

| Engine | Expected Speedup | Latency Improvement |
|--------|------------------|-------------------|
| lm.rs | 3-4x faster | 300ms → 100ms |
| Candle | 5-8x faster | 300ms → 50ms |
| Burn | 8-12x faster | 300ms → 30ms |
| Mistral.rs | 6-10x faster | 300ms → 40ms |

## Running

```bash
# Build baseline
cargo build --release --bin pytorch-baseline

# Run benchmark
echo '{"prompt":"Hello","iterations":50}' | ./target/release/pytorch-baseline
```

This baseline provides the **business case** for investing in Rust-based inference optimization.