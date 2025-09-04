# Candle Benchmark Engine

## Overview

**Candle CUDA inference** testing Hugging Face's minimalist Rust ML framework. This benchmark evaluates Candle's GPU performance against our baseline and other engines, focusing on production-ready PyTorch-like functionality in Rust.

## What We're Testing

### Candle Framework Capabilities
- **CUDA Backend**: GPU acceleration with cuDNN optimization
- **PyTorch-like API**: Familiar tensor operations and model definitions
- **Production Focus**: Designed for serverless and edge inference
- **Multi-Backend**: CUDA, Metal, CPU support (testing CUDA)
- **Model Compatibility**: Direct loading of PyTorch and SafeTensors models

### Performance Characteristics
- **Expected Latency**: 40-60ms per inference
- **Expected Throughput**: 150-250 tokens/second
- **Memory Usage**: Efficient VRAM utilization
- **GPU Utilization**: cuDNN optimized operations

## Architecture

```mermaid
graph LR
    A[candle-bench binary] --> B[Candle Framework]
    B --> C[CUDA Backend]
    C --> D[cuDNN Operations]
    D --> E[Tensor Core Usage]
    E --> F[JSON Results]
```

## Implementation Focus

### Candle Model Implementation
```rust
use candle_core::{Device, Tensor};
use candle_transformers::models::llama::LlamaModel;

pub struct CandleEngine {
    model: LlamaModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl CandleEngine {
    pub async fn infer(&self, prompt: &str) -> Result<String> {
        // Tokenize input
        let tokens = self.tokenizer.encode(prompt, false)?;
        let input_ids = Tensor::new(&tokens[..], &self.device)?;
        
        // GPU inference with Candle
        let logits = self.model.forward(&input_ids, 0)?;
        
        // Decode response
        self.decode_tokens(&logits)
    }
}
```

### Key Features Being Tested
- **GPU Memory Management**: Efficient VRAM allocation and deallocation
- **cuDNN Integration**: Optimized convolution and tensor operations
- **Tensor Operations**: Matrix multiplication, attention, activation functions
- **Model Loading**: Direct PyTorch/SafeTensors compatibility
- **Batch Processing**: Variable batch size handling
- **Mixed Precision**: FP16 support for memory efficiency

## What Makes This Interesting

### HuggingFace Rust Strategy
Candle represents HuggingFace's vision for Rust-based ML inference:
- **Minimal Dependencies**: Lightweight compared to full PyTorch
- **Cross-Platform**: Same code works on CUDA, Metal, CPU
- **Serverless Friendly**: Fast cold starts and small binaries
- **Production Ready**: Designed for actual deployment scenarios

### Current Performance Reality
Based on existing benchmarks, Candle faces performance challenges:
- **4-10x slower than PyTorch** in some GPU workloads (YOLOv8, RealESRGAN)
- **GPU kernel optimization** still maturing
- **Room for improvement** in CUDA utilization

## Benchmark Goals

### Performance Analysis
1. **Raw Speed**: Tokens/second vs PyTorch baseline
2. **GPU Utilization**: How well it uses available compute
3. **Memory Efficiency**: VRAM usage patterns
4. **Scaling**: Performance with different batch sizes
5. **Consistency**: Variance in inference times

### Business Viability Assessment
- **Deployment Benefits**: Binary size, dependencies, cold start
- **Performance Trade-offs**: Speed vs operational simplicity
- **Cost Efficiency**: Performance per dollar of hardware
- **Migration Path**: Effort required to switch from PyTorch

## Expected Results

### Performance Predictions
```
PyTorch Baseline: 300ms | 40 tok/s  (1.0x baseline)
Candle CUDA:      60ms  | 200 tok/s (5.0x speedup)
```

### Key Questions
- Can Candle overcome the 4-10x GPU performance gap seen in other models?
- Is the operational simplicity worth potential performance trade-offs?
- How does memory usage compare to optimized PyTorch?
- What's the actual GPU utilization percentage?

## Technical Specifications

### CUDA Features Used
- **cuDNN Backend**: Optimized deep learning operations
- **Tensor Operations**: Matrix multiplication, attention mechanisms
- **Memory Management**: GPU buffer allocation and reuse
- **Multi-precision**: FP32/FP16 support

### Model Architecture
- **Target Model**: Llama 3.2 1B Instruct
- **Quantization**: Testing both FP16 and FP32 precision
- **Sequence Length**: Variable length input/output testing
- **Batch Processing**: Single and batched inference comparison

## Running

```bash
# Build with CUDA support
cargo build --release --bin candle-bench --features cuda

# Run benchmark
echo '{"prompt":"Hello","iterations":50,"model_path":"./models/llama-3.2-1b"}' | \
    ./target/release/candle-bench
```

## Success Criteria

### Performance Targets
- **Baseline Improvement**: >5x faster than PyTorch baseline
- **GPU Utilization**: >60% of available compute
- **Memory Efficiency**: Reasonable VRAM usage for model size
- **Consistency**: <20% variance in inference times

### Business Value
- **Deployment Simplicity**: Demonstrate operational benefits
- **Cost Justification**: Performance improvement vs complexity
- **Migration Feasibility**: Practical path from PyTorch to Candle

This benchmark will provide real-world performance data on Candle's viability for production GPU inference workloads.