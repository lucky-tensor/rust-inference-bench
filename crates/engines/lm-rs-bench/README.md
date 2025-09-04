# lm.rs Benchmark Engine

## Overview

**Highly optimized CPU-only inference** using the lm.rs approach. This represents the best possible CPU performance without GPU acceleration, serving as a bridge between unoptimized PyTorch and GPU-accelerated solutions.

## What We're Testing

### CPU Optimization Excellence
- **Zero Dependencies**: No external ML frameworks or libraries
- **SIMD Optimization**: Hand-optimized vectorized operations
- **Memory Efficiency**: Direct memory mapping and efficient data structures
- **Parallel Processing**: Multi-core CPU utilization with rayon
- **Rust Performance**: Zero-cost abstractions and compile-time optimization

### Performance Characteristics
- **Expected Latency**: 80-120ms per inference
- **Expected Throughput**: 80-120 tokens/second
- **Memory Usage**: ~1GB for Llama 3.2 1B model
- **CPU Utilization**: High multi-core utilization

## Architecture

```mermaid
graph LR
    A[lm-rs-bench binary] --> B[Memory-mapped model]
    B --> C[SIMD operations]
    C --> D[Multi-core parallel processing]
    D --> E[Optimized matrix operations]
    E --> F[JSON Results]
```

## Implementation Approach

### Direct Model Implementation
```rust
pub struct LlamaModel {
    config: ModelConfig,
    weights: MemoryMappedWeights,
    tokenizer: FastTokenizer,
}

impl LlamaModel {
    fn forward(&self, tokens: &[u32]) -> Vec<f32> {
        // Direct implementation without framework overhead
        let embeddings = self.embed_tokens(tokens);
        let mut hidden = embeddings;
        
        for layer in &self.layers {
            // Hand-optimized attention and FFN
            hidden = layer.forward_optimized(&hidden);
        }
        
        self.output_projection(&hidden)
    }
}
```

### Key Optimizations
- **Memory Mapping**: Direct file access without loading into RAM
- **SIMD Instructions**: Vectorized math operations (AVX, AVX2)
- **Cache Optimization**: Data layout optimized for CPU cache hierarchy
- **Parallel Execution**: Multi-threaded matrix operations
- **Quantization**: 8-bit quantization for memory efficiency
- **No Heap Allocation**: Stack-allocated operations where possible

## Why This Matters

### CPU Performance Ceiling
lm.rs represents the **theoretical maximum CPU performance** for LLM inference, showing what's possible without GPU acceleration.

### Migration Path
Many businesses start with CPU inference and need to understand:
- When GPU acceleration becomes worth the complexity
- Performance gains available before investing in GPU infrastructure
- Cost-efficiency of CPU vs GPU for different workloads

### Optimization Techniques
Demonstrates advanced Rust optimization techniques:
- Zero-copy operations
- SIMD vectorization  
- Memory layout optimization
- Parallel processing patterns

## Expected Performance vs Other Engines

| Comparison | Performance | Reasoning |
|------------|-------------|-----------|
| vs PyTorch Baseline | **3-4x faster** | No Python overhead, optimized operations |
| vs Candle GPU | **2-3x slower** | CPU limited, but shows optimization value |
| vs Burn GPU | **4-6x slower** | Demonstrates GPU acceleration benefit |
| vs Mistral.rs GPU | **3-5x slower** | Shows specialized GPU optimization gains |

## Technical Highlights

### Memory Efficiency
```rust
// Memory-mapped model weights (no RAM loading)
let weights = unsafe {
    memmap2::Mmap::map(&file)?
};

// Zero-copy tokenization
let tokens = self.tokenizer.encode_fast(&prompt);
```

### SIMD Optimization
```rust
// Hand-optimized matrix multiplication with SIMD
fn matmul_simd(a: &[f32], b: &[f32], c: &mut [f32]) {
    use std::arch::x86_64::*;
    // AVX2 vectorized operations
    unsafe {
        // 8 floats per vector
        for i in (0..a.len()).step_by(8) {
            let va = _mm256_load_ps(&a[i]);
            let vb = _mm256_load_ps(&b[i]);
            let vc = _mm256_mul_ps(va, vb);
            _mm256_store_ps(&mut c[i], vc);
        }
    }
}
```

## Running

```bash
# Build optimized CPU engine (native optimization enabled by default)
cargo build --release --bin lm-rs-bench

# Execute benchmark
echo '{"prompt":"Hello","iterations":100}' | ./target/release/lm-rs-bench

# Manual native optimization (if needed)
RUSTFLAGS="-C target-cpu=native -C target-feature=+avx2,+fma" \
  cargo build --release --bin lm-rs-bench
```

## Business Value

### Cost Analysis
- **No GPU Required**: Runs on standard server hardware
- **High Density**: Multiple models per server
- **Predictable Costs**: No GPU premium, standard CPU pricing
- **Easy Deployment**: No CUDA dependencies or driver complexity

### Use Cases
- **Development/Testing**: Fast iteration without GPU setup
- **Edge Deployment**: Resource-constrained environments  
- **Cost-Sensitive**: Workloads where GPU cost isn't justified
- **Regulatory**: Environments where GPU usage is restricted

This benchmark shows the **peak performance achievable on CPU** and provides the economic comparison point for GPU acceleration investment.