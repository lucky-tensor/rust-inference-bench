# Getting Started Guide

## Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (RTX 4090 or A100 recommended)
- **VRAM**: Minimum 8GB, recommended 24GB+
- **CPU**: 16+ cores for baseline comparisons
- **RAM**: 32GB+ system memory
- **Storage**: 50GB+ free space for models and results

### Software Requirements
- **Rust**: Latest stable version (1.70+)
- **CUDA Toolkit**: Version 12.0+
- **cuDNN**: Version 8.6+
- **Python**: 3.9+ (for PyTorch baseline)
- **Git LFS**: For downloading model files

## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/your-org/cuda-inference-benchmarks.git
cd cuda-inference-benchmarks
```

### 2. Environment Setup
```bash
# Install Rust dependencies
cargo check

# Setup Python environment for baseline
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA installation
nvidia-smi
nvcc --version
```

### 3. Download Models
```bash
# Download Llama 3.2 1B model (primary test model)
mkdir -p models
cd models
git lfs clone https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
cd ..
```

### 4. Run Quick Test
```bash
# Run minimal benchmark to verify setup
cargo run --bin quick-benchmark -- \
  --config configs/quick-test.yaml \
  --output ./results/quick-test
```

Expected output:
```
🚀 CUDA Inference Benchmark
===========================
✅ Hardware: NVIDIA RTX 4090 (24GB)
✅ CUDA: 12.1
✅ Model: Llama-3.2-1B-Instruct loaded

📊 Quick Benchmark Results:
  🥇 burn_cuda     :   32.1ms |  312.0 tok/s (9.4x vs PyTorch)
  🥈 candle_cuda   :   48.7ms |  205.3 tok/s (6.2x vs PyTorch) 
  🥉 lm_rs         :  101.2ms |   98.8 tok/s (3.0x vs PyTorch)
  🏁 pytorch_base  :  301.5ms |   33.2 tok/s (1.0x baseline)

⚡ Best speedup: 9.4x (burn vs pytorch_baseline)
💾 Results saved to: ./results/quick-test/
```

## Configuration

### Basic Configuration File
```yaml
# configs/basic-benchmark.yaml
benchmark:
  name: "Basic Inference Comparison"
  output_dir: "./results/basic"
  model_path: "./models/Llama-3.2-1B-Instruct"

engines:
  - name: "pytorch_baseline"
    type: "pytorch"
    enabled: true
    config:
      device: "cuda:0"
      optimization: "none"
  
  - name: "lm_rs"
    type: "lm_rs"  
    enabled: true
    config:
      threads: 16
      
  - name: "candle_cuda"
    type: "candle"
    enabled: true
    config:
      device: "cuda:0"
      precision: "f16"

workloads:
  - name: "interactive"
    description: "Single request latency test"
    batch_sizes: [1]
    sequence_lengths: [10, 50, 100]
    iterations: 50
    warmup_iterations: 5
    
  - name: "throughput"
    description: "Batch processing test"  
    batch_sizes: [8, 16, 32]
    sequence_lengths: [50]
    iterations: 20
    warmup_iterations: 3

analysis:
  confidence_level: 0.95
  remove_outliers: true
  generate_plots: true
```

### Environment Variables
```bash
# Optional configuration via environment
export CUDA_DEVICE=0
export MAX_MEMORY_GB=20
export OUTPUT_DIR="./results"
export MODEL_CACHE_DIR="./models"
export LOG_LEVEL="info"
```

## Running Benchmarks

### Full Benchmark Suite
```bash
# Run comprehensive comparison (30-60 minutes)
cargo run --release --bin benchmark-suite -- \
  --config configs/comprehensive.yaml \
  --threads 8 \
  --verbose
```

### Individual Engine Testing
```bash
# Test single engine
cargo run --bin single-engine -- \
  --engine candle_cuda \
  --prompt "Hello, how are you?" \
  --iterations 100
```

### Custom Workload
```bash
# Create custom test configuration
cargo run --bin benchmark-suite -- \
  --config configs/custom.yaml \
  --engines "candle_cuda,burn_cuda" \
  --workloads "interactive,throughput" \
  --output ./results/custom
```

## Interpreting Results

### Performance Report Structure
```
results/
├── summary.json           # Machine-readable results
├── report.html           # Interactive dashboard  
├── raw_data/            
│   ├── timings.csv       # Raw timing measurements
│   ├── gpu_metrics.csv   # GPU utilization data
│   └── memory_usage.csv  # Memory consumption
├── plots/
│   ├── performance_comparison.png
│   ├── latency_distribution.png  
│   └── throughput_scaling.png
└── logs/
    └── benchmark.log     # Detailed execution log
```

### Key Metrics Explained

**Timing Statistics:**
- **Mean**: Average inference time
- **Median (P50)**: Middle value, less affected by outliers
- **P90/P99**: 90th/99th percentile latency (tail latency)
- **Std Dev**: Consistency measure (lower is better)

**Throughput Metrics:**
- **Tokens/Second**: Raw generation speed
- **Requests/Second**: Concurrent request handling capacity
- **GPU Utilization**: Percentage of compute capacity used

**Efficiency Metrics:**
- **Speedup**: Performance vs baseline (higher is better)
- **Memory Efficiency**: VRAM usage per token
- **Cost Efficiency**: Performance per dollar of hardware

### Example Analysis
```json
{
  "summary": {
    "fastest_engine": "burn_cuda",
    "best_speedup": 9.4,
    "most_efficient": "candle_cuda",
    "recommendation": "burn_cuda for throughput, candle_cuda for memory-constrained"
  },
  "detailed_results": {
    "burn_cuda": {
      "mean_latency_ms": 32.1,
      "throughput_tok_s": 312.0,
      "gpu_utilization": 87.3,
      "vram_usage_gb": 4.2
    }
  }
}
```

## Troubleshooting

### Common Issues

**CUDA Out of Memory:**
```bash
# Reduce batch size or use smaller model
export MAX_BATCH_SIZE=8
cargo run --bin benchmark-suite -- --config configs/small-memory.yaml
```

**Engine Compilation Errors:**
```bash
# Clean and rebuild specific engine
cargo clean -p candle-engine
cargo build -p candle-engine --features cuda
```

**Model Loading Failures:**
```bash
# Verify model files
ls -la models/Llama-3.2-1B-Instruct/
# Re-download if corrupted
rm -rf models/Llama-3.2-1B-Instruct/
git lfs clone https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct models/
```

**Performance Issues:**
```bash
# Check GPU state
nvidia-smi
# Ensure GPU clocks are not throttled
sudo nvidia-smi -pm 1  # Enable persistence mode
sudo nvidia-smi -lgc 1500,1500  # Lock GPU clocks
```

### Debug Mode
```bash
# Run with detailed logging
RUST_LOG=debug cargo run --bin benchmark-suite -- \
  --config configs/debug.yaml \
  --verbose
```

### Validation
```bash
# Verify benchmark accuracy
cargo test --release
cargo run --bin validate-results -- --results ./results/latest/
```

## Next Steps

1. **Customize Configuration**: Modify `configs/` files for your specific use case
2. **Add Custom Engines**: Implement additional inference libraries
3. **Extend Analysis**: Add custom metrics and analysis plugins
4. **Production Deployment**: Use results to guide production inference setup

For detailed implementation guides, see:
- [Architecture Documentation](architecture.md)
- [Code Conventions](code-conventions.md)
- [Test Methodology](test-methodology.md)