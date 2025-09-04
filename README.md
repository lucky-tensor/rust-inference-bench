# CUDA Inference Engine Benchmarks

## Overview

Comprehensive benchmarking suite comparing CUDA kernel performance across different Rust inference libraries against an unoptimized PyTorch baseline. Designed to provide **business-relevant performance data** for inference engine selection and GPU optimization investment decisions.

## 🎯 Project Goals

### Business-Focused Benchmarking
- **Realistic Baseline**: Unoptimized PyTorch representing typical business deployments
- **GPU Utilization Focus**: Sustained throughput vs latency trade-offs for cost optimization
- **Production Readiness**: Real-world deployment scenarios and constraints
- **Investment Justification**: Clear ROI data for GPU acceleration and Rust adoption

### Technical Excellence
- **Process Isolation**: Each engine runs independently to ensure fair comparison
- **Statistical Rigor**: 95% confidence intervals, outlier detection, significance testing  
- **Hardware Monitoring**: GPU utilization, memory usage, power consumption tracking
- **Reproducible Results**: Version-locked dependencies and controlled environment

## 🏗️ Architecture

### Mono-Repo Structure
```
cuda-inference-benchmarks/
├── crates/
│   ├── shared-protocol/         # Minimal communication types
│   ├── benchmark-runner/        # Orchestration and analysis
│   └── engines/
│       ├── pytorch-baseline/    # Unoptimized PyTorch reference
│       ├── lm-rs-bench/        # CPU-optimized Rust inference  
│       ├── candle-bench/       # Candle CUDA framework
│       ├── burn-bench/         # Burn CubeCL advanced optimization
│       └── mistral-rs-bench/   # Specialized LLM inference
├── docs/                       # Comprehensive documentation
├── configs/                    # Benchmark configurations
├── models/                     # Shared model files
└── results/                    # Benchmark outputs
```

### Process Isolation Benefits
- **Fair Comparison**: Each engine only includes its specific dependencies
- **Clean GPU State**: Fresh CUDA context for each benchmark
- **Independent Compilation**: Build specific engines without dependency conflicts
- **Reproducible Results**: Process boundaries prevent cross-contamination

## 🚀 Quick Start

### Prerequisites
```bash
# Hardware Requirements
# - NVIDIA GPU with 8GB+ VRAM (RTX 4090/A100 recommended)
# - 16+ CPU cores, 32GB+ RAM
# - CUDA 12.0+, cuDNN 8.6+

# Software Requirements
# - Rust 1.70+
# - Python 3.9+ (for PyTorch baseline)
# - Git LFS
```

### Setup
```bash
# Clone repository
git clone <repository-url>
cd cuda-inference-benchmarks

# Download models
mkdir models && cd models
git lfs clone https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct

# Setup Python environment
python -m venv venv && source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Build all engines with native CPU optimization
cargo build --release
```

### Quick Test
```bash
cargo run --bin benchmark-runner -- \
  --config configs/quick-test.yaml \
  --output ./results/quick-test
```

Expected output:
```
🚀 CUDA Inference Benchmark Results
===================================
🥇 burn_cuda     :   32.1ms |  312.0 tok/s (9.4x vs PyTorch)
🥈 mistral_rs    :   41.7ms |  267.3 tok/s (7.2x vs PyTorch)  
🥉 candle_cuda   :   52.1ms |  198.6 tok/s (5.7x vs PyTorch)
   lm_rs         :   95.3ms |   89.4 tok/s (3.1x vs PyTorch)
🏁 pytorch_base  :  298.7ms |   41.2 tok/s (1.0x baseline)
```

## 📊 Benchmark Engines

### [PyTorch Baseline](crates/engines/pytorch-baseline/README.md)
**Unoptimized reference** - Represents typical business PyTorch deployments
- Python subprocess overhead + GIL
- No kernel fusion or optimization  
- Standard PyTorch operations
- Expected: 250-400ms, 30-50 tok/s

### [lm.rs](crates/engines/lm-rs-bench/README.md) 
**CPU optimization excellence** - Best possible CPU-only performance
- Zero dependencies, SIMD optimization
- Memory mapping and parallel processing
- Expected: 80-120ms, 80-120 tok/s

### [Candle](crates/engines/candle-bench/README.md)
**HuggingFace Rust ML** - PyTorch-like API with CUDA acceleration  
- cuDNN optimized operations
- Multi-backend support
- Expected: 40-60ms, 150-250 tok/s

### [Burn](crates/engines/burn-bench/README.md)
**Next-gen GPU optimization** - Advanced CubeCL backend with tensor cores
- Runtime kernel optimization
- Hardware-specific acceleration
- Expected: 25-40ms, 250-400 tok/s

### [Mistral.rs](crates/engines/mistral-rs-bench/README.md)
**Specialized LLM engine** - Purpose-built transformer inference
- LLM-native optimizations
- Production async API
- Expected: 35-50ms, 200-300 tok/s

## 📈 Business Intelligence

### Performance Analysis
- **Speedup Calculations**: Performance multiplier vs PyTorch baseline
- **Cost Efficiency**: Performance per dollar of GPU hardware
- **Resource Utilization**: GPU/CPU/memory usage patterns
- **Scaling Characteristics**: Performance under different batch sizes

### Deployment Recommendations
- **Performance Tiers**: Matching engines to business requirements
- **Migration Strategy**: Transition paths from existing infrastructure  
- **Risk Assessment**: Framework maturity and stability analysis
- **Investment Justification**: ROI calculations for GPU acceleration

## 🔧 Configuration

### Basic Benchmark
```yaml
benchmark:
  name: "GPU Inference Comparison"
  model_path: "./models/Llama-3.2-1B-Instruct"

engines: ["pytorch_baseline", "candle_cuda", "burn_cuda"]

workloads:
  - name: "interactive"
    batch_sizes: [1]
    sequence_lengths: [10, 50, 100]
    iterations: 50
```

### Advanced Configuration
```yaml
analysis:
  confidence_level: 0.95
  remove_outliers: true
  generate_plots: true
  statistical_tests: ["mann_whitney_u", "welch_t_test"]

system:
  gpu_device: 0
  thermal_limit: 80
  memory_limit_gb: 20
```

## 📋 Test Scenarios

### 1. Latency Testing
Single request response time measurement
- Interactive chat scenarios
- Real-time applications
- User experience optimization

### 2. Throughput Testing  
Sustained inference performance
- Batch processing workloads
- Server capacity planning
- Cost optimization analysis

### 3. Concurrent Load Testing
Multiple simultaneous requests
- Production server simulation
- Scaling behavior analysis
- Resource contention effects

### 4. Memory Efficiency Testing
VRAM and bandwidth utilization
- Model size vs performance
- Memory optimization opportunities
- Hardware requirement planning

## 📊 Results and Analysis

### Output Structure
```
results/
├── summary.json           # Key findings and recommendations
├── report.html           # Interactive dashboard
├── raw_data/             # Complete measurement datasets
├── plots/                # Performance visualizations
└── logs/                 # Detailed execution logs
```

### Key Metrics
- **Latency**: Time to first token (TTFT), total inference time
- **Throughput**: Tokens/second, requests/second
- **Efficiency**: GPU utilization %, memory bandwidth usage
- **Consistency**: Standard deviation, percentile distributions

## 🎯 Success Criteria

### Performance Targets
- **Baseline Improvement**: >3x speedup for Rust engines vs PyTorch
- **GPU Utilization**: >70% compute utilization under sustained load  
- **Statistical Significance**: p < 0.05 for performance differences
- **Reproducibility**: <10% variance across benchmark runs

### Business Value
- **Clear ROI**: Quantified performance improvement vs implementation cost
- **Deployment Guidance**: Specific recommendations for different use cases
- **Risk Mitigation**: Stability and maturity assessment for each engine

## 📚 Documentation

- [Benchmark Specification](docs/benchmark-specification.md) - Complete technical requirements
- [Test Methodology](docs/test-methodology.md) - Statistical rigor and procedures
- [Architecture Design](docs/architecture.md) - System design and isolation strategy
- [Code Conventions](docs/code-conventions.md) - Development standards
- [Getting Started Guide](docs/getting-started.md) - Setup and usage instructions

## 🤝 Contributing

### Adding New Engines
1. Create new crate in `crates/engines/`
2. Implement `shared-protocol` communication
3. Add binary configuration to workspace
4. Update benchmark runner engine registry

### Extending Analysis  
1. Add new metrics to `shared-protocol`
2. Implement collection in engine binaries
3. Extend statistical analysis in `benchmark-runner`
4. Update visualization and reporting

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

**Providing data-driven insights for GPU acceleration and Rust inference engine adoption decisions.**