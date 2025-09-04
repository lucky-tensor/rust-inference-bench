# Code Conventions and Standards

## Project Structure

### Crate Organization
```
cuda-inference-benchmarks/
├── crates/
│   ├── shared-types/          # Common interfaces and data structures
│   ├── benchmark-harness/     # Core benchmarking infrastructure
│   ├── pytorch-baseline/      # Unoptimized PyTorch reference
│   ├── lm-rs-engine/         # lm.rs adapter
│   ├── candle-engine/        # Candle framework adapter
│   ├── burn-engine/          # Burn framework adapter
│   └── mistral-rs-engine/    # Mistral.rs adapter
├── docs/                     # Documentation
├── models/                   # Model files and configurations
├── results/                  # Benchmark results and analysis
└── scripts/                  # Automation and analysis scripts
```

## Rust Code Standards

### Naming Conventions
- **Crate Names**: kebab-case (`benchmark-harness`)
- **Function Names**: snake_case (`run_benchmark_suite`)
- **Type Names**: PascalCase (`BenchmarkResult`)
- **Constants**: SCREAMING_SNAKE_CASE (`MAX_CONCURRENT_REQUESTS`)
- **Module Names**: snake_case (`cuda_profiler`)

### Error Handling
```rust
use anyhow::{Context, Result};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum BenchmarkError {
    #[error("CUDA operation failed: {0}")]
    CudaError(String),
    
    #[error("Inference engine failed: {source}")]
    InferenceError {
        #[from]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    
    #[error("Configuration invalid: {message}")]
    ConfigError { message: String },
}

// Use Result<T, BenchmarkError> for domain errors
// Use anyhow::Result for general errors with context
pub fn run_inference(config: &BenchmarkConfig) -> Result<BenchmarkResult> {
    let engine = load_engine(&config.engine_type)
        .context("Failed to load inference engine")?;
        
    engine.infer(&config.prompt)
        .context("Inference execution failed")?;
        
    Ok(result)
}
```

### Async Patterns
```rust
// Use async/await throughout for non-blocking operations
#[async_trait::async_trait]
pub trait InferenceEngine: Send + Sync {
    async fn infer(&self, prompt: &str) -> Result<String, InferenceError>;
    
    async fn infer_batch(&self, prompts: Vec<&str>) -> Result<Vec<String>, InferenceError>;
    
    fn name(&self) -> &str;
}

// Use structured concurrency
pub async fn run_concurrent_benchmark(
    engines: Vec<Box<dyn InferenceEngine>>,
    prompts: Vec<String>,
) -> Result<Vec<BenchmarkResult>> {
    let tasks = engines.into_iter()
        .map(|engine| {
            let prompts = prompts.clone();
            tokio::spawn(async move {
                benchmark_engine(engine, prompts).await
            })
        })
        .collect::<Vec<_>>();
        
    let results = futures::future::try_join_all(tasks).await?;
    Ok(results)
}
```

### Memory Management
```rust
// Use Arc for shared ownership of heavy objects
use std::sync::Arc;

pub struct BenchmarkSuite {
    engines: Vec<Arc<dyn InferenceEngine>>,
    profiler: Arc<CudaProfiler>,
}

// Avoid unnecessary clones, prefer borrowing
impl BenchmarkSuite {
    pub async fn run_all_benchmarks(&self, config: &BenchmarkConfig) -> Result<SuiteResults> {
        let mut results = Vec::new();
        
        for engine in &self.engines {
            let result = self.benchmark_single_engine(engine.as_ref(), config).await?;
            results.push(result);
        }
        
        Ok(SuiteResults::new(results))
    }
}
```

## Configuration Management

### Structured Configuration
```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    pub engines: Vec<EngineConfig>,
    pub workloads: Vec<WorkloadConfig>,
    pub hardware: HardwareConfig,
    pub analysis: AnalysisConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadConfig {
    pub name: String,
    pub batch_sizes: Vec<usize>,
    pub sequence_lengths: Vec<usize>,
    pub concurrent_requests: Vec<usize>,
    pub iterations: usize,
    pub warmup_iterations: usize,
}

// Use builder pattern for complex configurations
impl BenchmarkConfig {
    pub fn builder() -> BenchmarkConfigBuilder {
        BenchmarkConfigBuilder::default()
    }
}
```

### Environment Variables
```rust
// Use environment variables for system-specific settings
use std::env;

pub struct SystemConfig {
    pub cuda_device: usize,
    pub max_memory_gb: usize,
    pub output_dir: PathBuf,
}

impl SystemConfig {
    pub fn from_env() -> Result<Self> {
        Ok(Self {
            cuda_device: env::var("CUDA_DEVICE")
                .unwrap_or_else(|_| "0".to_string())
                .parse()?,
            max_memory_gb: env::var("MAX_MEMORY_GB")
                .unwrap_or_else(|_| "16".to_string()) 
                .parse()?,
            output_dir: env::var("OUTPUT_DIR")
                .unwrap_or_else(|_| "./results".to_string())
                .into(),
        })
    }
}
```

## Testing Standards

### Unit Tests
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use tempfile::TempDir;
    
    #[tokio::test]
    async fn test_benchmark_statistical_analysis() {
        let timings = vec![100.0, 105.0, 98.0, 102.0, 99.0];
        let stats = calculate_statistics(&timings);
        
        assert_relative_eq!(stats.mean, 100.8, epsilon = 0.1);
        assert_relative_eq!(stats.std_dev, 2.77, epsilon = 0.1);
        assert_eq!(stats.sample_count, 5);
    }
    
    #[test]
    fn test_config_serialization() {
        let config = BenchmarkConfig::builder()
            .add_engine("candle", EngineType::Candle)
            .add_workload("interactive", WorkloadType::Interactive)
            .build();
            
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: BenchmarkConfig = serde_json::from_str(&json).unwrap();
        
        assert_eq!(config.engines.len(), deserialized.engines.len());
    }
}
```

### Integration Tests
```rust
// tests/integration_test.rs
use cuda_inference_benchmarks::*;

#[tokio::test]
async fn test_end_to_end_benchmark() {
    // Requires GPU and model files
    if !has_cuda_gpu() {
        return;
    }
    
    let config = load_test_config("test-configs/minimal.yaml").unwrap();
    let suite = BenchmarkSuite::new(config).await.unwrap();
    
    let results = suite.run_quick_test().await.unwrap();
    
    assert!(!results.is_empty());
    assert!(results.iter().all(|r| r.timing.mean_ms > 0.0));
}
```

## Documentation Standards

### Code Documentation
```rust
/// Runs a comprehensive benchmark comparing inference engines.
///
/// This function executes multiple test scenarios across all configured
/// inference engines, measuring performance, memory usage, and GPU utilization.
///
/// # Arguments
/// * `config` - Benchmark configuration specifying engines and workloads
/// * `output_dir` - Directory for saving results and analysis
///
/// # Returns
/// * `Ok(BenchmarkResults)` - Complete benchmark results with statistical analysis
/// * `Err(BenchmarkError)` - Error during benchmark execution
///
/// # Examples
/// ```rust
/// let config = BenchmarkConfig::load("config.yaml")?;
/// let results = run_comprehensive_benchmark(&config, "./results").await?;
/// println!("Best engine: {}", results.fastest_engine().name());
/// ```
pub async fn run_comprehensive_benchmark(
    config: &BenchmarkConfig,
    output_dir: &Path,
) -> Result<BenchmarkResults, BenchmarkError> {
    // Implementation
}
```

### README Structure
```markdown
# Section Structure for Crate READMEs
## Overview
## Quick Start  
## Configuration
## Examples
## Performance Characteristics
## Contributing
```

## Performance Standards

### Optimization Guidelines
```rust
// Use const generics for compile-time optimization
pub struct FixedSizeBatch<const N: usize> {
    prompts: [String; N],
}

// Prefer slice operations over Vec when possible
pub fn calculate_percentiles(timings: &[f32]) -> Percentiles {
    let mut sorted = timings.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    Percentiles {
        p50: percentile(&sorted, 0.50),
        p90: percentile(&sorted, 0.90),
        p99: percentile(&sorted, 0.99),
    }
}

// Use zero-copy where possible
pub fn format_results(results: &[BenchmarkResult]) -> String {
    results.iter()
        .map(|r| format!("{}: {:.2}ms", r.engine_name, r.mean_time_ms))
        .collect::<Vec<_>>()
        .join("\n")
}
```

### Memory Efficiency
- **Avoid Unnecessary Clones**: Use references and borrowing
- **Pool Resources**: Reuse CUDA events and GPU memory buffers  
- **Lazy Initialization**: Load models and engines on-demand
- **Streaming**: Process large datasets without loading entirely into memory

## Security Considerations

### Safe FFI Practices
```rust
// When interfacing with C++ libraries (e.g., potential TensorRT bindings)
use std::ffi::{CStr, CString};

pub unsafe fn call_cpp_function(input: &str) -> Result<String, BenchmarkError> {
    let c_input = CString::new(input)
        .map_err(|e| BenchmarkError::ConfigError { 
            message: format!("Invalid string: {}", e) 
        })?;
    
    let result_ptr = cpp_inference_function(c_input.as_ptr());
    if result_ptr.is_null() {
        return Err(BenchmarkError::InferenceError { /* ... */ });
    }
    
    let c_result = CStr::from_ptr(result_ptr);
    let rust_result = c_result.to_str()
        .map_err(|e| BenchmarkError::ConfigError { 
            message: format!("Invalid UTF-8: {}", e) 
        })?
        .to_owned();
        
    // Cleanup C++ allocated memory
    cpp_free_result(result_ptr);
    
    Ok(rust_result)
}
```

### Input Validation
```rust
// Always validate external inputs
pub fn validate_benchmark_config(config: &BenchmarkConfig) -> Result<(), BenchmarkError> {
    if config.workloads.is_empty() {
        return Err(BenchmarkError::ConfigError {
            message: "At least one workload must be specified".to_string(),
        });
    }
    
    for workload in &config.workloads {
        if workload.iterations == 0 {
            return Err(BenchmarkError::ConfigError {
                message: format!("Workload '{}' must have iterations > 0", workload.name),
            });
        }
    }
    
    Ok(())
}
```

These conventions ensure maintainable, performant, and safe code across all benchmark components.