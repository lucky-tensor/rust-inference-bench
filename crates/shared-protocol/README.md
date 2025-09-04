# Shared Protocol

## Overview

Minimal shared types for communication between the benchmark runner and individual inference engines. This crate contains **zero heavy dependencies** to ensure fair benchmarking.

## Purpose

- **Process Communication**: JSON serialization for stdin/stdout communication
- **Minimal Overhead**: No ML frameworks, no async traits, no complex dependencies
- **Fair Benchmarking**: Each engine only includes its specific dependencies

## Key Types

### BenchmarkRequest
Sent from benchmark runner to engine process:
```json
{
  "prompt": "Hello, how are you?",
  "max_tokens": 100,
  "temperature": 0.7,
  "iterations": 50,
  "model_path": "./models/llama-3.2-1b"
}
```

### BenchmarkResponse  
Returned from engine process to benchmark runner:
```json
{
  "engine_name": "candle",
  "generated_text": "Hello! I'm doing well...",
  "timings_ms": [45.2, 43.8, 46.1],
  "token_count": 15,
  "memory_usage_mb": 1024.5,
  "gpu_utilization": 87.3
}
```

## Architecture Benefits

- **Process Isolation**: Engines run as separate processes
- **Clean Communication**: Simple JSON over stdin/stdout
- **No Shared State**: Each engine has independent memory and GPU context
- **Reproducible Results**: Process boundaries prevent cross-contamination

This ensures fair performance comparisons with minimal shared infrastructure.