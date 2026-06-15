# Legacy AI Kernel

Legacy AI Kernel is a systems project for running AI efficiently on older hardware using Rust and Zig. It focuses on quantization, memory management, kernel batching, and CPU/GPU optimization without requiring new chips.

## Why this exists

Most AI software assumes modern GPUs, large VRAM, and expensive hardware. This project takes the opposite approach: it tries to make legacy CPUs and consumer GPUs useful again for AI workloads.

That means:
- Lower memory use.
- Better performance on older machines.
- Smaller binaries and more control over compute paths.
- A practical path for edge deployments, labs, and cost-sensitive teams.

## Who this is for

This project is for:
- Engineers building low-level AI infrastructure.
- People running AI on older CPUs or consumer GPUs.
- Teams that care about cost, memory footprint, and deployment control.
- Technical founders exploring compute efficiency as a moat.

If you just want a simple AI app, this is probably too low-level. If you want to shape the compute layer itself, this is the right place.

## What it does

Legacy AI Kernel provides:
- INT4 quantization for reduced model memory.
- Paged KV-cache management for constrained VRAM.
- CPU matmul paths tuned for older hardware.
- GPU batching logic for better throughput.
- Rust wrappers with Zig/C-style low-level kernels.

## Current status

This project is under active development and intended to become deployable on:
- Older Intel and AMD CPUs.
- Consumer NVIDIA GPUs with limited VRAM.
- Mixed CPU/GPU systems with memory pressure.

## Getting started

### Requirements
- Rust toolchain
- A Linux environment recommended
- A C compiler for native kernel bindings

### Build
```bash
cargo build --release
```

### Test
```bash
cargo test
```

### Run
```bash
cargo run
```

### Example
```bash
cargo run --example legacy_llm
```

## Architecture

The project is organized into a few layers:

- `quantization/`: reduces model size and prepares weights for legacy hardware.
- `memory/`: manages paged memory and KV-cache behavior.
- `cpu/`: runs optimized compute on older processors.
- `gpu/`: batches work to reduce GPU launch overhead.
- `zig_bindings/`: native low-level implementations exposed to Rust.

## What makes it different

Legacy AI Kernel is not trying to invent new chips. It is trying to make old hardware valuable again by improving the software layer around inference and compute.

That gives you:
- Better hardware utilization.
- Lower deployment cost.
- More control over memory and performance.
- A path to AI systems that run where modern stacks struggle.

## Example use cases

- Running smaller LLMs on old desktops.
- Serving inference on consumer GPUs.
- Building local-first AI tools.
- Exploring kernel-level AI optimizations.
- Reducing infrastructure cost for edge deployments.

## Benchmarks

Benchmark numbers will be added as the implementation hardens. The goal is to measure:
- Tokens per second.
- Memory footprint.
- KV-cache efficiency.
- CPU and GPU utilization.
- End-to-end latency.

## Development goals

- Make the code compile cleanly.
- Keep the runtime minimal.
- Reduce memory overhead.
- Improve support for older hardware.
- Add real benchmark suites and integration tests.

## Contributing

Contributions are welcome, especially in:
- Systems programming.
- Rust performance work.
- Zig native kernels.
- Memory management.
- AI inference optimization.

## License

MIT

## Contact

Built by Julius Hill.
