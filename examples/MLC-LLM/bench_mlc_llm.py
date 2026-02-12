#!/usr/bin/env python3
"""
MLC LLM Benchmark for Jetson Orin AGX.
Tests decode throughput for comparison against tinygrad and llama.cpp.

Usage:
    python3 bench_mlc_llm.py [model] [num_tokens]

    # Examples:
    python3 bench_mlc_llm.py HF://mlc-ai/Llama-3.2-1B-Instruct-q4f16_1-MLC 25
    python3 bench_mlc_llm.py HF://mlc-ai/Llama-3.2-3B-Instruct-q4f16_1-MLC 25
"""
import sys, time, json

def benchmark_mlc_llm(model_path, num_tokens=25, prompt="Hello, how are you?"):
    try:
        from mlc_llm import MLCEngine
    except ImportError:
        print("ERROR: mlc_llm not installed. Run: pip install mlc-ai-nightly -f https://mlc.ai/wheels")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"MLC LLM Benchmark")
    print(f"Model: {model_path}")
    print(f"Prompt: '{prompt}'")
    print(f"Max tokens: {num_tokens}")
    print(f"{'='*70}\n")

    # Create engine
    print("Loading model...", flush=True)
    t0 = time.time()
    engine = MLCEngine(model_path)
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s\n")

    # Warmup
    print("Warming up...", flush=True)
    for resp in engine.chat.completions.create(
        messages=[{"role": "user", "content": "Hi"}],
        model=model_path,
        max_tokens=5,
        stream=True
    ):
        pass

    # Benchmark
    print("Benchmarking decode...", flush=True)
    times = []
    tokens_generated = 0
    
    t_start = time.time()
    for resp in engine.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model_path,
        max_tokens=num_tokens,
        stream=True
    ):
        times.append(time.time())
        if resp.choices[0].delta.content:
            tokens_generated += 1
    t_end = time.time()

    # Calculate metrics
    total_time = t_end - t_start
    if tokens_generated > 3:
        # Skip first 2 tokens (prefill), measure rest
        decode_times = [times[j] - times[j-1] for j in range(3, len(times))]
        if decode_times:
            avg_ms = sum(decode_times) / len(decode_times) * 1000
            tok_s = len(decode_times) / sum(decode_times)
        else:
            avg_ms = total_time / tokens_generated * 1000
            tok_s = tokens_generated / total_time
    else:
        avg_ms = total_time / max(tokens_generated, 1) * 1000
        tok_s = max(tokens_generated, 1) / total_time

    print(f"\nResults:")
    print(f"  Tokens generated: {tokens_generated}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Decode throughput: {tok_s:.2f} tok/s")
    print(f"  Decode latency: {avg_ms:.1f} ms/token")
    print(f"  Model load time: {load_time:.1f}s")

    result = {
        "engine": "mlc-llm",
        "model": model_path,
        "tokens": tokens_generated,
        "tok_s": tok_s,
        "ms_token": avg_ms,
        "load_time_s": load_time
    }
    
    print(f"\nJSON: {json.dumps(result)}")
    
    engine.terminate()
    return result

if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "HF://mlc-ai/Llama-3.2-1B-Instruct-q4f16_1-MLC"
    num_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    
    benchmark_mlc_llm(model, num_tokens)
