"""
demo/benchmark_latency.py
Phase 2 — Latency benchmark: p50/p95/p99 from chunk-in to score-out.

Usage:
    python demo/benchmark_latency.py [--n 50] [--audio data/samples/synthetic_00.wav]
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


async def run_benchmark(host: str, port: int, n_windows: int, audio_path: Path):
    import websockets
    import numpy as np

    uri = f"ws://{host}:{port}/ws/stream"
    SAMPLE_RATE   = 16_000
    WINDOW_SAMPLES = SAMPLE_RATE * 2    # 2-second windows
    HOP_SAMPLES    = SAMPLE_RATE * 1    # 1-second hop

    # Load or generate test audio
    if audio_path and audio_path.exists():
        try:
            from src.preprocessing import load_audio, to_mono_16k
            waveform, sr = load_audio(audio_path)
            waveform = to_mono_16k(waveform, sr)
            audio_np = waveform.numpy()
        except Exception:
            import soundfile as sf
            audio_np, _ = sf.read(str(audio_path), dtype="float32")
            if len(audio_np.shape) > 1:
                audio_np = audio_np.mean(axis=1)
    else:
        # White noise as test signal
        print("No audio file provided; using white noise test signal.")
        audio_np = np.random.randn(WINDOW_SAMPLES * n_windows).astype(np.float32) * 0.1

    latencies = []

    print(f"Benchmarking {n_windows} windows -> {uri}")
    print("=" * 50)

    try:
        async with websockets.connect(uri) as ws:
            offset = 0
            for i in range(n_windows):
                # Send one window worth of audio in chunks
                window = audio_np[offset:offset + WINDOW_SAMPLES]
                if len(window) < WINDOW_SAMPLES:
                    window = np.pad(window, (0, WINDOW_SAMPLES - len(window)))

                offset = (offset + HOP_SAMPLES) % max(1, len(audio_np) - WINDOW_SAMPLES)

                int16 = (window * 32767).clip(-32768, 32767).astype("int16")

                t_send = time.perf_counter()
                # Send in 100ms chunks to simulate real-time
                chunk_size = SAMPLE_RATE // 10   # 100ms chunks
                for j in range(0, len(int16), chunk_size):
                    await ws.send(int16[j:j+chunk_size].tobytes())
                    await asyncio.sleep(0.1)   # simulate real-time rate

                # Wait for score
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    t_recv = time.perf_counter()
                    data = json.loads(msg)
                    if data.get("type") == "score":
                        latency_ms = data.get("latency_ms") or (t_recv - t_send) * 1000
                        latencies.append(latency_ms)
                        if (i + 1) % 10 == 0:
                            print(f"  Window {i+1:3d}/{n_windows}  latency={latency_ms:.1f}ms")
                except asyncio.TimeoutError:
                    print(f"  Window {i+1:3d}: timeout - server may be overloaded")

    except ConnectionRefusedError:
        print(f"✗ Cannot connect to {uri}. Start the server: python demo/server.py")
        sys.exit(1)

    if not latencies:
        print("No latency measurements collected.")
        return

    import numpy as np
    lats = np.array(latencies)
    print()
    print("=" * 50)
    print("LATENCY REPORT")
    print("=" * 50)
    print(f"  Windows measured: {len(lats)}")
    print(f"  p50:  {np.percentile(lats, 50):6.1f} ms")
    print(f"  p90:  {np.percentile(lats, 90):6.1f} ms")
    print(f"  p95:  {np.percentile(lats, 95):6.1f} ms")
    print(f"  p99:  {np.percentile(lats, 99):6.1f} ms")
    print(f"  min:  {lats.min():6.1f} ms")
    print(f"  max:  {lats.max():6.1f} ms")
    print(f"  mean: {lats.mean():6.1f} ms")
    print("=" * 50)
    print()

    # Target comparison
    budget_ms = 600   # from architecture §4
    p95 = float(np.percentile(lats, 95))
    if p95 <= budget_ms:
        print(f"✓ p95 ({p95:.0f}ms) is within the {budget_ms}ms latency budget.")
    else:
        print(f"⚠ p95 ({p95:.0f}ms) exceeds the {budget_ms}ms budget. Consider batching or GPU.")


def main():
    parser = argparse.ArgumentParser(description="Benchmark detection pipeline latency.")
    parser.add_argument("--n",     type=int, default=20, help="Number of windows. [default: 20]")
    parser.add_argument("--audio", default=None,         help="Test audio file path.")
    parser.add_argument("--host",  default="localhost",  help="Server host.")
    parser.add_argument("--port",  type=int, default=8000)
    args = parser.parse_args()

    audio_path = Path(args.audio) if args.audio else None
    asyncio.run(run_benchmark(args.host, args.port, args.n, audio_path))


if __name__ == "__main__":
    main()
