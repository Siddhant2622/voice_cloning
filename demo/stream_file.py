"""
demo/stream_file.py
Phase 2 — File-to-WebSocket audio streamer.

Simulates a real-time audio stream by reading a file and sending
it to the demo server over WebSocket in real-time-rate chunks.

Usage:
    python demo/stream_file.py <audio_file> [--host localhost] [--port 8000]
    python demo/stream_file.py data/samples/synthetic_00.wav
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


async def stream_file(audio_path: Path, host: str, port: int, chunk_ms: int = 100):
    """
    Load audio, resample to 16 kHz mono, send to WebSocket in real-time-rate chunks.
    """
    import websockets
    import numpy as np

    uri = f"ws://{host}:{port}/ws/stream"
    print(f"Streaming: {audio_path.name} -> {uri}")

    # Load and preprocess audio
    try:
        from src.preprocessing import load_audio, to_mono_16k
        waveform, sr = load_audio(audio_path)
        waveform = to_mono_16k(waveform, sr)
        audio_np = waveform.numpy()
    except ImportError:
        # Fallback using soundfile
        import soundfile as sf
        import librosa
        audio_np, sr = sf.read(str(audio_path), dtype="float32")
        if len(audio_np.shape) > 1:
            audio_np = audio_np.mean(axis=1)
        if sr != 16000:
            audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=16000)

    sample_rate   = 16_000
    chunk_samples = int(sample_rate * chunk_ms / 1000)
    total_chunks  = (len(audio_np) + chunk_samples - 1) // chunk_samples
    duration_s    = len(audio_np) / sample_rate

    print(f"  Duration:   {duration_s:.1f}s  |  {total_chunks} chunks × {chunk_ms}ms")

    try:
        async with websockets.connect(uri) as ws:
            print("  Connected. Streaming...\n")
            t_start = time.perf_counter()

            for i in range(total_chunks):
                chunk = audio_np[i * chunk_samples : (i + 1) * chunk_samples]
                # Pad last chunk if needed
                if len(chunk) < chunk_samples:
                    chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))

                # Convert to 16-bit PCM
                int16 = (chunk * 32767).clip(-32768, 32767).astype("int16")
                await ws.send(int16.tobytes())

                # Receive any score updates
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=0.05)
                    import json
                    data = json.loads(msg)
                    if data.get("type") == "score":
                        fusion = data.get("fusion_score", 0.5)
                        dec    = data.get("decision", "-")
                        lat    = data.get("latency_ms", 0)
                        bar_len = int(fusion * 30)
                        bar = "#" * bar_len + "-" * (30 - bar_len)
                        color = "\033[92m" if fusion < 0.35 else ("\033[93m" if fusion < 0.65 else "\033[91m")
                        print(f"\r  [{bar}] {color}{fusion:.3f}\033[0m  {dec:8s}  {lat:5.0f}ms", end="", flush=True)
                except asyncio.TimeoutError:
                    pass

                # Real-time pacing
                expected_time = (i + 1) * chunk_ms / 1000
                elapsed = time.perf_counter() - t_start
                sleep_s = expected_time - elapsed
                if sleep_s > 0:
                    await asyncio.sleep(sleep_s)

            # Wait for final score
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                import json
                data = json.loads(msg)
                if data.get("type") == "score":
                    print(f"\n\n  Final score: {data.get('fusion_score', 0):.3f}  ->  {data.get('decision', '-')}")
                    print(f"  Reason: {data.get('reason', '')}")
            except asyncio.TimeoutError:
                pass

            print(f"\n\n  Streaming complete ({time.perf_counter()-t_start:.1f}s)")

    except ConnectionRefusedError:
        print(f"\n  ✗ Could not connect to {uri}")
        print("  Is the demo server running? Run: python demo/server.py")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Stream an audio file to the demo server.")
    parser.add_argument("audio_file", help="Path to audio file.")
    parser.add_argument("--host",     default="localhost", help="Server host. [default: localhost]")
    parser.add_argument("--port",     type=int, default=8000, help="Server port. [default: 8000]")
    parser.add_argument("--chunk-ms", type=int, default=100,  help="Chunk size in ms. [default: 100]")
    args = parser.parse_args()

    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"✗ File not found: {audio_path}")
        sys.exit(1)

    asyncio.run(stream_file(audio_path, args.host, args.port, args.chunk_ms))


if __name__ == "__main__":
    main()
