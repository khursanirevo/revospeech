"""Streaming synthesis — generate audio in chunks for low-latency playback.

Some TTS engines override BaseTTS.synthesize_streaming() to yield Audio
chunks as they are produced. This example shows the pattern.

Usage:
    python examples/10_streaming_tts.py
"""

from revospeech.tts import TTS

# Try streaming synthesis
tts = TTS()

text = "First sentence. Second sentence. Third sentence."

try:
    chunk_iter = tts.synthesize_streaming(text)
    chunks = list(chunk_iter)
    print(f"Got {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}: {chunk.duration:.2f}s, {chunk.sample_rate}Hz")
        if i == 0:
            chunk.save("streaming_first.wav")
except NotImplementedError:
    print("This TTS engine does not support streaming synthesis.")
    print("Falling back to non-streaming:")
    audio = tts.synthesize(text)
    print(f"Generated {audio.duration:.2f}s of audio")
    audio.save("streaming_fallback.wav")
