import io
import pathlib
import modal

# ---------- Modal App and Image ----------

stub = modal.App("dia-tts-app")

# Base CUDA‑ready Debian image + Python 3.10
# We install torch with GPU wheels, Dia itself and a couple of audio helpers.
image = (
    modal.Image.debian_slim(python_version="3.10")
    # SoundFile requires libsndfile at the OS level.
    .apt_install("git", "ffmpeg", "libsndfile1").pip_install(
        # CUDA enabled torch/torchaudio wheels
        "torch==2.6.0",  # Modal already has the CUDA runtime so the PyPI wheel is fine
        "torchaudio==2.6.0",
        "soundfile",
        # Install Dia straight from GitHub ― keeps you on the latest commit
        "git+https://github.com/nari-labs/dia.git",
    )
)

# ---------- Remote function ----------


@stub.function(
    image=image,
    gpu="A10G",  # ~10 GB VRAM, plenty for Dia 1.6 B
    timeout=60 * 15,  # 15‑minute max runtime (first run downloads ~4 GB weights)
)
def tts(text: str) -> bytes:
    """Generate speech from *text* and return WAV bytes."""
    from dia.model import Dia
    import soundfile as sf

    # Load the pretrained model (cached inside the container after the first call)
    model = Dia.from_pretrained("nari-labs/Dia-1.6B")

    # Dia uses 44.1 kHz, 16‑bit WAV output
    wav = model.generate(text)

    # Serialise the NumPy array returned by Dia to an in‑memory WAV file
    buf = io.BytesIO()
    sf.write(buf, wav, 44100, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


# ---------- Local entry‑point ----------


@stub.local_entrypoint()
def main(
    script: str = "[S1] Dia running on Modal! [S2] Sounds good, doesn’t it?",
    out: str = "output.wav",
):
    """CLI entry‑point executed on your MacBook via `modal run`.

    Examples
    --------
    $ modal run dia_modal_stub.py --script "[S1] Hello world" --out hello.wav
    """
    audio = tts.remote(script)  # Remote GPU call ➜ audio bytes back to your laptop
    pathlib.Path(out).write_bytes(audio)
    print(f"\N{MUSICAL NOTE}  Saved {out} ({len(audio) // 1024} KB)")
