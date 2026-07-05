"""
Real-time AI noise reduction: mic -> de-emphasis -> denoiser -> speakers/headphones.

⚠️ USE HEADPHONES. Routing mic -> speakers directly will cause feedback
(howling) if the mic can hear the speakers.

Backends:
  rnnoise  - tiny RNN, CPU only, near-zero latency. GPU would not help here
             (model is too small; PCIe transfer per 10ms frame costs more
             than the compute saves).
  dpdfnet  - stronger DeepFilterNet2-based model, better quality on hard
             noise (TV/speech-like background, etc). NOTE: `pip install
             dpdfnet` is CPU-only ONNX inference per the project's own docs.
             Real GPU acceleration requires running their PyTorch checkpoint
             from source on .cuda() -- there's a stub for this below
             (--backend dpdfnet-gpu) but you must clone their repo and point
             --checkpoint at a downloaded .pth file.

Install (pick what you need):
    pip install sounddevice numpy pyrnnoise      # rnnoise backend
    pip install sounddevice numpy dpdfnet        # dpdfnet backend (CPU/ONNX)
    # for dpdfnet-gpu: clone github.com/ceva-ip/DPDFNet, install its
    # requirements.txt with a CUDA build of torch, download a checkpoint

Run:
    python realtime_denoise.py --list
    python realtime_denoise.py --backend rnnoise --in 2 --out 4
    python realtime_denoise.py --backend dpdfnet --model dpdfnet2_48khz_hr
    python realtime_denoise.py --deemphasis 0     # disable de-emphasis
"""

import argparse
import sys
import threading
from collections import deque
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi
import sounddevice as sd

SAMPLE_RATE = 48000
FRAME_SIZE = 480  # 10ms @ 48kHz -- required frame size for both backends


class BandpassFilter:
    """Butterworth bandpass, run via sosfilt (vectorized, not a per-sample
    Python loop -- fast, and state persists correctly across blocks via zi).

    This matters most for weak/narrowband signals (SSB especially): energy
    outside the actual voice bandwidth is pure noise as far as the model is
    concerned, and stripping it before the denoiser sees it directly raises
    the effective SNR the model has to work with. It also keeps AGC's level
    detector from being thrown off by out-of-band hiss or subaudible tones
    (e.g. CTCSS) on FM.
    """

    def __init__(self, low_hz: float, high_hz: float, sample_rate: int, order: int = 4):
        self.enabled = True
        self.sample_rate = sample_rate
        self.set_band(low_hz, high_hz, order)

    def set_band(self, low_hz: float, high_hz: float, order: int = 4):
        self.sos = butter(order, [low_hz, high_hz], btype="band",
                           fs=self.sample_rate, output="sos")
        self.zi = sosfilt_zi(self.sos) * 0.0  # start at rest, no startup click
        self.low_hz = low_hz
        self.high_hz = high_hz

    def toggle(self) -> bool:
        self.enabled = not self.enabled
        return self.enabled

    def process(self, x: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return x
        y, self.zi = sosfilt(self.sos, x, zi=self.zi)
        return y.astype(np.float32)


MODE_PRESETS = {
    # mode: (deemphasis_enabled, bandpass_low_hz, bandpass_high_hz)
    "nbfm": (True, 300.0, 3000.0),   # narrowband FM: repeaters/direct, strip
                                       # sub-audible CTCSS and above-voice hiss
    "ssb": (False, 300.0, 2900.0),   # SSB: no pre-emphasis on TX, so
                                       # de-emphasis must be OFF; tight
                                       # bandpass matched to TX audio filter
    "none": (None, None, None),      # no preset -- use explicit CLI flags
}


class DeEmphasis:
    """Single-pole de-emphasis filter, unity gain at DC:
        y[n] = (1-a)*x[n] + a*y[n-1]

    Keeps state across calls so there's no discontinuity/click at frame
    boundaries. Use this to undo a pre-emphasis stage applied upstream
    (e.g. by a codec or transmitter that boosted high frequencies), or
    to tame harsh high-frequency hiss/sibilance before the denoiser sees it.

    IMPORTANT: without the (1-a) term this filter has DC gain of 1/(1-a)
    -- e.g. 20x at a=0.95 -- which will blow up low-frequency content past
    the float ceiling, hard-clip it, and then have the denoiser mistake the
    clipping distortion for noise and suppress that whole band. That bug
    was in the previous version of this script; this version is unity-gain
    at DC so loud low end passes through cleanly instead of clipping.
    """

    def __init__(self, coeff: float = 0.95):
        self.a = coeff
        self.enabled = True
        self._prev = 0.0

    def reset(self):
        self._prev = 0.0

    def toggle(self) -> bool:
        self.enabled = not self.enabled
        return self.enabled

    def process(self, x: np.ndarray) -> np.ndarray:
        if not self.enabled or self.a == 0.0:
            return x
        # Plain Python floats in a list are faster here than numpy scalar
        # indexing (x[i]/y[i] on a numpy array has real per-element overhead
        # that adds up across a tight real-time budget).
        prev = self._prev
        a = self.a
        gain = 1.0 - a
        y = [0.0] * len(x)
        for i, xi in enumerate(x.tolist()):
            prev = gain * xi + a * prev
            y[i] = prev
        self._prev = prev
        return np.asarray(y, dtype=np.float32)


class AGC:
    """Automatic gain control: envelope-follower based, fast attack / slow
    release (broadcast-style) so it reacts quickly to loud peaks but doesn't
    pump/breathe during quiet passages.

    Pipeline position matters: running this before the denoiser will also
    amplify noise during quiet stretches (the denoiser then has more work
    to do right after a quiet->loud transition). That's an expected
    tradeoff of pre-denoiser AGC, not a bug.
    """

    def __init__(self, sample_rate: int, target_rms: float = 0.15,
                 attack_ms: float = 5, release_ms: float = 200,
                 gain_attack_ms: float = 5, gain_release_ms: float = 300,
                 max_gain_db: float = 24, min_gain_db: float = -24):
        self.target = target_rms
        self.env_attack = np.exp(-1.0 / (sample_rate * attack_ms / 1000))
        self.env_release = np.exp(-1.0 / (sample_rate * release_ms / 1000))
        self.gain_attack = np.exp(-1.0 / (sample_rate * gain_attack_ms / 1000))
        self.gain_release = np.exp(-1.0 / (sample_rate * gain_release_ms / 1000))
        self.max_gain = 10 ** (max_gain_db / 20)
        self.min_gain = 10 ** (min_gain_db / 20)
        self.enabled = True
        self._env = 0.0
        self._gain = 1.0

    def reset(self):
        self._env = 0.0
        self._gain = 1.0

    def toggle(self) -> bool:
        self.enabled = not self.enabled
        return self.enabled

    def process(self, x: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return x
        env = self._env
        gain = self._gain
        env_attack, env_release = self.env_attack, self.env_release
        gain_attack, gain_release = self.gain_attack, self.gain_release
        target, max_gain, min_gain = self.target, self.max_gain, self.min_gain

        y = [0.0] * len(x)
        for i, xi in enumerate(x.tolist()):
            rectified = abs(xi)
            if rectified > env:
                env = env_attack * env + (1 - env_attack) * rectified
            else:
                env = env_release * env + (1 - env_release) * rectified

            desired = target / max(env, 1e-6)
            desired = min(max(desired, min_gain), max_gain)

            if desired < gain:
                gain = gain_attack * gain + (1 - gain_attack) * desired
            else:
                gain = gain_release * gain + (1 - gain_release) * desired

            y[i] = xi * gain
        self._env = env
        self._gain = gain
        return np.asarray(y, dtype=np.float32)


class RNNoiseBackend:
    """CPU. Tiny model, effectively free latency-wise."""

    def __init__(self):
        from pyrnnoise import RNNoise
        self.denoiser = RNNoise(sample_rate=SAMPLE_RATE)

    def process(self, mono_f32: np.ndarray) -> np.ndarray:
        int16_frame = np.clip(mono_f32 * 32767, -32768, 32767).astype(np.int16).reshape(1, -1)
        _, denoised = next(self.denoiser.denoise_chunk(int16_frame))
        return (denoised.astype(np.float32) / 32767.0).ravel()


class DPDFNetBackend:
    """CPU-only ONNX inference (per dpdfnet's own docs). Better quality,
    more CPU cost than RNNoise. Not GPU-accelerated in this form."""

    def __init__(self, model: str = "dpdfnet2_48khz_hr"):
        import dpdfnet
        self.enhancer = dpdfnet.StreamEnhancer(model=model)

    def process(self, mono_f32: np.ndarray) -> np.ndarray:
        enhanced = self.enhancer.process(mono_f32, sample_rate=SAMPLE_RATE)
        if len(enhanced) < len(mono_f32):
            # buffer still filling on the first couple of frames
            pad = np.zeros(len(mono_f32) - len(enhanced), dtype=np.float32)
            enhanced = np.concatenate([enhanced, pad])
        return enhanced[: len(mono_f32)]


class DPDFNetGPUBackend:
    """Stub: run DPDFNet's PyTorch checkpoint directly on CUDA.

    This is NOT provided by the pip package -- you need to clone
    github.com/ceva-ip/DPDFNet, install its requirements.txt (with a CUDA
    build of torch), download a .pth checkpoint, and adapt their model's
    forward() for causal frame-by-frame streaming with persistent hidden
    state. That's real engineering work, not a pip install -- happy to help
    build it out if RNNoise/dpdfnet-CPU aren't fast enough for your case.
    """

    def __init__(self, checkpoint: str):
        raise NotImplementedError(
            "GPU streaming requires building a custom causal-inference loop "
            "around DPDFNet's PyTorch model. See the class docstring."
        )


def soft_limit(x: np.ndarray, threshold: float = 0.9) -> np.ndarray:
    """Gentle safety limiter: passes signal through untouched below
    `threshold`, then smoothly compresses anything above it with tanh
    instead of hard-clipping. This is a backstop for a hot mic input or
    gain buildup anywhere upstream -- it should rarely engage in normal use."""
    over = np.abs(x) > threshold
    if not np.any(over):
        return x
    y = x.copy()
    sign = np.sign(x[over])
    excess = np.abs(x[over]) - threshold
    y[over] = sign * (threshold + (1 - threshold) * np.tanh(excess / (1 - threshold)))
    return y


def command_listener(deemph: "DeEmphasis", agc: "AGC | None", bandpass: "BandpassFilter | None"):
    """Reads single-key commands from stdin in a background thread so the
    audio callback thread is never touched. Toggling a plain Python bool is
    atomic under the GIL, so no lock is needed here."""
    print("\nLive controls (type a letter/number + Enter):")
    print("  d = toggle de-emphasis   a = toggle AGC   b = toggle bandpass")
    print("  1 = switch to NBFM preset   2 = switch to SSB preset   q = quit\n")
    for line in sys.stdin:
        cmd = line.strip().lower()
        if cmd == "d":
            state = deemph.toggle()
            print(f"De-emphasis: {'ON' if state else 'OFF'}")
        elif cmd == "a" and agc is not None:
            state = agc.toggle()
            print(f"AGC: {'ON' if state else 'OFF'}")
        elif cmd == "b" and bandpass is not None:
            state = bandpass.toggle()
            print(f"Bandpass ({bandpass.low_hz:.0f}-{bandpass.high_hz:.0f}Hz): {'ON' if state else 'OFF'}")
        elif cmd in ("1", "2") and bandpass is not None:
            mode = "nbfm" if cmd == "1" else "ssb"
            preset_deemph, low, high = MODE_PRESETS[mode]
            deemph.enabled = preset_deemph
            bandpass.set_band(low, high)
            bandpass.enabled = True
            print(f"Switched to {mode.upper()} preset: de-emphasis {'ON' if preset_deemph else 'OFF'}, "
                  f"bandpass {low:.0f}-{high:.0f}Hz")
        elif cmd == "q":
            print("Quitting...")
            import os
            os._exit(0)


def list_devices():
    print(sd.query_devices())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--in", dest="in_dev", type=int, default=None)
    parser.add_argument("--out", dest="out_dev", type=int, default=None)
    parser.add_argument("--backend", choices=["rnnoise", "dpdfnet", "dpdfnet-gpu"], default="rnnoise")
    parser.add_argument("--model", default="dpdfnet2_48khz_hr", help="dpdfnet model name")
    parser.add_argument("--checkpoint", default=None, help="dpdfnet-gpu: path to .pth checkpoint")
    parser.add_argument("--mode", choices=["none", "nbfm", "ssb"], default="none",
                         help="Preset for de-emphasis + bandpass tuned to the signal type. "
                              "'nbfm': de-emphasis ON, bandpass 300-3000Hz (repeaters/direct FM). "
                              "'ssb': de-emphasis OFF (no pre-emphasis on TX side), bandpass "
                              "300-2900Hz matched to typical SSB TX audio filtering. "
                              "Explicit --deemphasis/--bandpass-low/--bandpass-high flags "
                              "override the preset if given.")
    parser.add_argument("--deemphasis", type=float, default=None,
                         help="de-emphasis coefficient (0 disables it), typical range 0.9-0.97. "
                              "Overrides --mode's preset if given.")
    parser.add_argument("--bandpass-low", type=float, default=None, help="bandpass low corner (Hz), overrides --mode preset")
    parser.add_argument("--bandpass-high", type=float, default=None, help="bandpass high corner (Hz), overrides --mode preset")
    parser.add_argument("--strength", type=float, default=1.0,
                         help="Wet/dry blend of denoiser output vs pre-denoise signal, 0-1. "
                              "1.0 = fully processed (best for weak/noisy signals like distant "
                              "SSB). Lower it (e.g. 0.5-0.7) on already-strong signals (local FM "
                              "repeaters) to reduce neural-denoiser artifacts -- you're asking "
                              "the model to do less work on a signal that barely needs it.")
    parser.add_argument("--agc", action="store_true", default=True, help="enable AGC (default on)")
    parser.add_argument("--no-agc", dest="agc", action="store_false")
    parser.add_argument("--agc-target", type=float, default=0.15, help="AGC target RMS level (0-1)")
    parser.add_argument("--agc-max-gain-db", type=float, default=24)
    parser.add_argument("--agc-min-gain-db", type=float, default=-24)
    parser.add_argument("--agc-attack-ms", type=float, default=5)
    parser.add_argument("--agc-release-ms", type=float, default=200)
    parser.add_argument("--blocksize", type=int, default=0,
                         help="PortAudio callback block size in samples. 0 = let PortAudio "
                              "choose (recommended). Does NOT need to be 480 -- audio is "
                              "internally buffered into the 480-sample frames the model needs, "
                              "so a larger/auto blocksize just gives PortAudio more slack and "
                              "should fix 'output underflow' warnings. Try e.g. 2048 if 0 "
                              "still underflows.")
    parser.add_argument("--latency", choices=["low", "high"], default="high",
                         help="PortAudio latency hint. 'high' trades a bit of extra delay for "
                              "much more underrun resistance -- start here if you're seeing "
                              "underflows, drop to 'low' once things are stable if you want to "
                              "shave off latency.")
    args = parser.parse_args()

    if args.list:
        list_devices()
        return

    if args.backend == "rnnoise":
        engine = RNNoiseBackend()
    elif args.backend == "dpdfnet":
        engine = DPDFNetBackend(model=args.model)
    else:
        engine = DPDFNetGPUBackend(checkpoint=args.checkpoint)  # raises for now

    preset_deemph, preset_low, preset_high = MODE_PRESETS[args.mode]

    deemph_coeff = args.deemphasis if args.deemphasis is not None else (
        0.95 if preset_deemph in (True, None) else 0.0
    )
    deemph = DeEmphasis(coeff=deemph_coeff)
    if args.mode != "none" and args.deemphasis is None:
        deemph.enabled = preset_deemph

    bp_low = args.bandpass_low if args.bandpass_low is not None else preset_low
    bp_high = args.bandpass_high if args.bandpass_high is not None else preset_high
    # Always construct the filter (defaulting to a sane voice band) so the
    # live '1'/'2' preset-switch keys work even if you started in --mode none;
    # `enabled` reflects whether it should actually be active at startup.
    bandpass = BandpassFilter(bp_low or 300.0, bp_high or 3000.0, SAMPLE_RATE)
    bandpass.enabled = bool(bp_low and bp_high)

    strength = min(max(args.strength, 0.0), 1.0)

    agc = AGC(
        sample_rate=SAMPLE_RATE,
        target_rms=args.agc_target,
        attack_ms=args.agc_attack_ms,
        release_ms=args.agc_release_ms,
        max_gain_db=args.agc_max_gain_db,
        min_gain_db=args.agc_min_gain_db,
    ) if args.agc else None

    in_buf = deque()
    out_buf = deque()

    def callback(indata, outdata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)

        mono = indata[:, 0] if indata.ndim > 1 else indata.ravel()
        in_buf.extend(mono.tolist())

        # Process every complete 480-sample frame currently buffered.
        while len(in_buf) >= FRAME_SIZE:
            chunk = np.asarray([in_buf.popleft() for _ in range(FRAME_SIZE)], dtype=np.float32)
            chunk = deemph.process(chunk)
            if bandpass is not None:
                chunk = bandpass.process(chunk)
            if agc is not None:
                chunk = agc.process(chunk)
            pre_denoise = chunk
            chunk = engine.process(chunk)
            if strength < 1.0:
                chunk = strength * chunk + (1.0 - strength) * pre_denoise
            chunk = soft_limit(chunk)
            out_buf.extend(chunk.tolist())

        # Pull exactly `frames` samples for this callback; pad with silence
        # only if the model pipeline hasn't produced enough yet (e.g. the
        # very first callback or two while the buffer fills).
        n = min(len(out_buf), frames)
        for i in range(n):
            outdata[i, 0] = out_buf.popleft()
        if n < frames:
            outdata[n:, 0] = 0.0

    print(f"Mode: {args.mode} | de-emphasis: {'ON' if deemph.enabled else 'OFF'} (coeff {deemph_coeff}) | "
          f"AGC: {'on' if args.agc else 'off'} | bandpass: "
          f"{f'{bandpass.low_hz:.0f}-{bandpass.high_hz:.0f}Hz' if bandpass.enabled else 'off'} | "
          f"strength: {strength}")
    print(f"Model frame size: {FRAME_SIZE} samples (~10ms) @ {SAMPLE_RATE} Hz.")
    print(f"PortAudio blocksize: {'auto' if args.blocksize == 0 else args.blocksize} | latency hint: {args.latency}")
    print("Wear headphones. Press Ctrl+C to stop.")

    with sd.Stream(
        samplerate=SAMPLE_RATE,
        blocksize=args.blocksize,
        latency=args.latency,
        channels=1,
        dtype="float32",
        device=(args.in_dev, args.out_dev),
        callback=callback,
    ):
        listener = threading.Thread(target=command_listener, args=(deemph, agc, bandpass), daemon=True)
        listener.start()
        try:
            while True:
                sd.sleep(1000)
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
