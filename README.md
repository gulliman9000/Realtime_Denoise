# Realtime AI Noise Reduction

Real-time speech denoising for a live audio stream (mic, SDR/discriminator
audio, etc.) with a processing chain built for radio use: de-emphasis,
a bandpass filter matched to voice bandwidth, AGC, a neural denoiser
(RNNoise or DPDFNet), a wet/dry blend, and a minimum-gain floor to stop
the denoiser from muting weak/fading speech.

Two entry points:

- **`realtime_denoise.py`** — command-line tool, live keyboard toggles
- **`realtime_denoise_gui.py`** — PySide6 GUI: device pickers, live sliders,
  input/output VU meters, bypass button

## Install

```bash
pip install -r requirements.txt
```

On Windows, if installing PySide6 fails with a long-path / `.obj` file
error, install the lighter `PySide6-Essentials` package instead of full
`PySide6` (this project only uses QtWidgets/QtCore/QtGui, not the
QML/Addons package that trips the path-length limit):

```bash
pip install PySide6-Essentials
```

If that still fails, enable Windows long-path support (run PowerShell as
Administrator):

```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

## Usage

### GUI

```bash
python realtime_denoise_gui.py
```

Pick your input/output devices, choose a mode preset, hit Start. All
sliders/checkboxes update live while the stream is running. The Bypass
button routes raw input straight to output for instant A/B comparison.

### Command line

```bash
python realtime_denoise.py --list                     # list audio devices
python realtime_denoise.py --in 2 --out 4              # basic run
python realtime_denoise.py --mode nbfm --strength 0.6  # FM repeaters/direct
python realtime_denoise.py --mode ssb --min-gain-db -20  # weak SSB DX
```

Live keyboard controls while running (type a letter/number + Enter):

| Key | Action |
|---|---|
| `d` | toggle de-emphasis |
| `a` | toggle AGC |
| `b` | toggle bandpass |
| `1` | switch to NBFM preset |
| `2` | switch to SSB preset |
| `q` | quit |

## Signal chain

```
input -> de-emphasis -> bandpass -> AGC -> denoiser -> min-gain floor -> wet/dry blend -> soft limiter -> output
```

### Mode presets

| Mode | De-emphasis | Bandpass | Use case |
|---|---|---|---|
| `none` | on, 0.95 coeff | off | manual configuration |
| `nbfm` | on | 300–3000 Hz | narrowband FM repeaters/direct contacts |
| `ssb` | **off** (no TX pre-emphasis) | 300–2900 Hz | SSB, matched to typical TX audio filtering |

### Key parameters

- **`--strength`** (0–1): wet/dry blend of denoiser output vs. the
  pre-denoiser signal. Lower it (0.5–0.7) on already-strong signals to
  reduce neural-denoiser artifacts; keep at 1.0 for weak/noisy signals.
- **`--min-gain-db`**: floor on how much the denoiser may suppress a
  sample. Default `-60` (effectively no floor — full suppression allowed).
  On fading/weak signals (SSB, distant FM), the denoiser's own confidence
  gating can mistake a weak-but-real syllable for noise-only and mute it
  completely; raising this to `-15` to `-20` lets faded speech through at
  reduced volume instead of disappearing.
- **`--backend`**: `rnnoise` (tiny RNN, negligible CPU/latency) or
  `dpdfnet` (heavier model, better on hard noise, CPU-only ONNX inference).

## Example Audio

<video controls width="800">
    <source src="Realtime Denoise Example.mp4" type="video/mp4">
</video>

## Notes / known limitations

- Use headphones for direct mic monitoring — routing mic straight to
  speakers will feed back.
- GPU acceleration isn't meaningfully available here: RNNoise is too small
  to benefit (transfer overhead would dominate), and the `dpdfnet` pip
  package is CPU-only ONNX inference per its own docs. Real GPU
  acceleration would require running DPDFNet's PyTorch checkpoint directly
  on CUDA with a custom streaming loop — not provided out of the box.
- Generic speech-enhancement models (RNNoise, DPDFNet) are trained on
  room/mic noise, not ionospheric/atmospheric static — for weak SSB
  signals, the bandpass filter and min-gain floor generally help more than
  pushing the neural stage harder.
- Changing input/output device in the GUI requires Stop then Start.