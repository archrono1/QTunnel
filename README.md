# Quantum Tunneling Simulation

A fully standalone 2D quantum mechanical simulation of a particle tunneling through potential barriers. Built with NumPy and Matplotlib, with optional GPU acceleration via CuPy.

---

## What It Simulates

A Gaussian wavepacket travels toward zero, one, or two rectangular potential barriers. Number of potential barriers is chosen at runtime.

| Mode | Setup | Physics |
|---|---|---|
| Free propagation | No barriers | Wavepacket spreads via dispersion only |
| Single barrier | One wall at V₀ = 1.7 eV | Standard tunneling, exponential decay through the wall |
| Double barrier | Two walls separated by a quantum well | Resonant tunneling, transmission spikes at resonance energies |

The wavepacket travels at **E = 0.8 eV**, which is below the barrier height of **1.7 eV**, normally unable to cross, but quantum mechanically able to tunnel through. In double-barrier mode, the well between the two walls supports quasi-bound states at specific resonance energies; when the particle energy matches one of these, transmission probability jumps dramatically. This is the operating principle behind resonant tunneling diodes (RTDs).

**Double barrier geometry:**

```
|<-- a=3Å -->|<-- b=5Å -->|<-- a=3Å -->|
             quantum well
V₀=1.7 eV                 V₀=1.7 eV
```

---

## Physics: Split-Step Fourier Method

The time-dependent Schrödinger equation is solved using **Strang splitting** (split-step Fourier method). Each timestep applies three operators in sequence:

```
ψ  →  e^{-i V dt/2ℏ} ψ           (half-step: potential, position space)
   →  FFT  →  e^{-i ℏk²dt/2m} ψ̃  (full-step: kinetic, momentum space)
   →  IFFT →  e^{-i V dt/2ℏ} ψ   (half-step: potential, position space)
```

Second-order accurate in time. Unitary — norm is conserved to machine precision.

The simulation runs in **atomic units** (`ℏ = mₑ = 1`), which avoids floating point issues from mixing SI-scale constants.

---

## Visualization

The wavefunction is rendered using **HSV coloring**:

- **Hue** → quantum phase `arg(ψ)`
- **Brightness** → amplitude `|ψ|` (zero amplitude = pure black)

This encodes both the probability density and the phase structure of the wavepacket simultaneously (and also looks pretty). Green vertical lines mark the potential barriers.

---

## Parameters

| Parameter | Value | Description |
|---|---|---|
| Grid points | 256 × 256 or 512 × 512 | Chosen at runtime |
| Spatial extent | 700 Å | Domain size |
| Wavepacket energy | 0.8 eV | E₀ |
| Wavepacket width | 25 Å | σ |
| Barrier height | 1.7 eV | V₀ |
| Barrier width | 3 Å | a |
| Well width | 5 Å | b (double barrier only) |
| Total timesteps | 8000 | Sub-steps per run |
| Stored frames | 800 | Frames written to memory |

---

## Requirements

```
numpy
matplotlib
```

Install with:

```bash
pip install numpy matplotlib
```

**Optional — GPU acceleration:**

```bash
pip install cupy-cuda12x   # match your CUDA version
```

See [CuPy installation docs](https://docs.cupy.dev/en/stable/install.html) for other CUDA versions.

**Optional — export:**

```bash
pip install pillow          # GIF export
# FFmpeg must be installed separately for MP4 export
# https://ffmpeg.org/download.html
```

Python 3.8+ recommended.

---

## Usage

```bash
python simulation.py
```

You will be prompted before the simulation runs (I got sick of changing numbers):

```
Barrier Configuration:
  0. No barriers (free propagation)
  1. Single barrier
  2. Double barrier
Select number of barriers (0/1/2):

Grid Resolution:
  256  (medium)
  512  (fine, slow)
Select grid resolution (256/512):

Compute Backend:
  numpy  (CPU)
  cupy   (GPU — requires NVIDIA GPU + CUDA)
Select backend (numpy/cupy):
```

After the simulation finishes, you'll be prompted to export as GIF, MP4, or skip.

### Performance note

| Hardware | N=256 | N=512 |
|---|---|---|
| Modern laptop CPU (NumPy) | ~2–5 min | ~15–30 min |
| NVIDIA GPU (CuPy) | ~10–20 sec | ~1–2 min |

Halving N from 512 to 256 cuts runtime by ~16× with minimal visual difference at normal playback speed.

---

## File Structure

```
.
├── simulation.py     
├── requirements.txt
└── README.md
```

---

## Background Reading

- [Split-operator method — Algorithm Archive](https://www.algorithm-archive.org/contents/split-operator_method/split-operator_method.html)
- [Split-step method — Wikipedia](https://en.wikipedia.org/wiki/Split-step_method)
- [Resonant tunneling diode — Wikipedia](https://en.wikipedia.org/wiki/Resonant-tunneling_diode)
- [QMSolve — original library](https://github.com/quantum-visualizations/qmsolve)

---

## License

MIT
