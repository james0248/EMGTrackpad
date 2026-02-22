# EMGTrackpad

Decode motor intention from forearm surface EMG signals to control a macOS trackpad in real time.

<p align="center">
  <img src="resources/demo.webp" alt="EMGTrackpad demo" width="600">
</p>

## Overview

EMGTrackpad captures surface EMG signals from a [MindRove](https://mindrove.com/) armband, trains neural networks to decode cursor movement and discrete actions (click, scroll), and drives the macOS trackpad in real time.

The repo has two components:

- **Model** (`src/emg/`) — data collection, signal processing, model training, and real-time inference
- **Platform** (`apps/platform/`) — web app that presents structured tasks (e.g., click targets, drag paths) during data collection

## Setup

### Prerequisites

- macOS (Apple Silicon)
- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- [MindRove](https://mindrove.com/) EMG armband
- [Bun](https://bun.sh/) (for the Platform only)

### Installation

```bash
git clone https://github.com/your-username/EMGTrackpad.git
cd EMGTrackpad && uv sync && uv pip install .
cd apps/platform && bun i
```

## Data Collection

Collect synchronized EMG and trackpad events into an HDF5 session file:

```bash
uv run python src/emg/track.py
```

Sessions are saved to `data/session_YYYYMMDD_HHMMSS.h5`.

## Training

Train a continuous controller (cursor movement + click/scroll actions):

```bash
uv run python -m emg.train_controller --config-name channel_attention
```

Available configs: `rms`, `freq_rms`, `channel_attention`, `freq_rms_lstm`, `channel_attention_lstm`

## Inference

Run the trained model to control the macOS trackpad:

```bash
uv run python -m emg.inference.controller checkpoint=path/to/checkpoint.pt
```

## Platform

Run the web app for presenting structured tasks while recording EMG data:

```bash
cd apps/platform
bun dev
```

Then open `http://localhost:3000`.
