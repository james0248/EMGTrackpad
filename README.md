## EMGTrackpad

This repo contains:

- **Python** scripts for EMG + input collection and offline analysis (under `src/emg-trackpad/`)
- **Web app** for structured task / stimulus presentation (under `apps/platform/`)

### Web app (task playground)

```bash
cd apps/platform
bun install
bun dev
```

Then open `http://localhost:3000`.

### Python (collector / analysis)

```bash
# from repo root
uv run python src/emg-trackpad/track.py
```

Data files are written to `data/` (ignored by git).
