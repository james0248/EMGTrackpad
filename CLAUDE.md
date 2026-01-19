# Project: sEMG Neural Interface Replication (Nature 2025)

## 1. Project Overview
This project aims to replicate and implement the non-invasive neural interface methodology described in **Kaifosh et al. (Nature, 2025)**. The system decodes motor intention from surface Electromyography (sEMG) signals collected from the wrist, focusing on continuous control and discrete gesture recognition (e.g., Trackpad Click).

## 2. Done
- **Data collection:** MindRove EMG → HDF5 with synchronized mouse events
- **Preprocessing:** 40Hz highpass, 60Hz notch filter
- **Feature extraction:** RMS with temporal sub-windowing (50ms segments)
- **Model:** MLP click classifier (8ch → 256 → 256 → 3 classes)
- **Training:** Hydra config, class balancing, checkpointing

## 3. Next Goal
Improve RMS MLP perfomance
- [x] Implement `FrequencyRMSFeature`
- [ ] Create MLP model using the new feature


## 4. Convention
- Use `uv` for tooling and executing
- Keep `__init__.py` empty
