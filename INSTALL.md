# Environment Setup Guide for LTC Training with Robomimic/Robosuite

## Quick Start

### 1. Create Conda Environment

```bash
conda create -n robomimic_venv python=3.10
conda activate robomimic_venv
```

### 2. Install PyTorch with CUDA Support

**For CUDA 12.1:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CUDA 11.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CPU only:**
```bash
pip install torch torchvision torchaudio
```

### 3. Install Core Dependencies

```bash
cd /work/liquid_time-constant_networks
pip install -r requirements.txt
```

### 4. Install Robomimic and Robosuite from Source

```bash
# Install robomimic
cd /work/robosuite/robomimic
pip install -e .

# Install robosuite
cd /work/robosuite
pip install -e .
```

### 5. Verify Installation

```bash
# Test PyTorch
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

# Test robosuite
python -c "import robosuite; print('Robosuite version:', robosuite.__version__)"

# Test ncps
python -c "import ncps; print('NCP/LTC imported successfully')"
```

## Alternative: Full Installation

If you want all dependencies (including optional packages):

```bash
pip install -r requirements_full.txt
```

## Troubleshooting

### Issue: ModuleNotFoundError for robosuite

**Solution:**
```bash
export PYTHONPATH="/work/robosuite:$PYTHONPATH"
```

Or add to your `.bashrc`:
```bash
echo 'export PYTHONPATH="/work/robosuite:$PYTHONPATH"' >> ~/.bashrc
source ~/.bashrc
```

### Issue: CUDA version mismatch

**Solution:**
Check your CUDA version:
```bash
nvidia-smi
```

Then install the matching PyTorch version from https://pytorch.org/get-started/locally/

### Issue: EGL/OSMesa rendering errors

**Solution:**
For training (no rendering needed), use:
```python
env = suite.make(
    "Lift",
    has_renderer=False,
    has_offscreen_renderer=False,
    use_camera_obs=False,
)
```

### Issue: Protobuf version conflicts

**Solution:**
Force install the correct version:
```bash
pip install protobuf==3.20.3 --force-reinstall
```

## Package Version Summary

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.10.x | Base environment |
| PyTorch | >=2.0.0 | Deep learning framework |
| pytorch-lightning | >=2.0.0 | Training framework |
| ncps | >=1.0.0 | LTC/NCP neural networks |
| tensorflow | 2.14.0 | Robomimic compatibility |
| mujoco | >=3.0.0 | Physics simulation |
| robosuite | 1.5.1 | Robotics environments |
| hydra-core | >=1.3.0 | Configuration management |

## Testing Your Setup

### Test 1: Train on UCI-HAR Dataset

```bash
cd /work/liquid_time-constant_networks
python train.py experiment=uci_har/ncp dataset=uci_har train.seed=1
```

### Test 2: Train on Robomimic Dataset

```bash
cd /work/liquid_time-constant_networks
./train_scripts/robomimic/ncp_lift.sh
```

### Test 3: Evaluate with Rollouts (requires robosuite)

```bash
cd /work/liquid_time-constant_networks
./eval_robosuite.sh
```

## Known Issues

1. **torchvision image.so warning**: This is a known issue with pre-built torchvision wheels. It doesn't affect training. To fix, rebuild torchvision from source or ignore the warning.

2. **Pydantic warnings**: UnsupportedFieldAttributeWarning can be safely ignored. These are compatibility warnings between pydantic and other packages.

3. **TF32 precision warnings**: PyTorch 2.9+ changed default precision settings. These warnings can be safely ignored for most use cases.

## Conda Environment Export

To export your exact environment:

```bash
conda activate robomimic_venv
conda env export > environment.yml
pip freeze > requirements_frozen.txt
```

To recreate from export:

```bash
conda env create -f environment.yml
```

## Docker Alternative

If you prefer Docker:

```bash
# Build image
docker build -t ltc-robomimic .

# Run container
docker run --gpus all -it -v /work:/work ltc-robomimic
```

## Additional Resources

- PyTorch Installation: https://pytorch.org/get-started/locally/
- Robosuite Documentation: https://robosuite.ai/
- NCP/LTC Paper: https://www.nature.com/articles/s42256-020-00237-3
- Hydra Documentation: https://hydra.cc/

## License

See individual package licenses for details.

## Citation

If you use this codebase, please cite:

```bibtex
@article{hasani2020liquid,
  title={Liquid time-constant networks},
  author={Hasani, Ramin and Lechner, Mathias and Amini, Alexander and Rus, Daniela and Grosu, Radu},
  journal={arXiv preprint arXiv:2006.04439},
  year={2020}
}

@inproceedings{robosuite2020,
  title={robosuite: A modular simulation framework and benchmark for robot learning},
  author={Zhu, Yuke and Wong, Josiah and Mandlekar, Ajay and Mart{\'i}n-Mart{\'i}n, Roberto and Joshi, Abhishek and Nasiriany, Soroush and Zhu, Yifeng},
  booktitle={arXiv preprint arXiv:2009.12293},
  year={2020}
}
```
