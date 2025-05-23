# Pre-built Mamba SSM Wheel

This repository contains a pre-built wheel for `mamba-ssm` that works without compilation issues.

## Installation

### Google Colab
```python
!pip install https://github.com/yourusername/mamba-wheels/raw/main/mamba_ssm-2.2.2-cp311-cp311-linux_x86_64.whl
```

### Local Installation
```bash
pip install https://github.com/yourusername/mamba-wheels/raw/main/mamba_ssm-2.2.2-cp311-cp311-linux_x86_64.whl
```

## Wheel Details
- **Version**: 2.2.2
- **Python**: 3.11
- **Platform**: Linux x86_64
- **Built from**: Working Singularity container

## Usage
```python
from mamba_ssm import Mamba
import torch

model = Mamba(d_model=256)
x = torch.randn(2, 100, 256)
output = model(x)
```

## Compatibility
- ✅ Google Colab
- ✅ Linux x86_64 with Python 3.11
- ⚠️ May work with Python 3.10, 3.12 (untested)
- ❌ Not compatible with Windows/macOS (different wheel needed)