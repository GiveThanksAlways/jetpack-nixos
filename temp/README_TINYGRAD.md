# TinyGrad on Jetson

## TL;DR - Deploy Options

**Option 1: Manual Dev (fastest)**
```bash
nix flake clone github:path/to/jetpack-nixos#tinygrad-flake.nix
nix develop
pip install -e vendor/tinygrad
CUDA=1 python examples/jetson_device_check.py
```

**Option 2: Build & Deploy (clean, isolated)**
```bash
nix develop github:path/to/jetpack-nixos#tinygrad-deploy-flake.nix
CUDA=1 tinygrad-run examples/jetson_device_check.py
```

**Option 3: Complete (one command, everything handled)**
```bash
nix run github:path/to/jetpack-nixos#tinygrad-complete
```

## Files

- `tinygrad-flake.nix` - Dev environment (manual)
- `tinygrad-deploy-flake.nix` - Build in Nix, run with Python
- `tinygrad-complete-flake.nix` - Fully automated, one-shot deployment
