# NV Backend on Jetson Orin AGX 64GB — Investigation Report

**Date:** 2026-02-10  
**Device:** NVIDIA Jetson Orin AGX 64GB Developer Kit  
**Driver:** NVIDIA UNIX Open Kernel Module for aarch64 540.4.0 (L4T/Tegra)  
**Kernel:** 5.15.148  
**CUDA:** 12.6  
**tinygrad commit:** `cc9bf8ccbc0b7eb0e3b8510d475fa56263ef8cab`

## TL;DR

**NV backend cannot work on Jetson Orin.** Both interfaces (NVK and PCI) fail for fundamental architectural reasons — the Orin iGPU is not a discrete PCI GPU and doesn't expose UVM. CUDA backend works fine and is the correct path.

---

## Error Output

```bash
NV=1 python3 -c 'from tinygrad import Tensor; print(Tensor([1,2,3]).numpy())'
```

```
ExceptionGroup: No interface for NV:0 is available (2 sub-exceptions)
  1) FileNotFoundError: [Errno 2] No such file or directory: '/dev/nvidia-uvm'
  2) IndexError: list index out of range
```

---

## Interface 1: NVK — Missing `/dev/nvidia-uvm`

NVK uses the NVIDIA kernel driver via `/dev/nvidiactl`, `/dev/nvidia0`, and `/dev/nvidia-uvm`.

**Problem:** The Jetson L4T driver does not ship the `nvidia-uvm` kernel module.

```bash
# What exists:
$ ls /dev/nvidia*
/dev/nvidia0  /dev/nvidiactl

# UVM module does NOT exist for this kernel:
$ modinfo nvidia-uvm
modinfo: ERROR: Module nvidia-uvm not found.

# The nvidia module loaded is the Tegra variant:
$ lsmod | grep nvidia
nvidia               1613824  0
tegra_dce             114688  2 nvidia

# A desktop nvidia-uvm.ko exists in nix store but for wrong kernel:
$ find /nix/store -name 'nvidia-uvm*' 2>/dev/null
/nix/store/.../lib/modules/6.12.69/misc/nvidia-uvm.ko   # ← kernel 6.12.69, NOT 5.15.148

# Driver version confirms Tegra/L4T variant:
$ cat /proc/driver/nvidia/version
NVRM version: NVIDIA UNIX Open Kernel Module for aarch64  540.4.0
```

**Root cause:** Jetson uses a Tegra-specific NVIDIA driver. The GPU memory management goes through `nvgpu`/`host1x`, not UVM. UVM is a desktop/server GPU feature.

---

## Interface 2: PCI — iGPU Not on PCI Bus

PCI interface scans `/sys/bus/pci/devices` for NVIDIA GPUs with class `0x03` (display controller).

**Problem:** The Orin GPU is an integrated SoC GPU on the platform bus, not a PCIe device.

```bash
# PCI devices on this Jetson:
$ for d in /sys/bus/pci/devices/*/; do
    echo "$(cat $d/vendor) $(cat $d/device) $(cat $d/class) $(basename $d)"
  done
0x10de 0x229e 0x060400 0001:00:00.0   # PCIe bridge
0x10ec 0xc822 0x028000 0001:01:00.0   # Realtek WiFi
0x10de 0x229c 0x060400 0004:00:00.0   # PCIe bridge
0xc0a9 0x560a 0x010802 0004:01:00.0   # NVMe SSD

# No GPU (class 0x03) on PCI bus at all.

# The GPU is on the platform bus:
$ cat /sys/class/drm/card0/device/uevent
DRIVER=drm
OF_NAME=host1x
OF_FULLNAME=/bus@0/host1x@13e00000
OF_COMPATIBLE_0=nvidia,tegra234-host1x
```

**The tinygrad PCI backend only supports discrete GPUs:**

```python
# From ops_nv.py line 542:
devices=[(0xff00, [0x2200, 0x2400, 0x2500, 0x2600, 0x2700, 0x2800, 0x2b00, 0x2c00, 0x2d00, 0x2f00])]
# These are Ampere/Ada/Blackwell DISCRETE GPU device IDs (RTX 3000/4000/5000 series)
```

**GPU identity:**
```bash
$ nvidia-smi -q | head -10
GPU 00000000:00:00.0
    Product Name                          : Orin (nvgpu)
    Product Architecture                  : Ampere
```

The GPU is Ampere architecture but accessed via `nvgpu` platform driver, not PCIe.

---

## Why CUDA Works

The CUDA backend uses `libcuda.so` + `libnvrtc.so` (runtime compilation). These libraries abstract away whether the GPU is discrete or integrated — NVIDIA's CUDA driver handles the Tegra/iGPU path internally. This is why `CUDA=1` works perfectly.

---

## Roadmap: Getting NV Backend Working on Jetson

This would require a new "Tegra" interface (`TegraIface`) alongside `NVKIface` and `PCIIface`. Here's the iteration loop:

### Phase 1: Understand the Tegra GPU Interface

1. **Map the kernel driver interface**
   - Study `/dev/nvidia0` and `/dev/nvidiactl` ioctls on Jetson
   - Compare with desktop NVK ioctls to find overlap
   - Key question: does the Jetson's `/dev/nvidiactl` support the same RM (Resource Manager) API?
   ```bash
   # Check if RM ioctls work at all:
   python3 -c "
   from tinygrad.runtime.support.hcq import FileIOInterface
   import os
   fd = FileIOInterface('/dev/nvidiactl', os.O_RDWR | os.O_CLOEXEC)
   print('opened nvidiactl OK')
   "
   ```

2. **Check if NVK partially works without UVM**
   - The NVK interface opens 3 device files: `/dev/nvidiactl`, `/dev/nvidia-uvm`, `/dev/nvidia0`
   - UVM is used for memory management — could an alternative memory path use Tegra's unified memory?
   - Try patching `NVKIface.__init__` to skip UVM init and see how far it gets

### Phase 2: Build a TegraIface (Fail Loop)

3. **Stub out a minimal `TegraIface`**
   - Inherit from `NVKIface` or create fresh
   - Skip `/dev/nvidia-uvm` entirely
   - Use the Tegra unified memory model (CPU and GPU share physical memory)
   - Map GPU memory via `/dev/nvidia0` mmap or `nvmap`/`nvhost` IOCTLs

4. **Iterate on RM API calls**
   - Each `rm_alloc` / `rm_control` call may fail differently on Tegra
   - Log every IOCTL, note which ones succeed/fail
   - The Tegra RM may support a subset of desktop RM commands
   ```bash
   # Enable debug logging:
   NV_DEBUG=4 NV=1 python3 -c "..."
   ```

5. **Memory allocation without UVM**
   - UVM provides `UVM_CREATE_RANGE_GROUP`, `UVM_MAP_EXTERNAL_ALLOCATION`, etc.
   - On Tegra, memory is unified — you may be able to use `NvRmMemAlloc` + direct mmap
   - Or use the CUDA driver's memory allocation and pass pointers to the NV command queue

### Phase 3: GPU Command Submission

6. **GPFIFO setup**
   - The NV backend submits work via GPU FIFOs (ring buffers)
   - Check if the Tegra driver supports GPFIFO allocation via the same RM path
   - The `AMPERE_CHANNEL_GPFIFO_A` class may or may not be supported on Orin

7. **Compute kernel dispatch**
   - QMD (Queue Meta Data) format should be the same (Ampere SM)
   - Shader compilation via NVRTC should work (same CUDA 12.6)
   - The challenge is getting the command queue to the GPU

### Phase 4: Testing & Upstream

8. **Validate with simple kernels first**
   - Start with a trivial kernel (copy buffer, add two vectors)
   - Gradually move to full tensor operations
   
9. **Upstream to tinygrad**
   - Would need to add Jetson/Tegra device detection
   - Add `TegraIface` as a third option in `_select_iface()`
   - Modify `PCIIfaceBase` scan to also check platform bus devices

### Key Resources

- [NVIDIA open-gpu-kernel-modules](https://github.com/NVIDIA/open-gpu-kernel-modules) — RM API source
- [L4T kernel source](https://developer.nvidia.com/embedded/jetson-linux) — Tegra-specific driver
- `/dev/nvhost-*` and `/dev/nvmap` — Tegra-specific device nodes
- `tinygrad/runtime/support/c.py` — DLL loading mechanism
- `tinygrad/runtime/ops_nv.py` — NVKIface and PCIIface implementations

### Difficulty Assessment

**Hard.** The main challenge is that tinygrad's NV backend assumes either:
- A desktop NVIDIA kernel driver with UVM (NVK), or
- Direct PCI BAR access to a discrete GPU (PCI)

Jetson is neither. It's an SoC with unified memory and a platform-bus GPU. The RM API *might* partially overlap with desktop, but memory management is fundamentally different. A realistic estimate is **weeks of iteration** for someone familiar with both tinygrad internals and Tegra driver architecture.
