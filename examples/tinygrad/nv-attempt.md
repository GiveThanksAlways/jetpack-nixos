# NV Backend on Jetson Orin AGX 64GB — Investigation Report

**Date:** 2026-02-10  
**Device:** NVIDIA Jetson Orin AGX 64GB Developer Kit  
**JetPack:** 6 (L4T)  
**Drivers loaded:** `nvidia.ko` 540.4.0 (Tegra variant) + `nvgpu.ko` (actual compute) + `nvmap.ko` (memory)  
**Kernel:** 5.15.148  
**CUDA:** 12.6  
**GPU:** Orin iGPU, Ampere arch, SM 8.7, platform bus `17000000.gpu`  
**tinygrad commit:** `cc9bf8ccbc0b7eb0e3b8510d475fa56263ef8cab`

## TL;DR

**NV backend cannot work on Jetson Orin AGX (JetPack 6)** without significant porting work. Both interfaces (NVK and PCI) fail for fundamental architectural reasons. The Orin uses **two separate kernel drivers** — `nvidia.ko` provides a partial RM API (display/modesetting), while `nvgpu.ko` + `nvmap.ko` handle actual GPU compute and memory. NVIDIA's `nvidia-uvm` module is **only available on JetPack 7 (Thor)**. CUDA backend is the correct path today.

---

## System Architecture

The Jetson Orin has a fundamentally different GPU driver stack than desktop NVIDIA:

```
Desktop NVIDIA (dGPU)                    Jetson Orin (JetPack 6)
─────────────────────                    ──────────────────────
nvidia.ko (full RM API)                  nvidia.ko (PARTIAL RM – display only)
nvidia-uvm.ko (memory mgmt)             nvgpu.ko (actual GPU compute driver)
/dev/nvidiactl                           nvmap.ko (memory allocator)
/dev/nvidia-uvm                          
/dev/nvidia0                             /dev/nvidiactl (limited RM)
GPU on PCIe bus                          /dev/nvidia0 (limited RM)
                                         /dev/nvhost-gpu (compute ioctls)
                                         /dev/nvhost-ctrl-gpu (GPU control)
                                         /dev/nvhost-as-gpu (address space)
                                         /dev/nvmap (memory)
                                         GPU on platform bus @ 0x17000000
```

**nvidia-uvm is JetPack 7 (Thor) only** — confirmed in `jetpack-nixos/modules/default.nix` line 283:
```nix
boot.kernelModules = if (jetpackAtLeast "7") then [ "nvidia-uvm" ] else [ "nvgpu" ];
```
And line 291:
```nix
softdep nvidia pre: governor_pod_scaling post: nvidia-uvm  # JetPack 7 only
```

---

## Error Output

```bash
$ NV=1 python3 -c 'from tinygrad import Tensor; print(Tensor([1,2,3]).numpy())'

ExceptionGroup: No interface for NV:0 is available (2 sub-exceptions)
  1) FileNotFoundError: [Errno 2] No such file or directory: '/dev/nvidia-uvm'
  2) IndexError: list index out of range
```

---

## Interface 1: NVK — Partial RM API, No UVM

NVK requires `/dev/nvidiactl`, `/dev/nvidia-uvm`, and `/dev/nvidia0`.

### What works (RM basic ops)

```bash
# Device nodes exist:
$ ls /dev/nvidia*
/dev/nvidia0  /dev/nvidiactl

# nvidia.ko IS loaded (Tegra variant):
$ lsmod | grep nvidia
nvidia               1613824  0
tegra_dce             114688  2 nvidia
```

**RM API test results** (ran actual tinygrad RM ioctls via `NV_ESC_RM_ALLOC` / `NV_ESC_RM_CONTROL`):

| Operation | ioctl | Status | Meaning |
|---|---|---|---|
| Root client alloc | `NV01_ROOT_CLIENT` | **0** | **NV_OK** ✓ |
| Card info | `NV_ESC_CARD_INFO` | - | GPU found (gpu_id=131072, valid=1, minor=0) ✓ |
| GPU ID info | `NV0000_CTRL_CMD_GPU_GET_ID_INFO_V2` | **31** | NV_ERR_INVALID_ARGUMENT ✗ |
| Device alloc | `NV01_DEVICE_0` | **34** | **NV_ERR_INVALID_CLASS** ✗ |
| Subdevice alloc | `NV20_SUBDEVICE_0` | **87** | NV_ERR_OBJECT_NOT_FOUND ✗ (cascading failure) |
| Class list query | `NV0080_CTRL_CMD_GPU_GET_CLASSLIST` | **61** | NV_ERR_INVALID_POINTER ✗ (cascading failure) |

**The critical failure is `NV01_DEVICE_0` → NV_ERR_INVALID_CLASS (34).** This is the core GPU device abstraction that tinygrad's NVKIface builds everything on (device → subdevice → VA space → channels → compute). The Tegra nvidia.ko driver doesn't support this class at all.

### What doesn't exist

```bash
# nvidia-uvm module is NOT available for this kernel:
$ sudo modprobe nvidia-uvm
modprobe: FATAL: Module nvidia-uvm not found in directory /run/booted-system/kernel-modules/lib/modules/5.15.148

# Only these nvidia modules exist:
$ ls /run/booted-system/kernel-modules/lib/modules/5.15.148/updates/
nvidia.ko.xz  nvidia-drm.ko.xz  nvidia-modeset.ko.xz  nvgpu.ko.xz  nvhwpm.ko.xz
# ↑ NO nvidia-uvm.ko
```

### Why NV01_DEVICE_0 fails

The `nvidia.ko` on Tegra/JetPack 6 is a **display-oriented RM** (modesetting, DRM, display engine). It exposes the RM ioctl endpoint (`/dev/nvidiactl`) and can enumerate GPUs, but the **compute-side RM object classes** (`NV01_DEVICE_0`, `NV20_SUBDEVICE_0`, `AMPERE_CHANNEL_GPFIFO_A`, etc.) are not implemented in this driver. Those capabilities live in `nvgpu.ko` via completely different ioctls on `/dev/nvhost-gpu`.

---

## Interface 2: PCI — iGPU Not on PCI Bus

PCI interface scans `/sys/bus/pci/devices` for NVIDIA GPUs with class `0x03` (display controller).

**Problem:** The Orin GPU is an integrated SoC GPU on the platform bus, not a PCIe device.

```bash
# PCI devices on this Jetson — no GPU (class 0x03):
$ for d in /sys/bus/pci/devices/*/; do
    echo "$(cat $d/vendor) $(cat $d/device) $(cat $d/class) $(basename $d)"
  done
0x10de 0x229e 0x060400 0001:00:00.0   # PCIe bridge
0x10ec 0xc822 0x028000 0001:01:00.0   # Realtek WiFi
0x10de 0x229c 0x060400 0004:00:00.0   # PCIe bridge
0xc0a9 0x560a 0x010802 0004:01:00.0   # NVMe SSD

# GPU is on platform bus, addressed by MMIO:
$ cat /sys/class/devfreq/17000000.gpu/cur_freq
306000000

# tinygrad PCI backend only knows discrete GPU device IDs:
# ops_nv.py: devices=[(0xff00, [0x2200..0x2f00])]  ← RTX 3000/4000/5000 series
```

**GPU identity:**
```bash
$ nvidia-smi
|   0  Orin (nvgpu)                  N/A  | N/A              N/A |                  N/A |
# ↑ "nvgpu" confirms this GPU is accessed via nvgpu.ko, NOT nvidia.ko's PCIe path
```

---

## The Actual GPU Interface: nvgpu + nvhost + nvmap

On JetPack 6 Orin, the actual GPU compute stack is:

```bash
$ lsmod | grep nvgpu
nvgpu                2793472  0
nvmap                 262144  1 nvgpu

$ ls /dev/nvhost-*gpu* /dev/nvmap
/dev/nvhost-as-gpu              # Address Space management
/dev/nvhost-ctrl-gpu            # GPU control channel
/dev/nvhost-ctxsw-gpu           # Context switch
/dev/nvhost-dbg-gpu             # Debug
/dev/nvhost-gpu                 # GPU submission
/dev/nvhost-nvsched-gpu         # Scheduling
/dev/nvhost-nvsched_ctrl_fifo-gpu
/dev/nvhost-power-gpu           # Power management
/dev/nvhost-prof-ctx-gpu        # Profiling
/dev/nvhost-prof-dev-gpu        
/dev/nvhost-prof-gpu            
/dev/nvhost-sched-gpu           # Scheduling
/dev/nvhost-tsg-gpu             # TSG (Time Slice Group)
/dev/nvmap                      # Memory allocator
```

This is a **completely different ioctl interface** from the desktop nvidia.ko RM API. The nvgpu driver uses its own set of IOCTL codes for:
- Channel management (`/dev/nvhost-gpu`)
- Address space management (`/dev/nvhost-as-gpu`)
- Memory allocation (`/dev/nvmap`)
- GPU control and queries (`/dev/nvhost-ctrl-gpu`)
- TSG (Time Slice Group) scheduling (`/dev/nvhost-tsg-gpu`)

---

## Why CUDA Works

The CUDA backend uses `libcuda.so` + `libnvrtc.so` (runtime compilation). These libraries abstract away the entire driver difference — NVIDIA's CUDA userspace driver knows how to talk to both `nvidia.ko` (desktop) and `nvgpu.ko/nvhost` (Tegra). This is why `CUDA=1` works perfectly on Jetson.

---

## The JetPack 7 (Thor) Question

Per `jetpack-nixos/modules/default.nix` and the project README, JetPack 7 **only supports Thor AGX** (not Orin):

```
|       Device       | JetPack 5 | JetPack 6 | JetPack 7 |
| Jetson Thor AGX    |           |           |     ✓     |
| Jetson Orin AGX    |     ✓     |     ✓     |           |   ← NO JetPack 7
```

JetPack 7 loads `nvidia-uvm` instead of `nvgpu` and has full desktop-style RM. If Thor + JetPack 7 were available, the NV backend would likely work with no or minimal changes. But **this will never be available for Orin AGX** — Orin is stuck on JetPack 5/6 with the nvgpu architecture.

---

## Roadmap: Getting NV-Like Raw Kernel Submission on Jetson Orin

There are three possible approaches, ordered by feasibility:

### Option A: nvgpu Backend (New Backend — Hard, Most Promising)

Write a new tinygrad backend that uses the nvgpu/nvhost/nvmap ioctls directly. This is the "correct" approach for Orin.

#### Phase 1: Reverse-Engineer nvgpu IOCTLs

**Iteration loop:**

1. **Map the ioctl interface**
   - Study the nvgpu kernel source in [NVIDIA's L4T kernel](https://developer.nvidia.com/embedded/jetson-linux)
   - Key files: `drivers/gpu/nvgpu/os/linux/ioctl*.c`
   - Document every ioctl code, struct, and behavior for:
     - `/dev/nvhost-ctrl-gpu` — GPU properties, capabilities, SM version
     - `/dev/nvhost-gpu` — channel open, submit, etc.
     - `/dev/nvhost-as-gpu` — address space map/unmap
     - `/dev/nvhost-tsg-gpu` — TSG (compute queue) management
     - `/dev/nvmap` — memory alloc/free/pin/mmap

2. **Write Python ctypes bindings (autogen)**
   - Similar to how tinygrad has `autogen/nv_570.py` for desktop RM
   - Create `autogen/nvgpu.py` with struct definitions from nvgpu headers
   - Tool: adapt tinygrad's existing header-to-ctypes generation

3. **Test basic operations** (fail loop)
   ```python
   # Pseudo-iteration:
   fd = open("/dev/nvhost-ctrl-gpu", O_RDWR)
   ioctl(fd, NVGPU_GPU_IOCTL_GET_CHARACTERISTICS, ...)  # → does it return SM version?
   
   fd_as = open("/dev/nvhost-as-gpu", O_RDWR) 
   ioctl(fd_as, NVGPU_AS_IOCTL_BIND_CHANNEL, ...)  # → bind to GPU channel
   
   fd_map = open("/dev/nvmap", O_RDWR)
   ioctl(fd_map, NVMAP_IOC_CREATE, ...)  # → allocate memory handle
   ioctl(fd_map, NVMAP_IOC_ALLOC, ...)   # → back it with physical memory
   
   # Each call: check return value, decode error, adjust parameters, retry
   ```

#### Phase 2: Memory Management via nvmap

4. **Implement allocator**
   - nvmap provides handle-based memory allocation
   - Handles can be mapped into CPU VA and GPU VA
   - Key ioctls: `NVMAP_IOC_CREATE` → `NVMAP_IOC_ALLOC` → `mmap()` → `NVGPU_AS_IOCTL_MAP_BUFFER`
   - Orin has unified memory — CPU and GPU share the same DRAM
   - This replaces the entire UVM subsystem

5. **Set up GPU virtual address space**
   - Open `/dev/nvhost-as-gpu`, allocate VA ranges
   - Map nvmap handles into GPU address space
   - Test with known patterns: write from CPU, read GPU VA

#### Phase 3: Command Submission

6. **Create TSG + Channel**
   - Open `/dev/nvhost-tsg-gpu`, create a TSG
   - Open `/dev/nvhost-gpu`, create a GPU channel within the TSG
   - Bind the channel to an address space
   - Allocate a GPFIFO (command ring buffer) via nvmap

7. **Submit compute work**
   - Format QMD (Queue Meta Data) — Ampere format, same as desktop
   - Write push buffer entries pointing to QMD
   - Submit via `NVGPU_IOCTL_CHANNEL_SUBMIT_GPFIFO`
   - Key question: are the QMD fields identical to desktop Ampere?

8. **Shader compilation**
   - Use `libnvrtc.so` (already works via CUDA path) to compile PTX → SASS
   - Parse the ELF output to extract the kernel binary
   - This should be identical to desktop — SM 8.7 PTX assembly is SM 8.7

#### Phase 4: Integration

9. **Build `TegraIface` class**
   - Implement the same interface as `NVKIface` / `PCIIface`
   - Required methods: `rm_alloc` equivalent, memory alloc, GPU mapping, GPFIFO submit
   - Add detection: check for `/dev/nvhost-gpu` existence

10. **Test incrementally**
    - Vector add → matrix multiply → conv2d → full model
    - Compare outputs with CUDA backend for correctness

11. **Upstream to tinygrad**
    - Add to `_select_iface()` in `ops_nv.py`
    - PR to tinygrad with Jetson CI testing

**Difficulty:** Very Hard. Estimated 4-8 weeks for an experienced systems programmer familiar with GPU driver internals. The nvgpu ioctl interface is well-structured but poorly documented outside NVIDIA.

---

### Option B: Hybrid RM + nvmap (Experimental — Uncertain)

Try to use the partial RM API (which does work for root alloc + card info) combined with nvmap for memory, bypassing UVM entirely.

**Approach:**
1. Patch `NVKIface` to skip `/dev/nvidia-uvm` open
2. Replace all `self.uvm()` calls with nvmap equivalents
3. See if any higher-level RM classes work when memory is provided differently
4. **Problem:** `NV01_DEVICE_0` returns INVALID_CLASS — this blocks the entire RM object hierarchy. Without a device object, you can't create subdevices, channels, or anything.
5. **Verdict:** Probably a dead end on JetPack 6. The nvidia.ko RM on Tegra just doesn't support compute classes.

**Difficulty:** Medium effort to try, but high probability of dead end.

---

### Option C: Wait for Platform Changes (Easiest, No Code)

- **JetPack 7 on Orin:** Not planned by NVIDIA. Orin is limited to JetPack 5/6.
- **Thor AGX + JetPack 7:** Would have `nvidia-uvm` and full RM. NV backend should work with minimal changes.
- **Upstream tinygrad nvgpu support:** Watch tinygrad issues/PRs for Jetson support.

**Difficulty:** Zero effort, but depends on external factors.

---

### Recommended Path

For **production use today**: stick with `CUDA=1`. It works, it's stable, NVIDIA supports it.

For **research/hacking**: Option A (nvgpu backend) is the most promising. Start with Phase 1 step 1 — get the nvgpu kernel source and map the ioctl interface. The L4T kernel source tree has the complete nvgpu driver, and the structs/ioctls are defined in headers like:
- `include/uapi/linux/nvgpu.h`
- `include/uapi/linux/nvmap.h`
- `drivers/gpu/nvgpu/include/nvgpu/linux/ioctl*.h`

### Key Resources

- [NVIDIA L4T kernel source](https://developer.nvidia.com/embedded/jetson-linux) — contains full nvgpu driver
- [open-gpu-kernel-modules](https://github.com/NVIDIA/open-gpu-kernel-modules) — desktop RM source (for comparison)
- `/dev/nvhost-*` and `/dev/nvmap` — the actual Tegra GPU interfaces
- `tinygrad/runtime/ops_nv.py` — NVKIface and PCIIface implementations
- `jetpack-nixos/modules/default.nix` line 283 — confirms nvidia-uvm is JetPack 7 only
- `nvidia-smi` shows GPU as "Orin (nvgpu)" — confirming nvgpu is the real driver
