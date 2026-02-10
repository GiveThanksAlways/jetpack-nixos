# Phase 1: Reverse-Engineer nvgpu IOCTLs — Working Document

**Goal:** Map the complete ioctl interface for `/dev/nvhost-gpu`, `/dev/nvhost-ctrl-gpu`, `/dev/nvhost-as-gpu`, `/dev/nvhost-tsg-gpu`, and `/dev/nvmap` so we can build a tinygrad `TegraIface`.

## TODO Checklist

- [ ] Download & extract L4T BSP kernel sources (public_sources.tbz2)
- [ ] Find nvgpu ioctl header files (struct definitions, ioctl codes)
- [ ] Find nvmap ioctl header files
- [ ] Strace a CUDA program to capture real ioctl sequence
- [ ] Decode strace output — map ioctl numbers to names
- [ ] Document the ioctl flow: init → alloc memory → create channel → submit work
- [ ] Write Python ctypes structs for key ioctls
- [ ] Test basic ioctls from Python (GPU characteristics, memory alloc)

## Approach

### Step 0: Strace CUDA as a Rosetta Stone

Before reading source, strace the CUDA backend doing a simple operation. This shows us:
1. Which device files CUDA opens and in what order
2. Exactly which ioctls it calls (ioctl number, direction, size)
3. The sequence: init → memory → channel → submit → sync
4. Real struct sizes and field values as a reference

```bash
# Strace a minimal CUDA program, filtering for ioctl/open/mmap on nvhost/nvmap:
strace -f -e trace=ioctl,openat,mmap -o /tmp/cuda_trace.txt \
  python3 -c "CUDA=1; from tinygrad import Tensor; print(Tensor([1,2,3]).numpy())"
```

### Step 1: Get Kernel Source Headers

Download BSP sources from NVIDIA, extract nvgpu + nvmap headers.

### Step 2: Map IOCTLs

Cross-reference strace output with header definitions to build a complete map.

### Step 3: Test from Python

Start calling ioctls directly using ctypes/fcntl.

---

## Work Log

(Entries added as work progresses)

