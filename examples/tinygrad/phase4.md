# Phase 4: TegraIface Integration

**Status:** NOT STARTED  
**Parent doc:** [nv-attempt.md](nv-attempt.md)  
**Previous phase:** [phase3.md](phase3.md)  
**Learning context:** [Learning-Phase1.md](Learning-Phase1.md)

## Goal

Build `TegraIface` class that plugs into tinygrad's NV runtime (`ops_nv.py`), enabling `NV=1` on Jetson Orin.

## Key Info

- tinygrad's NV backend already has Ampere QMD formatting, push buffer construction, shader compilation
- TegraIface only replaces the DRIVER LAYER (memory allocation, command submission)
- The GPU PROGRAMMING LAYER (QMD format, shader ISA, class methods) stays the same
- Study NVKIface and PCIIface in tinygrad/runtime/ops_nv.py as reference implementations

## Tasks

- [ ] Study NVKIface and PCIIface interface contracts
- [ ] Implement TegraIface class with nvgpu/nvmap backend
- [ ] Replace UVM memory ops with nvmap allocator
- [ ] Replace RM channel creation with nvgpu TSG+channel pipeline
- [ ] Replace RM GPFIFO submit with usermode submit
- [ ] Add detection: check for /dev/nvgpu/igpu0/ctrl in _select_iface()
- [ ] Test: vector add
- [ ] Test: matrix multiply
- [ ] Test: conv2d
- [ ] Test: GPT-2 end-to-end
- [ ] Compare outputs with CUDA backend for correctness
- [ ] Upstream PR to tinygrad

## Notes

(This file will be filled in during Phase 4 iteration work)
