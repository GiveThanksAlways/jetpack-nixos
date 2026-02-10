# Phase 3: Command Submission

**Status:** NOT STARTED  
**Parent doc:** [nv-attempt.md](nv-attempt.md)  
**Previous phase:** [phase2.md](phase2.md)  
**Learning context:** [Learning-Phase1.md](Learning-Phase1.md)

## Goal

Push actual GPU commands via GPFIFO and execute a compute shader. Prove we can dispatch work and wait for completion.

## Key Info From Phase 1

- CUDA uses usermode submit (no SUBMIT_GPFIFO ioctl)
- Work submit token (doorbell): 511
- Syncpoint ID: 17, max=30000, GPU VA=0xffffe10000
- GPFIFO: 1024 entries x 8 bytes = 8192 bytes
- Userd: 4096 bytes
- Compute class: 0xc7c0 (Ampere compute)
- GPFIFO class: 0xc76f
- DMA copy class: 0xc7b5
- The QMD format should match desktop Ampere — check tinygrad's ops_nv.py

## Tasks

- [ ] mmap userd region for doorbell writes
- [ ] Understand GPFIFO entry format (8 bytes: GPU_VA + length + flags)
- [ ] Study tinygrad's NVKIface QMD construction code
- [ ] Compile trivial shader via libnvrtc.so (PTX -> SASS for SM 8.7)
- [ ] Format a QMD with shader address, grid dims, etc.
- [ ] Write push buffer with inline methods or QMD launch
- [ ] Ring doorbell via userd write
- [ ] Wait for completion via syncpoint
- [ ] Verify shader output in memory

## Notes

(This file will be filled in during Phase 3 iteration work)
