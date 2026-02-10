# Phase 1: Reverse-Engineer nvgpu IOCTLs — Working Document

**Goal:** Map the complete ioctl interface for `/dev/nvhost-gpu`, `/dev/nvhost-ctrl-gpu`, `/dev/nvhost-as-gpu`, `/dev/nvhost-tsg-gpu`, and `/dev/nvmap` so we can build a tinygrad `TegraIface`.

## TODO Checklist

- [x] Download & extract L4T BSP kernel sources (public_sources.tbz2)
- [x] Find nvgpu ioctl header files (struct definitions, ioctl codes)
- [x] Find nvmap ioctl header files
- [x] Strace a CUDA program to capture real ioctl sequence
- [x] Decode strace output — map ioctl numbers to names
- [x] Document the ioctl flow: init → alloc memory → create channel → submit work
- [ ] Write Python ctypes structs for key ioctls
- [ ] Test basic ioctls from Python (GPU characteristics, memory alloc)
- [ ] Build minimal `TegraIface` prototype

## Key Finding: CUDA Uses Usermode Submit on Jetson!

**There are ZERO `SUBMIT_GPFIFO` ioctls (H:107) in the strace.** CUDA uses usermode submit
on Jetson — it writes GPFIFO entries directly to mapped memory and rings the hardware
doorbell through MMIO, just like tinygrad does on desktop NV via `NVKIface`. This is
enabled by `NVGPU_IOCTL_CHANNEL_SETUP_BIND` (H:128) with `USERMODE_SUPPORT` flag.

## Device Files Used by CUDA

```
fd=3  /dev/nvmap              — Memory allocator (nvmap)         Magic: 'N' (0x4e)
fd=4  /dev/nvgpu/igpu0/ctrl   — GPU control (ctrl-gpu)           Magic: 'G' (0x47)
fd=5  /dev/dri/renderD128     — DRM device                       (DRM ioctls)
fd=6  /dev/host1x-fence       → then reused as AS fd             Magic: 'X'→'A' (0x41)
fd=7  /dev/dri/renderD128     — DRM (second open)
fd=8  (TSG fd from OPEN_TSG)                                     Magic: 'T' (0x54)
fd=9+ (channel fds from OPEN_CHANNEL)                            Magic: 'H' (0x48)
```

## Complete IOCTL Decode

### Magic → Device Mapping

| Magic | Char | Device | Header |
|-------|------|--------|--------|
| 0x47 | 'G' | /dev/nvgpu/igpu0/ctrl | nvgpu-ctrl.h |
| 0x4e | 'N' | /dev/nvmap | nvmap.h |
| 0x41 | 'A' | address space (from ALLOC_AS) | nvgpu-as.h |
| 0x48 | 'H' | channel (from OPEN_CHANNEL) | nvgpu.h |
| 0x54 | 'T' | TSG (from OPEN_TSG) | nvgpu.h |
| 0x58 | 'X' | /dev/host1x-fence | upstream kernel |

### Ctrl-GPU Ioctls (Magic 'G' = 0x47)

| Nr | Hex | Name | Count | Notes |
|----|-----|------|-------|-------|
| 1 | 0x01 | ZCULL_GET_CTX_SIZE | 2 | ZCull context size |
| 2 | 0x02 | ZCULL_GET_INFO | 2 | ZCull geometry info |
| 5 | 0x05 | GET_CHARACTERISTICS | 2 | **KEY: returns arch, SM version, compute_class, gpfifo_class** |
| 8 | 0x08 | ALLOC_AS | 1 | Allocate address space, returns AS fd |
| 9 | 0x09 | OPEN_TSG | 3 | Create TSG, returns TSG fd |
| 10 | 0x0a | GET_TPC_MASKS | 2 | TPC topology |
| 11 | 0x0b | OPEN_CHANNEL | 16 | Create channel, returns channel fd |
| 19 | 0x13 | VSMS_MAPPING | 2 | Virtual SM mapping |
| 26 | 0x1a | GET_ENGINE_INFO | 2 | Engine list (GR, CE, etc.) |
| 28 | 0x1c | CLK_GET_RANGE | 4 | Clock frequency range |
| 29 | 0x1d | CLK_GET_VF_POINTS | 2 | Voltage-frequency points |
| 38 | 0x26 | GET_FBP_L2_MASKS | 2 | L2 cache topology |
| 40 | 0x28 | SET_DETERMINISTIC_OPTS | 1 | Determinism control |
| 41 | 0x29 | REGISTER_BUFFER | 171 | **HOT PATH: register nvmap buffer** |
| 43 | 0x2b | GET_GPC_LOCAL_TO_PHYSICAL_MAP | 2 | GPC mapping |
| 44 | 0x2c | GET_GPC_LOCAL_TO_LOGICAL_MAP | 2 | GPC mapping |

### nvmap Ioctls (Magic 'N' = 0x4e)

| Nr | Hex | Name | Count | Notes |
|----|-----|------|-------|-------|
| 0 | 0x00 | CREATE | 171 | **HOT PATH: create handle** |
| 3 | 0x03 | ALLOC | 171 | **HOT PATH: allocate physical memory** |
| 15 | 0x0f | GET_FD | 553 | **HOT PATH: get dmabuf fd** |
| 25 | 0x19 | GET_AVAILABLE_HEAPS | 1 | Discover memory heaps |
| 105 | 0x69 | QUERY_HEAP_PARAMS | 1 | Heap parameters |

### Address Space Ioctls (Magic 'A' = 0x41)

| Nr | Hex | Name | Count | Notes |
|----|-----|------|-------|-------|
| 1 | 0x01 | BIND_CHANNEL | 16 | Bind channel to AS |
| 6 | 0x06 | ALLOC_SPACE | 4 | Pre-allocate VA regions |
| 7 | 0x07 | MAP_BUFFER_EX | 138 | **HOT PATH: map buffer into GPU VA** |
| 8 | 0x08 | GET_VA_REGIONS | 2 | Query VA region layout |
| 12 | 0x0c | GET_SYNC_RO_MAP | 1 | Syncpoint read-only mapping |

### Channel Ioctls (Magic 'H' = 0x48)

| Nr | Hex | Name | Count | Notes |
|----|-----|------|-------|-------|
| 14 | 0x0e | SET_TIMEOUT_EX | 1 | Failed ENOTTY (wrong fd) |
| 108 | 0x6c | ALLOC_OBJ_CTX | 16 | **Allocate compute class** (class_num!) |
| 111 | 0x6f | SET_ERROR_NOTIFIER | 16 | Error notification setup |
| 119 | 0x77 | WDT | 16 | Watchdog timer config |
| 122 | 0x7a | SET_PREEMPTION_MODE | 1 | Preemption mode |
| 126 | 0x7e | GET_USER_SYNCPOINT | 16 | Syncpoint for completion |
| 128 | 0x80 | SETUP_BIND | 16 | **KEY: GPFIFO+userd+usermode submit** |

### TSG Ioctls (Magic 'T' = 0x54)

| Nr | Hex | Name | Count | Notes |
|----|-----|------|-------|-------|
| 7 | 0x07 | EVENT_ID_CTRL | 4 | Event notification |
| 9 | 0x09 | SET_TIMESLICE | 1 | Scheduling timeslice |
| 11 | 0x0b | BIND_CHANNEL_EX | 16 | Bind channel to TSG |
| 13 | 0x0d | SET_L2_MAX_WAYS_EVICT_LAST | 1 | L2 cache policy |
| 18 | 0x12 | CREATE_SUBCONTEXT | 1 | Create subcontext in TSG |

## CUDA Init Sequence (Decoded)

### Phase 1: Discovery
```
1. open(/dev/nvmap) → fd=3
2. N:25 GET_AVAILABLE_HEAPS          — discover heaps (SYSMEM, VPR, etc.)
3. DRM probe (renderD128 × 2)
4. open(/dev/nvgpu/igpu0/ctrl) → fd=4
5. G:5  GET_CHARACTERISTICS          — GPU arch, SM ver, compute_class, flags
6. G:10 GET_TPC_MASKS                — TPC topology
7. G:43 GET_GPC_LOCAL_TO_PHYSICAL    — GPC physical mapping
8. G:44 GET_GPC_LOCAL_TO_LOGICAL     — GPC logical mapping
9. G:38 GET_FBP_L2_MASKS             — L2 cache config
10. G:19 VSMS_MAPPING                — Virtual SM mapping
11. G:1  ZCULL_GET_CTX_SIZE          — ZCull context
12. G:2  ZCULL_GET_INFO              — ZCull geometry
13. G:26 GET_ENGINE_INFO             — Engine enumeration (GR, CE, etc.)
14. G:28 CLK_GET_RANGE (×2)          — Clock freq min/max
15. G:29 CLK_GET_VF_POINTS           — Voltage-frequency table
    (steps 5-15 repeated for second ctrl handle)
```

### Phase 2: Memory System Setup
```
16. N:105 QUERY_HEAP_PARAMS          — detailed heap info
17. open(/dev/host1x-fence)          — fence device
18. X:16  host1x init ioctl
19. G:8   ALLOC_AS                   — create address space → AS fd=6
20. A:12  GET_SYNC_RO_MAP            — syncpoint map for GPU
21. A:8   GET_VA_REGIONS (×2)        — understand VA layout
22. A:6   ALLOC_SPACE (×3)           — pre-reserve VA regions
```

### Phase 3: Buffer Allocation Loop (171 iterations!)
```
For each buffer:
  23. N:0  NVMAP_CREATE               — create handle (size → handle)
  24. N:3  NVMAP_ALLOC                — back with physical memory (heap, flags, align)
  25. N:15 NVMAP_GET_FD (×3)          — get dmabuf fd
  26. G:41 REGISTER_BUFFER            — register with GPU
  27. A:7  MAP_BUFFER_EX              — map into GPU VA space
```

### Phase 4: TSG + Channel Setup (16 channels on 1 TSG)
```
28. G:9   OPEN_TSG                   — create TSG → fd=8
29. T:18  CREATE_SUBCONTEXT          — create subcontext in TSG

For each of 16 channels:
  30. G:11  OPEN_CHANNEL              — create channel → fd=9..24
  31. T:11  BIND_CHANNEL_EX           — bind channel to TSG
  32. H:119 WDT                       — configure/disable watchdog
  33. H:128 SETUP_BIND                — **GPFIFO + userd + usermode submit!**
  34. H:126 GET_USER_SYNCPOINT        — get syncpoint for completion tracking
  35. H:108 ALLOC_OBJ_CTX             — **allocate compute class object!**
  36. H:111 SET_ERROR_NOTIFIER        — error notification

37. T:7   EVENT_ID_CTRL              — event setup
38. H:122 SET_PREEMPTION_MODE        — preemption config
39. T:9   SET_TIMESLICE              — scheduling timeslice
40. T:13  SET_L2_MAX_WAYS_EVICT_LAST — L2 cache policy
```

### Steady State: No SUBMIT_GPFIFO ioctls!
CUDA writes GPFIFO entries directly to mapped memory (from SETUP_BIND)
and rings the doorbell via MMIO. Zero ioctl overhead for submissions.

## Next Steps

1. **Write Python ctypes test** — call GET_CHARACTERISTICS and decode the result
2. **Test nvmap CREATE+ALLOC** — allocate GPU memory from Python
3. **Test ALLOC_AS + MAP_BUFFER_EX** — map into GPU virtual address
4. **Build minimal TegraIface** — enough to prove compute dispatch works

