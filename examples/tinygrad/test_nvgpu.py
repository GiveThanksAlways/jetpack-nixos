#!/usr/bin/env python3
"""
Phase 1 Test: Direct nvgpu/nvmap ioctl access from Python.

Proves we can talk to the Jetson Orin GPU without going through CUDA.
Tests the ioctl sequence discovered by stracing CUDA:
  1. Open /dev/nvmap, /dev/nvgpu/igpu0/ctrl
  2. GET_CHARACTERISTICS — read GPU info (arch, SM, compute_class)
  3. NVMAP_CREATE + NVMAP_ALLOC — allocate GPU memory
  4. ALLOC_AS — create address space

Prerequisites: run as user with access to /dev/nvgpu/* and /dev/nvmap
"""

import os
import sys
import struct
import ctypes
import ctypes.util
import fcntl
import mmap
from ctypes import c_uint8, c_uint16, c_uint32, c_uint64, c_int16, c_int32, c_int64

# ============================================================================
# ioctl helpers
# ============================================================================

# Linux ioctl direction bits (aarch64)
_IOC_NONE  = 0
_IOC_WRITE = 1
_IOC_READ  = 2

_IOC_NRBITS   = 8
_IOC_TYPEBITS = 8
_IOC_SIZEBITS = 14
_IOC_DIRBITS  = 2

_IOC_NRSHIFT   = 0
_IOC_TYPESHIFT = _IOC_NRSHIFT + _IOC_NRBITS      # 8
_IOC_SIZESHIFT = _IOC_TYPESHIFT + _IOC_TYPEBITS   # 16
_IOC_DIRSHIFT  = _IOC_SIZESHIFT + _IOC_SIZEBITS   # 30

def _IOC(d, t, nr, size):
    return (d << _IOC_DIRSHIFT) | (ord(t) << _IOC_TYPESHIFT) | (nr << _IOC_NRSHIFT) | (size << _IOC_SIZESHIFT)

def _IO(t, nr):         return _IOC(_IOC_NONE, t, nr, 0)
def _IOR(t, nr, size):  return _IOC(_IOC_READ, t, nr, size)
def _IOW(t, nr, size):  return _IOC(_IOC_WRITE, t, nr, size)
def _IOWR(t, nr, size): return _IOC(_IOC_READ | _IOC_WRITE, t, nr, size)


def nv_ioctl(fd, ioc_code, buf):
    """Call an ioctl, raise on error."""
    ret = fcntl.ioctl(fd, ioc_code, buf)
    if ret < 0:
        raise OSError(f"ioctl 0x{ioc_code:08x} failed with {ret}")
    return ret


# ============================================================================
# nvgpu ctrl-gpu structs & ioctls (Magic 'G' = 0x47)
# ============================================================================

class nvgpu_gpu_characteristics(ctypes.Structure):
    """From nvgpu-ctrl.h — returned by GET_CHARACTERISTICS."""
    _pack_ = 1
    _fields_ = [
        ("arch",                 c_uint32),
        ("impl",                 c_uint32),
        ("rev",                  c_uint32),
        ("num_gpc",              c_uint32),
        ("L2_cache_size",        c_uint64),
        ("on_board_video_memory_size", c_uint64),
        ("num_tpc_per_gpc",      c_uint32),
        ("bus_type",             c_uint32),
        ("big_page_size",        c_uint32),
        ("compression_stat_count", c_uint32),
        ("pde_coverage_bit_count", c_uint32),
        ("available_big_page_sizes", c_uint32),
        ("gpc_mask",             c_uint64),
        ("sm_arch_sm_version",   c_uint32),
        ("sm_arch_spa_version",  c_uint32),
        ("sm_arch_warp_count",   c_uint32),
        ("gpu_va_bit_count",     c_uint32),
        ("reserved",             c_uint32),
        ("flags",                c_uint64),
        ("twod_class",           c_uint32),
        ("threed_class",         c_uint32),
        ("compute_class",        c_uint32),
        ("gpfifo_class",         c_uint32),
        ("inline_to_memory_class", c_uint32),
        ("dma_copy_class",       c_uint32),
        ("max_fbps_count",       c_uint32),
        ("fbp_en_mask",          c_uint32),
        ("max_ltc_per_fbp",      c_uint32),
        ("max_lts_per_ltc",      c_uint32),
        ("max_tex_per_tpc",      c_uint32),
        ("max_gpc_count",        c_uint32),
        ("rop_l2_en_mask_0",     c_uint32),
        ("rop_l2_en_mask_1",     c_uint32),
        ("chipname",             c_uint32 * 8),
        ("gr_compbit_store_base_hw", c_uint64),
        ("gr_gobs_per_comptagline_per_slice", c_uint32),
        ("num_ltc",              c_uint32),
        ("lts_per_ltc",          c_uint32),
        ("cbc_cache_line_size",  c_uint32),
        ("cbc_comptags_per_line", c_uint32),
        ("padding", c_uint32),
        ("sm_version", c_uint32),
        ("max_gpfifo_entries", c_uint32),
        ("device_instance_id", c_uint32),
        # There may be more fields, but we pad to be safe
        ("_pad_to_256", c_uint8 * 64),
    ]


class nvgpu_gpu_get_characteristics(ctypes.Structure):
    """Wrapper for GET_CHARACTERISTICS ioctl."""
    _pack_ = 1
    _fields_ = [
        ("gpu_characteristics_buf_size", c_uint64),
        ("gpu_characteristics_buf_addr", c_uint64),
    ]


# GET_CHARACTERISTICS: _IOWR('G', 5, nvgpu_gpu_get_characteristics)
NVGPU_GPU_IOCTL_GET_CHARACTERISTICS = _IOWR('G', 5, ctypes.sizeof(nvgpu_gpu_get_characteristics))


class nvgpu_gpu_zcull_get_ctx_size_args(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("size", c_uint32)]

NVGPU_GPU_IOCTL_ZCULL_GET_CTX_SIZE = _IOR('G', 1, ctypes.sizeof(nvgpu_gpu_zcull_get_ctx_size_args))


# ============================================================================
# nvmap structs & ioctls (Magic 'N' = 0x4e)
# ============================================================================

class nvmap_create_handle(ctypes.Structure):
    """NVMAP_IOC_CREATE: create a memory handle."""
    _pack_ = 1
    _fields_ = [
        ("size",   c_uint32),   # in: requested size
        ("handle", c_uint32),   # out: handle id
    ]

NVMAP_IOC_CREATE = _IOWR('N', 0, ctypes.sizeof(nvmap_create_handle))


class nvmap_alloc_handle(ctypes.Structure):
    """NVMAP_IOC_ALLOC: back a handle with physical memory."""
    _pack_ = 1
    _fields_ = [
        ("handle",    c_uint32),  # in: handle from CREATE
        ("heap_mask", c_uint32),  # in: which heap(s) to use
        ("flags",     c_uint32),  # in: allocation flags
        ("align",     c_uint32),  # in: alignment requirement
        ("padding",   c_uint32),  # padding to match 0x14 = 20 bytes
    ]

# Heap masks from nvmap.h
NVMAP_HEAP_SYSMEM              = (1 << 31)
NVMAP_HEAP_IOVMM               = (1 << 30)
NVMAP_HEAP_CARVEOUT_GENERIC    = 1

# Alloc flags
NVMAP_HANDLE_UNCACHEABLE        = 0
NVMAP_HANDLE_WRITE_COMBINE      = 1
NVMAP_HANDLE_INNER_CACHEABLE    = 2
NVMAP_HANDLE_CACHEABLE          = 5

NVMAP_IOC_ALLOC = _IOW('N', 3, ctypes.sizeof(nvmap_alloc_handle))


class nvmap_handle_param(ctypes.Structure):
    """NVMAP_IOC_PARAM: query handle parameters."""
    _pack_ = 1
    _fields_ = [
        ("handle", c_uint32),
        ("param",  c_uint32),   # which param to query
        ("result", c_uint32),   # out: result
        ("_pad",   c_uint32),
    ]

# Param types
NVMAP_HANDLE_PARAM_SIZE       = 1
NVMAP_HANDLE_PARAM_ALIGNMENT  = 2
NVMAP_HANDLE_PARAM_BASE       = 3
NVMAP_HANDLE_PARAM_HEAP       = 4
NVMAP_HANDLE_PARAM_KIND       = 5
NVMAP_HANDLE_PARAM_COMPR      = 6

NVMAP_IOC_PARAM = _IOWR('N', 8, ctypes.sizeof(nvmap_handle_param))


class nvmap_get_fd_args(ctypes.Structure):
    """NVMAP_IOC_GET_FD: convert handle to dmabuf fd."""
    _pack_ = 1
    _fields_ = [
        ("handle", c_uint32),
        ("fd",     c_uint32),   # out: dmabuf fd
    ]

NVMAP_IOC_GET_FD = _IOWR('N', 15, ctypes.sizeof(nvmap_get_fd_args))


class nvmap_available_heaps(ctypes.Structure):
    """NVMAP_IOC_GET_AVAILABLE_HEAPS"""
    _pack_ = 1
    _fields_ = [
        ("heaps",  c_uint32),
        ("_pad",   c_uint32),
    ]

NVMAP_IOC_GET_AVAILABLE_HEAPS = _IOR('N', 25, ctypes.sizeof(nvmap_available_heaps))


# ============================================================================
# Test functions
# ============================================================================

def test_get_characteristics(ctrl_fd):
    """Call GET_CHARACTERISTICS to read GPU info."""
    print("\n=== GET_CHARACTERISTICS ===")

    chars = nvgpu_gpu_characteristics()
    ctypes.memset(ctypes.addressof(chars), 0, ctypes.sizeof(chars))

    # The ioctl takes a wrapper struct with buf_size and buf_addr
    req = nvgpu_gpu_get_characteristics()
    req.gpu_characteristics_buf_size = ctypes.sizeof(chars)
    req.gpu_characteristics_buf_addr = ctypes.addressof(chars)

    nv_ioctl(ctrl_fd, NVGPU_GPU_IOCTL_GET_CHARACTERISTICS, req)

    arch_names = {
        0x120: "GK100 (Kepler)", 0x130: "GK10B (Kepler mobile)",
        0x140: "GM200 (Maxwell)", 0x148: "GM20B (Maxwell mobile)",
        0x150: "GP100 (Pascal)", 0x152: "GP10B (Pascal mobile)",
        0x160: "GV100 (Volta)", 0x162: "GV11B (Volta mobile)",
        0x170: "GA100 (Ampere)", 0x172: "GA10B (Ampere mobile)",  # Orin! 
        0x190: "GH100 (Hopper)",
    }

    arch_name = arch_names.get(chars.arch, f"Unknown (0x{chars.arch:x})")

    print(f"  Architecture:     0x{chars.arch:04x} = {arch_name}")
    print(f"  Implementation:   0x{chars.impl:04x}")
    print(f"  Revision:         {chars.rev}")
    print(f"  Num GPC:          {chars.num_gpc}")
    print(f"  L2 cache size:    {chars.L2_cache_size // 1024} KB")
    print(f"  VRAM size:        {chars.on_board_video_memory_size} bytes")
    print(f"  Bus type:         {chars.bus_type}")
    print(f"  Big page size:    {chars.big_page_size}")
    print(f"  GPC mask:         0x{chars.gpc_mask:016x}")
    print(f"  SM arch version:  0x{chars.sm_arch_sm_version:08x}")
    print(f"  SM arch SPA ver:  0x{chars.sm_arch_spa_version:08x}")
    print(f"  SM arch warp cnt: {chars.sm_arch_warp_count}")
    print(f"  GPU VA bits:      {chars.gpu_va_bit_count}")
    print(f"  Flags:            0x{chars.flags:016x}")

    # Decode flags
    flag_names = {
        (1 << 0): "SUPPORT_PARTIAL_MAPPINGS",
        (1 << 1): "SUPPORT_SPARSE_ALLOCS",
        (1 << 2): "SUPPORT_SYNC_FENCE_FDS",
        (1 << 3): "SUPPORT_CYCLE_STATS",
        (1 << 4): "SUPPORT_CYCLE_STATS_SNAPSHOT",
        (1 << 5): "SUPPORT_USERMODE_SUBMIT",
        (1 << 6): "SUPPORT_IO_COHERENCE",
        (1 << 12): "SUPPORT_COMPUTE",
        (1 << 14): "SUPPORT_TSG",
        (1 << 20): "SUPPORT_DETERMINISTIC_SUBMIT_NO_JOBTRACKING",
        (1 << 21): "SUPPORT_DETERMINISTIC_SUBMIT_FULL",
        (1 << 22): "SUPPORT_DETERMINISTIC_OPTS",
        (1 << 24): "SUPPORT_DEVICE_EVENTS",
    }
    set_flags = []
    for bit, name in flag_names.items():
        if chars.flags & bit:
            set_flags.append(name)
    if set_flags:
        print(f"  Active flags:     {', '.join(set_flags)}")

    print(f"\n  === Classes (critical for compute) ===")
    print(f"  2D class:         0x{chars.twod_class:04x}")
    print(f"  3D class:         0x{chars.threed_class:04x}")
    print(f"  Compute class:    0x{chars.compute_class:04x}")
    print(f"  GPFIFO class:     0x{chars.gpfifo_class:04x}")
    print(f"  I2M class:        0x{chars.inline_to_memory_class:04x}")
    print(f"  DMA copy class:   0x{chars.dma_copy_class:04x}")

    print(f"\n  === Memory ===")
    print(f"  Max FBPs:         {chars.max_fbps_count}")
    print(f"  FBP en mask:      0x{chars.fbp_en_mask:08x}")
    print(f"  Max LTC/FBP:      {chars.max_ltc_per_fbp}")
    print(f"  Max LTS/LTC:      {chars.max_lts_per_ltc}")
    print(f"  Num LTC:          {chars.num_ltc}")
    print(f"  Max GPFIFO:       {chars.max_gpfifo_entries}")

    chipname_bytes = bytes(chars.chipname)
    chipname_str = chipname_bytes.split(b'\0')[0].decode('ascii', errors='replace')
    if chipname_str:
        print(f"  Chip name:        {chipname_str}")

    return chars


def test_zcull(ctrl_fd):
    """Call ZCULL_GET_CTX_SIZE."""
    print("\n=== ZCULL_GET_CTX_SIZE ===")
    args = nvgpu_gpu_zcull_get_ctx_size_args()
    nv_ioctl(ctrl_fd, NVGPU_GPU_IOCTL_ZCULL_GET_CTX_SIZE, args)
    print(f"  ZCull ctx size:   {args.size} bytes")
    return args.size


def test_nvmap_create_alloc(nvmap_fd, size=4096):
    """Test NVMAP_CREATE + NVMAP_ALLOC to allocate GPU memory."""
    print(f"\n=== NVMAP CREATE + ALLOC ({size} bytes) ===")

    # Step 1: Create handle
    create = nvmap_create_handle()
    create.size = size
    create.handle = 0
    nv_ioctl(nvmap_fd, NVMAP_IOC_CREATE, create)
    handle = create.handle
    print(f"  Created handle:   {handle}")

    # Step 2: Allocate physical memory
    alloc = nvmap_alloc_handle()
    alloc.handle = handle
    alloc.heap_mask = NVMAP_HEAP_SYSMEM   # System memory (unified on Jetson)
    alloc.flags = NVMAP_HANDLE_WRITE_COMBINE
    alloc.align = 4096
    alloc.padding = 0
    nv_ioctl(nvmap_fd, NVMAP_IOC_ALLOC, alloc)
    print(f"  Allocated on heap SYSMEM, alignment={alloc.align}")

    # Step 3: Get dmabuf fd
    get_fd = nvmap_get_fd_args()
    get_fd.handle = handle
    get_fd.fd = 0
    nv_ioctl(nvmap_fd, NVMAP_IOC_GET_FD, get_fd)
    print(f"  Got dmabuf fd:    {get_fd.fd}")

    return handle, get_fd.fd


def test_nvmap_heaps(nvmap_fd):
    """Query available memory heaps."""
    print("\n=== NVMAP GET_AVAILABLE_HEAPS ===")
    args = nvmap_available_heaps()
    nv_ioctl(nvmap_fd, NVMAP_IOC_GET_AVAILABLE_HEAPS, args)
    
    heap_names = {
        (1 << 31): "SYSMEM",
        (1 << 30): "IOVMM",
        1:         "CARVEOUT_GENERIC",
        (1 << 1):  "CARVEOUT_VPR",
    }
    print(f"  Heap bitmask:     0x{args.heaps:08x}")
    available = []
    for bit, name in heap_names.items():
        if args.heaps & bit:
            available.append(name)
    print(f"  Available heaps:  {', '.join(available) if available else 'none decoded'}")
    return args.heaps


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("Phase 1: Direct nvgpu/nvmap ioctl test")
    print("Testing GPU access WITHOUT CUDA")
    print("=" * 60)

    # Open devices
    try:
        nvmap_fd = os.open("/dev/nvmap", os.O_RDWR | os.O_SYNC)
        print(f"\nOpened /dev/nvmap → fd={nvmap_fd}")
    except OSError as e:
        print(f"ERROR: Cannot open /dev/nvmap: {e}")
        print("Make sure you have permission (user in 'video' group or root)")
        sys.exit(1)

    try:
        ctrl_fd = os.open("/dev/nvgpu/igpu0/ctrl", os.O_RDWR)
        print(f"Opened /dev/nvgpu/igpu0/ctrl → fd={ctrl_fd}")
    except OSError as e:
        print(f"ERROR: Cannot open /dev/nvgpu/igpu0/ctrl: {e}")
        os.close(nvmap_fd)
        sys.exit(1)

    success_count = 0
    total_tests = 0

    # Test 1: GET_CHARACTERISTICS
    total_tests += 1
    try:
        chars = test_get_characteristics(ctrl_fd)
        success_count += 1
        print("  ✓ GET_CHARACTERISTICS succeeded!")
    except Exception as e:
        print(f"  ✗ GET_CHARACTERISTICS failed: {e}")

    # Test 2: ZCULL_GET_CTX_SIZE
    total_tests += 1
    try:
        zcull_size = test_zcull(ctrl_fd)
        success_count += 1
        print("  ✓ ZCULL_GET_CTX_SIZE succeeded!")
    except Exception as e:
        print(f"  ✗ ZCULL_GET_CTX_SIZE failed: {e}")

    # Test 3: NVMAP heap query
    total_tests += 1
    try:
        heaps = test_nvmap_heaps(nvmap_fd)
        success_count += 1
        print("  ✓ GET_AVAILABLE_HEAPS succeeded!")
    except Exception as e:
        print(f"  ✗ GET_AVAILABLE_HEAPS failed: {e}")

    # Test 4: NVMAP CREATE + ALLOC
    total_tests += 1
    try:
        handle, dmabuf_fd = test_nvmap_create_alloc(nvmap_fd, size=4096)
        success_count += 1
        print("  ✓ NVMAP CREATE + ALLOC succeeded!")
        # Clean up
        os.close(dmabuf_fd)
    except Exception as e:
        print(f"  ✗ NVMAP CREATE + ALLOC failed: {e}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Results: {success_count}/{total_tests} tests passed")
    if success_count == total_tests:
        print("ALL TESTS PASSED — nvgpu/nvmap ioctl interface is accessible!")
        print("\nNext: test ALLOC_AS + MAP_BUFFER_EX to create GPU VA mappings")
    else:
        print("Some tests failed — check permissions and driver state")
    print(f"{'=' * 60}")

    # Cleanup
    os.close(ctrl_fd)
    os.close(nvmap_fd)

    return 0 if success_count == total_tests else 1


if __name__ == "__main__":
    sys.exit(main())
