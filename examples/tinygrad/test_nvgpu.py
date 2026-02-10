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
    """From nvgpu-ctrl.h — returned by GET_CHARACTERISTICS.
    
    MUST match the kernel struct exactly (natural alignment on aarch64).
    No _pack_ = 1 because the kernel uses natural alignment!
    """
    _fields_ = [
        # offset 0
        ("arch",                      c_uint32),
        ("impl",                      c_uint32),
        ("rev",                       c_uint32),
        ("num_gpc",                   c_uint32),
        # offset 16
        ("numa_domain_id",            c_int32),    # __s32, -1 = no NUMA info
        # 4 bytes padding inserted by compiler for u64 alignment
        ("_pad0",                     c_uint32),
        # offset 24
        ("L2_cache_size",             c_uint64),   # bytes
        ("on_board_video_memory_size", c_uint64),  # bytes
        # offset 40
        ("num_tpc_per_gpc",           c_uint32),   # architectural max
        ("bus_type",                  c_uint32),
        ("big_page_size",             c_uint32),   # default big page size
        ("compression_page_size",     c_uint32),
        # offset 56
        ("pde_coverage_bit_count",    c_uint32),
        ("available_big_page_sizes",  c_uint32),
        # offset 64
        ("flags",                     c_uint64),
        # offset 72
        ("twod_class",                c_uint32),
        ("threed_class",              c_uint32),
        ("compute_class",             c_uint32),
        ("gpfifo_class",              c_uint32),
        ("inline_to_memory_class",    c_uint32),
        ("dma_copy_class",            c_uint32),
        # offset 96
        ("gpc_mask",                  c_uint32),   # u32, NOT u64!
        ("sm_arch_sm_version",        c_uint32),
        ("sm_arch_spa_version",       c_uint32),
        ("sm_arch_warp_count",        c_uint32),
        # offset 112
        ("gpu_ioctl_nr_last",         c_int16),
        ("tsg_ioctl_nr_last",         c_int16),
        ("dbg_gpu_ioctl_nr_last",     c_int16),
        ("ioctl_channel_nr_last",     c_int16),
        ("as_ioctl_nr_last",          c_int16),
        # offset 122
        ("gpu_va_bit_count",          c_uint8),
        ("reserved",                  c_uint8),
        # offset 124
        ("max_fbps_count",            c_uint32),
        ("fbp_en_mask",               c_uint32),
        ("emc_en_mask",               c_uint32),
        ("max_ltc_per_fbp",           c_uint32),
        ("max_lts_per_ltc",           c_uint32),
        ("max_tex_per_tpc",           c_uint32),
        ("max_gpc_count",             c_uint32),
        # offset 152
        ("rop_l2_en_mask_DEPRECATED", c_uint32 * 2),
        # offset 160
        ("chipname",                  c_uint8 * 8),
        # offset 168
        ("gr_compbit_store_base_hw",  c_uint64),
        # offset 176
        ("gr_gobs_per_comptagline_per_slice", c_uint32),
        ("num_ltc",                   c_uint32),
        ("lts_per_ltc",               c_uint32),
        ("cbc_cache_line_size",       c_uint32),
        ("cbc_comptags_per_line",     c_uint32),
        ("map_buffer_batch_limit",    c_uint32),
        # offset 200
        ("max_freq",                  c_uint64),
        # offset 208
        ("graphics_preemption_mode_flags", c_uint32),
        ("compute_preemption_mode_flags",  c_uint32),
        ("default_graphics_preempt_mode",  c_uint32),
        ("default_compute_preempt_mode",   c_uint32),
        # offset 224
        ("local_video_memory_size",   c_uint64),  # non-zero only for dGPUs
        # offset 232
        ("pci_vendor_id",             c_uint16),
        ("pci_device_id",             c_uint16),
        ("pci_subsystem_vendor_id",   c_uint16),
        ("pci_subsystem_device_id",   c_uint16),
        ("pci_class",                 c_uint16),
        ("pci_revision",              c_uint8),
        ("vbios_oem_version",         c_uint8),
        ("vbios_version",             c_uint32),
        # offset 248
        ("reg_ops_limit",             c_uint32),
        ("reserved1",                 c_uint32),
        # offset 256
        ("event_ioctl_nr_last",       c_int16),
        ("pad",                       c_uint16),
        ("max_css_buffer_size",       c_uint32),
        # offset 264
        ("ctxsw_ioctl_nr_last",       c_int16),
        ("prof_ioctl_nr_last",        c_int16),
        ("nvs_ioctl_nr_last",         c_int16),
        ("reserved2",                 c_uint8 * 2),
        # offset 272
        ("max_ctxsw_ring_buffer_size", c_uint32),
        ("reserved3",                 c_uint32),
        # offset 280
        ("per_device_identifier",     c_uint64),
        # offset 288
        ("num_ppc_per_gpc",           c_uint32),
        ("max_veid_count_per_tsg",    c_uint32),
        ("num_sub_partition_per_fbpa", c_uint32),
        ("gpu_instance_id",           c_uint32),
        ("gr_instance_id",            c_uint32),
        ("max_gpfifo_entries",        c_uint32),
        ("max_dbg_tsg_timeslice",     c_uint32),
        ("reserved5",                 c_uint32),
        # offset 320
        ("device_instance_id",        c_uint64),
        # Total: 328 bytes
    ]


class nvgpu_gpu_get_characteristics(ctypes.Structure):
    """Wrapper for GET_CHARACTERISTICS ioctl."""
    _fields_ = [
        ("gpu_characteristics_buf_size", c_uint64),
        ("gpu_characteristics_buf_addr", c_uint64),
    ]


# GET_CHARACTERISTICS: _IOWR('G', 5, nvgpu_gpu_get_characteristics)
NVGPU_GPU_IOCTL_GET_CHARACTERISTICS = _IOWR('G', 5, ctypes.sizeof(nvgpu_gpu_get_characteristics))


class nvgpu_gpu_zcull_get_ctx_size_args(ctypes.Structure):
    _fields_ = [("size", c_uint32)]

NVGPU_GPU_IOCTL_ZCULL_GET_CTX_SIZE = _IOR('G', 1, ctypes.sizeof(nvgpu_gpu_zcull_get_ctx_size_args))


# ============================================================================
# nvmap structs & ioctls (Magic 'N' = 0x4e)
# ============================================================================

class nvmap_create_handle(ctypes.Structure):
    """NVMAP_IOC_CREATE: create a memory handle.
    Also used by NVMAP_IOC_GET_FD (nr=15) and NVMAP_IOC_FROM_FD (nr=16)."""
    _fields_ = [
        ("size",   c_uint32),   # in: requested size (CREATE) / unused (GET_FD)
        ("handle", c_uint32),   # out: handle id (CREATE) / in: handle (GET_FD)
    ]

NVMAP_IOC_CREATE = _IOWR('N', 0, ctypes.sizeof(nvmap_create_handle))


class nvmap_alloc_handle(ctypes.Structure):
    """NVMAP_IOC_ALLOC: back a handle with physical memory.
    sizeof = 20 (0x14) matching strace observation."""
    _pack_ = 1  # pack=1 is correct here: struct is 17+3pad = 20 bytes
    _fields_ = [
        ("handle",    c_uint32),  # in: handle from CREATE
        ("heap_mask", c_uint32),  # in: which heap(s) to use
        ("flags",     c_uint32),  # in: allocation flags
        ("align",     c_uint32),  # in: alignment requirement
        ("kind",      c_uint8),   # in: memory kind (0 = pitch)
        ("_pad",      c_uint8 * 3),  # trailing padding to match C sizeof = 20
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
    _fields_ = [
        ("handle", c_uint32),
        ("param",  c_uint32),   # which param to query
        ("result", c_uint64),   # out: result (u64 in kernel)
    ]

# Param types
NVMAP_HANDLE_PARAM_SIZE       = 1
NVMAP_HANDLE_PARAM_ALIGNMENT  = 2
NVMAP_HANDLE_PARAM_BASE       = 3
NVMAP_HANDLE_PARAM_HEAP       = 4
NVMAP_HANDLE_PARAM_KIND       = 5
NVMAP_HANDLE_PARAM_COMPR      = 6

NVMAP_IOC_PARAM = _IOWR('N', 8, ctypes.sizeof(nvmap_handle_param))


# NVMAP_IOC_GET_FD uses nvmap_create_handle struct
# handle field = input handle, size field reused as output fd

NVMAP_IOC_GET_FD = _IOWR('N', 15, ctypes.sizeof(nvmap_create_handle))


class nvmap_available_heaps(ctypes.Structure):
    """NVMAP_IOC_GET_AVAILABLE_HEAPS"""
    _fields_ = [
        ("heaps",  c_uint64),   # u64 bitmask, NOT u32!
    ]

NVMAP_IOC_GET_AVAILABLE_HEAPS = _IOR('N', 25, ctypes.sizeof(nvmap_available_heaps))


# ============================================================================
# Address Space structs & ioctls (Magic 'A' = 0x41)
# ============================================================================

class nvgpu_alloc_as_args(ctypes.Structure):
    """ALLOC_AS: create an address space. Returns AS fd."""
    _fields_ = [
        ("big_page_size", c_uint32),  # in: 0 = use default
        ("as_fd",         c_int32),   # out: fd for the new AS
        ("flags",         c_uint32),
        ("reserved",      c_uint32),
        ("va_range_start", c_uint64),
        ("va_range_end",   c_uint64),
        ("va_range_split", c_uint64),
        ("padding",       c_uint32 * 6),
    ]

NVGPU_GPU_IOCTL_ALLOC_AS = _IOWR('G', 8, ctypes.sizeof(nvgpu_alloc_as_args))

class nvgpu_as_bind_channel_args(ctypes.Structure):
    _fields_ = [
        ("channel_fd", c_uint32),
    ]

NVGPU_AS_IOCTL_BIND_CHANNEL = _IOWR('A', 1, ctypes.sizeof(nvgpu_as_bind_channel_args))

class nvgpu_as_alloc_space_args(ctypes.Structure):
    """Allocate VA space region."""
    _fields_ = [
        ("pages",     c_uint64),
        ("page_size", c_uint32),
        ("flags",     c_uint32),
        ("offset",    c_uint64),  # in/out: if FIXED_OFFSET, use this; else alignment
        ("padding",   c_uint32 * 2),
    ]

NVGPU_AS_IOCTL_ALLOC_SPACE = _IOWR('A', 6, ctypes.sizeof(nvgpu_as_alloc_space_args))

class nvgpu_as_map_buffer_ex_args(ctypes.Structure):
    """Map a dmabuf into the GPU address space."""
    _fields_ = [
        ("flags",          c_uint32),
        ("compr_kind",     c_int16),
        ("incompr_kind",   c_int16),
        ("dmabuf_fd",      c_uint32),
        ("page_size",      c_uint32),
        ("buffer_offset",  c_uint64),
        ("mapping_size",   c_uint64),
        ("offset",         c_uint64),  # in/out: GPU VA
    ]

NVGPU_AS_IOCTL_MAP_BUFFER_EX = _IOWR('A', 7, ctypes.sizeof(nvgpu_as_map_buffer_ex_args))

class nvgpu_as_get_va_regions_args(ctypes.Structure):
    """Query VA region layout."""
    _fields_ = [
        ("buf_addr", c_uint64),
        ("buf_size", c_uint32),
        ("reserved", c_uint32),
    ]

NVGPU_AS_IOCTL_GET_VA_REGIONS = _IOWR('A', 8, ctypes.sizeof(nvgpu_as_get_va_regions_args))


# ============================================================================
# TSG structs & ioctls (Magic 'T' = 0x54)
# ============================================================================

class nvgpu_gpu_open_tsg_args(ctypes.Structure):
    """OPEN_TSG: create a TSG, returns TSG fd."""
    _fields_ = [
        ("tsg_fd",   c_int32),
        ("flags",    c_uint32),
        ("token",    c_uint32),   # for sharing
        ("reserved", c_uint32),
        ("subctx_id", c_uint32),
        ("_pad",     c_uint32),
    ]

NVGPU_GPU_IOCTL_OPEN_TSG = _IOWR('G', 9, ctypes.sizeof(nvgpu_gpu_open_tsg_args))

class nvgpu_tsg_bind_channel_ex_args(ctypes.Structure):
    """Bind a channel to a TSG."""
    _fields_ = [
        ("channel_fd",  c_int32),
        ("padding",     c_uint32),
        ("subctx_id",   c_uint64),
        ("num_active_channels", c_uint32),     
        ("_pad",        c_uint32),
    ]

NVGPU_TSG_IOCTL_BIND_CHANNEL_EX = _IOWR('T', 11, ctypes.sizeof(nvgpu_tsg_bind_channel_ex_args))

class nvgpu_tsg_create_subcontext_args(ctypes.Structure):
    """Create a subcontext within a TSG."""
    _fields_ = [
        ("type",     c_uint32),   # in: SYNC(0) or ASYNC(1)
        ("as_fd",    c_int32),    # in: address space fd
        ("veid",     c_uint32),   # out: VEID for the subcontext
        ("reserved", c_uint32),
    ]

# Subcontext types
NVGPU_TSG_SUBCONTEXT_TYPE_SYNC  = 0
NVGPU_TSG_SUBCONTEXT_TYPE_ASYNC = 1

NVGPU_TSG_IOCTL_CREATE_SUBCONTEXT = _IOWR('T', 18, ctypes.sizeof(nvgpu_tsg_create_subcontext_args))


# ============================================================================
# Channel structs & ioctls (Magic 'H' = 0x48)
# ============================================================================

class nvgpu_gpu_open_channel_args(ctypes.Structure):
    """OPEN_CHANNEL: union of {in: runlist_id} and {out: channel_fd}, just one s32."""
    _fields_ = [
        ("channel_fd", c_int32),   # in: runlist_id (-1 = primary graphics), out: channel fd
    ]

NVGPU_GPU_IOCTL_OPEN_CHANNEL = _IOWR('G', 11, ctypes.sizeof(nvgpu_gpu_open_channel_args))

# NVGPU_IOCTL_MAGIC for channel ioctls = 'H'

class nvgpu_alloc_obj_ctx_args(ctypes.Structure):
    """Allocate a class object on a channel (e.g. compute class)."""
    _fields_ = [
        ("class_num", c_uint32),   # in: class to allocate (e.g. 0xc7c0 for compute)
        ("flags",     c_uint32),
        ("obj_id",    c_uint64),   # out: object handle
    ]

NVGPU_IOCTL_CHANNEL_ALLOC_OBJ_CTX = _IOWR('H', 108, ctypes.sizeof(nvgpu_alloc_obj_ctx_args))

class nvgpu_channel_setup_bind_args(ctypes.Structure):
    """Setup GPFIFO + userd binding with usermode submit support."""
    _fields_ = [
        ("num_gpfifo_entries",  c_uint32),
        ("num_inflight_jobs",   c_uint32),
        ("flags",               c_uint32),
        ("userd_dmabuf_fd",     c_int32),
        ("gpfifo_dmabuf_fd",    c_int32),
        ("work_submit_token",   c_uint32),  # out: token for usermode submit
        ("userd_dmabuf_offset", c_uint64),   # in
        ("gpfifo_dmabuf_offset", c_uint64),  # in
        ("gpfifo_gpu_va",       c_uint64),   # out
        ("userd_gpu_va",        c_uint64),   # out
        ("usermode_mmio_gpu_va", c_uint64),  # out
        ("reserved",            c_uint32 * 9),
    ]

NVGPU_IOCTL_CHANNEL_SETUP_BIND = _IOWR('H', 128, ctypes.sizeof(nvgpu_channel_setup_bind_args))

# SETUP_BIND flags
NVGPU_CHANNEL_SETUP_BIND_FLAGS_SUPPORT_VPR          = (1 << 0)
NVGPU_CHANNEL_SETUP_BIND_FLAGS_SUPPORT_DETERMINISTIC = (1 << 1)
NVGPU_CHANNEL_SETUP_BIND_FLAGS_REPLAYABLE_FAULTS_ENABLE = (1 << 2)
NVGPU_CHANNEL_SETUP_BIND_FLAGS_USERMODE_SUPPORT     = (1 << 3)

class nvgpu_set_error_notifier(ctypes.Structure):
    _fields_ = [
        ("offset", c_uint64),
        ("size",   c_uint64),
        ("mem",    c_uint32),
        ("_pad",   c_uint32),
    ]

NVGPU_IOCTL_CHANNEL_SET_ERROR_NOTIFIER = _IOWR('H', 111, ctypes.sizeof(nvgpu_set_error_notifier))

class nvgpu_channel_wdt_args(ctypes.Structure):
    _fields_ = [
        ("wdt_status", c_uint32),
        ("timeout_ms", c_uint32),
    ]

NVGPU_IOCTL_CHANNEL_WDT = _IOW('H', 119, ctypes.sizeof(nvgpu_channel_wdt_args))

class nvgpu_get_user_syncpoint_args(ctypes.Structure):
    _fields_ = [
        ("syncpoint_id",    c_uint32),
        ("syncpoint_value", c_uint32),
        ("gpu_va",          c_uint64),
    ]

NVGPU_IOCTL_CHANNEL_GET_USER_SYNCPOINT = _IOR('H', 126, ctypes.sizeof(nvgpu_get_user_syncpoint_args))


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
    print(f"  NUMA domain:      {chars.numa_domain_id}")
    print(f"  L2 cache size:    {chars.L2_cache_size} bytes ({chars.L2_cache_size // 1024} KB)")
    print(f"  VRAM size:        {chars.on_board_video_memory_size} bytes")
    print(f"  Num TPC/GPC:      {chars.num_tpc_per_gpc}")
    print(f"  Bus type:         {chars.bus_type}")
    print(f"  Big page size:    {chars.big_page_size}")
    print(f"  Compr page size:  {chars.compression_page_size}")
    print(f"  GPU VA bits:      {chars.gpu_va_bit_count}")
    print(f"  GPC mask:         0x{chars.gpc_mask:08x}")
    print(f"  SM arch version:  0x{chars.sm_arch_sm_version:08x}")
    print(f"  SM arch SPA ver:  0x{chars.sm_arch_spa_version:08x}")
    print(f"  SM arch warp cnt: {chars.sm_arch_warp_count}")
    print(f"  Flags:            0x{chars.flags:016x}")

    # Decode flags (from nvgpu-ctrl.h)
    flag_names = {
        (1 << 0):  "SUPPORT_PARTIAL_MAPPINGS",
        (1 << 1):  "SUPPORT_SPARSE_ALLOCS",
        (1 << 2):  "SUPPORT_SYNC_FENCE_FDS",
        (1 << 3):  "SUPPORT_CYCLE_STATS",
        (1 << 4):  "SUPPORT_CYCLE_STATS_SNAPSHOT",
        (1 << 5):  "SUPPORT_USERMODE_SUBMIT",  # bit 5 in old headers
        (1 << 6):  "SUPPORT_CLOCK_CONTROLS",
        (1 << 7):  "SUPPORT_GET_VOLTAGE",
        (1 << 8):  "SUPPORT_GET_CURRENT",
        (1 << 9):  "SUPPORT_GET_POWER",
        (1 << 10): "SUPPORT_GET_TEMPERATURE",
        (1 << 11): "SUPPORT_SET_THERM_ALERT_LIMIT",
        (1 << 14): "SUPPORT_TSG",
        (1 << 15): "SUPPORT_DEVICE_EVENTS",
        (1 << 16): "SUPPORT_FECS_CTXSW_TRACE",
        (1 << 18): "SUPPORT_DETERMINISTIC_SUBMIT_NO_JOBTRACKING",
        (1 << 19): "SUPPORT_DETERMINISTIC_SUBMIT_FULL",
        (1 << 20): "SUPPORT_IO_COHERENCE",
        (1 << 21): "SUPPORT_RESCHEDULE_RUNLIST",
        (1 << 22): "SUPPORT_TSG_SUBCONTEXTS",
        (1 << 24): "SUPPORT_DETERMINISTIC_OPTS",
        (1 << 25): "SUPPORT_SCG",
        (1 << 26): "SUPPORT_SYNCPOINT_ADDRESS",
        (1 << 27): "SUPPORT_VPR",
        (1 << 28): "SUPPORT_USER_SYNCPOINT",
        (1 << 29): "CAN_RAILGATE",
        (1 << 30): "SUPPORT_USERMODE_SUBMIT",  # bit 30
        (1 << 42): "SUPPORT_COMPUTE",
        (1 << 57): "SUPPORT_GPU_MMIO",
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
    print(f"  Max freq:         {chars.max_freq} Hz ({chars.max_freq / 1e6:.0f} MHz)")

    chipname_bytes = bytes(chars.chipname)
    chipname_str = chipname_bytes.split(b'\0')[0].decode('ascii', errors='replace')
    if chipname_str:
        print(f"  Chip name:        {chipname_str}")

    print(f"\n  === IOCTL interface levels ===")
    print(f"  GPU ioctl last:   {chars.gpu_ioctl_nr_last}")
    print(f"  TSG ioctl last:   {chars.tsg_ioctl_nr_last}")
    print(f"  Channel last:     {chars.ioctl_channel_nr_last}")
    print(f"  AS ioctl last:    {chars.as_ioctl_nr_last}")

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
    print(f"  Created handle:   {handle} (0x{handle:08x})")

    # Step 2: Try various heaps
    heap_options = [
        (NVMAP_HEAP_IOVMM, "IOVMM"),
        (NVMAP_HEAP_SYSMEM, "SYSMEM"),
        ((1 << 3), "CARVEOUT_GPU"),
        ((1 << 0), "CARVEOUT_GENERIC"),
        (0xFFFFFFFF, "ALL (0xFFFFFFFF)"),
    ]
    
    for heap_mask, heap_name in heap_options:
        alloc = nvmap_alloc_handle()
        alloc.handle = handle
        alloc.heap_mask = heap_mask
        alloc.flags = NVMAP_HANDLE_WRITE_COMBINE
        alloc.align = 4096
        alloc.kind = 0  # pitch linear
        try:
            nv_ioctl(nvmap_fd, NVMAP_IOC_ALLOC, alloc)
            print(f"  Allocated with heap={heap_name} (0x{heap_mask:08x}), alignment={alloc.align}")
            
            # Step 3: Get dmabuf fd (uses same struct as CREATE)
            get_fd = nvmap_create_handle()
            get_fd.handle = handle  # input: the handle
            get_fd.size = 0         # output: will be overwritten with fd
            nv_ioctl(nvmap_fd, NVMAP_IOC_GET_FD, get_fd)
            dmabuf_fd = get_fd.size  # fd is returned in the 'size' field
            print(f"  Got dmabuf fd:    {dmabuf_fd}")
            return handle, dmabuf_fd
        except OSError as e:
            print(f"  Heap {heap_name} (0x{heap_mask:08x}) failed: {e}")
    
    raise RuntimeError("All heap options failed")


def test_alloc_as(ctrl_fd):
    """Create a GPU address space."""
    print("\n=== ALLOC_AS (create address space) ===")
    args = nvgpu_alloc_as_args()
    # big_page_size=0 means no big pages (default for ga10b which reports 0)
    # flags=UNIFIED_VA required for compute
    # VA ranges must be PDE-aligned (2^21 = 2MB for ga10b)
    # va_range_split must be 0 for UNIFIED_VA
    PDE_SIZE = 1 << 21  # 2MB - determined empirically for ga10b
    args.big_page_size = 0
    args.flags = 2  # NVGPU_GPU_IOCTL_ALLOC_AS_FLAGS_UNIFIED_VA
    args.va_range_start = PDE_SIZE        # 0x200000 (2MB)
    args.va_range_end = (1 << 40) - PDE_SIZE  # 0xFFFFE00000 (almost 1TB)
    args.va_range_split = 0
    nv_ioctl(ctrl_fd, NVGPU_GPU_IOCTL_ALLOC_AS, args)
    print(f"  AS fd:            {args.as_fd}")
    print(f"  VA range:         0x{args.va_range_start:012x} - 0x{args.va_range_end:012x}")
    return args.as_fd


def test_map_buffer(as_fd, dmabuf_fd, size=4096):
    """Map a dmabuf into the GPU address space."""
    print(f"\n=== MAP_BUFFER_EX (map {size} bytes into GPU VA) ===")
    args = nvgpu_as_map_buffer_ex_args()
    args.flags = 0  # let kernel choose address
    args.compr_kind = -1   # NV_KIND_INVALID
    args.incompr_kind = 0  # pitch linear
    args.dmabuf_fd = dmabuf_fd
    args.page_size = 4096
    args.buffer_offset = 0
    args.mapping_size = 0  # 0 = whole buffer
    args.offset = 0        # kernel picks address
    nv_ioctl(as_fd, NVGPU_AS_IOCTL_MAP_BUFFER_EX, args)
    gpu_va = args.offset
    print(f"  GPU VA:           0x{gpu_va:012x}")
    return gpu_va


def test_open_tsg(ctrl_fd):
    """Create a TSG."""
    print("\n=== OPEN_TSG ===")
    args = nvgpu_gpu_open_tsg_args()
    args.flags = 0
    args.token = 0
    nv_ioctl(ctrl_fd, NVGPU_GPU_IOCTL_OPEN_TSG, args)
    print(f"  TSG fd:           {args.tsg_fd}")
    return args.tsg_fd


def test_open_channel(ctrl_fd):
    """Create a channel."""
    print("\n=== OPEN_CHANNEL ===")
    args = nvgpu_gpu_open_channel_args()
    args.channel_fd = -1  # in: runlist_id = -1 (auto/primary graphics)
    nv_ioctl(ctrl_fd, NVGPU_GPU_IOCTL_OPEN_CHANNEL, args)
    print(f"  Channel fd:       {args.channel_fd}")
    return args.channel_fd


def test_full_channel_setup(ctrl_fd, as_fd, nvmap_fd, compute_class):
    """Full channel setup: TSG → channel → bind → compute class."""
    print("\n" + "=" * 60)
    print("FULL CHANNEL + COMPUTE SETUP")
    print("=" * 60)

    # 1. Open TSG
    tsg_fd = test_open_tsg(ctrl_fd)

    # 2. Create subcontext in TSG (ASYNC for compute)
    print("\n=== CREATE_SUBCONTEXT ===")
    subctx = nvgpu_tsg_create_subcontext_args()
    subctx.type = NVGPU_TSG_SUBCONTEXT_TYPE_ASYNC  # compute
    subctx.as_fd = as_fd
    nv_ioctl(tsg_fd, NVGPU_TSG_IOCTL_CREATE_SUBCONTEXT, subctx)
    print(f"  VEID:             {subctx.veid}")

    # 3. Open channel
    ch_fd = test_open_channel(ctrl_fd)

    # 4. Bind channel to TSG
    print("\n=== BIND_CHANNEL_EX ===")
    bind = nvgpu_tsg_bind_channel_ex_args()
    bind.channel_fd = ch_fd
    bind.subctx_id = subctx.subctx_id
    nv_ioctl(tsg_fd, NVGPU_TSG_IOCTL_BIND_CHANNEL_EX, bind)
    print(f"  Bound channel {ch_fd} to TSG {tsg_fd}")

    # 5. Bind channel to AS
    print("\n=== AS BIND_CHANNEL ===")
    as_bind = nvgpu_as_bind_channel_args()
    as_bind.channel_fd = ch_fd
    nv_ioctl(as_fd, NVGPU_AS_IOCTL_BIND_CHANNEL, as_bind)
    print(f"  Bound channel {ch_fd} to AS {as_fd}")

    # 6. Disable watchdog
    print("\n=== CHANNEL WDT (disable) ===")
    wdt = nvgpu_channel_wdt_args()
    wdt.wdt_status = 1  # NVGPU_IOCTL_CHANNEL_DISABLE_WDT
    wdt.timeout_ms = 0
    nv_ioctl(ch_fd, NVGPU_IOCTL_CHANNEL_WDT, wdt)
    print(f"  Watchdog disabled")

    # 7. Allocate GPFIFO + userd buffers via nvmap
    print("\n=== Allocating GPFIFO + userd buffers ===")
    GPFIFO_ENTRIES = 1024  # reasonable default
    GPFIFO_SIZE = GPFIFO_ENTRIES * 8  # 8 bytes per GPFIFO entry
    USERD_SIZE = 4096  # one page for userd

    # GPFIFO buffer
    gpfifo_create = nvmap_create_handle()
    gpfifo_create.size = GPFIFO_SIZE
    nv_ioctl(nvmap_fd, NVMAP_IOC_CREATE, gpfifo_create)
    gpfifo_alloc = nvmap_alloc_handle()
    gpfifo_alloc.handle = gpfifo_create.handle
    gpfifo_alloc.heap_mask = NVMAP_HEAP_IOVMM
    gpfifo_alloc.flags = NVMAP_HANDLE_WRITE_COMBINE
    gpfifo_alloc.align = 4096
    gpfifo_alloc.kind = 0
    nv_ioctl(nvmap_fd, NVMAP_IOC_ALLOC, gpfifo_alloc)
    gpfifo_getfd = nvmap_create_handle()
    gpfifo_getfd.handle = gpfifo_create.handle
    nv_ioctl(nvmap_fd, NVMAP_IOC_GET_FD, gpfifo_getfd)
    gpfifo_dmabuf_fd = gpfifo_getfd.size
    print(f"  GPFIFO buffer:    handle={gpfifo_create.handle}, dmabuf_fd={gpfifo_dmabuf_fd}, size={GPFIFO_SIZE}")

    # userd buffer
    userd_create = nvmap_create_handle()
    userd_create.size = USERD_SIZE
    nv_ioctl(nvmap_fd, NVMAP_IOC_CREATE, userd_create)
    userd_alloc = nvmap_alloc_handle()
    userd_alloc.handle = userd_create.handle
    userd_alloc.heap_mask = NVMAP_HEAP_IOVMM
    userd_alloc.flags = NVMAP_HANDLE_WRITE_COMBINE
    userd_alloc.align = 4096
    userd_alloc.kind = 0
    nv_ioctl(nvmap_fd, NVMAP_IOC_ALLOC, userd_alloc)
    userd_getfd = nvmap_create_handle()
    userd_getfd.handle = userd_create.handle
    nv_ioctl(nvmap_fd, NVMAP_IOC_GET_FD, userd_getfd)
    userd_dmabuf_fd = userd_getfd.size
    print(f"  userd buffer:     handle={userd_create.handle}, dmabuf_fd={userd_dmabuf_fd}, size={USERD_SIZE}")

    # 8. SETUP_BIND — this is the big one! Sets up GPFIFO + userd + usermode submit
    print("\n=== SETUP_BIND (GPFIFO + userd + usermode submit) ===")
    setup = nvgpu_channel_setup_bind_args()
    setup.num_gpfifo_entries = GPFIFO_ENTRIES
    setup.num_inflight_jobs = 0
    setup.gpfifo_dmabuf_fd = gpfifo_dmabuf_fd
    setup.gpfifo_dmabuf_offset = 0
    setup.userd_dmabuf_fd = userd_dmabuf_fd
    setup.userd_dmabuf_offset = 0
    setup.flags = NVGPU_CHANNEL_SETUP_BIND_FLAGS_USERMODE_SUPPORT
    nv_ioctl(ch_fd, NVGPU_IOCTL_CHANNEL_SETUP_BIND, setup)
    print(f"  Work submit token: {setup.work_submit_token}")
    print(f"  GPFIFO GPU VA:     0x{setup.gpfifo_gpu_va:012x}")
    print(f"  USERD GPU VA:      0x{setup.userd_gpu_va:012x}")
    print(f"  Usermode MMIO VA:  0x{setup.usermode_mmio_gpu_va:012x}")

    # 9. Get user syncpoint
    print("\n=== GET_USER_SYNCPOINT ===")
    syncpt = nvgpu_get_user_syncpoint_args()
    nv_ioctl(ch_fd, NVGPU_IOCTL_CHANNEL_GET_USER_SYNCPOINT, syncpt)
    print(f"  Syncpoint ID:     {syncpt.syncpoint_id}")
    print(f"  Syncpoint value:  {syncpt.syncpoint_value}")
    print(f"  GPU VA:           0x{syncpt.gpu_va:012x}")

    # 10. Allocate compute class!
    print(f"\n=== ALLOC_OBJ_CTX (compute class 0x{compute_class:04x}) ===")
    obj = nvgpu_alloc_obj_ctx_args()
    obj.class_num = compute_class
    obj.flags = 0
    nv_ioctl(ch_fd, NVGPU_IOCTL_CHANNEL_ALLOC_OBJ_CTX, obj)
    print(f"  Compute object:   class=0x{obj.class_num:04x}, obj_id=0x{obj.obj_id:016x}")
    print(f"  ✓ COMPUTE CLASS ALLOCATED SUCCESSFULLY!")

    return {
        "tsg_fd": tsg_fd,
        "ch_fd": ch_fd,
        "gpfifo_dmabuf_fd": gpfifo_dmabuf_fd,
        "userd_dmabuf_fd": userd_dmabuf_fd,
        "work_submit_token": setup.work_submit_token,
        "syncpoint_id": syncpt.syncpoint_id,
        "compute_class": compute_class,
    }


def test_nvmap_heaps(nvmap_fd):
    """Query available memory heaps."""
    print("\n=== NVMAP GET_AVAILABLE_HEAPS ===")
    args = nvmap_available_heaps()
    nv_ioctl(nvmap_fd, NVMAP_IOC_GET_AVAILABLE_HEAPS, args)
    
    heap_names = {
        (1 << 31): "SYSMEM",
        (1 << 30): "IOVMM",
        (1 << 28): "CARVEOUT_VPR",
        (1 << 27): "CARVEOUT_TSEC",
        (1 << 26): "CARVEOUT_VIDMEM",
        (1 << 3):  "CARVEOUT_GPU",
        (1 << 2):  "CARVEOUT_FSI",
        (1 << 1):  "CARVEOUT_IVM",
        (1 << 0):  "CARVEOUT_GENERIC",
    }
    print(f"  Heap bitmask:     0x{args.heaps:016x}")
    available = []
    for bit, name in sorted(heap_names.items()):
        if args.heaps & bit:
            available.append(f"{name} (1<<{bit.bit_length()-1})")
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
    dmabuf_fd = None
    try:
        handle, dmabuf_fd = test_nvmap_create_alloc(nvmap_fd, size=4096)
        success_count += 1
        print("  ✓ NVMAP CREATE + ALLOC succeeded!")
    except Exception as e:
        print(f"  ✗ NVMAP CREATE + ALLOC failed: {e}")

    # Test 5: ALLOC_AS (create GPU address space)
    total_tests += 1
    as_fd = None
    try:
        as_fd = test_alloc_as(ctrl_fd)
        success_count += 1
        print("  ✓ ALLOC_AS succeeded!")
    except Exception as e:
        print(f"  ✗ ALLOC_AS failed: {e}")

    # Test 6: MAP_BUFFER_EX (map buffer into GPU VA)
    total_tests += 1
    if as_fd is not None and dmabuf_fd is not None:
        try:
            gpu_va = test_map_buffer(as_fd, dmabuf_fd, size=4096)
            success_count += 1
            print("  ✓ MAP_BUFFER_EX succeeded!")
        except Exception as e:
            print(f"  ✗ MAP_BUFFER_EX failed: {e}")
    else:
        print("  ✗ MAP_BUFFER_EX skipped (AS or dmabuf not available)")

    # Test 7: Full channel + compute class setup
    total_tests += 1
    compute_class = chars.compute_class if chars else 0xc7c0
    if as_fd is not None:
        try:
            channel_info = test_full_channel_setup(ctrl_fd, as_fd, nvmap_fd, compute_class)
            success_count += 1
            print("\n  ✓ FULL CHANNEL + COMPUTE SETUP succeeded!")
        except Exception as e:
            print(f"\n  ✗ FULL CHANNEL + COMPUTE SETUP failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("  ✗ Channel setup skipped (AS not available)")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Results: {success_count}/{total_tests} tests passed")
    if success_count == total_tests:
        print("ALL TESTS PASSED!")
        print("\nWe have proven:")
        print("  1. GPU characteristics readable (arch, compute_class, flags)")
        print("  2. Memory allocation works (nvmap IOVMM)")
        print("  3. GPU VA mapping works (MAP_BUFFER_EX)")
        print("  4. Channel + TSG + compute class setup works")
        print("  5. Usermode submit is available")
        print("\nNEXT: Build TegraIface for tinygrad!")
    else:
        print("Some tests failed — check errors above")
    print(f"{'=' * 60}")

    # Cleanup
    os.close(ctrl_fd)
    os.close(nvmap_fd)

    return 0 if success_count == total_tests else 1


if __name__ == "__main__":
    sys.exit(main())
