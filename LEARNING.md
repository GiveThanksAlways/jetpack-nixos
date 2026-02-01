# Understanding the jetpack-nixos Repository

This repository packages NVIDIA's JetPack SDK for NixOS, enabling you to run NixOS on Jetson devices (Orin, Xavier, Thor) with full GPU/CUDA support.

---

## Table of Contents

1. [Repository Structure](#1-repository-structure)
2. [Nix Fundamentals for This Repo](#2-nix-fundamentals-for-this-repo)
3. [The Overlay System](#3-the-overlay-system)
4. [Package Scopes and callPackage](#4-package-scopes-and-callpackage)
5. [Building from Debian Packages](#5-building-from-debian-packages)
6. [The NixOS Module System](#6-the-nixos-module-system)
7. [Kernel Architecture](#7-kernel-architecture)
8. [Flashing Workflow](#8-flashing-workflow)
9. [Package Categories](#9-package-categories)
10. [Development Tasks](#10-development-tasks)

---

## 1. Repository Structure

```
.
├── flake.nix                 # Entry point: defines outputs, NixOS configs, overlays
├── overlay.nix               # Main overlay applied to nixpkgs
├── mk-overlay.nix            # Builds the nvidia-jetpack package set
├── modules/                  # NixOS modules for system configuration
│   ├── default.nix           # Main module: hardware.nvidia-jetpack options
│   ├── flash-script.nix      # Firmware flashing configuration
│   ├── devices.nix           # Per-device (SOM) configurations
│   ├── graphics.nix          # GPU/graphics setup
│   ├── cuda.nix              # CUDA configuration
│   └── ...
├── pkgs/                     # Package definitions
│   ├── l4t/                  # L4T (Linux for Tegra) packages from NVIDIA debs
│   ├── kernels/              # Kernel builds for r35/r36/r38
│   ├── flash-tools/          # Flashing utilities
│   ├── uefi-firmware/        # EDK2-based UEFI firmware
│   ├── optee/                # OP-TEE (secure world) packages
│   ├── samples/              # CUDA/multimedia samples
│   └── ...
├── device-pkgs/              # Device-specific flashing scripts
├── sourceinfo/               # Metadata: deb sources and git repos per L4T version
│   ├── r35.6-debs.json
│   ├── r36.4.4-gitrepos.json
│   └── ...
```

### What Each Top-Level File Does

| File | Purpose |
|------|---------|
| `flake.nix` | Declares inputs (nixpkgs) and outputs (packages, NixOS configs, overlays) |
| `overlay.nix` | Maps JetPack major versions to overlays |
| `mk-overlay.nix` | The factory that builds `nvidia-jetpack` package set for a given L4T version |
| `overlay-with-config.nix` | Applies device-specific configuration to the overlay |

---

## 2. Nix Fundamentals for This Repo

### Functions and Currying

Nix functions take exactly one argument. Multi-argument functions are curried:

```nix
# Single argument
greet = name: "Hello, ${name}";
greet "world"  # => "Hello, world"

# "Multi-argument" via currying
add = a: b: a + b;
add 1 2        # => 3
addFive = add 5;  # Partial application
addFive 3      # => 8
```

### Attribute Sets (The Core Data Structure)

```nix
# Attribute set (like a dict/object)
person = {
  name = "Ada";
  age = 36;
};
person.name  # => "Ada"

# Nested access
config.hardware.nvidia-jetpack.som

# Recursive sets (can self-reference)
rec {
  x = 1;
  y = x + 1;  # y = 2
}
```

### The `{ arg1, arg2 }:` Pattern (Destructuring)

This is NOT a set literal—it's a function that takes a set and destructures it:

```nix
# Function that takes a set with 'name' and 'age' keys
describe = { name, age }: "${name} is ${toString age}";
describe { name = "Ada"; age = 36; }  # => "Ada is 36"

# With defaults
describe = { name, age ? 0 }: "${name} is ${toString age}";

# With extra args allowed
describe = { name, ... }: "Hello ${name}";
describe { name = "Ada"; ignored = true; }  # Works!
```

### `let ... in` Expressions

Define local bindings:

```nix
let
  x = 1;
  y = 2;
in
  x + y  # => 3
```

### `with` Expression

Brings an attrset's keys into scope:

```nix
let mySet = { a = 1; b = 2; };
in with mySet; a + b  # => 3

# Common pattern: with lib; ...
with lib; optional (x > 0) "positive"
```

### `inherit` Keyword

Shorthand for `x = x`:

```nix
let name = "Ada";
in {
  inherit name;      # Same as: name = name;
  inherit (lib) optional mkIf;  # Same as: optional = lib.optional; mkIf = lib.mkIf;
}
```

### `//` (Update Operator)

Merge attrsets (right side wins):

```nix
{ a = 1; b = 2; } // { b = 3; c = 4; }
# => { a = 1; b = 3; c = 4; }
```

---

## 3. The Overlay System

### What Is an Overlay?

An overlay is a function that modifies nixpkgs. It takes two arguments:

```nix
final: prev: {
  # final = the FINAL fixed-point of nixpkgs (after all overlays)
  # prev = nixpkgs BEFORE this overlay (use for overriding)
  
  myPackage = final.callPackage ./my-package.nix { };
}
```

**Key insight:** Use `final` when you want the final version of a package. Use `prev` when overriding an existing package to avoid infinite recursion.

### How This Repo Uses Overlays

```nix
# overlay.nix - Creates versioned overlays
{
  "6" = import ./mk-overlay.nix { l4tMajorMinorPatchVersion = "36.4.4"; };
  "5" = import ./mk-overlay.nix { l4tMajorMinorPatchVersion = "35.6.1"; };
  # ...
}
```

When you set `hardware.nvidia-jetpack.majorVersion = "6"`, the corresponding overlay is applied to nixpkgs, making `pkgs.nvidia-jetpack.*` available.

### The Overlay Factory: `mk-overlay.nix`

This file generates an overlay for a specific L4T version:

```nix
# Simplified mk-overlay.nix
{ l4tMajorMinorPatchVersion }:
final: prev: {
  nvidia-jetpack = final.lib.makeScope final.newScope (self: {
    # Version info available to all packages
    inherit l4tMajorMinorPatchVersion;
    
    # Version predicates
    l4tAtLeast = lib.versionAtLeast l4tMajorMinorPatchVersion;
    
    # Packages
    l4t-core = self.callPackage ./pkgs/l4t/l4t-core.nix { };
    flash-tools = self.callPackage ./pkgs/flash-tools { };
    # ...
  });
}
```

---

## 4. Package Scopes and callPackage

### `callPackage` Explained

`callPackage` is Nix's dependency injection. It:

1. Takes a function (usually from a .nix file)
2. Inspects its argument names
3. Auto-fills them from the package set

```nix
# pkgs/l4t/l4t-core.nix
{ lib, stdenv, fetchurl, autoPatchelfHook }:
stdenv.mkDerivation {
  pname = "l4t-core";
  # ...
}

# In nixpkgs or an overlay:
l4t-core = callPackage ./pkgs/l4t/l4t-core.nix { };
# Automatically passes: lib, stdenv, fetchurl, autoPatchelfHook

# Override specific args:
l4t-core = callPackage ./pkgs/l4t/l4t-core.nix {
  stdenv = clangStdenv;  # Use clang instead of gcc
};
```

### `makeScope` - Creating Package Sets

`makeScope` creates a new package set with its own `callPackage`:

```nix
nvidia-jetpack = lib.makeScope newScope (self: {
  # self.callPackage looks in THIS scope first, then falls back to nixpkgs
  
  l4t-core = self.callPackage ./l4t-core.nix { };
  l4t-cuda = self.callPackage ./l4t-cuda.nix { };
  # l4t-cuda.nix can have { l4t-core, ... } and it auto-resolves!
});
```

**Why this matters:** Packages within `nvidia-jetpack` can depend on each other by name without manual wiring.

---

## 5. Building from Debian Packages

NVIDIA distributes JetPack as Debian packages. This repo unpacks and adapts them for Nix.

### Source Tracking: `sourceinfo/`

```json
// sourceinfo/r36.4-debs.json
{
  "nvidia-l4t-core": {
    "url": "https://repo.download.nvidia.com/.../nvidia-l4t-core_36.4.0-xxx_arm64.deb",
    "hash": "sha256-..."
  }
}
```

These are generated by `debs-update.py` from NVIDIA's apt repository.

### The `buildFromDebs` Helper

```nix
# pkgs/buildFromDebs.nix (simplified)
{ pname, buildInputs ? [], postPatch ? "", ... }:
stdenv.mkDerivation {
  inherit pname;
  
  # Fetch the deb from sourceinfo
  src = fetchurl debInfo.${pname};
  
  nativeBuildInputs = [ dpkg autoPatchelfHook ];
  
  unpackPhase = ''
    dpkg-deb -x $src .
  '';
  
  # autoPatchelfHook automatically fixes library paths
  inherit buildInputs;
}
```

### Using `buildFromDebs`

```nix
# pkgs/l4t/l4t-multimedia.nix
{ buildFromDebs, l4t-core, l4t-cuda, glib, wayland }:
buildFromDebs {
  pname = "nvidia-l4t-multimedia";
  buildInputs = [ l4t-core l4t-cuda glib wayland ];
  
  # Fix hardcoded paths
  postPatch = ''
    substituteInPlace lib/libv4l2.so --replace /usr/lib /run/current-system/sw/lib
  '';
}
```

---

## 6. The NixOS Module System

### Module Structure

A NixOS module is an attrset (or function returning one) with these keys:

```nix
{ config, lib, pkgs, ... }:
{
  # Declare options this module provides
  options.myService.enable = lib.mkOption {
    type = lib.types.bool;
    default = false;
    description = "Enable my service";
  };

  # Define configuration when options are set
  config = lib.mkIf config.myService.enable {
    systemd.services.myService = { ... };
  };
}
```

### Key lib Functions for Modules

```nix
# mkOption - Declare an option
options.foo = lib.mkOption {
  type = lib.types.str;
  default = "bar";
  description = "The foo setting";
};

# mkEnableOption - Shorthand for boolean enable options
options.services.myService.enable = lib.mkEnableOption "my service";

# mkIf - Conditional configuration
config = lib.mkIf config.services.myService.enable {
  environment.systemPackages = [ pkgs.myService ];
};

# mkMerge - Combine multiple config fragments
config = lib.mkMerge [
  { environment.systemPackages = [ pkgs.base ]; }
  (lib.mkIf config.foo.enable { environment.systemPackages = [ pkgs.foo ]; })
];

# mkForce - Override lower-priority settings
boot.kernelPackages = lib.mkForce pkgs.linuxPackages_latest;

# mkDefault - Set default (can be overridden without mkForce)
boot.kernelPackages = lib.mkDefault pkgs.linuxPackages;
```

### This Repo's Main Module Options

```nix
# modules/default.nix provides:
hardware.nvidia-jetpack = {
  enable = lib.mkEnableOption "NVIDIA Jetson support";
  
  som = lib.mkOption {
    type = lib.types.enum [
      "orin-agx" "orin-agx-industrial" "orin-nx" "orin-nano"
      "xavier-agx" "xavier-agx-industrial" "xavier-nx" "xavier-nx-emmc"
    ];
    description = "Jetson System-on-Module type";
  };
  
  carrierBoard = lib.mkOption {
    type = lib.types.str;
    default = "devkit";
  };
  
  majorVersion = lib.mkOption {
    type = lib.types.enum [ "5" "6" "7" ];
    description = "JetPack major version";
  };
};
```

### How Modules Compose in This Repo

```
modules/default.nix          # Core: kernel, firmware, udev
    ├── imports devices.nix      # SOM-specific configs
    ├── imports flash-script.nix # Flashing options
    ├── imports graphics.nix     # GPU driver setup
    ├── imports cuda.nix         # CUDA assertions
    └── imports nvpmodel.nix     # Power profiles
```

---

## 7. Kernel Architecture

### Version Mapping

| JetPack | L4T Version | Kernel Dir | Devices |
|---------|-------------|------------|---------|
| 5.x | r35.x | `pkgs/kernels/r35/` | Xavier, Orin |
| 6.x | r36.x | `pkgs/kernels/r36/` | Orin |
| 7.x | r38.x | `pkgs/kernels/r38/` | Thor |

### Kernel Package Structure

```nix
# pkgs/kernels/r36/default.nix
{ lib, buildLinux, fetchgit, kernelPatches, ... }:
buildLinux {
  version = "5.15.136-l4t";
  
  src = fetchgit {
    url = "https://nv-tegra.nvidia.com/linux-nv-oot.git";
    rev = "jetson_36.4";
    hash = "sha256-...";
  };
  
  kernelPatches = [
    { name = "nix-paths"; patch = ./nix-paths.patch; }
  ];
  
  structuredExtraConfig = with lib.kernel; {
    TEGRA_BPMP = yes;
    NVGPU = module;
    # ...
  };
}
```

### Out-of-Tree Modules

NVIDIA ships some drivers as out-of-tree modules:

```nix
# pkgs/kernels/r36/oot-modules.nix
{ stdenv, kernel }:
stdenv.mkDerivation {
  pname = "nvidia-oot-modules";
  
  makeFlags = kernel.makeFlags ++ [
    "KERNEL_HEADERS=${kernel.dev}/lib/modules/${kernel.modDirVersion}/source"
    "KERNEL_OUTPUT=${kernel.dev}/lib/modules/${kernel.modDirVersion}/build"
  ];
  
  installPhase = ''
    make INSTALL_MOD_PATH=$out modules_install
  '';
}
```

---

## 8. Flashing Workflow

### The Flashing Pipeline

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│ NixOS Config │───▶│ Flash Script │───▶│ Jetson SOM  │
└─────────────┘    └──────────────┘    └─────────────┘
                          │
                   Uses these:
                   ├── UEFI Firmware
                   ├── OP-TEE
                   ├── Device Tree
                   └── Kernel + Initrd
```

### Flash Script Generation

```nix
# device-pkgs/default.nix
{ config, pkgs, ... }:
{
  # These become available on your NixOS config
  config.system.build = {
    flashScript = pkgs.callPackage ./flash-script.nix {
      inherit (config.hardware.nvidia-jetpack) som carrierBoard;
    };
    
    initrdFlashScript = pkgs.callPackage ./initrdflash-script.nix { };
  };
}
```

### Flashing a Device

```bash
# 1. Build the flash script
nix build .#nixosConfigurations.orin-agx-devkit.config.system.build.flashScript

# 2. Put device in recovery mode (hold REC button + power cycle)

# 3. Verify USB connection
lsusb | grep -i nvidia
# Should show: "NVIDIA Corp. APX"

# 4. Flash (uses initrd-flash method)
sudo ./result/bin/initrd-flash-orin-agx-devkit-cross

# 5. Build NixOS Installer ISO (takes time - cross-compiling ARM64)
nix build .#iso_minimal

# 6. Verify ISO was built
ls -lh ./result/iso/
# Should show: nixos-minimal-25.11.YYYYMMDD.HASH-aarch64-linux.iso (~1.3G)

# 7. Write ISO to USB drive (THIS WILL ERASE ALL DATA ON THE USB DRIVE!)
# First, identify your USB drive:
lsblk -o NAME,SIZE,MODEL,TRAN,VENDOR | grep -E "NAME|usb|sd"

# Set the USB device path (DOUBLE-CHECK THIS IS CORRECT!)
export DEV_USB="/dev/sdf"  # Replace with your actual USB device

# Verify you have the right device before proceeding:
lsblk $DEV_USB

# Write the ISO (This destroys all data on $DEV_USB!)
sudo dd if=./result/iso/nixos-minimal-*.iso of=$DEV_USB bs=1M oflag=sync status=progress

# Wait for dd to complete (shows progress), then eject safely:
sync
sudo eject $DEV_USB

# 8. Boot Jetson from USB installer
# - just plug into any USB port. I did the bottom one next to the ethernet port

# 9. Install NixOS to internal storage
# See below for detailed installation steps
```

### Installing NixOS to eMMC (No SSD)

The Jetson eMMC contains firmware partitions that **must not be destroyed**:
```
mmcblk0p1  (57.8G)  = UDA (User Data Area) - SAFE to use for NixOS
mmcblk0p2-p15       = Firmware partitions - DO NOT TOUCH!
mmcblk0boot0/boot1  = Boot partitions - DO NOT TOUCH!
```

**What happens if you mess up?**
- If you only format `mmcblk0p1`: You can always reinstall, device is fine
- If you destroy the partition table (`parted mklabel gpt`): You need to re-flash firmware from host PC
- The UEFI firmware lives in flash memory, not filesystem - device is NOT bricked, just needs re-flash

**Recovery path:** Put device in recovery mode → re-run flash script from host PC → device is restored

#### Step-by-Step eMMC Installation

```bash
# 1. First, check current partition layout
lsblk
sudo fdisk -l /dev/mmcblk0

# 2. Check if there's an existing ESP (EFI System Partition)
sudo blkid | grep -i fat

# 3. The UDA partition (mmcblk0p1) needs to be split into:
#    - ESP (512MB, FAT32) for bootloader
#    - Root (remainder, ext4) for NixOS
#
# We'll delete p1 and create two new partitions in its place.
# This is SAFE because we're only touching the UDA area.

# WARNING: Double-check partition numbers! p1 should be ~57.8GB

# 4. Use fdisk to repartition (safer than parted for this)
sudo fdisk /dev/mmcblk0

# In fdisk, type these commands:
#   p          (print current partitions - verify p1 is the 57.8G one!)
#   d          (delete partition)
#   1          (partition 1 - the UDA)
#   n          (new partition)
#   1          (partition number 1)
#   <enter>    (default first sector - fdisk picks the right one)
#   +512M      (512MB for ESP)
#   t          (change type)
#   1          (partition 1)
#   1          (EFI System type)
#   n          (new partition)
#   <enter>    (default partition number, likely 16 or uses free space)
#   <enter>    (default first sector)
#   <enter>    (default last sector - use remaining space)
#   p          (print to verify - should see new ESP + large partition)
#   w          (write changes - POINT OF NO RETURN for partition table)

# 5. Format the new partitions
sudo mkfs.fat -F 32 -n ESP /dev/mmcblk0p1
sudo mkfs.ext4 -L nixos /dev/mmcblk0p16  # Or whatever the new root partition number is

# 6. Mount filesystems
sudo mount /dev/disk/by-label/nixos /mnt
sudo mkdir -p /mnt/boot
sudo mount /dev/disk/by-label/ESP /mnt/boot

# 7. Generate hardware configuration
sudo nixos-generate-config --root /mnt

# 8. Edit configuration.nix
sudo nano /mnt/etc/nixos/configuration.nix
```

Add this configuration (for Orin AGX with JetPack 6):

```nix
{ config, lib, pkgs, ... }:

{
  imports = [
    ./hardware-configuration.nix
    (builtins.fetchTarball "https://github.com/anduril/jetpack-nixos/archive/master.tar.gz" + "/modules/default.nix")
  ];

  # Jetson configuration - CHANGE som IF NEEDED:
  # Options: "orin-agx", "orin-nx", "orin-nano"
  hardware.nvidia-jetpack.enable = true;
  hardware.nvidia-jetpack.som = "orin-agx";
  hardware.nvidia-jetpack.carrierBoard = "devkit";

  # GPU support (required for CUDA and containers too)
  hardware.graphics.enable = true;

  # Bootloader
  boot.loader.systemd-boot.enable = true;
  boot.loader.efi.canTouchEfiVariables = true;

  # Networking
  networking.hostName = "jetson";
  networking.networkmanager.enable = true;

  # User account - CHANGE THIS!
  users.users.spencer = {
    isNormalUser = true;
    extraGroups = [ "wheel" "video" "networkmanager" ];
    initialPassword = "changeme";  # Change after first login!
  };

  # Enable SSH for remote access
  services.openssh.enable = true;

  # Allow unfree packages (required for NVIDIA)
  nixpkgs.config.allowUnfree = true;

  system.stateVersion = "25.11";
}
```

```bash
# 9. Install NixOS
sudo nixos-install

# 10. Set root password when prompted

# 11. Reboot (remove USB when it powers off)
sudo reboot
```

### Installing NixOS to NVMe SSD (Recommended)

If you have an NVMe SSD, installation is simpler because you can partition freely:

```bash
# 1. Identify the NVMe drive
lsblk
# Should show /dev/nvme0n1
# Also shows /dev/mmcblk0 (eMMC with firmware) and /dev/sda (USB installer)

# 2. Partition the NVMe (this is SAFE - won't affect eMMC firmware)
sudo parted /dev/nvme0n1 -- mklabel gpt
sudo parted /dev/nvme0n1 -- mkpart ESP fat32 1MB 512MB
sudo parted /dev/nvme0n1 -- set 1 esp on
sudo parted /dev/nvme0n1 -- mkpart primary 512MB 100%

# 3. Format
sudo mkfs.fat -F 32 -n ESP /dev/nvme0n1p1
sudo mkfs.ext4 -L nixos /dev/nvme0n1p2

# 4. Mount
sudo mount /dev/disk/by-label/nixos /mnt
sudo mkdir -p /mnt/boot
sudo mount /dev/disk/by-label/ESP /mnt/boot

# 5. Generate hardware configuration
sudo nixos-generate-config --root /mnt

# 6. Create flake-based configuration structure
sudo mkdir -p /mnt/etc/nixos
```

#### Create flake.nix

```bash
sudo nano /mnt/etc/nixos/flake.nix
```

Paste this content:

```nix
{
  description = "Jetson Orin AGX NixOS Configuration";

  inputs = {
    # Using nixos-unstable for latest packages
    # Can pin to nixos-25.05 for stability
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    
    # Jetpack NixOS for Jetson support
    jetpack-nixos.url = "github:anduril/jetpack-nixos/master";
    jetpack-nixos.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, nixpkgs, jetpack-nixos, ... }@inputs: {
    nixosConfigurations.jetson = nixpkgs.lib.nixosSystem {
      system = "aarch64-linux";
      specialArgs = { inherit inputs; };
      modules = [
        jetpack-nixos.nixosModules.default
        ./configuration.nix
      ];
    };
  };
}
```

#### Create configuration.nix

```bash
sudo nano /mnt/etc/nixos/configuration.nix
```

Paste this content:

```nix
{ config, lib, pkgs, inputs, ... }:

{
  imports = [
    ./hardware-configuration.nix
  ];

  # ==========================================================================
  # JETSON HARDWARE CONFIGURATION
  # ==========================================================================
  
  hardware.nvidia-jetpack = {
    enable = true;
    som = "orin-agx";           # Options: "orin-agx", "orin-nx", "orin-nano"
    carrierBoard = "devkit";
    # modesetting.enable = true;  # Enable for Wayland support
  };

  # GPU support - required even for CUDA and containers
  hardware.graphics.enable = true;

  # ==========================================================================
  # BOOTLOADER
  # ==========================================================================
  
  boot.loader.systemd-boot.enable = true;
  boot.loader.efi.canTouchEfiVariables = true;
  
  # Use the Jetson-specific kernel (set by jetpack module, but explicit here)
  # boot.kernelPackages = pkgs.nvidia-jetpack.kernelPackages;

  # ==========================================================================
  # NETWORKING
  # ==========================================================================
  
  networking.hostName = "jetson";
  networking.networkmanager.enable = true;
  
  # Firewall - allow SSH, can add more ports later
  networking.firewall = {
    enable = true;
    allowedTCPPorts = [ 22 ];
  };

  # ==========================================================================
  # USER CONFIGURATION
  # ==========================================================================
  
  users.users.spencer = {
    isNormalUser = true;
    description = "Spencer";
    extraGroups = [ 
      "wheel"           # sudo access
      "video"           # GPU/display access
      "networkmanager"  # Network management
      "docker"          # Docker (if enabled)
    ];
    initialPassword = "changeme";  # CHANGE THIS after first login!
    openssh.authorizedKeys.keys = [
      # Add your SSH public key here for passwordless login:
      # "ssh-ed25519 AAAAC3Nza... your-key-comment"
    ];
  };

  # ==========================================================================
  # SERVICES
  # ==========================================================================
  
  # SSH - essential for headless operation
  services.openssh = {
    enable = true;
    settings = {
      PermitRootLogin = "no";
      PasswordAuthentication = true;  # Set to false after adding SSH keys
    };
  };

  # ==========================================================================
  # PACKAGES
  # ==========================================================================
  
  # Allow unfree packages (required for NVIDIA)
  nixpkgs.config.allowUnfree = true;

  # System packages
  environment.systemPackages = with pkgs; [
    # Essential tools
    vim
    git
    wget
    curl
    htop
    tmux
    
    # Networking
    iproute2
    ethtool
    
    # Hardware info
    pciutils
    usbutils
    lshw
  ];

  # ==========================================================================
  # NIX SETTINGS
  # ==========================================================================
  
  nix = {
    settings = {
      experimental-features = [ "nix-command" "flakes" ];
      auto-optimise-store = true;
      trusted-users = [ "root" "@wheel" ];
    };
    
    # Garbage collection
    gc = {
      automatic = true;
      dates = "weekly";
      options = "--delete-older-than 30d";
    };
  };

  # ==========================================================================
  # SYSTEM
  # ==========================================================================
  
  # Timezone - change to your location
  time.timeZone = "America/Los_Angeles";

  # Locale
  i18n.defaultLocale = "en_US.UTF-8";

  # This value determines the NixOS release from which the default
  # settings for stateful data, like file locations and database versions
  # on your system were taken. It's perfectly fine and recommended to leave
  # this value at the release version of the first install of this system.
  system.stateVersion = "25.11";
}
```

#### Install NixOS

```bash
# 7. Install NixOS (this will take a while - downloads packages)
sudo nixos-install --flake /mnt/etc/nixos#jetson

# 8. Set root password when prompted

# 9. Reboot (remove USB after power off)
sudo reboot
```

#### Post-Installation

After rebooting, your Jetson should boot from the SSD into NixOS!

```bash
# Login as your user (spencer / changeme)

# Change your password immediately!
passwd

# Verify Jetson hardware is working
cat /proc/device-tree/model
# Should show: NVIDIA Jetson AGX Orin Developer Kit

# Check GPU is accessible
ls /dev/nvidia*

# Check CUDA (if needed)
# nvidia-smi equivalent for Jetson:
cat /sys/class/tegra-gpe/device/load

# Future updates - from your dev machine or on device:
sudo nixos-rebuild switch --flake /etc/nixos#jetson
```

### Note on Determinate Nix

Determinate Nix (from Determinate Systems) provides a better out-of-box experience
for the `nix` CLI on non-NixOS systems. On NixOS itself, you already have nix
installed, but you can still benefit from their installer if you ever need to
set up nix on other machines (like your dev machine).

Key differences:
- Determinate installer: `curl --proto '=https' --tlsv1.2 -sSf -L https://install.determinate.systems/nix | sh`
- Enables flakes by default
- Better uninstall support
- Same nix underneath

For NixOS, we configure flakes in `nix.settings.experimental-features` (already done above).


### Flashing from WSL2 (Windows)

When flashing from WSL2, USB devices must be attached via `usbipd`. The flash process involves
multiple USB reconnects as the Jetson reboots through different stages, so **auto-attach is required**.

```powershell
# In PowerShell (Admin), first bind the devices to make them available:
usbipd list                           # Find the NVIDIA devices (APX and Tegra On-Platform Operator)
usbipd bind --busid <APX-busid>       # e.g., 4-6
usbipd bind --busid <serial-busid>    # e.g., 6-4

# Attach with auto-attach (re-attaches automatically after USB reconnects):
usbipd attach --wsl --busid <APX-busid> --auto-attach
usbipd attach --wsl --busid <serial-busid> --auto-attach
```

#### The Port-Hopping Problem

During flashing, the Jetson disconnects and reconnects multiple times. Sometimes it comes back on a
**different USB bus/port** (e.g., `4-6` becomes `1-3`). When this happens:

1. The device shows as "Shared" or "Not shared" instead of "Attached"
2. You need to quickly `bind` and `attach` the new busid
3. The flash script may time out waiting if you're too slow

**Practical approach:** Keep multiple PowerShell Admin windows open and monitor with `usbipd list`.
Listen for the USB disconnect/reconnect sound cues from Windows, then check which busid changed.

```powershell
# When you see a new device appear (e.g., COM10 on busid 1-3):
usbipd bind --busid 1-3
usbipd attach --wsl --busid 1-3 --auto-attach

# You can also use 'usbipd attach --wsl --hardware-id VID:PID' to attach by VID:PID instead:
usbipd attach --wsl --hardware-id 0955:7023 --auto-attach  # APX device
usbipd attach --wsl --hardware-id 0955:7045 --auto-attach  # Tegra On-Platform Operator
```

#### The "Flashing may have failed" False Alarm

Even when flashing succeeds, WSL2 may report "Flashing may have failed" because:
- The serial connection times out during the final reboot
- The Jetson disconnected before the expect script could confirm success

**Check the serial console** - if you see `Jetson UEFI firmware (version X.X.X)` and a boot menu,
the flash succeeded regardless of what the script reported.

#### Tips for WSL2 Flashing Success

1. **Use `--auto-attach`** on all NVIDIA devices before starting
2. **Keep `usbipd list` running** in a separate PowerShell window
3. **Listen for USB sounds** - Windows plays a sound on disconnect/reconnect
4. **Be ready to rebind** - have `usbipd bind --busid X-X` ready to paste
5. **Watch the serial console** - it shows the real status, not the flash script
6. **Consider native Linux** - a Linux USB stick or VM is more reliable for flashing

### What Gets Flashed

| Partition | Contents | Source |
|-----------|----------|--------|
| `esp` | UEFI firmware (EDK2) | `pkgs/uefi-firmware/` |
| `A_kernel` | Linux kernel | `pkgs/kernels/r36/` |
| `A_kernel-dtb` | Device tree blobs | Extracted from kernel |
| `OP-TEE` | Secure world | `pkgs/optee/` |
| `BCT` | Boot configuration | Flash tools |

### Understanding Initrd Flash (Why It Exists)

There are two ways to flash a Jetson from a host PC:

#### Traditional Flash (Old Method)
```
┌──────────┐  USB Recovery Protocol   ┌──────────┐
│  Host PC │ ──────────────────────▶  │  Jetson  │
│          │   (slow, proprietary)    │  (APX)   │
└──────────┘                          └──────────┘
```

The host PC uses NVIDIA's `tegrarcm_v2` tool to push data over a proprietary USB recovery protocol.
This is **slow** (~30-60 minutes for full flash) and unreliable with NVMe drives.

#### Initrd Flash (New Method)
```
Phase 1: Boot a minimal Linux on the Jetson
┌──────────┐   RCM Boot    ┌──────────┐
│  Host PC │ ────────────▶ │  Jetson  │  Boots initrd into RAM
└──────────┘               └──────────┘
                                 │
Phase 2: Jetson exposes its storage as USB Mass Storage
                                 ▼
┌──────────┐   USB Gadget  ┌──────────┐
│  Host PC │ ◀──────────── │  Jetson  │  "I'm a USB drive now!"
│          │  Block device │ (initrd) │
└──────────┘               └──────────┘
                                 │
Phase 3: Host writes partitions like a regular disk
                                 ▼
┌──────────┐   dd/mtd-utils ┌──────────┐
│  Host PC │ ─────────────▶ │  Jetson  │  Fast block writes!
└──────────┘                └──────────┘
```

**Why is this faster?**
- USB Mass Storage is a standard protocol - the kernel treats the Jetson's eMMC/NVMe as a local disk
- Block writes are much faster than the proprietary RCM protocol
- Uses standard Linux tools (`dd`, `mtd-utils`) instead of NVIDIA's closed-source tools

### The `/dev/serial/by-id/usb-NixOS_serial_0-if00` Device

This is a **Linux USB serial gadget** - let's break it down from first principles:

#### What is a USB Gadget?

Normally, your PC is a USB **host** and devices (keyboard, mouse, phone) are USB **peripherals**.
But some devices can switch roles - your phone can be a peripheral (file transfer) or a host (USB OTG).

The Jetson in initrd mode becomes a USB **peripheral** that presents itself as:
1. A **USB Mass Storage device** (its internal storage appears as a disk on your PC)
2. A **USB Serial device** (for communication with the flash script)

#### The Path Breakdown

```
/dev/serial/by-id/usb-NixOS_serial_0-if00
│    │       │     │   │      │       │
│    │       │     │   │      │       └── if00 = USB interface 0
│    │       │     │   │      └────────── serial_0 = first serial device
│    │       │     │   └───────────────── NixOS = USB vendor string (set in initrd)
│    │       │     └───────────────────── usb- prefix for USB devices
│    │       └─────────────────────────── by-id = persistent naming (vs /dev/ttyACM0)
│    └─────────────────────────────────── serial = serial port devices
└──────────────────────────────────────── /dev = device files
```

The initrd configures the Jetson's USB controller in "gadget mode" with:
- Vendor string: `NixOS`
- Product string: `serial_0`

This creates a **predictable device path** the flash script can wait for.

#### How the Flash Script Uses It

```bash
# From initrd-flash-orin-agx-devkit-cross:

# Wait for the serial device to appear (Jetson has booted initrd)
until [[ -e /dev/serial/by-id/usb-NixOS_serial_0-if00 || $counter -gt 480 ]] ; do
  echo -n "."
  sleep 0.5
done

# Use expect to communicate with the initrd over serial
expect -f /nix/store/.../expect-initrd-flash
```

The `expect` script sends commands over the serial port to:
1. Query the board info (`boardspec: 3701-300-0005--1---`)
2. Erase flash partitions (`Erasing /dev/mtd0...`)
3. Write firmware images (`Step 1/62... Writing mb1...`)
4. Trigger reboot when complete

#### Why Serial Instead of Just Mass Storage?

The Jetson exposes **both** serial and mass storage because:
- **Serial**: For control/status messages and coordinating the flash sequence
- **Mass Storage**: For the actual data transfer (fast block writes)

Think of it like this:
- Serial = "control channel" (small messages: "erase this", "write that", "done!")
- Mass Storage = "data channel" (large payloads: actual firmware blobs)

---

## 9. Package Categories

### L4T Packages (`pkgs/l4t/`)

Core NVIDIA libraries from Debian packages:

| Package | Purpose | Key Files |
|---------|---------|-----------|
| `l4t-core` | Base libraries | `libnvrm.so`, `libnvos.so` |
| `l4t-cuda` | CUDA runtime | `libcuda.so`, `libnvcudla.so` |
| `l4t-multimedia` | Video encode/decode | `libnvmm.so`, gstreamer plugins |
| `l4t-camera` | Argus camera API | `libnvargus.so` |
| `l4t-3d-core` | OpenGL ES | `libGLESv2_nvidia.so` |
| `l4t-gbm` | GBM (buffer management) | `libgbm_nvidia.so` |
| `l4t-nvpmodel` | Power management | `nvpmodel` binary |

### CUDA Extensions (`pkgs/cuda-extensions/`)

Additional CUDA libraries:

```nix
# Access via: pkgs.nvidia-jetpack.cudaPackages.*
cudnn          # Deep learning primitives
tensorrt       # Inference optimizer
vpi            # Vision programming interface
cuda-samples   # Example code
```

### Profiling Tools (`pkgs/cuda-packages-11-4/`)

```nix
nsight_systems_host   # System-wide profiler (runs on x86 host)
nsight_systems_target # Profiler agent (runs on Jetson)
nsight_compute_host   # Kernel profiler (runs on x86 host)
nsight_compute_target # Kernel profiler agent (runs on Jetson)
```

---

## 10. Development Tasks

### Adding a New L4T Package

1. **Verify it exists in source info:**
   ```bash
   grep "nvidia-l4t-yourpkg" sourceinfo/r36.4-debs.json
   ```

2. **Create the package file:**
   ```nix
   # pkgs/l4t/l4t-yourpkg.nix
   { buildFromDebs, l4t-core, someOtherDep }:
   buildFromDebs {
     pname = "nvidia-l4t-yourpkg";
     buildInputs = [ l4t-core someOtherDep ];
     
     # If needed: fix paths, permissions, etc.
     postPatch = ''
       chmod +x bin/yourtool
     '';
   }
   ```

3. **Add to the overlay** (or rely on `packagesFromDirectoryRecursive` in `mk-overlay.nix`)

### Updating Source Info

When NVIDIA releases a new L4T version:

```bash
# 1. Update apt sources and regenerate debs.json
python sourceinfo/debs-update.py r36.5

# 2. Update git repository info
nix build .#bspSrc-36-5
python sourceinfo/gitrepos-update.py r36.5 ./result/source/source_sync.sh
```

### Building and Testing

```bash
# Build a specific package
nix build .#nvidia-jetpack.l4t-core

# Build full system (native, on Jetson)
nix build .#nixosConfigurations.orin-agx-devkit.config.system.build.toplevel

# Cross-compile (from x86_64 to aarch64)
nix build .#nixosConfigurations.orin-agx-devkit-cross.config.system.build.toplevel

# Enter dev shell
nix develop

# Run tests
nix flake check
```

### Debugging Tips

```bash
# See what a package contains
nix build .#nvidia-jetpack.l4t-core && ls -la result/

# Check derivation details
nix show-derivation .#nvidia-jetpack.l4t-core

# Build with verbose output
nix build .#nvidia-jetpack.l4t-core -L

# Enter build environment
nix develop .#nvidia-jetpack.l4t-core
```

---

## 11. Flake Outputs Explained

Run `nix flake show` to see all outputs. Here's what they mean:

### Understanding the Naming Convention

Flash scripts and configs are named: `{som}[-super]-{carrierBoard}[-jp{version}]`

```nix
# From flake.nix line 51-53:
name = c.som 
     + lib.optionalString (c.super or false) "-super" 
     + "-${c.carrierBoard}" 
     + lib.optionalString (c ? majorVersion) "-jp${c.majorVersion}";
```

Examples:
| Name | SOM | Carrier | JetPack |
|------|-----|---------|---------|
| `orin-agx-devkit` | Orin AGX | Developer Kit | 6 (default) |
| `orin-agx-devkit-jp5` | Orin AGX | Developer Kit | 5 |
| `orin-nano-super-devkit` | Orin Nano (Super) | Developer Kit | 6 |
| `xavier-nx-devkit` | Xavier NX | Developer Kit | 5 |

### Key Flake Outputs

```bash
# NixOS Configurations (full system definitions)
.#nixosConfigurations.orin-agx-devkit

# Flash scripts (x86_64 only - for flashing from a PC)
.#flash-orin-agx-devkit           # Shorthand
.#nixosConfigurations.orin-agx-devkit.config.system.build.flashScript  # Full path

# Installer ISOs
.#iso_minimal          # JP6
.#iso_minimal_jp5      # JP5

# Individual packages
.#legacyPackages.x86_64-linux.nvidia-jetpack.flash-tools
.#legacyPackages.aarch64-linux.nvidia-jetpack.l4t-cuda
```

### Local vs Remote Builds

```bash
# LOCAL build (uses your cloned repo)
nix build .#flash-orin-agx-devkit

# REMOTE build (fetches from GitHub, ignores local changes)
nix build github:anduril/jetpack-nixos#flash-orin-agx-devkit
```

**Always use `.#` when developing locally!**

### JetPack Version Selection

| Your Device | Recommended | Command |
|-------------|-------------|---------|
| Orin AGX/NX/Nano | JetPack 6 | `nix build .#flash-orin-agx-devkit` |
| Orin (need JP5) | JetPack 5 | `nix build .#flash-orin-agx-devkit-jp5` |
| Xavier AGX/NX | JetPack 5 | `nix build .#flash-xavier-agx-devkit` |
| Thor | JetPack 7 | `nix build .#flash-thor-agx-devkit` |

The firmware version pre-installed on your device (like R35.4.1) doesn't matter—you're replacing it entirely.

---

## Quick Reference: Minimal Configuration

```nix
# configuration.nix
{ config, pkgs, ... }:
{
  imports = [
    # Import the jetpack-nixos module
    "${builtins.fetchTarball "https://github.com/anduril/jetpack-nixos/archive/master.tar.gz"}/modules/default.nix"
  ];

  hardware.nvidia-jetpack = {
    enable = true;
    som = "orin-agx";         # Your SOM type
    carrierBoard = "devkit";  # Your carrier board
    # majorVersion auto-detected from SOM
  };
  
  # Required for GPU acceleration
  hardware.graphics.enable = true;
  
  # Optional: Docker with GPU support
  virtualisation.docker.enable = true;
  hardware.nvidia-container-toolkit.enable = true;
}
```

---

## Glossary

| Term | Meaning |
|------|---------|
| **L4T** | Linux for Tegra - NVIDIA's BSP (Board Support Package) |
| **JetPack** | NVIDIA's SDK bundle (L4T + CUDA + cuDNN + TensorRT + ...) |
| **SOM** | System-on-Module - the compute module (Orin, Xavier, etc.) |
| **BSP** | Board Support Package - low-level software for hardware |
| **OP-TEE** | Open Portable Trusted Execution Environment |
| **DTB** | Device Tree Blob - hardware description for the kernel |
| **UEFI** | Unified Extensible Firmware Interface - the bootloader |
| **CBoot** | NVIDIA's older bootloader (replaced by UEFI in newer SOMs) |
| **CDI** | Container Device Interface - for GPU container access |
