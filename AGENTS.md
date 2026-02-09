# AGENTS.md

We are working with an Nvidia Jetson dev kit Orin AGX 64GB

## VERY IMPORTANT INFO

- **YOU MUST USE THE MCP TOOL TO ACCESS THE UART SERIAL CONSOLE WITH THE DEV KIT. (serial-uart-mcp)**

- **MAKE SURE TO BUILD USING THE FLAKE/CONFIGURATION inside jetpack-nixos/examples/nixos/**

- **USE SCP TO COPY THE LOCAL FILE CHANGES WE MAKE TO jetpack-nixos/examples/nixos/**

- I repeat, you must use the serial-uart-mcp tool to access the uart serial shell on device to run commands/ see the output.

## Quick UART Recovery Hint

If the UART serial console gets stuck in a pager or log view, spam 'q', RETURN, or CTRL+C to return to the shell prompt.

## More info

- I put the `configuration.nix` and the `hardware-configuration.nix` there for quick experimentation

```bash
# example build
sudo nixos-rebuild switch --flake /home/agent/jetpack-nixos/examples/nixos#nixos-telemetry --show-trace

# scp command (use this to transfer our file changes to the device)
# NOTE!!! when running this command, you should be in the jetpack-nixos root folder (since examples/nixos is relative and spencer is user on PC, agent on dev kit)
# be careful with what dir you are in/ what dir you are copying to on device. For ease of use, we just copy the entire nixos dir
scp -r examples/nixos spencer@192.168.8.162:/home/spencer/jetpack-nixos/examples/
```

Each subfolder is a self-contained flake. Pick the one that fits your use case.

## Layout

| Folder                        | What it does                                                                                                                        |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `nixos/`                      | Base NixOS system config with composable modules (performance, llama.cpp, TabbyAPI, telemetry). Pick a configuration at build time. |
| `llama-cpp-orin/`             | Minimal flake -- builds llama.cpp from upstream with CUDA/OpenSSL for Orin. Dev shell only.                                         |
| `llama-cpp-orin-nix-overlay/` | Full-featured overlay with wrapper scripts (`qwen3-coder`, `qwen3-server`, `llama-benchmark`). Reusable via `overlays.default`.     |
| `LLM/`                        | LLM's                                                                                                                               |
| `tinygrad/`                   | Tinygrad dev shell with CUDA + NV backend. Clone-and-go for LLM examples (GPT-2, LLaMA) on Orin AGX.                                |
| `telemetry-viewer/`           | SSH tunnel script to view Grafana dashboards from your PC. One command, then open Chrome.                                           |

## Quick reference

```bash
# NixOS system build (pick a config)
nixos-rebuild switch --flake ./nixos#nixos
nixos-rebuild switch --flake ./nixos#nixos-perf
nixos-rebuild switch --flake ./nixos#nixos-llama-cpp

# llama.cpp dev shell (simple)
cd llama-cpp-orin && nix develop

# llama.cpp dev shell (overlay + wrappers)
cd llama-cpp-orin-nix-overlay && nix develop

# vLLM dev shell
cd vLLM && nix develop

# tinygrad dev shell (CUDA + NV backend)
cd tinygrad && nix develop

# view telemetry dashboards from your PC (user named agent)
cd telemetry-viewer && ./connect-telemetry.sh <jetson-ip> <ssh-user>
