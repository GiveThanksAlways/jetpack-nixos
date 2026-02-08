import socket
import threading
import time
import re
from collections import deque
from typing import List

import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

server = Server("router_mcp")

class RouterState:
    def __init__(self):
        self.monitor_sock: socket.socket | None = None
        self.buffer: deque[str] = deque(maxlen=1000)
        self.partial: bytes = b""
        self.lock: threading.Lock = threading.Lock()

state = RouterState()

def cleanup_monitor():
    if state.monitor_sock is not None:
        try:
            state.monitor_sock.close()
        except Exception:
            pass
        state.monitor_sock = None

def connect_monitor():
    cleanup_monitor()
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect("/tmp/tio.sock")
    sock.settimeout(0.5)
    state.monitor_sock = sock

def monitoring_thread():
    """Background thread that keeps a persistent connection and captures ALL console output."""
    while True:
        if state.monitor_sock is None:
            try:
                connect_monitor()
            except Exception:
                time.sleep(1)
                continue

        try:
            data = state.monitor_sock.recv(4096)
            if not data:
                raise OSError("EOF")

            with state.lock:
                state.partial += data
                while True:
                    # Find the earliest line ending (\n or \r)
                    min_idx = -1
                    for sep in (b"\n", b"\r"):
                        idx = state.partial.find(sep)
                        if idx != -1 and (min_idx == -1 or idx < min_idx):
                            min_idx = idx

                    if min_idx == -1:
                        break

                    line_b = state.partial[:min_idx]
                    state.partial = state.partial[min_idx + 1:]

                    decoded = line_b.decode(errors="ignore").rstrip("\r\n")
                    if decoded.strip():  # Skip empty lines
                        state.buffer.append(decoded)

        except socket.timeout:
            continue
        except Exception:
            cleanup_monitor()
            time.sleep(1)

@server.list_tools()
async def tools() -> List[Tool]:
    return [
        Tool(
            name="send_serial_command",
            description="Send a command to the router serial console and return its clean output.",
            inputSchema={
                "type": "object",
                "properties": {"cmd": {"type": "string"}},
                "required": ["cmd"]
            }
        ),
        Tool(
            name="get_console_transcript",
            description="Retrieve ONLY NEW console output since the last call to this tool (clears buffer afterward for efficiency). On first call, returns all accumulated output. Use this for low-token monitoring. Set full_history=true ONLY if you need the complete buffer without clearing (e.g., for full-session summary).",
            inputSchema={
                "type": "object",
                "properties": {
                    "max_lines": {
                        "type": "integer",
                        "description": "Maximum number of new lines to return (default: all new)."
                    },
                    "full_history": {
                        "type": "boolean",
                        "description": "ONLY set to true if you specifically need the full accumulated buffer without clearing it afterward. Avoid for repeated calls to save tokens."
                    }
                },
                "additionalProperties": False  # Optional: prevents junk params
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> List[TextContent]:
    if name == "send_serial_command":
        cmd = arguments["cmd"]

        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as cmd_sock:
            cmd_sock.settimeout(5.0)
            cmd_sock.connect("/tmp/tio.sock")
            cmd_sock.sendall(f"{cmd}\r\n".encode())

            output = b""
            prompt_seen = False
            start_time = time.time()
            max_duration = 15  # 15 second failsafe timeout
            
            # Common prompt patterns: [user@host dir]$ or [user@host dir]# or root@host:~# etc
            prompt_patterns = [
                r'\[.+@.+\]\s*[\$#]\s*$',     # [user@host:~]$ or [user@host:~/path]$
                r'.+@.+[:#~]\s*[\$#]\s*$',    # user@host:~$ or root@host#
                r'[\$#]\s*$'                   # Simple $ or # at end
            ]

            while True:
                try:
                    # Check failsafe timeout
                    if time.time() - start_time > max_duration:
                        break
                    
                    data = cmd_sock.recv(4096)
                    if not data:
                        break
                    output += data

                    # Check if we've seen a prompt
                    text_so_far = output.decode(errors="ignore")
                    lines = text_so_far.splitlines()
                    
                    if lines:
                        last_line = lines[-1].strip()
                        # Check against multiple prompt patterns
                        for pattern in prompt_patterns:
                            if re.search(pattern, last_line):
                                prompt_seen = True
                                cmd_sock.settimeout(0.3)  # Short timeout to catch any trailing data
                                break

                except socket.timeout:
                    if prompt_seen:
                        break
                    # If no prompt seen yet, continue with longer timeout
                    continue

            # Clean up the output
            text = output.decode(errors="ignore")
            # Strip ANSI color codes
            ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
            text = ansi_escape.sub('', text)
            lines = text.splitlines()

            # Remove command echo (usually first line)
            if lines and cmd.strip() in lines[0]:
                lines = lines[1:]

            # Remove trailing prompt line
            if lines:
                last_line = lines[-1].strip()
                for pattern in prompt_patterns:
                    if re.search(pattern, last_line):
                        lines = lines[:-1]
                        break

            cleaned_text = "\n".join(lines).strip()
            return [TextContent(type="text", text=cleaned_text or "No output")]

    elif name == "get_console_transcript":
        max_lines = arguments.get("max_lines")
        full_history = arguments.get("full_history", False)

        with state.lock:
            if max_lines is not None:
                recent_lines = list(state.buffer)[-max_lines:]
            else:
                recent_lines = list(state.buffer)

            transcript_lines = recent_lines[:]

            # Default: clear buffer after reading (delta mode — only new stuff next time)
            if not full_history:
                state.buffer.clear()

        if not transcript_lines:
            transcript = "No new console output since last check (or tio not running)."
        else:
            transcript = "\n".join(transcript_lines)

        return [TextContent(type="text", text=transcript)]

    else:
        raise ValueError(f"Unknown tool: {name}")

async def main():
    thread = threading.Thread(target=monitoring_thread, daemon=True)
    thread.start()

    async with stdio_server() as streams:
        try:
            await server.run(streams[0], streams[1], server.create_initialization_options())
        finally:
            cleanup_monitor()

if __name__ == "__main__":
    asyncio.run(main())
