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
            description="Send a command to the router serial console and return its clean output. Optionally show prompt lines.",
            inputSchema={
                "type": "object",
                "properties": {
                    "cmd": {"type": "string"},
                    "show_prompt": {
                        "type": "boolean",
                        "description": "If true, show prompt lines at end. Default: false (clean view)."
                    }
                },
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
        show_prompt = arguments.get("show_prompt", False)

        reader, writer = await asyncio.open_unix_connection("/tmp/tio.sock")

        writer.write(f"{cmd}\r\n".encode())
        await writer.drain()

        output = bytearray()
        start_time = time.time()
        max_total = 5.0
        idle_timeout = 0.35

        last_data_time = start_time

        try:
            while time.time() - start_time < max_total:
                try:
                    data = await asyncio.wait_for(reader.read(8192), timeout=0.2)
                    if not data:
                        break
                    output.extend(data)
                    last_data_time = time.time()
                except asyncio.TimeoutError:
                    if time.time() - last_data_time > idle_timeout:
                        break
                    continue
        finally:
            # Quick final drain
            try:
                while True:
                    more = await asyncio.wait_for(reader.read(8192), timeout=0.1)
                    if not more:
                        break
                    output.extend(more)
            except asyncio.TimeoutError:
                pass

            writer.close()
            await writer.wait_closed()

        text = output.decode(errors="ignore")
        # Remove ANSI escapes
        text = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', text)
        # Remove common terminal noise
        text = re.sub(r'\[ESC\].*?\[', '', text, flags=re.DOTALL)
        # Remove lines like 'command output shows...'
        text = re.sub(r'^\s*command output shows.*?$', '', text, flags=re.MULTILINE | re.IGNORECASE)

        lines = [line.strip() for line in text.splitlines() if line.strip() and not line.startswith('[')]
        # Remove command echo (usually first line)
        if lines and cmd.strip() in lines[0]:
            lines = lines[1:]

        if not show_prompt:
            # Remove prompt lines
            prompt_re = re.compile(r'\[.+?@.+?:.+?\].*?[\$#] ?')
            lines = [line for line in lines if not prompt_re.match(line)]

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
