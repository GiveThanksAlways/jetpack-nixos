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
        # Persistent command connection for send_serial_command
        self.cmd_reader: asyncio.StreamReader | None = None
        self.cmd_writer: asyncio.StreamWriter | None = None
        self.cmd_lock: asyncio.Lock = asyncio.Lock()

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

async def get_cmd_connection():
    """Get or create a persistent connection for sending commands."""
    if state.cmd_reader is None or state.cmd_writer is None or state.cmd_writer.is_closing():
        try:
            state.cmd_reader, state.cmd_writer = await asyncio.open_unix_connection("/tmp/tio.sock")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to /tmp/tio.sock: {e}")
    return state.cmd_reader, state.cmd_writer

@server.list_tools()
async def tools() -> List[Tool]:
    return [
        Tool(
            name="send_serial_command",
            description="Send a command to the router serial console and return its clean output. Fast, main command for typical shell usage.",
            inputSchema={
                "type": "object",
                "properties": {
                    "cmd": {"type": "string"}
                },
                "required": ["cmd"]
            }
        ),
        Tool(
            name="send_serial_command_long",
            description="Send a command to the router serial console with extended timeout for edge cases or long-running commands.",
            inputSchema={
                "type": "object",
                "properties": {
                    "cmd": {"type": "string"}
                },
                "required": ["cmd"]
            }
        ),
        Tool(
            name="get_console_transcript",
            description="Retrieve ONLY NEW console output since the last call to this tool (clears buffer afterward for efficiency). On first call, returns all accumulated output. Use this for low-token monitoring. Set full_history=true ONLY if you need the complete buffer without clearing it afterward. Avoid for repeated calls to save tokens.",
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

    def _prompt_regex():
        return re.compile(r"^\[.*?@.*?:.*?\]\\$ ?", re.MULTILINE)

    async def _send_serial_command(cmd, max_total):
        async with state.cmd_lock:
            try:
                reader, writer = await get_cmd_connection()
            except RuntimeError as e:
                return [TextContent(type="text", text=f"Connection error: {e}")]

            # Clear buffer
            try:
                while True:
                    data = await asyncio.wait_for(reader.read(8192), timeout=0.3)
                    if not data:
                        break
            except asyncio.TimeoutError:
                pass

            writer.write(f"{cmd}\n".encode())
            await writer.drain()

            output = bytearray()
            start_time = time.time()
            prompt_regex = _prompt_regex()
            found_prompt = False
            last_data_time = start_time

            async def read_loop():
                nonlocal found_prompt, last_data_time
                while time.time() - start_time < max_total:
                    try:
                        data = await asyncio.wait_for(reader.read(8192), timeout=0.05)
                        if not data:
                            break
                        output.extend(data)
                        last_data_time = time.time()
                        decoded = output.decode(errors="ignore")
                        if prompt_regex.search(decoded):
                            found_prompt = True
                            break
                    except asyncio.TimeoutError:
                        if time.time() - last_data_time > 0.3:
                            break
                        continue

            await asyncio.gather(read_loop())

            # Drain remaining data
            try:
                while True:
                    more = await asyncio.wait_for(reader.read(8192), timeout=0.05)
                    if not more:
                        break
                    output.extend(more)
            except asyncio.TimeoutError:
                pass

        text = output.decode(errors="ignore")
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        text = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', text)
        text = text.replace('\u0007', '')

        # Remove prompt lines
        prompt_re = re.compile(r'^.*\[.*?@.*?:.*?\]\\$ ?.*$', re.MULTILINE)
        text = prompt_re.sub('', text)

        # Remove command echoes
        cmds = [c.strip() for c in cmd.split(';')]
        cleaned_lines = []
        for line in text.split('\n'):
            l = line.strip()
            if not l:
                continue
            if any(l == c for c in cmds):
                continue
            if re.match(r'^.*\[.*?@.*?:.*?\]\\$ ?.*$', l):
                continue
            if re.match(r'^[\u0007]+$', l):
                continue
            cleaned_lines.append(line.rstrip())

        while cleaned_lines and cleaned_lines[-1].strip() == '':
            cleaned_lines.pop()

        cleaned_text = "\n".join(cleaned_lines)
        return [TextContent(type="text", text=cleaned_text or "No output")]

    if name == "send_serial_command":
        return await _send_serial_command(arguments["cmd"], max_total=2.0)  # Fast, main command
    elif name == "send_serial_command_long":
        return await _send_serial_command(arguments["cmd"], max_total=15.0)  # Extended timeout for edge cases

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
            # Close persistent command connection
            if state.cmd_writer is not None and not state.cmd_writer.is_closing():
                state.cmd_writer.close()
                try:
                    await state.cmd_writer.wait_closed()
                except Exception:
                    pass

if __name__ == "__main__":
    asyncio.run(main())
