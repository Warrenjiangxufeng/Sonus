from __future__ import annotations

import json
import sys


TOOL = {
    "name": "echo",
    "description": "Echo text back.",
    "inputSchema": {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
}


for line in sys.stdin:
    try:
        request = json.loads(line)
    except json.JSONDecodeError:
        continue
    method = str(request.get("method") or "")
    if method == "notifications/initialized":
        continue
    request_id = request.get("id")
    if method == "initialize":
        result = {
            "protocolVersion": "2025-06-18",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "fake", "version": "1"},
        }
    elif method == "tools/list":
        result = {"tools": [TOOL]}
    elif method == "tools/call":
        arguments = dict((request.get("params") or {}).get("arguments") or {})
        result = {
            "content": [{"type": "text", "text": str(arguments.get("text") or "")}],
            "isError": False,
        }
    else:
        result = {}
    print(
        json.dumps(
            {"jsonrpc": "2.0", "id": request_id, "result": result},
            ensure_ascii=False,
        ),
        flush=True,
    )
