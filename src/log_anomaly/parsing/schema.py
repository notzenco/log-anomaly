"""The normalised event schema every parser produces.

Keeping one schema is what lets the feature extractor treat a Linux syslog
file, an OpenSSH server log and a proxy log the same way.
"""

from __future__ import annotations

import pandas as pd

#: Event taxonomy. ``other`` covers everything that is not security-relevant
#: enough to model but is still worth keeping for context.
EVENT_TYPES: tuple[str, ...] = (
    "auth_failure",
    "auth_success",
    "session_open",
    "session_close",
    "sudo_failure",
    "sudo_success",
    "process_exec",
    "network_conn",
    "other",
)

#: Column name -> pandas dtype for the normalised event frame.
EVENT_DTYPES: dict[str, str] = {
    "event_index": "int64",
    "ts": "datetime64[ns, UTC]",
    "host": "string",
    "program": "string",
    "pid": "Int64",
    "event_type": "string",
    "user": "string",
    "src_ip": "string",
    "src_port": "Int64",
    "dst_ip": "string",
    "dst_port": "Int64",
    "parent_process": "string",
    "process": "string",
    "bytes_sent": "Int64",
    "bytes_received": "Int64",
    "duration_s": "Float64",
    "message": "string",
    "raw": "string",
}

#: Columns that carry security-relevant semantics, in a stable order.
FEATURE_INPUT_COLUMNS: tuple[str, ...] = (
    "event_index",
    "ts",
    "host",
    "event_type",
    "user",
    "src_ip",
    "dst_ip",
    "process",
    "parent_process",
)


def empty_events() -> pd.DataFrame:
    """Return an empty event frame with the canonical dtypes."""
    frame = pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in EVENT_DTYPES.items()})
    return frame
