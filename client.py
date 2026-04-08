"""
Spectrum Orchestration Environment Client.

Connects to a running Spectrum Orchestration server (local Docker or HF Space).

Example:
    >>> from client import SpectrumEnv
    >>> with SpectrumEnv(base_url="http://localhost:8000") as env:
    ...     env.reset()
    ...     tools = env.list_tools()
    ...     print([t.name for t in tools])
    ...     # ['assign_spectrum', 'get_state', 'get_score']
    ...
    ...     # Take a random step
    ...     import random
    ...     N = 10  # easy difficulty
    ...     channels = [random.random() for _ in range(N)]
    ...     powers   = [random.random() for _ in range(N)]
    ...     result = env.call_tool("assign_spectrum", channels=channels, powers=powers)
    ...     print(result)

    >>> # Or connect directly to HF Space:
    >>> env = SpectrumEnv(base_url="https://<your-hf-space>.hf.space")
"""

from openenv.core.mcp_client import MCPToolClient


class SpectrumEnv(MCPToolClient):
    """
    Client for the Spectrum Orchestration Environment.

    Inherits all MCPToolClient functionality:
      - list_tools()            — discover available tools
      - call_tool(name, **kw)   — call assign_spectrum / get_state / get_score
      - reset(**kwargs)         — reset the episode
    """
    pass
