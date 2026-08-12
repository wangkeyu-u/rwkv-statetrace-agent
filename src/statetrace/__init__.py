"""RWKV StateTrace Agent.

The package keeps the model backend, agent controller, tools, persistence, and
reporting layers separate so that live RWKV inference and deterministic replay
can share the same auditable execution loop.
"""

__version__ = "0.1.0"
