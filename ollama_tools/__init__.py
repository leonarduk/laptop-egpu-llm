"""Fit-checked tooling around Ollama, for a laptop with an eGPU.

Cross-platform on purpose: the Windows driver diagnostics in
``diagnostics/`` stay PowerShell because Get-PnpDevice, pnputil and Device
Manager error codes have no Ubuntu equivalent, but everything here is HTTP,
JSON and nvidia-smi, which behave identically on both.

Standard library only, so it runs on a fresh box with nothing installed.
"""

__all__ = ["bench", "client", "envfile", "fit", "gpu"]
