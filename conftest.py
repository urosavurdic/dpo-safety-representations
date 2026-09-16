"""Anchors the pytest rootdir at the repository root.

Without this, pytest infers the rootdir from wherever it was invoked, which
changes how `pythonpath` resolves and lets a test pass from one directory and
fail from another.
"""
