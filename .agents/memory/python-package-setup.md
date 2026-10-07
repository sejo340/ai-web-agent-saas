---
name: Python package setup
description: Environment-specific guidance for installing Python dependencies in this workspace.
---

Use a full Python tools module such as `python-3.11` before installing packages. The base Python module can lack pip and reject installs because the interpreter is externally managed.

**Why:** The system Python is immutable, while the full tools module provides the project-local package environment used by workflows.

**How to apply:** Check available Python modules, install a full tools module, then install project dependencies through the package-management integration. Playwright also needs its browser binaries and native runtime libraries installed before Chromium can launch.