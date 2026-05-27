"""
path_setup.py

Centralized sys.path bootstrap for LlamaGuard.
Import this module first in any file that needs cross-package imports.
"""
import os
import sys

_workflow_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_workflow_dir)

for _p in (_project_dir, os.path.join(_project_dir, 'llama-model')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
