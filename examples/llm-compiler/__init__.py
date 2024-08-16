"""
Initialize the LLMCompiler package.

This module imports main functions from the created modules and provides package metadata.
"""

from .planner import create_planner, stream_plan
from .scheduler import schedule_tasks
from .joiner import create_joiner
from .utils import get_pass

__all__ = [
    'create_planner',
    'stream_plan',
    'schedule_tasks',
    'create_joiner',
    'get_pass'
]

__version__ = '0.1.0'
__author__ = 'Your Name'
__email__ = 'your.email@example.com'
__description__ = 'A modular implementation of LLMCompiler for production use.'
