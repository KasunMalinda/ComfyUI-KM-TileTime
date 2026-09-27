"""Run the nodes' tensor work on every CPU core.

Environments that set OMP_NUM_THREADS=1 leave torch on a single CPU thread,
which made a 4K merge about 5x slower. The nodes raise the count only while
they run and restore it afterwards, so nothing else in ComfyUI changes.
"""

import os
from contextlib import contextmanager

import torch


@contextmanager
def all_cpu_threads():
    before = torch.get_num_threads()
    want = os.cpu_count() or before
    if want > before:
        torch.set_num_threads(want)
    try:
        yield
    finally:
        if torch.get_num_threads() != before:
            torch.set_num_threads(before)
