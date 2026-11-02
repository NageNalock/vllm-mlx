# SPDX-License-Identifier: Apache-2.0
"""Current process footprint from Darwin's proc_pid_rusage API."""

import ctypes
import os
import re
import resource
import subprocess


class _RUsageInfoV0(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64)
        for name in (
            "user_time",
            "system_time",
            "pkg_idle_wkups",
            "interrupt_wkups",
            "pageins",
            "wired_size",
            "resident_size",
            "phys_footprint",
            "proc_start_abstime",
            "proc_exit_abstime",
        )
    ]


_libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
_libproc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]


def physical_footprint():
    usage = _RUsageInfoV0()
    if _libproc.proc_pid_rusage(os.getpid(), 0, ctypes.byref(usage)):
        raise OSError(ctypes.get_errno(), "proc_pid_rusage failed")
    return usage.phys_footprint


def system_state():
    vm = subprocess.check_output(["vm_stat"], text=True)
    fields = {
        name: int(value)
        for name, value in re.findall(r"^([^:\n]+):\s+(\d+)\.", vm, re.M)
    }
    page_size = int(re.search(r"page size of (\d+) bytes", vm).group(1))
    status = subprocess.check_output(
        ["sysctl", "vm.swapusage", "kern.memorystatus_vm_pressure_level"], text=True
    )
    return {
        "swap_used_bytes": int(
            float(re.search(r"used = ([\d.]+)M", status).group(1)) * 2**20
        ),
        "swapin_bytes": fields["Swapins"] * page_size,
        "swapout_bytes": fields["Swapouts"] * page_size,
        "compressed_bytes": fields["Pages occupied by compressor"] * page_size,
        "pressure_level": int(status.strip().rsplit(":", 1)[1]),
        "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
