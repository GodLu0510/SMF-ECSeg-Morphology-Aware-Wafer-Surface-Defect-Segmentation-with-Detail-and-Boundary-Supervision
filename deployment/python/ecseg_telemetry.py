"""Additional read-only board/process telemetry, streamed to disk during inference."""
import glob
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import threading
import time

from telemetry import Monitor, read_text, summarize_samples
from ecseg_common import dump_json


def numeric_fields(text, separator=":"):
    result = {}
    for line in (text or "").splitlines():
        parts = line.split(separator, 1)
        if len(parts) == 2:
            value = parts[1].strip().split()
            if value and re.fullmatch(r"\d+", value[0]):
                result[parts[0].strip()] = int(value[0])
    return result


def board_snapshot(out, suffix):
    paths = ["/proc/device-tree/model", "/proc/device-tree/compatible", "/proc/cpuinfo", "/proc/version",
             "/proc/meminfo", "/proc/cmdline", "/sys/kernel/debug/rknpu/version", "/sys/kernel/debug/rknpu/mm",
             "/sys/kernel/debug/rknpu/driver_version", "/sys/kernel/debug/clk/clk_summary",
             "/sys/kernel/debug/regulator/regulator_summary", "/sys/kernel/debug/dma_buf/bufinfo",
             f"/proc/{os.getpid()}/smaps_rollup", f"/proc/{os.getpid()}/status"]
    for pattern in ("/sys/class/devfreq/*/governor", "/sys/class/devfreq/*/available_frequencies",
                    "/sys/class/devfreq/*/min_freq", "/sys/class/devfreq/*/max_freq",
                    "/sys/devices/system/cpu/cpufreq/policy*/scaling_governor",
                    "/sys/devices/system/cpu/cpufreq/policy*/scaling_min_freq",
                    "/sys/devices/system/cpu/cpufreq/policy*/scaling_max_freq",
                    "/sys/class/thermal/thermal_zone*/trip_point_*_temp"):
        paths.extend(sorted(glob.glob(pattern)))
    files = {}
    for path in paths:
        value = read_text(path)
        files[path] = {"available": value is not None,
                       "text": value.replace("\u0000", " ")[:2_000_000] if value else value}
    dump_json(Path(out)/f"board_{suffix}.json", {"platform": platform.platform(), "machine": platform.machine(),
              "cpu_count": os.cpu_count(), "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
              "files": files})
    if platform.system() == "Linux":
        try:
            result = subprocess.run(["dmesg", "--color=never"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
                                    errors="replace", timeout=5)
            dump_json(Path(out)/f"dmesg_{suffix}.json", {"returncode": result.returncode,
                       "stdout": result.stdout[-2_000_000:], "stderr": result.stderr})
        except (OSError, subprocess.TimeoutExpired) as exc:
            dump_json(Path(out)/f"dmesg_{suffix}.json", {"available": False, "error": str(exc)})


class ExtendedMonitor(Monitor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extra_previous = None
        for pattern, unit, scale in (("in*_input", "v", 0.001), ("curr*_input", "a", 0.001),
                                      ("fan*_input", "rpm", 1)):
            for path in sorted(glob.glob("/sys/class/hwmon/hwmon*/"+pattern)):
                p = Path(path)
                label = read_text(p.with_name(p.stem.replace("_input", "_label"))) or p.stem
                self.add_sensor(f"{unit}_{p.parent.name}_{label}", path, scale)
        self.load_paths = sorted(glob.glob("/sys/class/devfreq/*/load"))
        for path in sorted(glob.glob("/sys/class/thermal/cooling_device*/cur_state")):
            self.add_sensor(f"cooling_{Path(path).parent.name}_state", path, 1)
        for path in self.load_paths:
            self.sources[f"devfreq_load_{Path(path).parent.name}"] = {"path": path, "unit": "driver_specific_raw"}
        self.sources["extended_proc"] = {"paths": [f"/proc/{self.pid}/status", f"/proc/{self.pid}/io",
                  f"/proc/{self.pid}/stat", "/proc/vmstat", "/proc/loadavg", "/proc/pressure/cpu",
                  "/proc/pressure/memory", "/proc/pressure/io"], "available": platform.system() == "Linux"}

    def sample(self):
        sample = super().sample()
        now = time.monotonic()
        sample["timestamp_monotonic_s"] = now
        counters = numeric_fields(read_text(f"/proc/{self.pid}/io"))
        status = numeric_fields(read_text(f"/proc/{self.pid}/status"))
        for name in ("VmHWM", "VmPeak", "VmSize", "VmSwap"):
            sample[f"process_{name}_mb"] = status[name]/1024 if name in status else None
        sample["process_threads"] = status.get("Threads")
        for name in ("voluntary_ctxt_switches", "nonvoluntary_ctxt_switches"):
            counters[name] = status.get(name)
        stat = read_text(f"/proc/{self.pid}/stat")
        if stat:
            fields = stat[stat.rfind(")")+2:].split()
            if len(fields) > 36:
                counters.update({"minor_faults": int(fields[7]), "major_faults": int(fields[9])})
                sample["process_last_cpu"] = int(fields[36])
        for name in ("read_bytes", "write_bytes", "rchar", "wchar", "syscr", "syscw",
                     "voluntary_ctxt_switches", "nonvoluntary_ctxt_switches", "minor_faults", "major_faults"):
            value = counters.get(name)
            sample[f"process_{name}_total"] = value
            previous = self.extra_previous
            old = previous[1].get(name) if previous else None
            sample[f"process_{name}_per_s"] = ((value-old)/(now-previous[0])
                    if previous and value is not None and old is not None and now > previous[0] else None)
        self.extra_previous = now, counters
        memory = numeric_fields(read_text("/proc/meminfo"))
        for name in ("MemTotal", "MemFree", "Buffers", "Cached", "SwapTotal", "SwapFree", "CmaTotal", "CmaFree"):
            sample[f"system_{name}_mb"] = memory[name]/1024 if name in memory else None
        vmstat = numeric_fields(read_text("/proc/vmstat"), separator=None)
        for name in ("pgfault", "pgmajfault", "pswpin", "pswpout", "pgpgin", "pgpgout"):
            sample[f"system_{name}_total"] = vmstat.get(name)
        load = read_text("/proc/loadavg")
        for i, name in enumerate(("loadavg_1m", "loadavg_5m", "loadavg_15m")):
            sample[name] = float(load.split()[i]) if load else None
        for resource in ("cpu", "memory", "io"):
            pressure = read_text(f"/proc/pressure/{resource}")
            for line in (pressure or "").splitlines():
                tokens = line.split()
                for token in tokens[1:]:
                    key, value = token.split("=", 1)
                    sample[f"pressure_{resource}_{tokens[0]}_{key}"] = float(value)
        for path in self.load_paths:
            sample[f"devfreq_load_{Path(path).parent.name}_raw"] = read_text(path)
        return sample


class Sampler:
    def __init__(self, out, interval, npu_path=None, power_path=None, power_scale=1):
        self.out, self.interval, self.phase = Path(out), interval, "initialization"
        self.monitor = ExtendedMonitor(os.getpid(), npu_path, power_path, power_scale)
        self.samples, self.errors = [], []
        self.lock, self.stop_event = threading.Lock(), threading.Event()
        self.stream = (self.out/"hardware_samples.jsonl").open("w", encoding="utf-8")
        self.thread = threading.Thread(target=self.loop, name="ecseg-telemetry", daemon=True)

    def start(self):
        dump_json(self.out/"hardware_sources.json", self.monitor.sources)
        self.thread.start()

    def take(self):
        with self.lock:
            try:
                started = time.perf_counter()
                sample = self.monitor.sample()
                sample["sample_read_ms"] = (time.perf_counter()-started)*1000
                sample["phase"] = self.phase
                self.samples.append(sample)
                self.stream.write(json.dumps(sample, ensure_ascii=False, allow_nan=False)+"\n")
                self.stream.flush()
            except Exception as exc:
                self.errors.append({"time": time.time(), "error": repr(exc)})

    def loop(self):
        while not self.stop_event.is_set():
            start = time.monotonic()
            self.take()
            self.stop_event.wait(max(0.001, self.interval-(time.monotonic()-start)))

    def set_phase(self, phase):
        self.phase = phase
        self.take()

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            raise RuntimeError("Telemetry reader did not stop")
        self.stream.close()
        from benchmark import write_csv
        write_csv(self.out/"hardware_samples.csv", self.samples)
        dump_json(self.out/"telemetry_errors.json", self.errors)

    def summary(self, start, end):
        # Integrate with a monotonic clock so NTP/time corrections cannot change energy.
        samples = ([{**row, "timestamp_unix_s": row["timestamp_monotonic_s"]} for row in self.samples]
                   if end > start else [])
        for row in samples:
            row.pop("timestamp_monotonic_s", None)
            row.pop("phase", None)
            for name in list(row):
                if name.endswith("_raw") and name != "npu_load_raw":
                    row.pop(name)
        result = summarize_samples(samples, start, end)
        result["scope"] = "device-wide CPU/NPU/sensors; process metrics include Python, RKNN runtime and CPU decoder"
        result["time_basis"] = "monotonic"
        result["sampler_errors"] = len(self.errors)
        times = [s["timestamp_unix_s"] for s in samples if start <= s["timestamp_unix_s"] <= end]
        result["largest_sample_gap_s"] = max((b-a for a, b in zip(times, times[1:])), default=None)
        return result
