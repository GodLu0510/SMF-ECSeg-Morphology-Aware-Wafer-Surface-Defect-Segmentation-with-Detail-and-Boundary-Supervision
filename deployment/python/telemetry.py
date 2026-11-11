"""Read-only Linux telemetry. Unavailable sensors stay null, never estimated."""
import glob
import math
import os
from pathlib import Path
import re
import time


def read_text(path):
    try:
        return Path(path).read_text().strip()
    except (OSError, UnicodeError):
        return None


def read_number(path, scale=1):
    value = read_text(path)
    try:
        number = float(value) * scale
        return number if math.isfinite(number) else None
    except (ValueError, TypeError):
        return None


def parse_npu_load(text):
    if not text:
        return {}
    return {f"npu_core{core}_percent": float(value)
            for core, value in re.findall(r"Core\s*(\d+)\s*:\s*(\d+(?:\.\d+)?)\s*%", text, flags=re.I)}


class Monitor:
    def __init__(self, pid, npu_load_path=None, power_path=None, power_scale=1.0):
        self.pid = pid
        self.prev_cpu = {}
        self.prev_process = None
        self.cpu_count = os.cpu_count() or 1
        self.ticks = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
        candidates = [npu_load_path] if npu_load_path else ["/sys/kernel/debug/rknpu/load", "/sys/kernel/debug/rknpu0/load"]
        self.npu_path = next((p for p in candidates if read_text(p) is not None), candidates[0])
        self.sensors = {}
        self.sources = {
            "npu_load": {"path": self.npu_path, "available": bool(parse_npu_load(read_text(self.npu_path)))},
            "cpu_system": {"path": "/proc/stat", "available": read_text("/proc/stat") is not None},
            "cpu_process": {"path": f"/proc/{pid}/stat", "available": read_text(f"/proc/{pid}/stat") is not None},
            "memory": {"path": "/proc/meminfo", "available": read_text("/proc/meminfo") is not None}}
        for path in sorted(glob.glob("/sys/class/thermal/thermal_zone*/temp")):
            label = read_text(Path(path).with_name("type")) or "unknown"
            self.add_sensor(f"temperature_{Path(path).parent.name}_{label}_c", path, 0.001)
        for path in sorted(glob.glob("/sys/class/devfreq/*/cur_freq")):
            device = Path(path).parent.name.lower()
            if any(name in device for name in ("npu", "gpu", "dmc", "ddr")):
                self.add_sensor(f"frequency_{device}_mhz", path, 1e-6)
        for path in sorted(glob.glob("/sys/devices/system/cpu/cpufreq/policy*/scaling_cur_freq")):
            self.add_sensor(f"frequency_cpu_{Path(path).parent.name}_mhz", path, 0.001)
        # Some drivers expose only an averaged reading. Prefer the instantaneous
        # reading when both exist, keeping each physical channel only once.
        power_inputs = sorted(glob.glob("/sys/class/hwmon/hwmon*/power*_input"))
        power_averages = sorted(glob.glob("/sys/class/hwmon/hwmon*/power*_average"))
        input_channels = {path[:-len("_input")] for path in power_inputs}
        power_paths = power_inputs + [path for path in power_averages
                                     if path[:-len("_average")] not in input_channels]
        for path in power_paths:
            base = Path(path)
            name = read_text(base.parent / "name") or "unknown"
            channel, kind = base.name.rsplit("_", 1)
            label = read_text(base.with_name(channel + "_label")) or channel
            key = f"power_{base.parent.name}_{name}_{label}_w"
            self.add_sensor(key, path, 1e-6)
            self.sources[key]["reading_kind"] = "instantaneous" if kind == "input" else "driver_average"
        if power_path:
            self.add_sensor("power_external_w", str(Path(power_path).resolve()), power_scale)

    def add_sensor(self, name, path, scale):
        self.sensors[name] = (path, scale)
        self.sources[name] = {"path": path, "scale_to_report_unit": scale, "available": read_number(path, scale) is not None}

    def sample(self):
        now = time.monotonic()
        sample = {"timestamp_unix_s": time.time(), "cpu_system_percent": None,
                  "cpu_process_percent": None, "cpu_process_machine_percent": None, "process_rss_mb": None}
        stat = read_text("/proc/stat")
        if stat:
            for line in stat.splitlines():
                fields = line.split()
                if not fields or not re.fullmatch(r"cpu\d*", fields[0]):
                    continue
                values = list(map(int, fields[1:9]))
                total, idle = sum(values), values[3]+values[4]
                previous = self.prev_cpu.get(fields[0])
                name = "cpu_system_percent" if fields[0] == "cpu" else f"{fields[0]}_percent"
                if previous and total > previous[0]:
                    sample[name] = max(0.0, min(100.0, 100*(1-(idle-previous[1])/(total-previous[0]))))
                self.prev_cpu[fields[0]] = (total, idle)
        process = read_text(f"/proc/{self.pid}/stat")
        if process:
            fields = process[process.rfind(")")+2:].split()
            if len(fields) > 21:
                ticks = int(fields[11]) + int(fields[12])
                sample["process_rss_mb"] = int(fields[21])*os.sysconf("SC_PAGE_SIZE")/(1024*1024)
                if self.prev_process and now > self.prev_process[0]:
                    percent = max(0.0, (ticks-self.prev_process[1])/self.ticks/(now-self.prev_process[0])*100)
                    sample["cpu_process_percent"] = percent
                    sample["cpu_process_machine_percent"] = percent/self.cpu_count
                self.prev_process = (now, ticks)
        memory = read_text("/proc/meminfo")
        if memory:
            match = re.search(r"MemAvailable:\s+(\d+)", memory)
            sample["system_memory_available_mb"] = int(match[1])/1024 if match else None
        npu_text = read_text(self.npu_path)
        sample["npu_load_raw"] = npu_text
        for core in range(3):
            sample[f"npu_core{core}_percent"] = None
        sample.update(parse_npu_load(npu_text))
        for name, (path, scale) in self.sensors.items():
            sample[name] = read_number(path, scale)
        return sample


def summarize_samples(samples, start, end):
    selected = [s for s in samples if start <= s["timestamp_unix_s"] <= end]
    names = sorted({key for sample in selected for key in sample
                    if key not in ("timestamp_unix_s", "npu_load_raw")})
    stats = {}
    for name in names:
        values = [s[name] for s in selected if isinstance(s.get(name), (float, int))]
        stats[name] = {"mean": sum(values)/len(values) if values else None,
                       "min": min(values) if values else None, "max": max(values) if values else None,
                       "samples": len(values)}
        if name.startswith("power_") and name.endswith("_w"):
            energy, duration = 0.0, 0.0
            for a, b in zip(selected, selected[1:]):
                if a.get(name) is not None and b.get(name) is not None:
                    dt = b["timestamp_unix_s"]-a["timestamp_unix_s"]
                    energy += (a[name]+b[name])*0.5*dt
                    duration += dt
            stats[name]["observed_energy_j"] = energy if duration else None
            stats[name]["integration_coverage_s"] = duration
    return {"samples_in_measurement": len(selected), "measurement_duration_s": end-start,
            "statistics": stats, "scope": "device-wide CPU/NPU/sensors; process CPU/RSS refer to C++ runner"}
