"""ORT-specific labels on the same TX2 hardware telemetry implementation."""
from tx2_telemetry import Sampler as BaseSampler,board_snapshot
class Sampler(BaseSampler):
    def summary(self,start,end):
        result=super().summary(start,end)
        result["scope"]="TX2 device-wide CPU/GPU/temperature/power rails; RSS includes Python/ONNX Runtime; unified-memory accounting may overlap"
        return result
