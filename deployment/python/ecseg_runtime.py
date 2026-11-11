"""RKNN C API binding for Linux aarch64, matching bundled include/rknn_api.h.

Uses the system/explicit runtime, synchronously copies and releases all outputs.
Performance queries are optional and retain unsupported query return codes.
"""
import ctypes as C
from pathlib import Path
import platform
import time

import numpy as np

from ecseg_common import sha256


class TensorAttr(C.Structure):
    _fields_ = [("index", C.c_uint32), ("n_dims", C.c_uint32), ("dims", C.c_uint32*16),
                ("name", C.c_char*256), ("n_elems", C.c_uint32), ("size", C.c_uint32),
                ("fmt", C.c_int), ("type", C.c_int), ("qnt_type", C.c_int), ("fl", C.c_int8),
                ("zp", C.c_int32), ("scale", C.c_float), ("w_stride", C.c_uint32),
                ("size_with_stride", C.c_uint32), ("pass_through", C.c_uint8), ("h_stride", C.c_uint32)]


class Input(C.Structure):
    _fields_ = [("index", C.c_uint32), ("buf", C.c_void_p), ("size", C.c_uint32),
                ("pass_through", C.c_uint8), ("type", C.c_int), ("fmt", C.c_int)]


class Output(C.Structure):
    _fields_ = [("want_float", C.c_uint8), ("is_prealloc", C.c_uint8), ("index", C.c_uint32),
                ("buf", C.c_void_p), ("size", C.c_uint32)]


class IONum(C.Structure):
    _fields_ = [("n_input", C.c_uint32), ("n_output", C.c_uint32)]


class Version(C.Structure):
    _fields_ = [("api_version", C.c_char*256), ("drv_version", C.c_char*256)]


class Memory(C.Structure):
    _fields_ = [("total_weight_size", C.c_uint32), ("total_internal_size", C.c_uint32),
                ("total_dma_allocated_size", C.c_uint64), ("total_sram_size", C.c_uint32),
                ("free_sram_size", C.c_uint32), ("reserved", C.c_uint32*10)]


class PerfDetail(C.Structure):
    _fields_ = [("perf_data", C.c_void_p), ("data_len", C.c_uint64)]


def attr_dict(attr):
    return {"index": attr.index, "name": attr.name.decode(errors="replace"),
            "shape": list(attr.dims[:attr.n_dims]), "elements": attr.n_elems,
            "size_bytes": attr.size, "format": attr.fmt, "type": attr.type,
            "quantization_type": attr.qnt_type, "zero_point": attr.zp, "scale": attr.scale,
            "width_stride": attr.w_stride, "height_stride": attr.h_stride,
            "size_with_stride": attr.size_with_stride}


def prepare_input(rgb, normalized, mode, native=None):
    """Use NHWC for conversion; only native mode owns normalization/quantization."""
    rgb = np.asarray(rgb)
    if rgb.ndim != 4 or rgb.shape[0] != 1 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Expected UINT8 RGB input with shape [1,H,W,3]")
    if mode == "uint8":
        return np.ascontiguousarray(rgb), 3, 1, 0
    if mode == "raw-float32":
        # Runtime conversion owns the compiler's mean/std; do not normalize twice.
        return np.ascontiguousarray(rgb, dtype=np.float32), 0, 1, 0
    if mode == "normalized-float32":
        raise ValueError("normalized-float32 was retired: its NCHW conversion path is invalid on this runtime; "
                         "use raw-float32 or native-normalized for diagnosis")
    if mode != "native-normalized":
        raise ValueError("Unknown RKNN input mode: " + str(mode))
    if normalized is None:
        raise ValueError("Native diagnostic mode requires ImageNet-normalized NCHW input")
    normalized = np.asarray(normalized, dtype=np.float32)
    if normalized.shape != (1, 3, rgb.shape[1], rgb.shape[2]) or not np.isfinite(normalized).all():
        raise ValueError("Invalid shape/non-finite normalized input")
    if native is None:
        raise ValueError("Native NHWC input attributes are unavailable; cannot safely bypass preprocessing")
    if (native.fmt != 1 or list(native.dims[:native.n_dims]) != list(rgb.shape)
            or native.n_elems != rgb.size or native.w_stride not in (0, rgb.shape[2])
            or native.h_stride not in (0, rgb.shape[1])):
        raise ValueError("Native diagnostic supports only contiguous, unpadded RGB NHWC input")
    data = np.ascontiguousarray(normalized.transpose(0, 2, 3, 1))
    if native.type in (0, 1):
        data = data.astype(np.float32 if native.type == 0 else np.float16)
    elif native.type == 2 and native.qnt_type == 2 and np.isfinite(native.scale) and native.scale > 0:
        data = np.clip(np.rint(data / native.scale + native.zp), -128, 127).astype(np.int8)
    else:
        raise ValueError("Unsupported native input type/quantization; refusing an assumed conversion")
    if native.size != data.nbytes or native.size_with_stride not in (0, data.nbytes) or not np.isfinite(data).all():
        raise ValueError("Native input byte size/stride/numeric values do not match the tensor contract")
    return np.ascontiguousarray(data), native.type, 1, 1


def reshape_output(flat, attr, expected):
    shape = list(attr.dims[:attr.n_dims])
    if attr.n_dims == 4 and attr.fmt == 1:
        # RKNN_QUERY_OUTPUT_ATTR describes logical NHWC layout in this case.
        value = flat.reshape(shape).transpose(0, 3, 1, 2)
    else:
        value = flat.reshape(shape)
    if list(value.shape) != list(expected):
        raise ValueError(f"RKNN output shape {list(value.shape)} differs from export {expected}")
    return np.ascontiguousarray(value)


class RKNNRuntime:
    def __init__(self, model, expected_shapes, expected_names, size, library="librknnrt.so", core_mask=0, profile=False):
        if platform.system() != "Linux" or platform.machine().lower() not in ("aarch64", "arm64"):
            raise RuntimeError("RKNN backend requires RK3588 Linux aarch64; use --backend onnx for PC validation")
        self.library, self.profile = library, profile
        self.lib = C.CDLL(library)
        self.ctx = C.c_uint64(0)
        ctx_type = C.c_uint64
        signatures = {
            "rknn_init": [C.POINTER(ctx_type), C.c_void_p, C.c_uint32, C.c_uint32, C.c_void_p],
            "rknn_query": [ctx_type, C.c_int, C.c_void_p, C.c_uint32],
            "rknn_set_core_mask": [ctx_type, C.c_int],
            "rknn_inputs_set": [ctx_type, C.c_uint32, C.POINTER(Input)],
            "rknn_run": [ctx_type, C.c_void_p],
            "rknn_outputs_get": [ctx_type, C.c_uint32, C.POINTER(Output), C.c_void_p],
            "rknn_outputs_release": [ctx_type, C.c_uint32, C.POINTER(Output)],
            "rknn_destroy": [ctx_type]}
        for name, parameters in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = parameters, C.c_int
        self.expected_shapes, self.query_errors = expected_shapes, {}
        try:
            path = C.create_string_buffer(str(Path(model).resolve()).encode())
            self.check("init", self.lib.rknn_init(C.byref(self.ctx), path, 0, 8 if profile else 0, None))
            self.check("core_mask", self.lib.rknn_set_core_mask(self.ctx, core_mask))
            counts = self.query(0, IONum())
            if counts.n_input != 1 or counts.n_output != len(expected_shapes):
                raise ValueError(f"RKNN I/O mismatch: {counts.n_input} inputs, {counts.n_output} outputs")
            self.input_attr = self.query(1, TensorAttr(index=0))
            shape = list(self.input_attr.dims[:self.input_attr.n_dims])
            expected_input = [1, size[0], size[1], 3] if self.input_attr.fmt == 1 else [1, 3, *size]
            if shape != expected_input:
                raise ValueError(f"RKNN input shape {shape} differs from manifest {expected_input}")
            attrs = [self.query(2, TensorAttr(index=i)) for i in range(counts.n_output)]
            # Resolve by tensor names, rejecting silent P3/P4/P5 or logits/boxes swaps.
            names = [a.name.decode(errors="replace") for a in attrs]
            if set(names) != set(expected_names):
                raise ValueError(f"RKNN output names {names} differ from export {expected_names}")
            self.attrs = [attrs[names.index(name)] for name in expected_names]
            version = self.query(5, Version(), optional=True)
            memory = self.query(6, Memory(), optional=True)
            self.metadata = {"library": library, "profile_enabled": profile, "core_mask": core_mask,
                             "input": attr_dict(self.input_attr), "outputs": [attr_dict(a) for a in self.attrs],
                             "version": {"api": version.api_version.decode(errors="replace"),
                                         "driver": version.drv_version.decode(errors="replace")} if version else None,
                             "memory_bytes": {name: getattr(memory, name) for name, _ in Memory._fields_
                                              if name != "reserved"} if memory else None,
                             "query_errors": self.query_errors}
            loaded = Path(library).resolve()
            if not loaded.is_file():
                try:
                    mapped = Path("/proc/self/maps").read_text().splitlines()
                except OSError:
                    mapped = []
                matches = [Path(line.split(None, 5)[-1]) for line in mapped if "librknnrt" in line and len(line.split(None, 5)) == 6]
                loaded = next((path for path in matches if path.is_file()), None)
            self.metadata["loaded_library"] = str(loaded) if loaded else None
            self.metadata["library_sha256"] = sha256(loaded) if loaded else None
        except BaseException:
            self.close()
            raise

    @staticmethod
    def check(stage, code):
        if code != 0:
            raise RuntimeError(f"RKNN {stage} failed: {code}")

    def query(self, command, value, optional=False):
        code = self.lib.rknn_query(self.ctx, command, C.byref(value), C.sizeof(value))
        if code != 0 and optional:
            self.query_errors[str(command)] = code
            return None
        self.check(f"query({command})", code)
        return value

    def infer(self, rgb, normalized=None, input_mode="uint8"):
        native = None
        if input_mode == "native-normalized":
            if not hasattr(self, "native_input_attr"):
                self.native_input_attr = self.query(10, TensorAttr(index=0), optional=True)
                self.metadata["native_nhwc_input"] = attr_dict(self.native_input_attr) if self.native_input_attr else None
            native = self.native_input_attr
        data, tensor_type, tensor_format, pass_through = prepare_input(rgb, normalized, input_mode, native)
        self.metadata["submitted_input"] = {"mode": input_mode, "shape": list(data.shape), "dtype": str(data.dtype),
                    "format": tensor_format, "type": tensor_type, "pass_through": pass_through,
                    "size_bytes": data.nbytes, "min": float(data.min()), "max": float(data.max())}
        inputs = (Input*1)(Input(index=0, buf=data.ctypes.data, size=data.nbytes, pass_through=pass_through,
                                type=tensor_type, fmt=tensor_format))
        start = time.perf_counter()
        self.check("inputs_set", self.lib.rknn_inputs_set(self.ctx, 1, inputs))
        submitted = time.perf_counter()
        self.check("run", self.lib.rknn_run(self.ctx, None))
        ran = time.perf_counter()
        outputs = (Output*len(self.attrs))(*[Output(want_float=1, index=a.index) for a in self.attrs])
        self.check("outputs_get", self.lib.rknn_outputs_get(self.ctx, len(outputs), outputs, None))
        received = time.perf_counter()
        try:
            values = []
            for output, attr, expected in zip(outputs, self.attrs, self.expected_shapes):
                if not output.buf or output.size != attr.n_elems*4:
                    raise RuntimeError("RKNN output float buffer has unexpected byte length")
                flat = np.ctypeslib.as_array(C.cast(output.buf, C.POINTER(C.c_float)), shape=(attr.n_elems,)).copy()
                values.append(reshape_output(flat, attr, expected))
        finally:
            self.check("outputs_release", self.lib.rknn_outputs_release(self.ctx, len(outputs), outputs))
        released = time.perf_counter()
        timing = {"input_ms": (submitted-start)*1000, "rknn_run_ms": (ran-submitted)*1000,
                  "output_get_ms": (received-ran)*1000, "output_copy_release_ms": (released-received)*1000,
                  "rknn_sync_ms": (received-submitted)*1000, "rknn_total_ms": (released-start)*1000}
        if self.profile:
            perf = self.query(4, C.c_int64(), optional=True)
            timing["driver_run_ms"] = perf.value/1000 if perf is not None else None
        return values, timing

    def perf_detail(self):
        if not self.profile:
            return None
        detail = self.query(3, PerfDetail(), optional=True)
        if detail is None or not detail.perf_data:
            return None
        return C.string_at(detail.perf_data, detail.data_len).decode(errors="replace")

    def memory_snapshot(self):
        memory = self.query(6, Memory(), optional=True)
        return {name: getattr(memory, name) for name, _ in Memory._fields_ if name != "reserved"} if memory else None

    def close(self):
        if self.ctx.value:
            self.lib.rknn_destroy(self.ctx)
            self.ctx.value = 0
