"""Compare production tanh GELU against native PyTorch over every finite FP16 value.

Build the library with build_vision_fa2_probe.sh. Optionally replay captured
ViT linear_fc1 outputs against linear_fc2 inputs (the native GELU boundary).
"""
import argparse
import ctypes as C


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True)
    parser.add_argument("--capture", help="Optional torch.save mapping of native linear inputs/outputs")
    args = parser.parse_args()

    class Exec(C.Structure):
        _fields_ = [("stream", C.c_uint64), ("scratch", C.c_uint64), ("size", C.c_size_t)]

    lib = C.CDLL(args.library)
    fn = lib.w34_vision_bias_gelu_f16
    fn.argtypes = [C.POINTER(Exec)] + [C.c_void_p] * 3 + [C.c_int] * 3
    fn.restype = C.c_int
    bits = torch.arange(65536, device="cuda").to(torch.uint16)
    values = bits.view(torch.float16)
    values = values[torch.isfinite(values)].contiguous()
    samples = {"all finite FP16": (values, torch.nn.functional.gelu(values, approximate="tanh"))}
    if args.capture:
        capture = torch.load(args.capture, map_location="cpu", weights_only=True)
        for key, tensors in capture.items():
            if key.endswith(".linear_fc1"):
                after = capture[key.replace(".linear_fc1", ".linear_fc2")]["input"]
                samples[key] = (tensors["output"].cuda().contiguous(), after.cuda().contiguous())
    for name, (x, expected) in samples.items():
        assert x.shape == expected.shape
        assert x.dtype == expected.dtype == torch.float16
        output = torch.empty_like(x)
        ex = Exec(torch.cuda.current_stream().cuda_stream, 0, 0)
        rc = fn(C.byref(ex), x.data_ptr(), None, output.data_ptr(), 1, x.numel(), 1)
        assert rc == 0, (name, rc)
        torch.cuda.synchronize()
        different = int((output.view(torch.int16) != expected.view(torch.int16)).sum())
        print(f"{name}: {different}/{output.numel()} differing half bits", flush=True)
        assert different == 0, name


if __name__ == "__main__":
    main()
