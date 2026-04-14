import os
import shutil
import sys


PASS = "[OK]"
FAIL = "[X]"
WARN = "[!]"


def _ok(msg: str) -> None:
    print(f"  {PASS} {msg}")


def _fail(msg: str) -> None:
    print(f"  {FAIL} {msg}")


def _warn(msg: str) -> None:
    print(f"  {WARN} {msg}")


def run_preflight() -> bool:
    all_ok = True
    print("=" * 55)
    print("  LiteLDM - Environment Preflight Check")
    print("=" * 55)

    print("\n[1] Python")
    vi = sys.version_info
    if vi.major == 3 and vi.minor >= 9:
        _ok(f"Python {vi.major}.{vi.minor}.{vi.micro}")
    else:
        _fail(f"Python {vi.major}.{vi.minor} - need 3.9+")
        all_ok = False

    print("\n[2] GPU / CUDA")
    try:
        import torch

        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            _ok(f"CUDA available - {gpu} ({vram:.1f} GB VRAM)")
            if vram < 8:
                _warn("Less than 8 GB VRAM - consider reducing batch size")
            elif vram < 12:
                _warn("8-12 GB VRAM - reduce VAE batch size if OOM")
            _ok(f"CUDA version: {torch.version.cuda}")
        else:
            _fail("CUDA not available - training will be slow on CPU")
            all_ok = False
    except ImportError:
        _fail("PyTorch not installed")
        all_ok = False

    print("\n[3] LOCAL storage")
    local = os.environ.get("LOCAL", "")
    if local:
        _ok(f"LOCAL is set: {local}")
        free = shutil.disk_usage(local).free / 1e9
        if free >= 10:
            _ok(f"Free space on LOCAL: {free:.1f} GB")
        else:
            _warn(f"Only {free:.1f} GB free on LOCAL")
    else:
        _fail("LOCAL is not set")
        _warn("Falling back to ./local_storage for local dev")

    print("\n[4] HuggingFace authentication")
    try:
        from huggingface_hub import HfApi, utils as hf_utils

        try:
            user = HfApi().whoami()
            _ok(f"Logged in as: {user['name']}")
        except hf_utils.HfHubHTTPError:
            _fail("Not authenticated - set token in env or run huggingface-cli login")
            all_ok = False
        except Exception as exc:
            _fail(f"Auth error: {exc}")
            all_ok = False
    except ImportError:
        _fail("huggingface_hub not installed")
        all_ok = False

    print("\n[5] Dataset repo access")
    try:
        from huggingface_hub import HfApi

        HfApi().dataset_info("deeplearningresearchproject/dataset_project")
        _ok("Access confirmed: deeplearningresearchproject/dataset_project")
    except Exception as exc:
        _fail(f"Cannot access dataset repo: {exc}")
        all_ok = False

    print("\n[6] Current directory write access")
    cwd = os.getcwd()
    test_file = os.path.join(cwd, ".write_test")
    try:
        with open(test_file, "w", encoding="utf-8") as handle:
            handle.write("ok")
        os.remove(test_file)
        free_cwd = shutil.disk_usage(cwd).free / 1e9
        _ok(f"Writable: {cwd} ({free_cwd:.1f} GB free)")
    except Exception as exc:
        _fail(f"Cannot write to current dir: {exc}")
        all_ok = False

    print("\n" + "=" * 55)
    if all_ok:
        print("  All checks passed - ready to run")
    else:
        print("  Fix the issues above before proceeding")
    print("=" * 55)
    return all_ok
