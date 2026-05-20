"""Import-time environment shims. Import this FIRST, before transformers.

The shared `dfc` conda env has a torchvision whose C++ ops are ABI-incompatible with
the installed torch (``RuntimeError: operator torchvision::nms does not exist``).
transformers' ``image_utils`` imports ``torchvision.transforms.InterpolationMode``
unconditionally, which crashes the whole import chain (incl. ModernBERT / DroidDetect).

We never touch vision. If real torchvision import fails, install a minimal stub into
sys.modules so the symbol resolves. We do NOT modify the shared environment on disk.
"""
from __future__ import annotations

import importlib.machinery
import sys
import types


def _install_torchvision_stub() -> None:
    if "torchvision" in sys.modules:
        return
    try:
        import torchvision  # noqa: F401  (works -> nothing to do)
        return
    except Exception:
        pass

    class _AutoStub(types.ModuleType):
        """Returns an empty stub submodule for any unknown attribute, so chains like
        ``from torchvision import io`` resolve without the broken real package.
        Stubs are callable (return None) and skip dunders (let those raise normally)."""
        def __getattr__(self, name):  # only called when normal lookup fails
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            sub = _AutoStub(f"{self.__name__}.{name}")
            sub.__spec__ = importlib.machinery.ModuleSpec(f"{self.__name__}.{name}", loader=None)
            sys.modules[f"{self.__name__}.{name}"] = sub
            setattr(self, name, sub)
            return sub
        def __call__(self, *a, **k):
            return None

    tv = _AutoStub("torchvision")
    tv.__version__ = "0.0.0-stub"
    tv.__spec__ = importlib.machinery.ModuleSpec("torchvision", loader=None)
    transforms = _AutoStub("torchvision.transforms")
    transforms.__spec__ = importlib.machinery.ModuleSpec("torchvision.transforms", loader=None)

    class InterpolationMode:  # mirrors the enum members transformers reads at import
        NEAREST = "nearest"
        NEAREST_EXACT = "nearest-exact"
        BILINEAR = "bilinear"
        BICUBIC = "bicubic"
        BOX = "box"
        HAMMING = "hamming"
        LANCZOS = "lanczos"

    transforms.InterpolationMode = InterpolationMode
    tv.transforms = transforms
    sys.modules["torchvision"] = tv
    sys.modules["torchvision.transforms"] = transforms


_install_torchvision_stub()
