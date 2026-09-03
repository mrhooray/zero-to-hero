"""Model and serving-image profiles for benchmark launchers."""
from importlib import import_module
from types import ModuleType

NAMES = ("deepseek_v4_flash", "glm53_flash")


def load(name: str) -> ModuleType:
    if name not in NAMES:
        raise ValueError(f"unknown benchmark profile: {name}")
    return import_module(f"{__name__}.{name}")
