import os
"""Load standalone V4.1 components without importing accelerator frameworks."""
import importlib
import importlib.util
from pathlib import Path
import sys

NAME = "dsv41_training_components"
PACKAGE = Path(os.environ.get("MINDSPEED_LLM_ROOT", Path(__file__).resolve().parents[4] / "third_party/MindSpeed-LLM")) / "mindspeed_llm/tasks/models/transformer/deepseek41"


def component(name):
    if NAME not in sys.modules:
        spec = importlib.util.spec_from_file_location(NAME, PACKAGE / "__init__.py",
                                                     submodule_search_locations=[str(PACKAGE)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[NAME] = module
        spec.loader.exec_module(module)
    return importlib.import_module(f"{NAME}.{name}")
