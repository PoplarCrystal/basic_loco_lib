"""Project paths for basic_loco_lib."""

import os
import sys
from pathlib import Path

# _sim_roots = []
# if "ISAACSIM_ROOT_PATH" in os.environ:
#     _sim_roots.append(Path(os.environ["ISAACSIM_ROOT_PATH"]))
# _sim_roots.extend(Path(sys.prefix).parents)
# for _root in _sim_roots:
#     _prebundle = _root / "exts" / "isaacsim.pip.newton" / "pip_prebundle"
#     if (_prebundle / "newton_usd_schemas").is_dir():
#         if str(_prebundle) not in sys.path:
#             sys.path.insert(0, str(_prebundle))
#         break

PROJECT_DIR = Path(__file__).resolve().parents[3]

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
