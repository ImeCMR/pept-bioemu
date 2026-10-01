"""pept-bioemu: BioEmu ensemble generation + conformational analysis for peptides."""
__version__ = "0.1.0"

# CRITICAL: Register bioemu as top-level package BEFORE any BioEmu imports
# This allows BioEmu's internal imports (from bioemu.chemgraph import ...) to work
import sys
import pkgutil
import importlib
from pathlib import Path

# Add vendor directory to path for alphafold and openfold imports
_vendor_dir = Path(__file__).parent / "bioemu" / "_vendor"
sys.path.insert(0, str(_vendor_dir))

from pept_bioemu import bioemu as _bioemu

sys.modules['bioemu'] = _bioemu
# Register all submodules (chemgraph, sample, etc.) as bioemu.submodule.
#
# Import each submodule directly under its aliased name (`bioemu.<name>`) rather
# than under its native name (`pept_bioemu.bioemu.<name>`) and copying the
# reference afterwards. Some submodules (e.g. so3_sde.py) import sibling
# submodules (e.g. sde_lib.py) via `from .sde_lib import ...`, which transitively
# resolves to and caches `bioemu.sde_lib` under sys.modules *while this loop is
# still running*. If the loop instead imported under the native name first, that
# transitive import would already have created and cached a `bioemu.sde_lib`
# module, and the loop's own native-name import would create a *second*,
# independent execution of sde_lib.py (a distinct module object under
# `pept_bioemu.bioemu.sde_lib`) -- silently overwriting the `bioemu.sde_lib` alias
# set by the transitive import with this second copy. Classes such as
# CosineVPSDE/SO3SDE would then exist as two distinct class objects depending on
# which import path reached them first, breaking `isinstance` checks between code
# that imported via `bioemu.X` and code that imported via `pept_bioemu.bioemu.X`.
#
# Importing directly under the aliased name lets importlib's own sys.modules
# cache do the deduplication: whichever import (this loop's, or a transitive one
# triggered while importing an earlier submodule) runs first wins, and every
# later import of the same name is guaranteed to return that same module object.
for _module_info in pkgutil.iter_modules(_bioemu.__path__):
    importlib.import_module(f'bioemu.{_module_info.name}')