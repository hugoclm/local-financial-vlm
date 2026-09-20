"""Exports paresseux : importer `src.parsing.chunking` ne charge ni PyMuPDF ni Pydantic."""
from importlib import import_module

_EXPORTS = {
    "DocumentChunk": "src.parsing.schemas",
    "ChunkType": "src.parsing.schemas",
    "BaseDocumentParser": "src.parsing.base",
    "KIDParser": "src.parsing.kid_parser",
    "FullProspectusParser": "src.parsing.prospectus_parser",
    "DocumentDispatcher": "src.parsing.dispatcher",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name in _EXPORTS:
        return getattr(import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module 'src.parsing' has no attribute {name!r}")
