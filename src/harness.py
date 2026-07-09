# src/harness.py — legacy shim; 실체는 neurogolf.scoring
from neurogolf.scoring import *          # noqa: F401,F403
from neurogolf.scoring import (          # noqa: F401 — star가 못 잡는 밑줄/상수 명시
    DATA_DIR, ROOT, GRID_SHAPE, IR_VERSION, OPSET_IMPORTS, DATA_TYPE,
    EXCLUDED_OP_TYPES, FILESIZE_LIMIT_IN_BYTES,
)
