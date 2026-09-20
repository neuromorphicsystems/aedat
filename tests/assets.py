from __future__ import annotations

import dataclasses
import hashlib
import pathlib
import typing

dirname = pathlib.Path(__file__).resolve().parent


@dataclasses.dataclass
class File:
    path: pathlib.Path
    id_to_stream: dict[int, dict[str, typing.Any]]
    field_to_digest: dict[str, str]

    def field_to_hasher(self, fields: list[str] | None = None):
        if fields is None:
            fields = list(self.field_to_digest.keys())
        return {field: hashlib.sha3_224() for field in fields}


files: list[File] = [
    File(
        path=dirname / "data" / "test_data.aedat4",
        id_to_stream={
            0: {"type": "events", "width": 346, "height": 260},
            1: {"type": "frame", "width": 346, "height": 260},
            2: {"type": "imus"},
            3: {"type": "triggers"},
        },
        field_to_digest={
            "t": "f1e093cad5afb6ecb971dfa2ef7646ab4ae0f467f73a48804e40bb68",
            "x": "1d8ea97b0febadfde24dd0b9e608682fc6934fc88656823b15f0e7a7",
            "y": "18f89da35f8f10b24c3407b03aa7f82bdd7c8e6ab5369e2c30f8bad0",
            "on": "6f99cf01187da8a05e1a032f3782de51b87e51bdc31356669bdd7cb9",
            "frame": "6dbd0c0ea251788515bce54edf50b9f29d1995a0330a8b623504379b",
            "imus": "9dffb33769bdb00c67404c3a15479bbd7e204cdc7725976c2ec563ef",
            "triggers": "8479de279528d9d1a04b987ed95d54e2c641124cda618d7072ebc3b7",
        },
    ),
]
