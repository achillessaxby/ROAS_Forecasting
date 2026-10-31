"""Versioned model registry with a ``CURRENT`` pointer (review findings P2, P3).

Layout (local folder or a GCS bucket mounted into Cloud Run as a volume)::

    registry/
      CURRENT                      -> "v2025-09"
      v2025-09/
        manifest.json              as_of, settings, per-game alpha / spend range / mix
        G001/metadata.json, q0.25.json, q0.50.json, q0.75.json
        ...

Every API response reports ``model_version`` and ``data_as_of`` from the manifest, and a
new version only becomes live when it is promoted (after the back-test gate).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .models import QuantileModelSet


@dataclass
class LoadedVersion:
    version: str
    manifest: dict
    models: dict[str, QuantileModelSet] = field(default_factory=dict)

    @property
    def data_as_of(self) -> str:
        return self.manifest["as_of"]

    def game(self, game_id: str) -> dict:
        return self.manifest["games"][game_id]


class Registry:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def version_dir(self, version: str) -> Path:
        return self.root / version

    def write_manifest(self, version: str, manifest: dict) -> None:
        d = self.version_dir(version)
        d.mkdir(parents=True, exist_ok=True)
        (d / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    def promote(self, version: str) -> None:
        if not (self.version_dir(version) / "manifest.json").exists():
            raise FileNotFoundError(f"version {version} has no manifest")
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "CURRENT").write_text(version)

    def current(self) -> str:
        p = self.root / "CURRENT"
        if not p.exists():
            raise FileNotFoundError(f"no CURRENT pointer in {self.root}")
        return p.read_text().strip()

    def load(self, version: str | None = None) -> LoadedVersion:
        version = version or self.current()
        d = self.version_dir(version)
        manifest = json.loads((d / "manifest.json").read_text())
        lv = LoadedVersion(version=version, manifest=manifest)
        for game_id in manifest["games"]:
            lv.models[game_id] = QuantileModelSet.load(d / game_id)
        return lv
