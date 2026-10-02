# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Filesystem regressions for manifest publication; no Mooncake binary or GPU needed."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store import master as master_module

pytestmark = pytest.mark.cpu_only


def test_startup_failure_retracts_partial_manifest(monkeypatch, tmp_path: Path) -> None:
    stopped = []
    master = SimpleNamespace(address="127.0.0.1:50051", stop=lambda: stopped.append(True))
    monkeypatch.setattr(master_module, "_launch_master", lambda *args, **kwargs: master)
    run_dir = tmp_path / "run"
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    sentinel = occupied / "keep"
    sentinel.write_text("unrelated")

    with pytest.raises(OSError):
        with master_module.running_master(str(run_dir), pool_file=str(occupied)):
            pytest.fail("publication over a directory must fail before yielding")

    assert not (run_dir / master_module.POOL_MANIFEST_NAME).exists()
    assert not (tmp_path / "occupied.partial").exists()
    assert sentinel.read_text() == "unrelated"
    assert stopped


def test_failed_publication_preserves_unvisited_manifest(tmp_path: Path) -> None:
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    untouched = tmp_path / "another-pool.json"
    other = {"master_server_address": "another-master:50051"}
    untouched.write_text(json.dumps(other))
    manifest = master_module.PoolManifest(master_server_address="127.0.0.1:50051")

    with pytest.raises(OSError):
        with master_module._published_manifest(manifest, [str(occupied), str(untouched)]):
            pytest.fail("publication over a directory must fail before yielding")

    assert json.loads(untouched.read_text()) == other
    assert occupied.is_dir()
    assert not (tmp_path / "occupied.partial").exists()


def test_published_manifest_is_readable_and_retracted_on_body_failure(tmp_path: Path) -> None:
    paths = [tmp_path / "run" / "pool.json", tmp_path / "shared" / "pool.json"]
    manifest = master_module.PoolManifest(master_server_address="127.0.0.1:50051", namespace="test")

    with pytest.raises(RuntimeError, match="body failed"):
        with master_module._published_manifest(manifest, [str(path) for path in paths]):
            for path in paths:
                assert json.loads(path.read_text()) == manifest.to_json()
                assert not Path(f"{path}.partial").exists()
            raise RuntimeError("body failed")

    assert all(not path.exists() for path in paths)
