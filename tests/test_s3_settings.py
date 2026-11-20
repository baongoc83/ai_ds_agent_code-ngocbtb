#!/usr/bin/env python
"""S3 settings: UI-over-.env priority, credential pairing, endpoint parsing,
scheme handling and path/virtual addressing auto-detection (no network).

Run:  python -m pytest tests/test_s3_settings.py -q
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pyarrow.fs as pafs

from config import Config
from Agents.BaseAgent.base_agent import BaseAgent as B

ENV = {"S3_ACCESS_KEY": "ENVKEY", "S3_SECRET_KEY": "envsecret", "S3_SESSION_TOKEN": "envtoken",
       "S3_REGION": "env-region", "S3_ENDPOINT_URL": "https://env.example", "S3_ADDRESSING": "auto"}


@pytest.fixture(autouse=True)
def env_config(monkeypatch):
    for k, v in ENV.items():
        monkeypatch.setattr(Config, k, v, raising=False)
    B._S3_STYLE_CACHE.clear()
    yield
    B.set_thread_s3_settings(None)
    B._S3_STYLE_CACHE.clear()


def settings(ui):
    B.set_thread_s3_settings(ui)
    return B.s3_settings()[:5]


# ── priority ──────────────────────────────────────────────────────────────

def test_empty_ui_falls_back_to_env():
    assert settings(None) == ("ENVKEY", "envsecret", "envtoken", "env-region", "https://env.example")
    assert settings({"S3_ENDPOINT_URL": "   ", "S3_ACCESS_KEY": ""})[4] == "https://env.example"


def test_ui_value_wins_per_field():
    s = settings({"S3_ENDPOINT_URL": "https://ui.example"})
    assert s == ("ENVKEY", "envsecret", "envtoken", "env-region", "https://ui.example")


def test_ui_credentials_are_one_set():
    # full UI pair: the .env session token must not ride along
    assert settings({"S3_ACCESS_KEY": "UIKEY", "S3_SECRET_KEY": "uisecret"})[:3] == ("UIKEY", "uisecret", "")
    # half pair from the UI stays half (the UI warns) instead of borrowing the .env secret
    assert settings({"S3_ACCESS_KEY": "UIKEY"})[:3] == ("UIKEY", "", "")


# ── paths / endpoints ─────────────────────────────────────────────────────

@pytest.mark.parametrize("path", ["s3a://bkt/dir/f.csv", "S3A://bkt/dir/f.csv", " s3://bkt/dir/f.csv/"])
def test_scheme_case_insensitive(path):
    assert B.is_s3_path(path)
    assert B._s3_bucket_key(path) == "bkt/dir/f.csv"


def test_not_s3():
    assert not B.is_s3_path("data/train.csv") and not B.is_s3_path(None)


@pytest.mark.parametrize("endpoint,expected", [
    ("https://devobs01.example.vn", ("https", "devobs01.example.vn")),
    ("HTTP://minio:9000/", ("http", "minio:9000")),
    ("https://gw.example/s3/api", ("https", "gw.example")),
    ("minio:9000", (None, "minio:9000")),
])
def test_parse_endpoint(endpoint, expected):
    assert B._parse_s3_endpoint(endpoint) == expected


# ── addressing auto-detection ─────────────────────────────────────────────

class FakeFS:
    def __init__(self, result):
        self.result, self.calls = result, 0

    def get_file_info(self, _bucket):
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return SimpleNamespace(type=self.result)


def patch_fs(monkeypatch, path_result, virtual_result):
    fss = {False: FakeFS(path_result), True: FakeFS(virtual_result)}
    monkeypatch.setattr(B, "_get_s3_filesystem", classmethod(lambda cls, virtual=False: fss[virtual]))
    return fss


D, NF = pafs.FileType.Directory, pafs.FileType.NotFound


@pytest.mark.parametrize("path_r,virtual_r,expect_virtual", [
    (D, D, False),                       # path-style works → keep it
    (NF, D, True),                       # only virtual-hosted finds the bucket
    (NF, NF, False),                     # bucket really missing → path (error reported later)
    (NF, OSError("dns"), False),         # bucket.host does not resolve
    (OSError("403"), D, True),           # no HeadBucket via path, virtual finds it
])
def test_auto_addressing(monkeypatch, path_r, virtual_r, expect_virtual):
    patch_fs(monkeypatch, path_r, virtual_r)
    B.set_thread_s3_settings(None)
    assert B._s3_fs_for("bkt")[1] is expect_virtual


def test_auto_decision_is_cached(monkeypatch):
    fss = patch_fs(monkeypatch, NF, D)
    B._s3_fs_for("bkt"); B._s3_fs_for("bkt")
    assert fss[False].calls == 1 and fss[True].calls == 1


@pytest.mark.parametrize("mode,expect_virtual", [("path", False), ("virtual", True)])
def test_forced_addressing_skips_probe(monkeypatch, mode, expect_virtual):
    monkeypatch.setattr(Config, "S3_ADDRESSING", mode)
    fss = patch_fs(monkeypatch, NF, D)
    assert B._s3_fs_for("bkt")[1] is expect_virtual
    assert fss[False].calls == fss[True].calls == 0


def test_aws_without_endpoint_skips_probe(monkeypatch):
    monkeypatch.setattr(Config, "S3_ENDPOINT_URL", "")
    fss = patch_fs(monkeypatch, NF, D)
    assert B._s3_fs_for("bkt")[1] is False and fss[False].calls == 0


def test_check_reports_missing_bucket_with_settings(monkeypatch):
    patch_fs(monkeypatch, NF, NF)
    with pytest.raises(FileNotFoundError, match=r"Bucket 'bkt'.*endpoint=https://env\.example.*addressing=path"):
        B.check_s3_access("s3a://bkt/dir/f.csv")


def test_check_reports_missing_object(monkeypatch):
    class ObjFS(FakeFS):
        def get_file_info(self, p):
            return SimpleNamespace(type=D if p == "bkt" else NF)
    monkeypatch.setattr(B, "_get_s3_filesystem", classmethod(lambda cls, virtual=False: ObjFS(None)))
    with pytest.raises(FileNotFoundError, match="Kết nối được, nhưng không thấy object"):
        B.check_s3_access("S3A://bkt/dir/f.csv")
