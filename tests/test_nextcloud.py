"""Tests for the Nextcloud WebDAV uploader (libs/nextcloud.py).

All HTTP goes through an injected fake session — no network, no credentials.
"""

import pytest

from libs import nextcloud
from libs.nextcloud import (
    NextcloudConfig,
    ensure_remote_dir,
    is_configured,
    load_config,
    probe,
    upload_file,
    upload_results,
)


class FakeResponse:
    def __init__(self, status_code=201):
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    """Records every request; MKCOL answers 405 for 'already exists' paths."""

    def __init__(self, existing_dirs=(), fail_put=False):
        self.calls = []
        self.existing = set(existing_dirs)
        self.fail_put = fail_put

    def request(self, method, url, **kwargs):
        body = kwargs.get("data")
        if hasattr(body, "read"):
            body = body.read()
        self.calls.append((method, url, body))
        if method == "MKCOL":
            name = url.rstrip("/").split("/files/")[-1]
            return FakeResponse(405 if name in self.existing else 201)
        if method == "PUT" and self.fail_put:
            return FakeResponse(403)
        return FakeResponse(201)


CONFIG = NextcloudConfig(
    base_url="https://nextcloud.vogelwarte.ch",
    user="dominik",
    password="app-pass",
    remote_dir="/docdig",
)


def test_config_from_environment(monkeypatch):
    monkeypatch.delenv("NEXTCLOUD_URL", raising=False)
    monkeypatch.delenv("NEXTCLOUD_USER", raising=False)
    monkeypatch.delenv("NEXTCLOUD_PASSWORD", raising=False)
    assert load_config() is None
    assert is_configured() is False

    monkeypatch.setenv("NEXTCLOUD_URL", "https://nextcloud.vogelwarte.ch")
    monkeypatch.setenv("NEXTCLOUD_USER", "dominik")
    monkeypatch.setenv("NEXTCLOUD_PASSWORD", "secret")
    cfg = load_config()
    assert cfg is not None
    assert cfg.remote_dir == "/docdig"
    assert cfg.dav_root == "https://nextcloud.vogelwarte.ch/remote.php/dav/files/dominik"
    assert is_configured() is True


def test_config_custom_dir_gets_leading_slash(monkeypatch):
    monkeypatch.setenv("NEXTCLOUD_URL", "https://x")
    monkeypatch.setenv("NEXTCLOUD_USER", "u")
    monkeypatch.setenv("NEXTCLOUD_PASSWORD", "p")
    monkeypatch.setenv("NEXTCLOUD_DIR", "scans/2026")
    assert load_config().remote_dir == "/scans/2026"


def test_ensure_remote_dir_creates_each_segment():
    session = FakeSession()
    ensure_remote_dir(CONFIG, "/docdig/scan_1972", session=session)
    methods_urls = [(m, u) for m, u, _ in session.calls]
    assert methods_urls == [
        ("MKCOL", f"{CONFIG.dav_root}/docdig"),
        ("MKCOL", f"{CONFIG.dav_root}/docdig/scan_1972"),
    ]


def test_ensure_remote_dir_tolerates_existing():
    session = FakeSession(existing_dirs={"docdig"})
    ensure_remote_dir(CONFIG, "/docdig", session=session)  # must not raise


def test_upload_file_puts_to_remote_dir(tmp_path):
    f = tmp_path / "scan.pdf"
    f.write_bytes(b"%PDF-fake")
    session = FakeSession()

    remote = upload_file(f, subdir="scan", config=CONFIG, session=session)

    assert remote == "/docdig/scan/scan.pdf"
    put = [c for c in session.calls if c[0] == "PUT"]
    assert len(put) == 1
    assert put[0][1] == f"{CONFIG.dav_root}/docdig/scan/scan.pdf"
    assert put[0][2] == b"%PDF-fake"


def test_upload_file_requires_config(tmp_path, monkeypatch):
    for var in ("NEXTCLOUD_URL", "NEXTCLOUD_USER", "NEXTCLOUD_PASSWORD"):
        monkeypatch.delenv(var, raising=False)
    f = tmp_path / "a.csv"
    f.write_text("x")
    with pytest.raises(RuntimeError, match="nicht konfiguriert"):
        upload_file(f, session=FakeSession())


def test_upload_file_missing_local_file():
    with pytest.raises(FileNotFoundError):
        upload_file("/nope/missing.pdf", config=CONFIG, session=FakeSession())


def test_upload_results_pdf_and_csv_side_by_side(tmp_path):
    pdf = tmp_path / "scan_1972_sample.pdf"
    pdf.write_bytes(b"%PDF")
    session = FakeSession()

    remote_pdf, remote_csv = upload_results(
        pdf_path=pdf,
        csv_bytes=b"a;b;c",
        csv_name="scan_1972_sample.csv",
        config=CONFIG,
        session=session,
    )

    assert remote_pdf == "/docdig/scan_1972_sample/scan_1972_sample.pdf"
    assert remote_csv == "/docdig/scan_1972_sample/scan_1972_sample.csv"
    puts = [(u, b) for m, u, b in session.calls if m == "PUT"]
    assert puts == [
        (f"{CONFIG.dav_root}/docdig/scan_1972_sample/scan_1972_sample.pdf", b"%PDF"),
        (f"{CONFIG.dav_root}/docdig/scan_1972_sample/scan_1972_sample.csv", b"a;b;c"),
    ]


class ProbeSession:
    def __init__(self, status=207, raise_exc=None):
        self.status = status
        self.raise_exc = raise_exc
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs.get("headers")))
        if self.raise_exc:
            raise self.raise_exc
        return FakeResponse(self.status)


def test_probe_success():
    session = ProbeSession(status=207)
    ok, msg = probe(CONFIG, session=session)
    assert ok
    assert "dominik" in msg
    method, url, headers = session.calls[0]
    assert method == "PROPFIND"
    assert url == CONFIG.dav_root
    assert headers == {"Depth": "0"}


def test_probe_wrong_credentials():
    ok, msg = probe(CONFIG, session=ProbeSession(status=401))
    assert not ok
    assert "App-Passwort" in msg


def test_probe_unreachable_server():
    import requests as _requests

    session = ProbeSession(raise_exc=_requests.exceptions.ConnectionError("boom"))
    ok, msg = probe(CONFIG, session=session)
    assert not ok
    assert "nicht erreichbar" in msg


def test_probe_ssl_error_gives_actionable_message():
    import requests as _requests

    session = ProbeSession(raise_exc=_requests.exceptions.SSLError("self-signed"))
    ok, msg = probe(CONFIG, session=session)
    assert not ok
    assert "Zertifikat" in msg


def test_verify_flag_is_passed_to_requests(tmp_path):
    class RecordingSession(FakeSession):
        def request(self, method, url, **kwargs):
            self.calls.append((method, url, kwargs.get("verify", "MISSING")))
            return FakeResponse(201)

    f = tmp_path / "a.pdf"
    f.write_bytes(b"%PDF")
    insecure = NextcloudConfig(
        base_url="https://x", user="u", password="p", remote_dir="/d", verify=False
    )
    session = RecordingSession()
    upload_file(f, config=insecure, session=session)
    assert all(v is False for _, _, v in session.calls), session.calls


def test_verify_from_env(monkeypatch):
    monkeypatch.setenv("NEXTCLOUD_URL", "https://x")
    monkeypatch.setenv("NEXTCLOUD_USER", "u")
    monkeypatch.setenv("NEXTCLOUD_PASSWORD", "p")

    monkeypatch.setenv("NEXTCLOUD_VERIFY_SSL", "false")
    assert load_config().verify is False

    monkeypatch.setenv("NEXTCLOUD_VERIFY_SSL", "true")
    assert load_config().verify is True

    monkeypatch.setenv("NEXTCLOUD_CA_BUNDLE", "/etc/ssl/vogelwarte.pem")
    assert load_config().verify == "/etc/ssl/vogelwarte.pem"


def test_upload_surfaces_server_errors(tmp_path):
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF")
    with pytest.raises(RuntimeError, match="HTTP 403"):
        upload_results(
            pdf_path=pdf,
            csv_bytes=b"x",
            csv_name="scan.csv",
            config=CONFIG,
            session=FakeSession(fail_put=True),
        )
