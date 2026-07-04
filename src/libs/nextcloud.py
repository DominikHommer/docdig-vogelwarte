"""Nextcloud upload via WebDAV.

Stores finalised CSVs together with the original PDF on the Vogelwarte
Nextcloud (https://nextcloud.vogelwarte.ch, folder ``/docdig``).

Configuration comes from the environment (put it in ``.env`` — the pipeline
already loads dotenv):

    NEXTCLOUD_URL=https://nextcloud.vogelwarte.ch
    NEXTCLOUD_USER=<login name>
    NEXTCLOUD_PASSWORD=<app password, NOT the account password>
    NEXTCLOUD_DIR=/docdig            # optional, default /docdig

Create the app password in Nextcloud under
Settings -> Security -> Devices & sessions -> "Create new app password".

No credentials are ever written to disk or the repo. When the environment
is not configured, `is_configured()` returns False and the UI simply hides
the upload button.

TLS: the Vogelwarte Nextcloud uses an internally-signed certificate that
Python's bundled CAs don't know. Resolution order:

1. `enable_system_truststore()` (called by the app) makes requests use the
   OS trust store — if the internal CA is installed there, plain HTTPS works.
2. ``NEXTCLOUD_CA_BUNDLE=/path/to/ca.pem`` pins the internal CA explicitly.
3. ``verify=False`` on the config (UI checkbox / NEXTCLOUD_VERIFY_SSL=false)
   disables verification as a last resort for self-signed setups.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union
from urllib.parse import quote

import requests


DEFAULT_REMOTE_DIR = "/docdig"
_TIMEOUT = 30  # seconds per request


def enable_system_truststore() -> bool:
    """Use the operating system's CA store (macOS Keychain, Windows, Linux)
    for all TLS verification instead of Python's bundled certifi list.

    Internally-signed certificates that IT has rolled out to the machine
    then verify without any further configuration. Safe to call more than
    once; returns False when the ``truststore`` package is missing.
    """
    try:
        import truststore

        truststore.inject_into_ssl()
        return True
    except Exception:
        return False


@dataclass
class NextcloudConfig:
    base_url: str
    user: str
    password: str
    remote_dir: str = DEFAULT_REMOTE_DIR
    # True = normal verification, False = skip (self-signed), str = CA bundle path
    verify: Union[bool, str] = True

    @property
    def dav_root(self) -> str:
        return f"{self.base_url.rstrip('/')}/remote.php/dav/files/{quote(self.user)}"


def _verify_from_env() -> Union[bool, str]:
    ca_bundle = (os.environ.get("NEXTCLOUD_CA_BUNDLE") or "").strip()
    if ca_bundle:
        return ca_bundle
    flag = (os.environ.get("NEXTCLOUD_VERIFY_SSL") or "").strip().lower()
    if flag in ("0", "false", "no", "off"):
        return False
    return True


def load_config() -> Optional[NextcloudConfig]:
    """Read the Nextcloud settings from the environment; None if incomplete."""
    base_url = (os.environ.get("NEXTCLOUD_URL") or "").strip()
    user = (os.environ.get("NEXTCLOUD_USER") or "").strip()
    password = (os.environ.get("NEXTCLOUD_PASSWORD") or "").strip()
    remote_dir = (os.environ.get("NEXTCLOUD_DIR") or DEFAULT_REMOTE_DIR).strip()
    if not (base_url and user and password):
        return None
    if not remote_dir.startswith("/"):
        remote_dir = "/" + remote_dir
    return NextcloudConfig(base_url, user, password, remote_dir, verify=_verify_from_env())


def is_configured() -> bool:
    return load_config() is not None


def _request(config: NextcloudConfig, http, method: str, url: str, **kwargs):
    """All WebDAV traffic goes through here so TLS handling stays in one place."""
    kwargs.setdefault("auth", (config.user, config.password))
    kwargs.setdefault("timeout", _TIMEOUT)
    kwargs.setdefault("verify", config.verify)
    if config.verify is False:
        # The user explicitly opted out (self-signed cert) — don't spam the
        # log with an InsecureRequestWarning per request.
        with warnings.catch_warnings():
            try:
                from urllib3.exceptions import InsecureRequestWarning

                warnings.simplefilter("ignore", InsecureRequestWarning)
            except ImportError:
                pass
            return http.request(method, url, **kwargs)
    return http.request(method, url, **kwargs)


def probe(
    config: NextcloudConfig, session: Optional[requests.Session] = None
) -> Tuple[bool, str]:
    """Cheap credential/connectivity check (PROPFIND depth 0 on the DAV root).

    Returns (ok, message) — never raises, so the UI can show the message
    directly next to the login form.
    """
    http = session or requests
    try:
        response = _request(
            config, http, "PROPFIND", config.dav_root, headers={"Depth": "0"}
        )
    except requests.exceptions.SSLError:
        return False, (
            "Zertifikatsprüfung fehlgeschlagen (intern signiertes Zertifikat). "
            "Entweder das CA-Zertifikat der Vogelwarte installieren / über "
            "NEXTCLOUD_CA_BUNDLE angeben — oder im Formular "
            "„Zertifikatsprüfung deaktivieren“ anhaken."
        )
    except requests.exceptions.RequestException as e:
        return False, f"Server nicht erreichbar: {e}"

    if response.status_code in (207, 200):
        return True, f"Verbunden als {config.user}."
    if response.status_code == 401:
        return False, "Login fehlgeschlagen — Benutzername/App-Passwort prüfen."
    return False, f"Unerwartete Antwort: HTTP {response.status_code}"


def _remote_url(config: NextcloudConfig, remote_path: str) -> str:
    remote_path = remote_path.lstrip("/")
    return f"{config.dav_root}/{quote(remote_path)}"


def ensure_remote_dir(
    config: NextcloudConfig, remote_dir: str, session: Optional[requests.Session] = None
) -> None:
    """MKCOL every path segment (existing segments answer 405 — fine)."""
    http = session or requests
    parts = [p for p in remote_dir.strip("/").split("/") if p]
    current = ""
    for part in parts:
        current = f"{current}/{part}"
        response = _request(config, http, "MKCOL", _remote_url(config, current))
        # 201 = created, 405 = already exists; everything else is a problem.
        if response.status_code not in (201, 405):
            response.raise_for_status()


def upload_file(
    local_path,
    remote_name: Optional[str] = None,
    subdir: str = "",
    config: Optional[NextcloudConfig] = None,
    session: Optional[requests.Session] = None,
) -> str:
    """Upload one file; returns the remote path.

    Args:
        local_path: file on disk.
        remote_name: name in Nextcloud (defaults to the local file name).
        subdir: optional folder below the configured remote dir, e.g. the
            scan name — keeps each scan's PDF + CSV together.
        config: explicit config (defaults to environment).
        session: injectable HTTP session (for tests).

    Raises RuntimeError when Nextcloud is not configured and
    requests.HTTPError when the server rejects the upload.
    """
    config = config or load_config()
    if config is None:
        raise RuntimeError(
            "Nextcloud ist nicht konfiguriert — NEXTCLOUD_URL, NEXTCLOUD_USER "
            "und NEXTCLOUD_PASSWORD in .env setzen."
        )

    local_path = Path(local_path)
    if not local_path.is_file():
        raise FileNotFoundError(local_path)

    target_dir = config.remote_dir
    if subdir:
        target_dir = f"{target_dir.rstrip('/')}/{subdir.strip('/')}"

    http = session or requests
    ensure_remote_dir(config, target_dir, session=http)

    remote_path = f"{target_dir.rstrip('/')}/{remote_name or local_path.name}"
    with open(local_path, "rb") as f:
        response = _request(
            config,
            http,
            "PUT",
            _remote_url(config, remote_path),
            data=f,
            timeout=_TIMEOUT * 4,  # PDFs can be large
        )
    if response.status_code not in (200, 201, 204):
        response.raise_for_status()
    return remote_path


def upload_results(
    pdf_path,
    csv_bytes: bytes,
    csv_name: str,
    config: Optional[NextcloudConfig] = None,
    session: Optional[requests.Session] = None,
) -> Tuple[str, str]:
    """Upload the original PDF and the finalised CSV side by side.

    Both land in ``<remote_dir>/<scan name without extension>/`` so every
    scan keeps its source and its transcription together.

    Returns (remote_pdf_path, remote_csv_path).
    """
    config = config or load_config()
    if config is None:
        raise RuntimeError(
            "Nextcloud ist nicht konfiguriert — NEXTCLOUD_URL, NEXTCLOUD_USER "
            "und NEXTCLOUD_PASSWORD in .env setzen."
        )

    pdf_path = Path(pdf_path)
    subdir = pdf_path.stem

    remote_pdf = upload_file(pdf_path, subdir=subdir, config=config, session=session)

    http = session or requests
    target_dir = f"{config.remote_dir.rstrip('/')}/{subdir}"
    remote_csv = f"{target_dir}/{csv_name}"
    response = _request(
        config, http, "PUT", _remote_url(config, remote_csv), data=csv_bytes
    )
    if response.status_code not in (200, 201, 204):
        response.raise_for_status()
    return remote_pdf, remote_csv
