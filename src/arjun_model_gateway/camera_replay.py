"""Deterministic, offline replay fixtures for the Camera history pilot."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import stat
import socket
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Iterator

from arjun_model_gateway.replay_eval import PRIVATE_IPV4_RE, SECRET_RE

CAMERA_REPO = Path("/Users/arjundivecha/Dropbox/AAA Backup/A Working/Camera")
PLAYWRIGHT_CLI = Path(
    "/Users/arjundivecha/.claude/skills/playwright/scripts/playwright_cli.sh"
)
PILOT_COMMITS = {
    "camera_dock_and_ios_regression": {
        "base": "7972767a10f58945342a39a818ae2c5e14654977",
        "gold": "9d814d0c07f30be620fbc57a9149b42ed6fa896d",
    },
    "camera_ios_clip_playback": {
        "base": "9d814d0c07f30be620fbc57a9149b42ed6fa896d",
        "gold": "c86be73998c005579eab25cd0657da20c58151d6",
    },
}
CAPSULE_PATHS = (
    "package.json",
    "package-lock.json",
    "next.config.ts",
    "tsconfig.json",
    "eslint.config.mjs",
    "postcss.config.mjs",
    "public",
    "src",
)
FORBIDDEN_TEXT = (
    re.compile(r"(?i)\b[\w.-]+\.ts\.net\b"),
    re.compile(r"(?i)\brtsp://\S+"),
)


@dataclass
class VerificationResult:
    task_id: str
    commit: str
    passed: bool
    checks: dict[str, bool]
    details: dict[str, object]


def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
    timeout: int = 120,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _safe_text_files(root: Path) -> Iterator[Path]:
    for path in root.rglob("*"):
        relative_parts = path.relative_to(root).parts
        if "node_modules" in relative_parts or ".next" in relative_parts:
            continue
        if path.is_file() and not path.is_symlink() and path.stat().st_size <= 2_000_000:
            try:
                path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            yield path


def assert_capsule_safe(root: Path) -> None:
    violations: list[str] = []
    for path in _safe_text_files(root):
        text = path.read_text(encoding="utf-8")
        if PRIVATE_IPV4_RE.search(text) or SECRET_RE.search(text):
            violations.append(str(path.relative_to(root)))
            continue
        if any(pattern.search(text) for pattern in FORBIDDEN_TEXT):
            violations.append(str(path.relative_to(root)))
    if violations:
        raise ValueError(f"capsule contains forbidden private material: {violations}")


def _sanitize_capsule(root: Path) -> None:
    for path in _safe_text_files(root):
        text = path.read_text(encoding="utf-8")
        sanitized = PRIVATE_IPV4_RE.sub("198.51.100.10", text)
        sanitized = FORBIDDEN_TEXT[0].sub("camera.test", sanitized)
        sanitized = FORBIDDEN_TEXT[1].sub("rtsp://synthetic.invalid/stream", sanitized)
        sanitized = re.sub(
            r"/Users/[^/\s\"']+(?:/[^\s\"']+)*",
            "/tmp/camera-replay",
            sanitized,
        )
        if sanitized != text:
            path.write_text(sanitized, encoding="utf-8")


def materialize_capsule(commit: str, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    archive = subprocess.Popen(
        ["git", "-C", str(CAMERA_REPO), "archive", commit, *CAPSULE_PATHS],
        stdout=subprocess.PIPE,
    )
    assert archive.stdout is not None
    extract = subprocess.run(
        ["tar", "-x", "-C", str(destination)],
        stdin=archive.stdout,
        capture_output=True,
        text=True,
        check=False,
    )
    archive.stdout.close()
    archive_code = archive.wait(timeout=30)
    if archive_code or extract.returncode:
        raise RuntimeError(extract.stderr or f"git archive failed with {archive_code}")

    # Preserve launcher logic when the commit has it, but replace all operator
    # constants. Earlier commits receive the same neutral known-bad stub.
    ops = destination / "ops"
    ops.mkdir(exist_ok=True)
    launcher_result = _run(
        ["git", "-C", str(CAMERA_REPO), "show", f"{commit}:ops/mac-launcher.sh"]
    )
    if launcher_result.returncode:
        launcher_text = """#!/bin/bash
# Benchmark stub: repair this launcher without using real services.
APP_URL="${APP_URL:-http://camera.test}"
# BUG: exits when a profile exists even if no app window exists.
if pgrep -f "CameraReplayProfile" >/dev/null 2>&1; then
  exit 0
fi
exec open "$APP_URL"
"""
    else:
        launcher_text = launcher_result.stdout
        launcher_text = re.sub(
            r'^APP_URL=.*$', 'APP_URL="${APP_URL:-http://camera.test}"',
            launcher_text, flags=re.MULTILINE,
        )
        launcher_text = re.sub(
            r'^PROJECT=.*$', 'PROJECT="${PROJECT:-/tmp/camera-replay}"',
            launcher_text, flags=re.MULTILINE,
        )
        launcher_text = re.sub(
            r'^LOG=.*$', 'LOG="${LOG:-/tmp/camera-replay.log}"',
            launcher_text, flags=re.MULTILINE,
        )
        launcher_text = re.sub(
            r'^PROFILE=.*$', 'PROFILE="${PROFILE:-$HOME/CameraReplayProfile}"',
            launcher_text, flags=re.MULTILINE,
        )
        launcher_text = re.sub(
            r'^CHROME=.*$', 'CHROME="${CHROME:-/Applications/Google Chrome}"',
            launcher_text, flags=re.MULTILINE,
        )
    (ops / "mac-launcher.sh").write_text(launcher_text, encoding="utf-8")
    (destination / "REPLAY_TASK.md").write_text(
        "Offline benchmark. Do not access networks, credentials, or production "
        "services. Use only local files and verifier commands.\n",
        encoding="utf-8",
    )
    _sanitize_capsule(destination)
    assert_capsule_safe(destination)
    current_modules = CAMERA_REPO / "node_modules"
    if current_modules.is_dir():
        copied = _run(
            ["cp", "-cR", str(current_modules), str(destination / "node_modules")],
            timeout=120,
        )
        if copied.returncode:
            raise RuntimeError(f"could not clone node_modules: {copied.stderr}")
    return destination


def _make_media(media_dir: Path) -> None:
    media_dir.mkdir(parents=True, exist_ok=True)
    result = _run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "color=c=#305070:s=320x180:d=2",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            "-hls_time", "1", "-hls_list_size", "0",
            str(media_dir / "index.m3u8"),
        ],
        timeout=60,
    )
    if result.returncode:
        raise RuntimeError(f"could not generate HLS fixture: {result.stderr}")
    image = _run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "color=c=#305070:s=320x180",
            "-frames:v", "1", str(media_dir / "frame.jpg"),
        ],
        timeout=30,
    )
    if image.returncode:
        raise RuntimeError(f"could not generate image fixture: {image.stderr}")


class _FixtureHandler(BaseHTTPRequestHandler):
    media_dir: Path
    requests_seen: list[str]

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        type(self).requests_seen.append(self.path)
        now = int(time.time())
        if self.path.startswith("/api/events?"):
            payload = [
                {
                    "id": "event-ready",
                    "camera": "front_door",
                    "label": "person",
                    "start_time": now - 20,
                    "end_time": now - 10,
                    "has_clip": True,
                    "has_snapshot": False,
                },
                {
                    "id": "event-missing",
                    "camera": "carport",
                    "label": "car",
                    "start_time": now - 40,
                    "end_time": now - 30,
                    "has_clip": False,
                    "has_snapshot": False,
                },
            ]
            self._send(200, json.dumps(payload).encode(), "application/json")
        elif self.path.endswith("/latest.jpg") or self.path.endswith("/thumbnail.jpg"):
            self._send(200, (self.media_dir / "frame.jpg").read_bytes(), "image/jpeg")
        elif self.path.endswith("/snapshot.jpg"):
            self._send(404, b"missing", "text/plain")
        elif self.path.endswith("/clip.mp4"):
            # Deliberately violate Safari's byte-range requirement.
            self._send(200, b"not-a-range-capable-mp4", "video/mp4")
        elif self.path.endswith("/index.m3u8") and "/vod/carport/" in self.path:
            self._send(404, b"missing", "text/plain")
        elif self.path.endswith("/index.m3u8"):
            self._send(
                200,
                (self.media_dir / "index.m3u8").read_bytes(),
                "application/vnd.apple.mpegurl",
            )
        elif self.path.endswith(".ts"):
            segment = self.media_dir / Path(self.path).name
            if segment.exists():
                self._send(200, segment.read_bytes(), "video/mp2t")
            else:
                self._send(404, b"missing", "text/plain")
        else:
            self._send(404, b"missing", "text/plain")


@contextlib.contextmanager
def fixture_server() -> Iterator[tuple[str, list[str]]]:
    with tempfile.TemporaryDirectory(prefix="camera-replay-media-") as temp:
        media_dir = Path(temp)
        _make_media(media_dir)
        handler = type(
            "FixtureHandler",
            (_FixtureHandler,),
            {"media_dir": media_dir, "requests_seen": []},
        )
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}", handler.requests_seen
        finally:
            server.shutdown()
            thread.join(timeout=5)


def _wait_for_http(url: str, timeout: int = 45) -> None:
    import urllib.request

    deadline = time.monotonic() + timeout
    error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status < 500:
                    return
        except Exception as exc:  # noqa: BLE001
            error = exc
        time.sleep(0.5)
    raise TimeoutError(f"app did not start at {url}: {error}")


@contextlib.contextmanager
def next_server(capsule: Path, fixture_origin: str) -> Iterator[str]:
    port = _free_port()
    env = os.environ.copy()
    env["FRIGATE_ORIGIN"] = fixture_origin
    env["NEXT_TELEMETRY_DISABLED"] = "1"
    process = subprocess.Popen(
        [
            str(capsule / "node_modules/.bin/next"), "dev",
            "--webpack", "--hostname", "127.0.0.1", "--port", str(port),
        ],
        cwd=capsule,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        _wait_for_http(url)
        yield url
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def _browser_eval(session: str, url: str, code: str) -> str:
    opened = _run(
        [
            str(PLAYWRIGHT_CLI), f"-s={session}", "open", url,
            "--browser", "webkit", "--device", "iPhone 15",
        ],
        timeout=60,
    )
    if opened.returncode:
        raise RuntimeError(f"Playwright open failed: {opened.stderr or opened.stdout}")
    try:
        result = _run(
            [str(PLAYWRIGHT_CLI), f"-s={session}", "run-code", code],
            timeout=60,
        )
        if result.returncode:
            raise RuntimeError(
                f"Playwright evaluation failed: {result.stderr or result.stdout}"
            )
        return result.stdout
    finally:
        _run([str(PLAYWRIGHT_CLI), f"-s={session}", "close"], timeout=30)


def _parse_browser_result(output: str) -> dict[str, object]:
    marker = "### Result\n"
    if marker not in output:
        raise ValueError("Playwright output did not contain a result")
    body = output.split(marker, 1)[1].split("\n### ", 1)[0].strip()
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise ValueError("Playwright result was not an object")
    return parsed


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _launcher_scenario(
    launcher: Path,
    *,
    profile_running: bool,
    focused: bool,
) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="camera-launcher-fixture-") as temp:
        root = Path(temp)
        bin_dir = root / "bin"
        bin_dir.mkdir()
        log = root / "calls.log"
        chrome = bin_dir / "chrome"
        _write_executable(
            chrome,
            '#!/bin/sh\nprintf "chrome:%s\\n" "$*" >> "$STUB_LOG"\n',
        )
        _write_executable(bin_dir / "curl", "#!/bin/sh\nexit 0\n")
        _write_executable(
            bin_dir / "pgrep",
            '#!/bin/sh\n[ "$STUB_PROFILE_RUNNING" = "1" ]\n',
        )
        _write_executable(
            bin_dir / "osascript",
            """#!/bin/sh
script=$(cat)
printf "osascript:%s\\n" "$script" >> "$STUB_LOG"
[ "$STUB_FOCUSED" = "1" ] && printf "found\\n"
exit 0
""",
        )
        _write_executable(
            bin_dir / "open",
            '#!/bin/sh\nprintf "open:%s\\n" "$*" >> "$STUB_LOG"\n',
        )
        _write_executable(
            bin_dir / "docker",
            '#!/bin/sh\nprintf "docker:%s\\n" "$*" >> "$STUB_LOG"\nexit 1\n',
        )
        env = {
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
            "HOME": str(root / "home"),
            "APP_URL": "http://camera.test/dashboard",
            "CHROME": str(chrome),
            "PROFILE": str(root / "profile"),
            "PROJECT": str(root / "project"),
            "LOG": str(root / "launcher.log"),
            "STUB_LOG": str(log),
            "STUB_PROFILE_RUNNING": "1" if profile_running else "0",
            "STUB_FOCUSED": "1" if focused else "0",
        }
        result = _run(["bash", str(launcher)], env=env, timeout=10)
        calls = log.read_text(encoding="utf-8") if log.exists() else ""
        return {
            "exit_code": result.returncode,
            "calls": calls,
            "stderr": result.stderr[-1000:],
        }


def verify_launcher(capsule: Path) -> dict[str, bool]:
    launcher = capsule / "ops/mac-launcher.sh"
    if not launcher.is_file():
        return {
            "launcher_opens_clean_session": False,
            "launcher_opens_when_profile_has_no_window": False,
            "launcher_focuses_without_duplicate": False,
            "launcher_matches_configured_host": False,
        }
    clean = _launcher_scenario(launcher, profile_running=False, focused=False)
    stale = _launcher_scenario(launcher, profile_running=True, focused=False)
    focused = _launcher_scenario(launcher, profile_running=True, focused=True)
    return {
        "launcher_opens_clean_session": (
            clean["exit_code"] == 0
            and ("chrome:" in str(clean["calls"]) or "open:" in str(clean["calls"]))
        ),
        "launcher_opens_when_profile_has_no_window": (
            stale["exit_code"] == 0
            and ("chrome:" in str(stale["calls"]) or "open:" in str(stale["calls"]))
        ),
        "launcher_focuses_without_duplicate": (
            focused["exit_code"] == 0
            and "osascript:" in str(focused["calls"])
            and "chrome:" not in str(focused["calls"])
            and "open:" not in str(focused["calls"])
        ),
        "launcher_matches_configured_host": (
            "camera.test" in str(stale["calls"]) or "camera.test" in str(focused["calls"])
        ),
    }


def verify_mobile_css_contract(capsule: Path) -> dict[str, bool]:
    sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (capsule / "src").rglob("*")
        if path.is_file() and path.suffix in {".ts", ".tsx", ".css"}
    )
    return {
        "uses_dynamic_mobile_viewport_unit": bool(
            re.search(r"(?:\b(?:dvh|svh)\b|h-(?:dvh|svh))", sources)
        ),
        "reserves_ios_safe_area": "safe-area-inset-bottom" in sources,
    }


def verify_capsule(task_id: str, commit: str, capsule: Path) -> VerificationResult:
    assert_capsule_safe(capsule)
    checks: dict[str, bool] = {}
    if task_id == "camera_dock_and_ios_regression":
        checks.update(verify_launcher(capsule))
        checks.update(verify_mobile_css_contract(capsule))
    details: dict[str, object] = {}
    with fixture_server() as (origin, requests_seen), next_server(capsule, origin) as url:
        session = f"camera-{os.getpid()}-{int(time.time() * 1000)}"
        browser_output = _browser_eval(
            session,
            url,
            """async (page) => {
              await page.waitForTimeout(2500);
              const heading = page.getByText('Recent Detections', { exact: true });
              const visible = await heading.isVisible().catch(() => false);
              const box = visible ? await heading.boundingBox() : null;
              const viewport = page.viewportSize();
              let readyModal = false;
              let missingModal = false;
              let fallbackVisible = false;
              let fallbackImage = "";
              const cards = page.getByTitle(/Play clip/);
              if (await cards.count()) {
                await cards.nth(0).click();
                await page.waitForTimeout(3500);
                readyModal = await page.getByLabel('Close').isVisible().catch(() => false);
                if (readyModal) await page.getByLabel('Close').click();
              }
              if (await cards.count() > 1) {
                await cards.nth(1).click();
                await page.waitForTimeout(3500);
                missingModal = await page.getByLabel('Close').isVisible().catch(() => false);
                fallbackVisible = await page.getByText(/SHOWING DETECTION IMAGE/)
                  .isVisible().catch(() => false);
                fallbackImage = await page.locator('[role="dialog"] img, .fixed.inset-0 img')
                  .first().getAttribute('src').catch(() => "") || "";
              }
              return {
                visible, box, viewport, readyModal, missingModal,
                fallbackVisible, fallbackImage
              };
            }""",
        )
        parsed = _parse_browser_result(browser_output)
        details["browser_output"] = browser_output[-4000:]
        details["fixture_requests"] = requests_seen
        checks["browser_loaded_detections"] = any(
            path.startswith("/api/events?limit=30") for path in requests_seen
        )
        checks["browser_opened_ready_event"] = parsed.get("readyModal") is True
        if task_id == "camera_dock_and_ios_regression":
            checks["mobile_strip_visible"] = parsed.get("visible") is True
        else:
            checks["ready_event_requested_hls"] = any(
                path.endswith("/index.m3u8") and "/vod/front_door/" in path
                for path in requests_seen
            )
            checks["ready_event_avoided_direct_mp4"] = not any(
                path.endswith("/clip.mp4") and "event-ready" in path for path in requests_seen
            )
            checks["missing_event_opened"] = parsed.get("missingModal") is True
            checks["missing_event_shows_fallback"] = parsed.get("fallbackVisible") is True
            checks["missing_event_uses_thumbnail"] = (
                "event-missing/thumbnail.jpg" in str(parsed.get("fallbackImage"))
                and any(
                    "event-missing/thumbnail.jpg" in path for path in requests_seen
                )
            )
            checks["missing_event_avoids_snapshot"] = not any(
                "event-missing/snapshot.jpg" in path for path in requests_seen
            )

    return VerificationResult(
        task_id=task_id,
        commit=commit,
        passed=all(checks.values()),
        checks=checks,
        details=details,
    )


def verify_commit(task_id: str, commit: str) -> VerificationResult:
    with tempfile.TemporaryDirectory(prefix="camera-replay-capsule-") as temp:
        capsule = materialize_capsule(commit, Path(temp))
        return verify_capsule(task_id, commit, capsule)


def prove_goldens() -> dict[str, object]:
    results: list[VerificationResult] = []
    for task_id, commits in PILOT_COMMITS.items():
        results.append(verify_commit(task_id, commits["base"]))
        results.append(verify_commit(task_id, commits["gold"]))
    passed = all(
        (not result.passed)
        if result.commit == PILOT_COMMITS[result.task_id]["base"]
        else result.passed
        for result in results
    )
    return {
        "schema_version": 1,
        "passed": passed,
        "results": [asdict(result) for result in results],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prove-goldens", "materialize"))
    parser.add_argument("--commit")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "prove-goldens":
        report = prove_goldens()
        print(json.dumps(report, indent=2))
        return 0 if report["passed"] else 1
    if not args.commit or not args.output:
        parser.error("materialize requires --commit and --output")
    materialize_capsule(args.commit, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
