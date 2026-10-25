"""Thin wrapper around the `adb` CLI for talking to an Android Automotive OS
(AAOS) emulator or a physical device.

Kept deliberately thin (subprocess calls) rather than pulling in a binary
protocol client library, since `adb` itself is the stable, documented
interface and is what most AAOS test rigs already have on PATH.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from voiceqa import config

logger = logging.getLogger(__name__)


class AdbError(RuntimeError):
    pass


class AdbClient:
    def __init__(self, serial: str | None = None, adb_binary: str | None = None):
        self.serial = serial or config.DEFAULT_DEVICE_SERIAL or None
        self.adb_binary = adb_binary or config.ADB_BINARY

    def _base_cmd(self) -> list[str]:
        cmd = [self.adb_binary]
        if self.serial:
            cmd += ["-s", self.serial]
        return cmd

    def _run(self, args: list[str], timeout: int = 30) -> subprocess.CompletedProcess:
        cmd = self._base_cmd() + args
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)
        except FileNotFoundError as exc:
            raise AdbError(
                f"'{self.adb_binary}' not found on PATH. Install the Android platform-tools "
                "or run the pipeline with --mock to skip real device interaction."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise AdbError(f"adb command timed out: {' '.join(cmd)}") from exc
        if result.returncode != 0:
            raise AdbError(f"adb command failed ({' '.join(cmd)}): {result.stderr.decode(errors='replace')}")
        return result

    @staticmethod
    def is_available(adb_binary: str | None = None) -> bool:
        return shutil.which(adb_binary or config.ADB_BINARY) is not None

    def devices(self) -> list[str]:
        result = self._run(["devices"])
        lines = result.stdout.decode(errors="replace").splitlines()[1:]
        return [line.split("\t")[0] for line in lines if line.strip() and "\tdevice" in line]

    def push(self, local_path: str, remote_path: str) -> None:
        self._run(["push", local_path, remote_path])

    def shell(self, command: str, timeout: int = 30) -> str:
        result = self._run(["shell", command], timeout=timeout)
        return result.stdout.decode(errors="replace")

    def screenshot(self, output_path: str) -> str:
        remote_tmp = "/sdcard/voiceqa_screen.png"
        self._run(["shell", "screencap", "-p", remote_tmp])
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        self._run(["pull", remote_tmp, output_path])
        return output_path

    def dump_ui(self) -> str:
        remote_tmp = "/sdcard/voiceqa_ui.xml"
        self._run(["shell", "uiautomator", "dump", remote_tmp])
        result = self._run(["shell", "cat", remote_tmp])
        return result.stdout.decode(errors="replace")
