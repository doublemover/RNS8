"""Explicit runtime-only, recoverable IFEO heap tracing configuration lease."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import re

from json_evidence import same_evidence_value

SOURCE = "https://learn.microsoft.com/en-us/windows-hardware/test/wpt/recording-for-heap-analysis"
IFEO = r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Image File Execution Options"


def configuration(image_name):
    if type(image_name) is not str or not re.fullmatch(
        r"[A-Za-z0-9_.-]{1,124}\.exe", image_name, re.I
    ):
        raise ValueError("one bounded native executable basename required for heap tracing")
    return dict(
        schema="rns8_windows_image_heap_configuration_v1",
        hive="HKEY_LOCAL_MACHINE",
        registry_view_bits=64,
        key=IFEO + "\\" + image_name,
        value="TracingFlags",
        enabled_value=1,
        enabled_type="REG_DWORD",
        image_name=image_name,
        scope="persistent_image_basename_setting_applies_to_new_processes_until_restored",
        requires_explicit_authorization=True,
        source=SOURCE,
    )


def snapshot_valid(value):
    if (
        not isinstance(value, dict)
        or set(value) != {"key_exists", "value_exists", "type", "value"}
        or type(value["key_exists"]) is not bool
        or type(value["value_exists"]) is not bool
        or value["value_exists"]
        and not value["key_exists"]
    ):
        raise ValueError("typed exact heap setting snapshot required")
    if value["value_exists"]:
        if (
            value["type"] != "REG_DWORD"
            or type(value["value"]) is not int
            or value["value"] not in {0, 1}
        ):
            raise ValueError("unknown existing heap setting type/bits require operator review")
    elif value["type"] is not None or value["value"] is not None:
        raise ValueError("absent heap setting cannot carry a value/type")
    return value


class NativeRegistry:
    """Only the documented64-bit IFEO value; imports do not touch the registry."""

    def __init__(self, policy):
        if platform.system() != "Windows":
            raise ValueError("native Windows registry required for image heap configuration")
        import winreg

        self.registry = winreg
        self.policy = policy
        self.mutex = None

    def __enter__(self):
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateMutexW.argtypes = (ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR)
        kernel.CreateMutexW.restype = wintypes.HANDLE
        kernel.ReleaseMutex.argtypes = (wintypes.HANDLE,)
        kernel.ReleaseMutex.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel.CloseHandle.restype = wintypes.BOOL
        name = (
            "Global\\RNS8_IFEO_"
            + hashlib.sha256(self.policy["key"].casefold().encode("utf-8")).hexdigest()
        )
        ctypes.set_last_error(0)
        handle = kernel.CreateMutexW(None, True, name)
        error = ctypes.get_last_error()
        if not handle:
            raise OSError(error, "could not acquire owned heap settings mutex")
        if error == 183:
            kernel.CloseHandle(handle)
            raise ValueError("another RNS8 image heap settings lease exists; do not overwrite it")
        self.kernel, self.mutex = kernel, handle
        return self

    def __exit__(self, *error):
        if self.mutex is not None:
            self.kernel.ReleaseMutex(self.mutex)
            self.kernel.CloseHandle(self.mutex)
            self.mutex = None

    def snapshot(self):
        reg = self.registry
        try:
            key = reg.OpenKey(
                reg.HKEY_LOCAL_MACHINE,
                self.policy["key"],
                0,
                reg.KEY_QUERY_VALUE | reg.KEY_WOW64_64KEY,
            )
        except FileNotFoundError:
            return snapshot_valid(dict(key_exists=False, value_exists=False, type=None, value=None))
        with key:
            try:
                value, kind = reg.QueryValueEx(key, self.policy["value"])
            except FileNotFoundError:
                return snapshot_valid(
                    dict(key_exists=True, value_exists=False, type=None, value=None)
                )
        return snapshot_valid(
            dict(
                key_exists=True,
                value_exists=True,
                type="REG_DWORD" if kind == reg.REG_DWORD else str(kind),
                value=value,
            )
        )

    def enable(self, prior):
        if self.snapshot() != prior:
            raise ValueError("heap setting changed before enable; no overwrite authorized")
        reg = self.registry
        with reg.CreateKeyEx(
            reg.HKEY_LOCAL_MACHINE,
            self.policy["key"],
            0,
            reg.KEY_QUERY_VALUE | reg.KEY_SET_VALUE | reg.KEY_WOW64_64KEY,
        ) as key:
            reg.SetValueEx(key, self.policy["value"], 0, reg.REG_DWORD, 1)
            reg.FlushKey(key)

    def restore(self, prior):
        current = self.snapshot()
        if current == prior:
            return "prior_state_already_present"
        if (
            not prior["key_exists"]
            and current["key_exists"]
            and all(current[k] == prior[k] for k in ("value_exists", "type", "value"))
        ):
            return "restored_value_empty_created_container_retained"
        if current != dict(key_exists=True, value_exists=True, type="REG_DWORD", value=1):
            raise ValueError(
                "heap setting differs from owned enabled state; preserve concurrent change"
            )
        reg = self.registry
        with reg.OpenKey(
            reg.HKEY_LOCAL_MACHINE,
            self.policy["key"],
            0,
            reg.KEY_QUERY_VALUE | reg.KEY_SET_VALUE | reg.KEY_WOW64_64KEY,
        ) as key:
            if prior["value_exists"]:
                reg.SetValueEx(key, self.policy["value"], 0, reg.REG_DWORD, prior["value"])
            else:
                reg.DeleteValue(key, self.policy["value"])
            reg.FlushKey(key)
        if not prior["key_exists"]:
            # Deleting a key also deletes values added by another writer between
            # an emptiness check and DeleteKey. Preserve the harmless container;
            # only the setting this lease changed is restored automatically.
            return "restored_value_created_container_retained"
        if self.snapshot() != prior:
            raise ValueError("heap settings restoration readback differs from saved state")
        return "restored_exact_prior_state"


def write_journal(path, document):
    # Publish the recoverable prior value before any persistent mutation. A
    # journal failure therefore cannot strand an unrecorded owned transition.
    path = Path(path)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def temporary_configuration(
    policy, directory, *, authorized=False, registry_factory=NativeRegistry, journal=write_journal
):
    if authorized is not True:
        raise ValueError("persistent heap configuration requires explicit authorization")
    if not isinstance(policy, dict) or not same_evidence_value(
        policy, configuration(policy.get("image_name"))
    ):
        raise ValueError("image heap configuration policy changed")
    directory = Path(directory)
    with registry_factory(policy) as registry:
        before = snapshot_valid(registry.snapshot())
        prior = dict(
            schema="rns8_windows_heap_settings_journal_v1",
            policy=policy,
            prior=before,
            owner_pid=os.getpid(),
            mutation_may_follow=True,
            recovery="restore only this saved TracingFlags value when current value still matches the owned enabled state",
        )
        journal(directory / "heap-settings-before.json", prior)
        changed = not (before["value_exists"] and before["value"] == 1)
        armed = False
        try:
            if changed:
                armed = True  # A failed native write/flush may have changed the value.
                registry.enable(before)
            if registry.snapshot() != dict(
                key_exists=True, value_exists=True, type="REG_DWORD", value=1
            ):
                raise ValueError("heap configuration did not read back the owned enabled state")
            yield prior
        finally:
            receipt = dict(
                schema="rns8_windows_heap_settings_restore_v1",
                policy=policy,
                prior=before,
                mutation_attempted=armed,
                owner_pid=os.getpid(),
            )
            try:
                receipt["state"] = registry.restore(before) if armed else "not_modified"
                receipt["after"] = registry.snapshot()
                receipt["prior_value_restored"] = all(
                    receipt["after"][k] == before[k] for k in ("value_exists", "type", "value")
                )
                receipt["prior_container_presence_restored"] = (
                    receipt["after"]["key_exists"] == before["key_exists"]
                )
            except BaseException as error:
                receipt.update(
                    state="restore_incomplete",
                    error=str(error) or type(error).__name__,
                    prior_value_restored=False,
                )
                journal(directory / "heap-settings-after.json", receipt)
                raise
            journal(directory / "heap-settings-after.json", receipt)
            if not receipt["prior_value_restored"]:
                raise ValueError(
                    "heap setting was not restored; retained journal requires operator recovery"
                )


def restore_journal(
    path, directory, *, authorized=False, registry_factory=NativeRegistry, journal=write_journal
):
    if authorized is not True:
        raise ValueError("heap settings recovery requires explicit authorization")
    from json_evidence import parse_evidence_json

    path = Path(path).resolve(strict=True)
    with path.open("rb") as stream:
        raw = stream.read(65537)
    if not raw or len(raw) > 65536:
        raise ValueError("bounded saved heap settings journal required")
    prior = parse_evidence_json(raw.decode("utf-8"))
    if (
        not isinstance(prior, dict)
        or prior.get("schema") != "rns8_windows_heap_settings_journal_v1"
        or not isinstance(prior.get("policy"), dict)
        or not same_evidence_value(
            prior["policy"], configuration(prior["policy"].get("image_name"))
        )
        or type(prior.get("owner_pid")) is not int
        or not 0 < prior["owner_pid"] < 2**32
        or prior.get("mutation_may_follow") is not True
    ):
        raise ValueError("typed original heap settings journal required")
    before = snapshot_valid(prior.get("prior"))
    directory = Path(directory)
    request = dict(
        schema="rns8_windows_heap_settings_recovery_v1",
        policy=prior["policy"],
        saved_journal=dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw)),
        original_owner_pid=prior["owner_pid"],
        prior=before,
        owner_pid=os.getpid(),
    )
    # An active cooperative lease owns the same named mutex and blocks recovery.
    with registry_factory(prior["policy"]) as registry:
        journal(directory / "heap-settings-recovery-before.json", request)
        result = dict(request)
        try:
            result["state"] = registry.restore(before)
            result["after"] = registry.snapshot()
            result["prior_value_restored"] = all(
                result["after"][k] == before[k] for k in ("value_exists", "type", "value")
            )
            result["prior_container_presence_restored"] = (
                result["after"]["key_exists"] == before["key_exists"]
            )
        except BaseException as error:
            result.update(
                state="restore_incomplete",
                error=str(error) or type(error).__name__,
                prior_value_restored=False,
            )
            journal(directory / "heap-settings-recovery-after.json", result)
            raise
        journal(directory / "heap-settings-recovery-after.json", result)
        if not result["prior_value_restored"]:
            raise ValueError("saved heap setting recovery did not match its prior value")
        return result


def main(argv=None):
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image-name")
    group.add_argument("--restore-journal", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--authorize-image-heap-configuration", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.image_name:
            if args.out or args.authorize_image_heap_configuration:
                raise ValueError("image-name mode is plan-only and cannot mutate settings")
            print(json.dumps(configuration(args.image_name), indent=2))
            return 0
        root = (Path(__file__).resolve().parents[1] / "temp").resolve()
        if not args.out or not args.authorize_image_heap_configuration:
            raise ValueError("recovery needs --out and --authorize-image-heap-configuration")
        out = args.out.resolve()
        if out == root or not out.is_relative_to(root) or out.exists():
            raise ValueError("recovery needs a fresh ignored child of temp/")
        out.mkdir(parents=True, exist_ok=False)
        print(json.dumps(restore_journal(args.restore_journal, out, authorized=True), indent=2))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
