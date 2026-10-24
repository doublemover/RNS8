"""Reviewed dependency pins and observed receipts; no network or installation.

Latest release, selected candidate, observed artifact and qualification evidence
remain separate. This module consumes the existing doctor and CMake selections.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import re

SCHEMA = "rns8_dependency_policy_v1"
ROLES = {"required", "tests", "build", "bootstrap_only", "package_manager", "optional_reference", "transitive_optional", "ci", "inventory_only", "accelerator"}


def hexadecimal(value: object, length: int) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{" + str(length) + "}", value) is not None


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_policy(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    errors = validate_policy(value)
    if errors: raise ValueError("invalid dependency policy: " + "; ".join(errors))
    return value


def validate_policy(value: dict) -> list[str]:
    errors = []
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        return ["unsupported dependency policy schema"]
    if value.get("promotion_allowed") is not False:
        errors.append("dependency policy must not authorize backend/cache promotion")
    components = value.get("components")
    if not isinstance(components, list): return errors + ["components must be an array"]
    names = set()
    for component in components:
        if not isinstance(component, dict): errors.append("component must be an object"); continue
        name = component.get("name")
        if not isinstance(name, str) or not name or name in names:
            errors.append("missing or duplicate component: " + str(name)); continue
        names.add(name)
        role = component.get("role")
        if not isinstance(role, str) or role not in ROLES: errors.append(name + ": invalid role")
        latest = component.get("latest_stable", {})
        if not isinstance(latest, dict) or not isinstance(latest.get("version"), str) or not isinstance(latest.get("source"), str) or not latest["source"].startswith("https://"):
            errors.append(name + ": latest stable needs version and primary source")
        elif re.search(r"rc|alpha|beta|dev|nightly|preview", latest["version"], re.I):
            errors.append(name + ": prerelease is not stable")
        selected = component.get("selected")
        if role == "inventory_only" and selected is not None:
            errors.append(name + ": unused inventory must not be selected or installed")
        if role != "inventory_only" and (not isinstance(selected, dict) or not isinstance(selected.get("version"), str) or not selected["version"] or not selected.get("status")):
            errors.append(name + ": selected candidate needs version and validation status")
        if isinstance(selected, dict) and isinstance(selected.get("version"), str) and selected["version"] in {"main", "master", "latest"}:
            errors.append(name + ": moving selected version is forbidden")
        if isinstance(selected, dict) and isinstance(latest, dict) and selected.get("version") != latest.get("version"):
            exception = component.get("exception")
            if not isinstance(exception, dict) or not all(exception.get(key) for key in ("reason", "owner", "revisit")):
                errors.append(name + ": older selected version needs an explicit exception and revisit condition")
        if role == "ci" and not hexadecimal(component.get("commit"), 40):
            errors.append(name + ": CI action needs immutable commit")
    required = {"python", "cmake", "ninja", "pip", "git", "vcpkg-tool", "boost-multiprecision", "catch2", "nlohmann-json", "gmp", "flint", "mpfr"}
    if not required.issubset(names): errors.append("required policy components are missing: " + ", ".join(sorted(required - names)))
    registry = value.get("vcpkg", {})
    if not isinstance(registry, dict): return errors + ["invalid vcpkg registry"]
    if not hexadecimal(registry.get("baseline"), 40):
        errors.append("vcpkg baseline must be immutable")
    if not hexadecimal(registry.get("baseline_manifest_sha256"), 64): errors.append("vcpkg baseline manifest hash is missing")
    ports = registry.get("ports", {})
    if not isinstance(ports, dict) or not ports: return errors + ["declared vcpkg closure is missing"]
    if not {"boost-multiprecision", "catch2", "nlohmann-json", "gmp", "flint", "mpfr"}.issubset(ports):
        errors.append("selected direct ports are missing from declared closure")
    for name, port in ports.items():
        if not isinstance(port, dict): errors.append(name + ": invalid port"); continue
        for field in ("manifest_sha256", "portfile_sha256"):
            if not hexadecimal(port.get(field), 64): errors.append(name + ": invalid " + field)
        if type(port.get("port_version")) is not int or port["port_version"] < 0:
            errors.append(name + ": invalid port revision")
        if not isinstance(port.get("source_sha512"), list) or any(not hexadecimal(item, 128) for item in port.get("source_sha512", [])):
            errors.append(name + ": invalid published source SHA512")
        dependencies = port.get("dependencies", [])
        if not isinstance(dependencies, list): errors.append(name + ": dependencies must be an array"); continue
        for dependency in dependencies:
            target = dependency if isinstance(dependency, str) else dependency.get("name") if isinstance(dependency, dict) else None
            if not isinstance(target, str) or target not in ports: errors.append(name + ": dependency missing from declared closure: " + str(target))
    bootstrap = value.get("bootstrap", {})
    if not isinstance(bootstrap, dict): return errors + ["invalid bootstrap lock"]
    if bootstrap.get("requirements") != "scripts/requirements-build-tools.txt": errors.append("unsupported bootstrap requirements path")
    if not hexadecimal(bootstrap.get("catch2_commit"), 40): errors.append("Catch2 bootstrap needs an immutable commit")
    if not hexadecimal(bootstrap.get("catch2_source_sha512"), 128): errors.append("Catch2 source hash is missing")
    artifacts = bootstrap.get("artifacts", [])
    if not isinstance(artifacts, list) or not artifacts: errors.append("hashed bootstrap artifacts are missing")
    if not isinstance(artifacts, list): return errors
    seen = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict): errors.append("invalid bootstrap artifact"); continue
        filename = artifact.get("filename", "")
        if not isinstance(filename, str): errors.append("artifact filename must be a string"); continue
        if filename in seen: errors.append("duplicate bootstrap artifact: " + filename)
        seen.add(filename)
        if not hexadecimal(artifact.get("sha256"), 64): errors.append(filename + ": invalid SHA256")
        if not isinstance(artifact.get("url"), str) or not artifact["url"].startswith("https://"): errors.append(filename + ": artifact URL must use HTTPS")
    return errors


def check_repository(root: Path, policy: dict) -> list[str]:
    """Detect pin drift offline; does not claim installed compatibility."""
    errors = validate_policy(policy)
    if errors: return errors
    components = {item["name"]: item for item in policy["components"]}
    if policy["bootstrap"]["catch2_tag"] != "v" + components["catch2"]["selected"]["version"]:
        errors.append("Catch2 bootstrap version differs from selected candidate")
    for artifact in policy["bootstrap"]["artifacts"]:
        component = components.get(artifact.get("component"))
        if not component or not component["selected"] or artifact.get("version") != component["selected"]["version"]:
            errors.append("bootstrap artifact differs from selected component version")
    manifest = json.loads((root / "vcpkg.json").read_text(encoding="utf-8"))
    if manifest.get("builtin-baseline") != policy["vcpkg"]["baseline"]:
        errors.append("vcpkg.json baseline differs from reviewed policy")
    overrides = {item["name"]: item for item in manifest.get("overrides", [])}
    if len(overrides) != len(manifest.get("overrides", [])): errors.append("duplicate manifest overrides")
    for name in ("boost-multiprecision", "catch2", "nlohmann-json", "gmp", "flint", "mpfr"):
        pin, declared = overrides.get(name, {}), policy["vcpkg"]["ports"][name]
        if pin.get("version") != declared["version"] or pin.get("port-version", 0) != declared["port_version"]:
            errors.append(name + ": manifest override differs from reviewed port")
        if components[name]["selected"]["version"] != declared["version"]:
            errors.append(name + ": component and port versions disagree")
    requirement_path = root / policy["bootstrap"]["requirements"]
    content = requirement_path.read_text(encoding="utf-8")
    lines = [line.strip() for line in content.splitlines() if line.strip() and not line.startswith("#")]
    expected = {}
    for artifact in policy["bootstrap"]["artifacts"]:
        expected.setdefault(artifact["component"] + "==" + artifact["version"], set()).add(artifact["sha256"])
    actual = {}
    for line in lines:
        parts = line.split()
        if parts[0] in actual: errors.append("duplicate build-tool requirement: " + parts[0])
        actual[parts[0]] = set(part.removeprefix("--hash=sha256:") for part in parts[1:] if part.startswith("--hash=sha256:"))
        if any(not part.startswith("--hash=sha256:") for part in parts[1:]): errors.append("unsupported build-tool requirement option")
    if actual != expected: errors.append("build-tool requirements differ from exact reviewed versions/hashes")
    for workflow in (root / ".github/workflows").glob("*.yml"):
        workflow_text = workflow.read_text(encoding="utf-8")
        for name, ref in re.findall(r"uses:\s*(actions/[\w-]+)@([^\s#]+)", workflow_text):
            item = components.get(name)
            if not item or ref != item.get("commit"): errors.append(str(workflow.relative_to(root)) + ": unreviewed action " + name + "@" + ref)
        if "actions/setup-python@" not in workflow_text or "python-version-file: .python-version" not in workflow_text:
            errors.append(str(workflow.relative_to(root)) + ": selected Python is not explicitly configured")
    if (root / ".python-version").read_text(encoding="utf-8").strip() != components["python"]["selected"]["version"]:
        errors.append(".python-version differs from selected Python")
    return errors


def installed_vcpkg_receipt(installed_root: Path) -> dict:
    """Read a selected installation's complete status/ABI metadata, not PATH."""
    metadata = installed_root / "vcpkg"
    status = metadata / "status"
    if not status.is_file():
        return {"status": "unavailable", "root": str(installed_root), "packages": [], "runtime_abi_validated": False}
    content = status.read_text(encoding="utf-8")
    packages = []; seen = set()
    for stanza in re.split(r"\r?\n\s*\r?\n", content.strip()):
        fields = {}; key = None
        for line in stanza.splitlines():
            if line.startswith((" ", "\t")) and key:
                fields[key] += "\n" + line.strip()
            elif ":" in line:
                key, value = line.split(":", 1); fields[key] = value.strip()
        if fields.get("Status") != "install ok installed": continue
        name, architecture = fields.get("Package"), fields.get("Architecture")
        if not name or not architecture: raise ValueError("installed vcpkg status lacks package/triplet")
        identity = name, architecture, fields.get("Feature", "core")
        if identity in seen: raise ValueError("duplicate installed package/triplet/feature: " + str(identity))
        seen.add(identity)
        version = fields.get("Version") or fields.get("Version-Semver") or fields.get("Version-Date") or fields.get("Version-String")
        port_version = int(fields.get("Port-Version", "0"))
        if port_version < 0: raise ValueError("negative installed port revision")
        packages.append({"name": name, "version": version, "port_version": port_version,
                         "triplet": architecture, "feature": fields.get("Feature", "core"), "abi": fields.get("Abi"),
                         "dependencies": fields.get("Depends", ""), "source": fields.get("Source")})
    identities = {path.relative_to(metadata).as_posix(): sha256(path) for path in sorted((metadata / "info").glob("*.list")) if path.is_file()}
    return {"status": "observed", "root": str(installed_root), "status_sha256": sha256(status),
            "package_metadata_sha256": identities, "packages": sorted(packages, key=lambda item: (item["name"], item["triplet"], item["feature"])),
            "scope": "all installed status entries, feature/triplet/ABI and file-list identities; not runtime loaded libraries",
            "runtime_abi_validated": False}


def configured_installed_root(cache_path: Path | None) -> Path | None:
    if cache_path is None: return None
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("VCPKG_INSTALLED_DIR:") and "=" in line:
            value = line.split("=", 1)[1]
            return Path(value) if value and not value.endswith("-NOTFOUND") else None
    return None


def selected_prefix_receipt(prefix: str) -> dict:
    """Find only a selected prefix's own bounded vcpkg metadata ancestors."""
    root = Path(prefix)
    for candidate in (root, root.parent, root.parent.parent):
        if (candidate / "vcpkg/status").is_file(): return installed_vcpkg_receipt(candidate)
    return {"status": "unavailable", "root": prefix, "packages": [], "runtime_abi_validated": False}


def selected_source_version(name: str, selected: dict) -> str | None:
    """Read the selected component's declaration, never a different root."""
    if selected.get("version"): return selected["version"]
    source = Path(selected["canonical_root"]) / "CMakeLists.txt"
    if source.is_file():
        variable = "version" if name == "ck" else "VERSION_STRING"
        match = re.search(r"^\s*set\s*\(\s*" + variable + r'\s+"?(\d+\.\d+\.\d+)"?\s*\)',
                          source.read_text(encoding="utf-8"), re.M | re.I)
        return match.group(1) if match else None
    return None


def evaluate_report(root: Path, policy: dict, report: dict, cache_path: Path | None = None) -> dict:
    """Attach reviewable version drift to the existing doctor evidence."""
    observations = []
    commands, packages = report.get("commands", {}), report.get("python_packages", {})
    installation = configured_installed_root(cache_path)
    receipt = installed_vcpkg_receipt(installation) if installation else {"status": "not_requested", "packages": [], "runtime_abi_validated": False}
    target_triplet = report.get("dependency_resolution", {}).get("vcpkg", {}).get("selected")
    target_triplet = target_triplet.get("paths", {}).get("triplet") if target_triplet else None
    installed = {item["name"]: item for item in receipt["packages"] if item["feature"] == "core" and item["triplet"] == target_triplet}
    features = report.get("dependency_resolution", {}).get("feature_policy", {})
    optional = report.get("dependency_resolution", {}).get("optional_components", {})
    selected_receipts = {}
    for name in ("gmp", "flint"):
        component = optional.get(name, {}).get("selected")
        if component:
            selected_receipts[name] = selected_prefix_receipt(component["canonical_root"])
    for component in policy["components"]:
        name, selected = component["name"], component["selected"]
        host = report.get("dependency_resolution", {}).get("request", {}).get("host")
        platform_selection = component.get("platform_selections", {}).get(host)
        if selected and platform_selection:
            selected = {**selected, "version": platform_selection["version"]}
        observed = None; source = "unavailable"
        if name == "python":
            observed = report.get("system", {}).get("python"); source = "running_interpreter"
        elif (command_name := "vcpkg" if name == "vcpkg-tool" else name) in commands:
            match = re.search(r"\b(\d{4}-\d{2}-\d{2}|\d+\.\d+(?:\.\d+)?(?:\.windows\.\d+)?)", commands[command_name].get("detail", ""))
            observed = match.group(1) if match else None; source = "existing_doctor_version_output"
        elif name in {"gmp", "flint", "mpfr"} and (dependency := "flint" if name == "mpfr" else name) in selected_receipts:
            matching = [item for item in selected_receipts[dependency]["packages"]
                        if item["name"] == name and item["feature"] == "core" and item["triplet"] == target_triplet]
            if matching:
                installed[name] = matching[0]; observed = matching[0]["version"]; source = "selected_component_installed_status"
            else:
                installed.pop(name, None); source = "selected_component_has_no_matching_installed_receipt"
        elif name in installed:
            observed = installed[name]["version"]; source = "configured_installation_status"
        elif name in packages:
            observed = packages[name]; source = "current_interpreter_package_metadata"
        elif name == "pip":
            try: observed = importlib.metadata.version("pip"); source = "current_interpreter_package_metadata"
            except importlib.metadata.PackageNotFoundError: pass
        elif name in {"ck", "rocwmma"} and optional.get(name, {}).get("selected"):
            observed = selected_source_version(name, optional[name]["selected"])
            source = "selected_source_declaration_not_runtime"
        port = installed.get(name)
        expected_port = policy["vcpkg"]["ports"].get(name)
        matches_version = observed == selected["version"] if selected and observed else None
        matches_pin = (matches_version and port["port_version"] == expected_port["port_version"]) if port and expected_port else matches_version
        revision = optional.get(name, {}).get("selected") or {}
        if selected and selected.get("revision"):
            matches_pin = matches_version and revision.get("revision") == selected["revision"]
        observations.append({"name": name, "role": component["role"], "latest_stable": component["latest_stable"]["version"],
                             "selected": selected, "observed_version": observed, "observation_source": source,
                             "observed_port_version": port["port_version"] if port else None,
                             "observed_revision": revision.get("revision"),
                             "matches_selected_version": matches_version, "matches_selected_pin": matches_pin,
                             "required_for_selected_configuration": component["role"] in {"build", "tests", "required"} or
                                 (component["role"] == "package_manager" and host == "Windows") or
                                 (name in {"ck", "rocwmma"} and features.get("RNS8_ENABLE_" + name.upper(), False)) or
                                 (name == "gmp" and (features.get("RNS8_ENABLE_GMP", False) or features.get("RNS8_ENABLE_FLINT", False))) or
                                 (name in {"flint", "mpfr"} and features.get("RNS8_ENABLE_FLINT", False)),
                             "compatibility": "not_established_by_version_match"})
    return {"schema": "rns8_dependency_version_receipt_v1", "policy_status": policy["status"], "promotion_allowed": False,
            "policy_sha256": sha256(root / "metadata/dependencies.json"), "consistency_errors": check_repository(root, policy),
            "components": observations, "installed_vcpkg": receipt, "selected_reference_installations": selected_receipts,
            "remaining_gates": policy["remaining_gates"]}
