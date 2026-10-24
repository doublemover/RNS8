"""Check first-party documentation ownership, local links/anchors and command names.

Network checks are opt-in. A skipped/blocked requested URL is unresolved, not a pass.
This tool never configures a build, installs dependencies or launches GPU work.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import subprocess
import time
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = "docs/documentation-inventory.json"
OFFICIAL_DOMAINS = ("amd.com", "microsoft.com", "cmake.org", "ninja-build.org", "git-scm.com", "python.org")


def document_paths(root):
    """Git knows first-party files and ignores generated output/submodule contents."""
    files = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "-z"]
    ).decode("utf-8").split("\0")
    return sorted({p for p in files if p and (
        Path(p).suffix.lower() in (".md", ".html", ".rst") or p in ("LICENSE", "NOTICE", ".github/FUNDING.yml")
        or (p.startswith(".github/ISSUE_TEMPLATE/") and Path(p).suffix in (".yml", ".yaml")))})


def prose(text):
    lines, fence = [], None
    for line in text.splitlines():
        match = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
        if match:
            delimiter = match[1]
            if fence is None:
                fence = delimiter
            elif delimiter[0] == fence[0] and len(delimiter) >= len(fence):
                fence = None
            lines.append("")
        else:
            lines.append(line if fence is None else "")
    return "\n".join(lines)


def slug(text):
    text = re.sub(r"<[^>]+>", "", text).strip().lower()
    text = re.sub(r"!?\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = "".join(c for c in text if c in " -_" or unicodedata.category(c)[0] in "LNM")
    return text.replace(" ", "-")


class HtmlReferences(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links, self.ids = [], set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        for attribute in ("href", "src"):
            if attrs.get(attribute):
                self.links.append((self.getpos()[0], attrs[attribute]))
        for attribute in ("id", "name"):
            if attrs.get(attribute):
                self.ids.add(attrs[attribute])


def anchors(text, html=False):
    parsed = HtmlReferences()
    parsed.feed(text if html else prose(text))
    result, counts = set(parsed.ids), Counter()
    lines = prose(text).splitlines()
    for index, line in enumerate(lines):
        heading = re.match(r"^\s{0,3}#{1,6}\s+(.+?)(?:\s+#+\s*)?$", line)
        title = heading[1] if heading else None
        if title is None and index + 1 < len(lines) and line.strip() and re.fullmatch(r"\s{0,3}(?:=+|-+)\s*", lines[index + 1]):
            title = line.strip()
        if title is not None:
            base = slug(title)
            identity = base if not counts[base] else f"{base}-{counts[base]}"
            while identity in result:
                counts[base] += 1
                identity = f"{base}-{counts[base]}"
            result.add(identity)
            counts[base] += 1
    return result


def links(text, html=False):
    source = text if html else prose(text)
    parsed = HtmlReferences()
    parsed.feed(source)
    found = list(parsed.links)
    if html:
        return found
    definitions = {}
    for index, line in enumerate(source.splitlines(), 1):
        match = re.match(r"^\s{0,3}\[([^\]]+)\]:\s*(<[^>]+>|\S+)", line)
        if match:
            definitions[match[1].casefold()] = match[2].strip("<>")
            found.append((index, match[2].strip("<>")))
    # Balanced parentheses accommodate source paths and URL query parameters.
    for match in re.finditer(r"!?\[([^\]\n]*)\]\(", source):
        start = match.end()
        depth, cursor = 1, start
        while cursor < len(source) and depth:
            if source[cursor] == "\\":
                cursor += 2
                continue
            if source[cursor] == "(":
                depth += 1
            elif source[cursor] == ")":
                depth -= 1
            cursor += 1
        if depth:
            continue
        target = source[start:cursor - 1].strip()
        target = target[1:target.index(">")] if target.startswith("<") and ">" in target else target.split()[0] if target else ""
        found.append((source.count("\n", 0, match.start()) + 1, target))
    for match in re.finditer(r"!?\[([^\]\n]+)\]\[([^\]\n]*)\]", source):
        key = (match[2] or match[1]).casefold()
        found.append((source.count("\n", 0, match.start()) + 1, definitions.get(key, "MISSING_REFERENCE:" + key)))
    for match in re.finditer(r'https?://[^\s<>"`|]+', source):
        target = match[0].rstrip(".,;:")
        while target.endswith(")") and target.count(")") > target.count("("):
            target = target[:-1]
        # Markdown closers are punctuation, not part of a bare URL.
        target = target.rstrip("]")
        found.append((source.count("\n", 0, match.start()) + 1, target))
    return found


def local_target(root, source, reference):
    parsed = urlsplit(reference)
    if parsed.scheme or parsed.netloc:
        return None
    path = unquote(parsed.path).replace("\\", "/")
    target = (root / path.lstrip("/") if path.startswith("/") else source.parent / path).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("link escapes repository")
    return target, unquote(parsed.fragment)


def command_errors(root, source, text, presets):
    """Static name checks only; commands are not executed."""
    errors = []
    for match in re.finditer(r"\b(cmake|ctest)\s+(?:(--build)\s+)?--preset\s+([\w-]+)", text):
        family = "testPresets" if match[1] == "ctest" else "buildPresets" if match[2] else "configurePresets"
        if match[3] not in presets.get(family, set()):
            errors.append(f"{source}: unknown {family} name {match[3]}")
    for match in re.finditer(r"\b(?:tools|scripts)[/\\][\w./\\-]+\.(?:py|sh)\b", text):
        path = match[0].replace("\\", "/")
        if not (root / path).is_file():
            errors.append(f"{source}: missing referenced script {path}")
    targets = set(re.findall(r"\brns8-example-[a-z0-9-]+\b", text))
    examples = (root / "examples/CMakeLists.txt").read_text(encoding="utf-8")
    available = {"rns8-example-" + name for name in re.findall(r"^  ([a-z0-9]+(?:-[a-z0-9]+)+)$", examples, re.M)}
    for target in targets - available:
        errors.append(f"{source}: unknown example target {target}")
    bench_args = (root / "benchmarks/rns8_bench_args.cpp")
    if bench_args.exists():
        declared = set(re.findall(r'"(--[a-z0-9-]+)"', bench_args.read_text(encoding="utf-8")))
        for match in re.finditer(r"rns8-bench(?:\.exe)?([^\n]*(?:[\\`]\n[^\n]*)*)", text):
            for option in set(re.findall(r"--[a-z][a-z0-9-]*", match[1])) - declared:
                errors.append(f"{source}: unknown rns8-bench option {option}")
    return errors


def template_errors(root):
    # The form deliberately uses a small, inspectable YAML subset. PyYAML is not
    # required by the documentation workflow; GitHub renders the final form.
    path = root / ".github/ISSUE_TEMPLATE/hardware_validation_offer.yml"
    if not path.exists():
        return ["hardware offer form missing"]
    text = path.read_text(encoding="utf-8")
    errors = []
    ids = re.findall(r"^    id: (\w+)$", text, re.M)
    if len(ids) != len(set(ids)) or set(ids) != {"offer_type", "hardware", "availability", "contact", "self_run"}:
        errors.append("hardware offer form has missing/duplicate field ids")
    for required in ("Do not post passwords", "SSH keys", "tokens", "account credentials", "private connection endpoints",
                     "billing data", "Not sure", "https://github.com/sponsors/doublemover", "/RNS8/issues/58"):
        if required not in text:
            errors.append("hardware offer form missing required notice/coordination: " + required)
    if re.search(r"^\s*\t", text, re.M) or not text.startswith("name:") or "\nbody:\n" not in text:
        errors.append("hardware offer form has invalid top-level/indentation structure")
    return errors


def check(root=ROOT):
    paths = document_paths(root)
    errors, external, inbound = [], set(), {path: [] for path in paths}
    parsed_anchors = {}
    presets = json.loads((root / "CMakePresets.json").read_text(encoding="utf-8"))
    presets = {key: {entry["name"] for entry in presets[key]} for key in ("configurePresets", "buildPresets", "testPresets")}
    inventory_path = root / INVENTORY
    inventory = json.loads(inventory_path.read_text(encoding="utf-8")) if inventory_path.exists() else {}
    records = inventory.get("documents", [])
    known = [record.get("path") for record in records]
    if len(known) != len(set(known)):
        errors.append("duplicate documentation inventory entries")
    if set(known) != set(paths):
        errors.append(f"documentation inventory mismatch: missing={sorted(set(paths)-set(known))}; stale={sorted(set(known)-set(paths))}")
    def verify(source, reference, location):
        if reference.startswith("MISSING_REFERENCE:"):
            errors.append(f"{location}: {reference}")
            return
        if urlsplit(reference).scheme in ("http", "https"):
            external.add(reference)
            return
        if urlsplit(reference).scheme:
            return
        try:
            resolved = local_target(root, source, reference)
        except ValueError as error:
            errors.append(f"{location}: {error}: {reference}")
            return
        if resolved is None:
            return
        target, anchor = resolved
        if not target.exists():
            errors.append(f"{location}: missing target {reference}")
            return
        relative = target.relative_to(root.resolve()).as_posix()
        if relative in inbound:
            inbound[relative].append({"source": source.relative_to(root.resolve()).as_posix(), "reference": reference})
        if anchor and target.suffix.lower() in (".md", ".html"):
            if target not in parsed_anchors:
                parsed_anchors[target] = anchors(target.read_text(encoding="utf-8"), target.suffix == ".html")
            if anchor not in parsed_anchors[target]:
                errors.append(f"{location}: missing anchor {reference}")
    for path in paths:
        source = root / path
        text = source.read_text(encoding="utf-8")
        for line, reference in links(text, source.suffix == ".html"):
            verify(source, reference, f"{path}:{line}")
        # Preserved dated material may name removed historical commands. Current
        # instructions are checked, while history remains link/anchor checked.
        record = next((r for r in records if r.get("path") == path), {})
        if record.get("disposition") not in ("historical", "preserve", "redirect"):
            errors += command_errors(root, path, text, presets)
    for record in records:
        for key in ("purpose", "audience", "disposition", "canonical", "owner", "reason"):
            if not isinstance(record.get(key), str) or not record[key]:
                errors.append(f"{record.get('path')}: missing inventory {key}")
        if record.get("disposition") not in ("keep", "rewrite", "merge", "historical", "preserve", "redirect"):
            errors.append(f"{record.get('path')}: invalid disposition")
        if record.get("canonical"):
            verify(root / INVENTORY, "../" + record["canonical"], "inventory canonical " + str(record.get("path")))
        if record.get("owner") and not (root / record["owner"]).exists():
            errors.append(f"{record.get('path')}: missing capability/evidence owner {record['owner']}")
    errors += template_errors(root)
    all_anchors = {p: sorted(anchors((root / p).read_text(encoding="utf-8"), p.endswith('.html')))
                   for p in paths if p.endswith((".md", ".html"))}
    for record in records:
        path = record.get("path")
        if record.get("inbound_links") != inbound.get(path, []) or record.get("anchors") != all_anchors.get(path, []):
            errors.append(f"{path}: stale derived inventory links/anchors; use --refresh-inventory")
    return dict(schema="rns8_documentation_check_v1", documents=len(paths), errors=errors,
                external_urls=sorted(external), inbound_links=inbound,
                anchors=all_anchors,
                command_check="static names only; no configure/build/GPU execution")


def external_status(url, *, timeout=12, attempts=2):
    """Retry transient responses once; do not hammer a blocked endpoint."""
    for attempt in range(attempts):
        try:
            request = Request(url.split("#", 1)[0], headers={"User-Agent": "RNS8-documentation-check/1.0"})
            with urlopen(request, timeout=timeout) as response:
                response.read(1024)
                return dict(url=url, status="reachable", http_status=response.status)
        except HTTPError as error:
            transient = error.code in (429, 500, 502, 503, 504)
            if transient and attempt + 1 < attempts:
                retry = error.headers.get("Retry-After", "1") if error.headers else "1"
                time.sleep(min(int(retry), 5) if retry.isdecimal() else 1)
                continue
            return dict(url=url, status="unresolved" if transient or error.code in (401, 403) else "failed", http_status=error.code)
        except (URLError, TimeoutError, OSError) as error:
            if attempt + 1 < attempts:
                continue
            return dict(url=url, status="unresolved", detail=str(error))
    return dict(url=url, status="unresolved")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external", action="store_true", help="check HTTP links; unresolved/blocked URLs fail this check")
    parser.add_argument("--official-only", action="store_true", help="scope external checks to official dependency/installer domains")
    parser.add_argument("--refresh-inventory", action="store_true", help="refresh derived inbound links/anchors in the authored inventory")
    parser.add_argument("--json", action="store_true", help="emit inventory inbound links/anchors and check results")
    args = parser.parse_args()
    if args.official_only and not args.external:
        parser.error("--official-only requires --external")
    report = check()
    if args.refresh_inventory:
        inventory_path = ROOT / INVENTORY
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        for record in inventory["documents"]:
            record["inbound_links"] = report["inbound_links"].get(record["path"], [])
            record["anchors"] = report["anchors"].get(record["path"], [])
        inventory_path.write_text(json.dumps(inventory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        report = check()
    results = []
    if args.external:
        urls = sorted({url.split("#", 1)[0] for url in report["external_urls"] if not args.official_only or any(
            urlsplit(url).hostname == domain or (urlsplit(url).hostname or "").endswith("." + domain) for domain in OFFICIAL_DOMAINS)})
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(external_status, urls))
    report["external_results"] = results
    report["external_scope"] = "official dependency domains" if args.official_only else "all linked HTTP URLs" if args.external else "not requested; external reachability unverified"
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"Documentation: {report['documents']} inventoried; {len(report['errors'])} local errors; {report['command_check']}")
        for error in report["errors"]:
            print(error)
        print("External: " + report["external_scope"])
        if args.external:
            print(dict(Counter(result["status"] for result in results)))
            for result in results:
                if result["status"] != "reachable":
                    print(result)
    return 1 if report["errors"] else 2 if any(r["status"] != "reachable" for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
