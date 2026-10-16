"""Create complete immutable package snapshots and verify their local imports."""

import argparse
import ast
import hashlib
import json
import shutil
from pathlib import Path


def missing_local_modules(package):
    missing = set()
    for path in package.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            names = []
            if isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            elif isinstance(node, ast.Import):
                names = [item.name for item in node.names]
            for name in names:
                if name.startswith("hip_orb."):
                    relative = name.removeprefix("hip_orb.").replace(".", "/")
                    if not (package / (relative + ".py")).is_file() and not (package / relative / "__init__.py").is_file():
                        missing.add(name)
    return sorted(missing)


def verify_snapshot(root):
    root = Path(root)
    package = root / "hip_orb"
    manifest = json.loads((root / "manifest.json").read_text())
    missing = missing_local_modules(package)
    mismatched = [name for name, expected in manifest["modules"].items()
                  if not (package / name).is_file()
                  or hashlib.sha256((package / name).read_bytes()).hexdigest() != expected]
    if missing or mismatched or not manifest["complete"]:
        raise ValueError(f"Incomplete runtime snapshot: missing={missing}, changed={mismatched}")
    return manifest


def create_snapshot(source, output):
    source, output = Path(source), Path(output)
    # Never overwrite an allocation's existing runtime or replace a partial copy.
    output.mkdir(parents=True, exist_ok=False)
    package = output / "hip_orb"
    shutil.copytree(source, package, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    missing = missing_local_modules(package)
    modules = {str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted(package.rglob("*.py"))}
    manifest = {"complete": not missing, "modules": modules, "missing_modules": missing}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return verify_snapshot(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-package", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        manifest = verify_snapshot(args.output)
    else:
        if args.source_package is None:
            parser.error("Creating a snapshot requires --source-package")
        manifest = create_snapshot(args.source_package, args.output)
    print(json.dumps({"complete": manifest["complete"], "modules": len(manifest["modules"])}))


if __name__ == "__main__":
    main()
