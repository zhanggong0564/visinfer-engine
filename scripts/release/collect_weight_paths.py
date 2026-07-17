"""Collect concrete deployment weight files referenced by plugin configs."""

import argparse
import ast
from pathlib import Path


WEIGHT_PREFIX = "./weights/"


def collect_weight_paths(config_paths: list[Path], weights_root: Path) -> list[Path]:
    """Return sorted files below weights_root referenced by Python string literals."""
    root = weights_root.resolve(strict=True)
    referenced: set[Path] = set()
    for config_path in config_paths:
        tree = ast.parse(config_path.read_text(encoding="utf-8"), filename=str(config_path))
        for node in ast.walk(tree):
            value = node.value if isinstance(node, ast.Constant) else None
            if not isinstance(value, str) or not value.startswith(WEIGHT_PREFIX):
                continue
            relative = Path(value[len(WEIGHT_PREFIX):])
            target = (root / relative).resolve(strict=False)
            if target != root and root not in target.parents:
                raise ValueError(f"weight path escapes root: {value}")
            if not target.exists():
                raise FileNotFoundError(f"missing deployment weight: {target}")
            if target.is_dir():
                referenced.update(
                    path.relative_to(root)
                    for path in target.rglob("*")
                    if path.is_file()
                )
            elif target.is_file():
                referenced.add(target.relative_to(root))
    if not referenced:
        raise ValueError("no ./weights references found in plugin configs")
    return sorted(referenced, key=lambda path: path.as_posix())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+", type=Path)
    parser.add_argument("--root", type=Path, default=Path("weights"))
    args = parser.parse_args()
    for path in collect_weight_paths(args.configs, args.root):
        print(path.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
