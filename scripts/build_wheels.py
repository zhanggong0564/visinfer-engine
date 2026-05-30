"""一键构建 vie-framework 与全部场景插件的二进制 wheel(.so) 到 dist/。

用法（仓库根目录）：
    python scripts/build_wheels.py [--out dist] [--no-isolation]

  --no-isolation  用当前环境构建（需已装 Cython/setuptools/wheel），更快、复用本机编译器。

产物：dist/*.whl —— 业务模块均为 .so，无明文。部署时：
    pip install -r requirements.txt
    pip install dist/vie_framework-*.whl dist/vie_plugin_*.whl

本脚本取代旧的 scripts/pack.py（后者仅加密 services/，插件化后已不覆盖 plugins/）。
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLUGINS = sorted((ROOT / "plugins").glob("vie-plugin-*"))


def clean(project: Path):
    """清理上次构建残留（build/ 与 egg-info，以及包内可能的 .c/.so 残留）。"""
    shutil.rmtree(project / "build", ignore_errors=True)
    for egg in project.glob("*.egg-info"):
        shutil.rmtree(egg, ignore_errors=True)
    for stray in list(project.rglob("*.c")) + list(project.rglob("*.so")):
        if "build" not in stray.parts:
            stray.unlink()


def build_one(project: Path, out: Path, no_isolation: bool):
    clean(project)
    cmd = [sys.executable, "-m", "pip", "wheel", str(project), "--no-deps", "-w", str(out)]
    if no_isolation:
        cmd.append("--no-build-isolation")
    print(f"\n>>> 构建 {project.name or 'vie-framework'}")
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def main():
    ap = argparse.ArgumentParser(description="构建框架与插件二进制 wheel")
    ap.add_argument("--out", default="dist", help="wheel 输出目录（默认 dist/）")
    ap.add_argument("--no-isolation", action="store_true",
                    help="用当前环境构建（需已装 Cython/setuptools/wheel）")
    args = ap.parse_args()

    out = (ROOT / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    build_one(ROOT, out, args.no_isolation)        # vie-framework
    for plugin in PLUGINS:                          # 各场景插件
        build_one(plugin, out, args.no_isolation)

    wheels = sorted(out.glob("*.whl"))
    print(f"\n构建完成，共 {len(wheels)} 个 wheel → {out}")
    for w in wheels:
        print(f"  {w.name}")


if __name__ == "__main__":
    main()
