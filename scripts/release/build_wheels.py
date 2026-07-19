"""一键构建 vie-framework 与场景插件的二进制 wheel(.so) 到 dist/。

用法（仓库根目录）：
    python scripts/release/build_wheels.py [--out dist] [--no-isolation] [--plugins NAME ...]

  --no-isolation  用当前环境构建（需已装 Cython/setuptools/wheel），更快、复用本机编译器。
  --plugins       只构建指定插件（可空格分隔多个），名字接受
                  `panel-label` / `panel_label` / `vie-plugin-panel-label` 任一写法；
                  不传则构建全部插件。框架 wheel 始终构建（所有插件都依赖它）。
  --framework-only 只构建 vie_framework，跳过所有插件（统一 runtime 镜像用：
                  镜像不烤任何场景，插件经 pkg/ 覆盖层按服务装卸）。与 --plugins 互斥。
  --plugins-only  只构建 --plugins 指定的插件，不重复构建框架（基础 overlay 用）。

产物：dist/*.whl —— 业务模块均为 .so，无明文。部署时：
    pip install -r requirements.txt
    pip install dist/vie_framework-*.whl dist/vie_plugin_*.whl
    # 只上线单场景时只装需要的插件 wheel，框架按 entry_points 只挂载已装插件路由

本脚本取代旧的 scripts/pack.py（后者仅加密 services/，插件化后已不覆盖 plugins/）。
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# 本脚本位于 scripts/release/，距仓库根两级（release → scripts → root）
ROOT = Path(__file__).resolve().parent.parent.parent
PLUGINS = sorted((ROOT / "plugins").glob("vie-plugin-*"))


def _norm(name: str) -> str:
    """归一化插件名：去掉 vie-plugin- 前缀，- 与 _ 等价，便于匹配多种写法。"""
    return name.lower().removeprefix("vie-plugin-").replace("_", "-")


def select_plugins(wanted):
    """按 --plugins 过滤插件目录；wanted 为空则返回全部。未匹配到则报错退出。"""
    if not wanted:
        return PLUGINS
    wanted_norm = {_norm(w) for w in wanted}
    selected = [p for p in PLUGINS if _norm(p.name) in wanted_norm]
    matched = {_norm(p.name) for p in selected}
    missing = wanted_norm - matched
    if missing:
        available = ", ".join(_norm(p.name) for p in PLUGINS)
        sys.exit(f"未找到插件: {', '.join(sorted(missing))}；可选: {available}")
    return selected


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
    ap.add_argument("--plugins", nargs="*", default=None, metavar="NAME",
                    help="只构建指定插件（如 panel-label）；不传则构建全部")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--framework-only", action="store_true",
                      help="只构建 vie_framework，跳过所有插件（统一 runtime 镜像用）")
    mode.add_argument("--plugins-only", action="store_true",
                      help="只构建 --plugins 指定的插件，跳过 vie_framework")
    args = ap.parse_args()

    if args.framework_only and args.plugins is not None:
        sys.exit("--framework-only 与 --plugins 互斥：前者不构建任何插件")
    if args.plugins_only and args.plugins is None:
        sys.exit("--plugins-only 必须同时指定 --plugins")

    out = (ROOT / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    plugins = [] if args.framework_only else select_plugins(args.plugins)

    if not args.plugins_only:
        build_one(ROOT, out, args.no_isolation)
    for plugin in plugins:                          # 选中的场景插件
        build_one(plugin, out, args.no_isolation)

    wheels = sorted(out.glob("*.whl"))
    print(f"\n构建完成，共 {len(wheels)} 个 wheel → {out}")
    for w in wheels:
        print(f"  {w.name}")


if __name__ == "__main__":
    main()
