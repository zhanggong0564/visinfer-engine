'''
@Author       : zhanggong
@Date         : 2026-01-07 09:54:40
@Description  : 一键加密 services/ 目录并生成 encrypted/ 部署包

[已弃用 / DEPRECATED] 场景插件化后业务逻辑迁至 plugins/，本脚本仅加密 services/，
不再覆盖 5 个插件，产出的 encrypted/ 会缺失这些场景。请改用二进制 wheel 构建：

    python scripts/build_wheels.py --no-isolation     # 产出 dist/*.whl(.so)

部署见 PACKAGING.md。本脚本仅保留供 plate_screw 等仍在 services/ 的历史排查参考。
'''
import sys

print(__doc__, file=sys.stderr)

import os
import shutil
import time
from pathlib import Path
from distutils.core import setup
from Cython.Build import cythonize


PROJECT_ROOT = Path(__file__).parent.parent

# 非加密模块（完整复制，排除 __pycache__）
COPY_MODULES = ["schemas", "routers", "config", "utils"]

# 非代码资源（直接复制）
COPY_RESOURCES = ["weights"]

# 启动文件（直接复制）
COPY_FILES = ["app.py", "requirements.txt"]


def collect_py_files(source_dir: str):
    """递归收集需要编译的 .py 文件，排除 __init__.py。
    返回相对于 source_dir 父目录的路径（即从项目根看到 services/panel_label/xxx.py）。"""
    base = PROJECT_ROOT / source_dir
    parent = base.parent
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != "__pycache__"]
        for f in files:
            if f.endswith('.py') and not f.startswith('__'):
                yield os.path.relpath(os.path.join(root, f), parent)


def compile_services():
    """编译 services/ 下所有 .py 文件为 .so"""
    build_dir = PROJECT_ROOT / "services" / "build"
    build_tmp = build_dir / "temp"
    os.chdir(PROJECT_ROOT)

    py_files = list(collect_py_files("services"))
    if not py_files:
        print("警告: 未找到需要编译的文件")
        return False

    print(f"待编译文件 ({len(py_files)}):")
    for f in py_files:
        print(f"  {f}")

    setup(ext_modules=cythonize(py_files),
          script_args=["build_ext", "-b", str(build_dir), "-t", str(build_tmp)])

    # 清理 Cython 生成的 .c 文件（残留在源码目录）
    for root, _, files in os.walk(PROJECT_ROOT / "services"):
        for f in files:
            if f.endswith('.c'):
                os.remove(os.path.join(root, f))

    # 清理 build 内的临时文件
    for root, _, files in os.walk(build_dir):
        for f in files:
            if f.endswith('.c'):
                os.remove(os.path.join(root, f))
    if build_tmp.exists():
        shutil.rmtree(build_tmp)

    print(f"编译完成，输出目录: {build_dir}")
    return True


def build_encrypted():
    """收集编译产物和非加密文件到 encrypted/"""
    encrypted = PROJECT_ROOT / "encrypted"
    if encrypted.exists():
        shutil.rmtree(encrypted)

    # 1. 复制 .so 文件，去掉 build_dir 内多余的 services/ 前缀
    build_dir = PROJECT_ROOT / "services" / "build"
    for root, _, files in os.walk(build_dir):
        for f in files:
            if f.endswith('.so'):
                src = Path(root) / f
                # build 内的相对路径如 services/panel_label/xxx.so
                rel = src.relative_to(build_dir)
                parts = list(rel.parts)
                # 去掉前导的 services/ 前缀
                if parts[0] == "services":
                    parts = parts[1:]
                dst = encrypted / "services" / Path(*parts)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

    # 2. 复制 services/ 的 __init__.py（保留包结构，排除 __pycache__/build）
    services_dir = PROJECT_ROOT / "services"
    for root, dirs, files in os.walk(services_dir):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", "build")]
        for f in files:
            if f == "__init__.py":
                src = Path(root) / f
                rel = src.relative_to(services_dir)
                dst = encrypted / "services" / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

    # 3. 复制非加密模块
    for module in COPY_MODULES:
        src = PROJECT_ROOT / module
        if not src.exists():
            continue
        dst = encrypted / module
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    # 4. 复制资源文件（weights 等）
    for resource in COPY_RESOURCES:
        src = PROJECT_ROOT / resource
        if not src.exists():
            continue
        dst = encrypted / resource
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    # 5. 复制启动文件（app.py 切换为生产模式：关闭 reload）
    for f in COPY_FILES:
        src = PROJECT_ROOT / f
        if not src.exists():
            continue
        dst = encrypted / src.name
        if src.name == "app.py":
            content = src.read_text(encoding="utf-8")
            prod_content = content.replace(
                "reload=True,  # 开发模式开启热重载",
                "reload=False,  # 生产模式关闭热重载",
            )
            if prod_content == content:
                print("警告: app.py 未找到 reload=True 标记，未做生产模式替换")
            dst.write_text(prod_content, encoding="utf-8")
            shutil.copystat(src, dst)
        else:
            shutil.copy2(src, dst)

    so_count = sum(1 for _ in encrypted.rglob("*.so"))
    print(f"已生成 encrypted/ 目录，包含 {so_count} 个 .so 文件")


def cleanup():
    """清理 build 临时目录"""
    build_dir = PROJECT_ROOT / "services" / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
        print("已清理临时 build 目录")


def main():
    print("=" * 50)
    print("开始打包 services/ → encrypted/")
    print("=" * 50)

    start = time.time()

    if not compile_services():
        print("编译失败")
        return

    build_encrypted()
    cleanup()

    print(f"打包完成，耗时 {time.time() - start:.1f}s")
    print(f"输出目录: {PROJECT_ROOT / 'encrypted'}")


if __name__ == "__main__":
    main()
