"""vie-framework 二进制 wheel 构建。

把 services/schemas/routers/utils/config 五个顶层包的业务模块 cython 编译为 .so，
wheel 仅含各包 __init__.py + .so + 元数据，不落明文业务源码。app.py 作为启动器
随部署单独提供，不打入本 wheel。
"""
from pathlib import Path

from setuptools import setup, find_packages
from setuptools.command.build_py import build_py
from Cython.Build import cythonize

PACKAGES = ["services", "schemas", "routers", "utils", "config"]

# dc_fuse「服务形态」示例：保留在仓库作老框架（非插件）用法参考，但【不打入】
# framework wheel —— 框架 contract 为「不含具体场景」（见 pyproject 描述），dc_fuse 由
# vie-plugin-dc-fuse 经 entry_points 提供。若打入会与插件双重注册同名 dc_fuse 路由/检测器。
EXAMPLE_EXCLUDE_FILES = {
    Path("routers/dc_fuse_routers.py"),
    Path("schemas/dc_fuse_schemas.py"),
    Path("config/dc_fuse_config.py"),
}
EXAMPLE_EXCLUDE_DIRS = (Path("services/dc_fuse"),)


def _is_example(p: Path) -> bool:
    """判定某源文件是否为「不入 wheel」的示例（dc_fuse 服务形态）。"""
    return p in EXAMPLE_EXCLUDE_FILES or any(d in p.parents for d in EXAMPLE_EXCLUDE_DIRS)


# 保留各包 __init__.py 为纯 py，其余业务模块编译成 .so（排除 build/ 中间产物与示例）
py_sources = [
    str(p)
    for pkg in PACKAGES
    for p in Path(pkg).rglob("*.py")
    if p.name != "__init__.py" and "build" not in p.parts and not _is_example(p)
]


class BuildPyInitOnly(build_py):
    """只把 __init__.py 作为源码打入 wheel；其余 .py 已编成 .so，剔除以防明文泄露。"""

    def find_package_modules(self, package, package_dir):
        return [m for m in super().find_package_modules(package, package_dir) if m[1] == "__init__"]


setup(
    packages=find_packages(
        include=[f"{p}*" for p in PACKAGES],
        exclude=["services.dc_fuse", "services.dc_fuse.*"],  # 示例不入 wheel，见上
    ),
    # annotation_typing=False：关闭 Cython3 默认的注解类型强制，否则 FastAPI Form()/File()
    # 默认值与 `x: str` 注解冲突报 "Expected str, got Form"，pydantic 字段注解同理。
    ext_modules=cythonize(
        py_sources, build_dir="build",
        compiler_directives={"language_level": "3", "annotation_typing": False},
    ),
    cmdclass={"build_py": BuildPyInitOnly},
)
