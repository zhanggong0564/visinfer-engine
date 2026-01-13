'''
@Author       : gongzhang4
@Date         : 2026-01-07 09:54:40
@LastEditors  : zhanggong1 zhanggong1@sungrowpower.com
@LastEditTime : 2026-01-07 11:34:08
@FilePath     : pack.py
@Description  : 批量加密多个文件夹的Python代码（适配模块内build结构）
'''

import os
import sys
import shutil
import subprocess
from pathlib import Path

# 需要加密的文件夹列表
FOLDERS_TO_ENCRYPT = ["services"]

# 需要排除的文件（启动文件和配置文件）
EXCLUDE_FILES = ["app.py", "config.py", "start_app.sh", "requirements.txt", "Dockerfile", "test.json"]


def run_py2so_on_folder(folder_path):
    """对单个文件夹运行py2so加密"""
    try:
        print(f"正在加密文件夹: {folder_path}")

        # 切换到项目根目录
        original_dir = os.getcwd()
        project_root = Path(__file__).parent.parent
        os.chdir(project_root)

        # 执行py2so.py
        cmd = [sys.executable, "tools/py2so.py", folder_path]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"✅ 文件夹 {folder_path} 加密完成")
            print(result.stdout)
        else:
            print(f"❌ 文件夹 {folder_path} 加密失败")
            print(result.stderr)

        os.chdir(original_dir)
        return result.returncode == 0

    except Exception as e:
        print(f"❌ 加密文件夹 {folder_path} 时出错: {e}")
        return False


def collect_encrypted_files_from_module_builds():
    """从各个模块的build目录收集加密文件"""
    project_root = Path(__file__).parent.parent

    # 创建统一的加密文件目录结构
    encrypted_root = project_root / "encrypted"
    if encrypted_root.exists():
        shutil.rmtree(encrypted_root)
    encrypted_root.mkdir(exist_ok=True)

    encrypted_files_count = 0

    # 遍历每个需要加密的文件夹，收集其build目录中的.so文件
    for folder in FOLDERS_TO_ENCRYPT:
        module_dir = project_root / folder
        build_dir = module_dir / "build"

        if build_dir.exists():
            print(f"📁 处理模块: {folder}")

            # 针对services模块的build目录结构进行智能处理
            for root, dirs, files in os.walk(build_dir):
                for file in files:
                    if file.endswith('.so'):
                        source_file = Path(root) / file
                        relative_path = source_file.relative_to(build_dir)

                        # services模块的build目录结构处理：
                        # - services/build/lap_surf_core.so → encrypted/services/lap_surf_core.so
                        # - services/build/services/base.so → encrypted/services/base.so
                        # - services/build/services/dc_fuse/business_logic.so → encrypted/services/dc_fuse/business_logic.so
                        parts = list(relative_path.parts)
                        if len(parts) > 1 and parts[0] == "services":
                            # 去掉重复的services目录，但保留子目录结构
                            target_file = encrypted_root / folder / Path(*parts[1:])
                        else:
                            # 直接复制根目录的文件
                            target_file = encrypted_root / folder / relative_path

                        # 确保目标目录存在
                        target_file.parent.mkdir(parents=True, exist_ok=True)

                        shutil.copy2(source_file, target_file)
                        print(f"  📄 复制: {file} -> {target_file.relative_to(encrypted_root)}")
                        encrypted_files_count += 1

            # 复制模块的__init__.py文件（如果存在）
            init_file = module_dir / "__init__.py"
            if init_file.exists():
                target_init = encrypted_root / folder / "__init__.py"
                target_init.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(init_file, target_init)
                print(f"  📄 复制: __init__.py")

            # 对于services模块，还需要复制子目录的__init__.py文件
            if folder == "services":
                for subdir in ["dc_fuse", "lap_surf", "plate_screw"]:
                    subdir_init = module_dir / subdir / "__init__.py"
                    if subdir_init.exists():
                        target_subdir_init = encrypted_root / folder / subdir / "__init__.py"
                        target_subdir_init.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(subdir_init, target_subdir_init)
                        print(f"  📄 复制: {subdir}/__init__.py")

    return encrypted_files_count


def organize_encrypted_structure():
    """整理加密后的文件结构，保持原有模块结构"""
    project_root = Path(__file__).parent.parent
    encrypted_root = project_root / "encrypted"

    if not encrypted_root.exists():
        print("❌ 未找到encrypted目录，请先运行加密")
        return

    # 复制必要的启动文件和配置文件
    for file in ["app.py", "config.py", "requirements.txt", "start_app.sh"]:
        source_file = project_root / file
        if source_file.exists():
            shutil.copy2(source_file, encrypted_root)
            print(f"📄 已复制配置文件: {file}")

    # 复制权重文件
    weights_dir = project_root / "weights"
    if weights_dir.exists():
        shutil.copytree(weights_dir, encrypted_root / "weights", dirs_exist_ok=True)
        print("📦 已复制权重文件")

    # 复制其他未加密的模块（保持目录结构）
    modules_to_copy = ["schemas", "utils", "routers"]
    for module in modules_to_copy:
        module_dir = project_root / module
        if module_dir.exists():
            shutil.copytree(module_dir, encrypted_root / module, dirs_exist_ok=True)
            print(f"📁 已复制未加密模块: {module}")

    # 创建必要的__init__.py文件
    for folder in FOLDERS_TO_ENCRYPT:
        init_file = encrypted_root / folder / "__init__.py"
        if not init_file.exists():
            init_file.parent.mkdir(parents=True, exist_ok=True)
            with open(init_file, 'w') as f:
                f.write("# Auto-generated init file for encrypted module\n")
            print(f"📄 创建: {folder}/__init__.py")

    print(f"🎉 加密文件整理完成，目录: {encrypted_root}")


def cleanup_module_build_dirs():
    """清理各个模块的build临时目录"""
    project_root = Path(__file__).parent.parent

    for folder in FOLDERS_TO_ENCRYPT:
        build_dir = project_root / folder / "build"
        if build_dir.exists():
            shutil.rmtree(build_dir)
            print(f"🧹 已清理: {folder}/build")

    # 清理根目录的build（如果有）
    root_build_dir = project_root / "build"
    if root_build_dir.exists():
        shutil.rmtree(root_build_dir)
        print("🧹 已清理: build/")


def main():
    """主函数：批量加密并整理文件结构"""
    print("🚀 开始批量加密Python代码...")
    print("💡 注意：加密文件将生成在各个模块的build目录中")

    # 检查py2so.py是否存在
    py2so_path = Path(__file__).parent / "py2so.py"
    if not py2so_path.exists():
        print("❌ 未找到py2so.py文件")
        return

    # 检查需要加密的文件夹是否存在
    project_root = Path(__file__).parent.parent
    existing_folders = []

    for folder in FOLDERS_TO_ENCRYPT:
        folder_path = project_root / folder
        if folder_path.exists():
            existing_folders.append(folder)
        else:
            print(f"⚠️  文件夹不存在: {folder}")

    if not existing_folders:
        print("❌ 没有找到需要加密的文件夹")
        return

    print(f"📂 将加密以下文件夹: {', '.join(existing_folders)}")

    # 批量加密
    success_count = 0
    for folder in existing_folders:
        if run_py2so_on_folder(folder):
            success_count += 1

    print(f"\n📊 加密完成统计: {success_count}/{len(existing_folders)} 个文件夹成功")

    if success_count > 0:
        # 收集各个模块build目录中的加密文件
        print("\n📦 正在收集加密文件...")
        file_count = collect_encrypted_files_from_module_builds()

        if file_count > 0:
            # 整理加密文件结构
            organize_encrypted_structure()

            print(f"\n📊 共收集 {file_count} 个加密文件")

            # 清理临时文件（可选）
            cleanup_choice = input("\n是否清理各个模块的build临时目录? (y/n): ").lower()
            if cleanup_choice == 'y':
                cleanup_module_build_dirs()

            print("\n🎊 批量加密完成！")
            print("📁 加密后的文件位于: encrypted/ 目录")
            print("💡 启动应用时请使用 encrypted/ 目录中的文件")
            print("📋 目录结构:")
            print("encrypted/")
            print("├── services/     # 加密后的服务模块")
            print("├── schemas/      # 未加密的数据模型模块")
            print("├── utils/        # 未加密的工具模块")
            print("├── routers/      # 未加密的路由模块")
            print("├── app.py        # 启动文件")
            print("├── config.py     # 配置文件")
            print("└── weights/      # 权重文件")
        else:
            print("❌ 未找到任何加密文件，请检查加密过程")
    else:
        print("❌ 加密失败，请检查错误信息")


if __name__ == "__main__":
    main()
