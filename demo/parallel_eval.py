#!/usr/bin/env python3
"""Panel Label 并发评测脚本

采用多进程按型号分开测试，避免单进程长时间运行导致的性能退化。
支持配置并发 worker 数量、可视化模式、超时设置等。
"""
import os
import sys
import json
import time
import argparse
import subprocess
import multiprocessing
from pathlib import Path
from datetime import datetime
from tqdm import tqdm


def run_product_task(args):
    """执行单个型号的评测任务

    参数:
        args: (product_type, config) 元组

    返回:
        {
            "product": "型号名",
            "status": "success" | "timeout" | "error",
            "result": {解析的 JSON 结果} | None,
            "error": "错误信息" (仅 status=error 时),
            "duration": 耗时秒数
        }
    """
    product_type, config = args

    json_path = Path(config["log_dir"]) / f"{product_type}.json"
    log_path = Path(config["log_dir"]) / f"{product_type}.log"

    # 构造命令
    cmd = [
        "/data/zhanggong/miniconda3/envs/padocr/bin/python",
        "plugins/vie-plugin-panel-label/examples/run.py",
        "--batch", f"{config['test_dir']}/{product_type}",
        "--vis-dir", f"{config['vis_dir']}/{product_type}",
        "--vis-mode", config["vis_mode"],
        "--output-json", str(json_path),
        "--rule", config["rule"],
    ]

    # 环境变量
    env = os.environ.copy()
    env.update({
        "PANEL_LABEL_GUIDELINE_FILTER": "false",
        "QT_QPA_PLATFORM": "offscreen",
    })

    start_time = time.time()

    try:
        # 执行子进程，日志写入文件
        with open(log_path, "w") as log_file:
            result = subprocess.run(
                cmd,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=config["timeout"],
                cwd="/data/zhanggong/workspace/project/move_vsion/mobile_vision"
            )

        duration = time.time() - start_time

        # 检查退出码
        if result.returncode != 0:
            return {
                "product": product_type,
                "status": "error",
                "error": f"进程退出码 {result.returncode}",
                "duration": duration
            }

        # 读取 JSON 结果
        if not json_path.exists():
            return {
                "product": product_type,
                "status": "error",
                "error": f"JSON 结果文件不存在: {json_path}",
                "duration": duration
            }

        with open(json_path, encoding="utf-8") as f:
            json_result = json.load(f)

        return {
            "product": product_type,
            "status": "success",
            "result": json_result,
            "duration": duration
        }

    except subprocess.TimeoutExpired:
        return {
            "product": product_type,
            "status": "timeout",
            "duration": config["timeout"]
        }
    except Exception as e:
        return {
            "product": product_type,
            "status": "error",
            "error": str(e),
            "duration": time.time() - start_time
        }


def format_duration(seconds):
    """将秒数格式化为易读的时长字符串"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs:02d}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins:02d}m"


def generate_report(results, args, total_duration):
    """生成 Markdown 报告"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 统计数据
    success_count = sum(1 for r in results if r["status"] == "success")
    timeout_count = sum(1 for r in results if r["status"] == "timeout")
    error_count = sum(1 for r in results if r["status"] == "error")

    total_images = 0
    pass_images = 0

    for r in results:
        if r["status"] == "success" and r.get("result"):
            total_images += r["result"].get("total", 0)
            pass_images += r["result"].get("pass", 0)

    # 生成报告内容
    report_lines = [
        "# Panel Label 并发评测报告",
        "",
        f"> **评测时间**: {timestamp}",
        f"> **测试集**: {args.test_dir}",
        f"> **并发进程数**: {args.workers}",
        f"> **总耗时**: {format_duration(total_duration)}",
        "",
        "---",
        "",
        "## 总体统计",
        "",
        "| 指标 | 数值 |",
        "|------|------|",
        f"| 总型号数 | {len(results)} |",
        f"| 成功型号 | {success_count} |",
        f"| 超时型号 | {timeout_count} |",
        f"| 错误型号 | {error_count} |",
        f"| 总图片数 | {total_images} |",
        f"| 通过图片 | {pass_images} ({pass_images / max(total_images, 1):.2%}) |",
        f"| 失败图片 | {total_images - pass_images} ({(total_images - pass_images) / max(total_images, 1):.2%}) |",
        "",
        "---",
        "",
        "## 各型号详情",
        "",
        "| 型号 | 状态 | 通过/总数 | 通过率 | 耗时 | 详细日志 |",
        "|------|------|----------|--------|------|---------|",
    ]

    # 按型号名排序
    sorted_results = sorted(results, key=lambda r: r["product"])

    for r in sorted_results:
        product = r["product"]
        duration_str = format_duration(r["duration"])
        log_link = f"[{args.log_dir}/{product}.log]({args.log_dir}/{product}.log)"

        if r["status"] == "success" and r.get("result"):
            result_data = r["result"]
            pass_count = result_data.get("pass", 0)
            total_count = result_data.get("total", 0)
            rate = result_data.get("rate", 0.0)
            status_icon = "✓"
            report_lines.append(
                f"| {product} | {status_icon} | {pass_count}/{total_count} | {rate:.2%} | {duration_str} | {log_link} |"
            )
        elif r["status"] == "timeout":
            report_lines.append(
                f"| {product} | ⏱ TIMEOUT | - | - | {duration_str} | {log_link} |"
            )
        else:  # error
            report_lines.append(
                f"| {product} | ✗ ERROR | - | - | {duration_str} | {log_link} |"
            )

    report_lines.extend([
        "",
        "---",
        "",
        "## 异常型号",
        "",
    ])

    # 超时型号
    timeout_products = [r for r in sorted_results if r["status"] == "timeout"]
    if timeout_products:
        report_lines.append(f"### 超时型号 ({len(timeout_products)})")
        report_lines.append("")
        for r in timeout_products:
            report_lines.append(f"- **{r['product']}**: 超过 {args.timeout} 秒未完成")
        report_lines.append("")

    # 错误型号
    error_products = [r for r in sorted_results if r["status"] == "error"]
    if error_products:
        report_lines.append(f"### 错误型号 ({len(error_products)})")
        report_lines.append("")
        for r in error_products:
            error_msg = r.get("error", "未知错误")
            report_lines.append(f"- **{r['product']}**: {error_msg}")
        report_lines.append("")

    if not timeout_products and not error_products:
        report_lines.append("无异常型号")
        report_lines.append("")

    report_lines.extend([
        "---",
        "",
        "## 文件索引",
        "",
        "| 路径 | 说明 |",
        "|------|------|",
        f"| `{args.log_dir}/` | 各型号详细日志与 JSON 统计 |",
        f"| `{args.vis_dir}/` | 可视化结果（vis-mode={args.vis_mode}） |",
        f"| `{args.summary}` | JSON 格式汇总统计 |",
        "",
    ])

    # 写入报告文件
    with open(args.report, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))


def generate_summary_json(results, args, total_duration):
    """生成 JSON 汇总文件"""
    timestamp = datetime.now().isoformat()

    # 统计数据
    success_count = sum(1 for r in results if r["status"] == "success")
    timeout_count = sum(1 for r in results if r["status"] == "timeout")
    error_count = sum(1 for r in results if r["status"] == "error")

    total_images = 0
    pass_images = 0

    for r in results:
        if r["status"] == "success" and r.get("result"):
            total_images += r["result"].get("total", 0)
            pass_images += r["result"].get("pass", 0)

    # 构造 JSON 数据
    summary_data = {
        "meta": {
            "timestamp": timestamp,
            "test_dir": args.test_dir,
            "workers": args.workers,
            "vis_mode": args.vis_mode,
            "rule": args.rule,
            "timeout": args.timeout,
            "total_duration_seconds": total_duration
        },
        "summary": {
            "total_products": len(results),
            "success": success_count,
            "timeout": timeout_count,
            "error": error_count,
            "total_images": total_images,
            "pass_images": pass_images,
            "fail_images": total_images - pass_images,
            "pass_rate": pass_images / max(total_images, 1)
        },
        "products": []
    }

    # 添加各型号详情
    for r in sorted(results, key=lambda x: x["product"]):
        product_info = {
            "product": r["product"],
            "status": r["status"],
            "duration": r["duration"],
            "log": f"{args.log_dir}/{r['product']}.log"
        }

        if r["status"] == "success" and r.get("result"):
            result_data = r["result"]
            product_info.update({
                "pass": result_data.get("pass", 0),
                "total": result_data.get("total", 0),
                "rate": result_data.get("rate", 0.0),
                "json": f"{args.log_dir}/{r['product']}.json"
            })
        elif r["status"] == "error":
            product_info["error"] = r.get("error", "未知错误")

        summary_data["products"].append(product_info)

    # 写入 JSON 文件
    with open(args.summary, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)


def print_summary(results, total_duration):
    """打印终端摘要"""
    success_count = sum(1 for r in results if r["status"] == "success")
    timeout_count = sum(1 for r in results if r["status"] == "timeout")
    error_count = sum(1 for r in results if r["status"] == "error")

    total_images = 0
    pass_images = 0

    for r in results:
        if r["status"] == "success" and r.get("result"):
            total_images += r["result"].get("total", 0)
            pass_images += r["result"].get("pass", 0)

    print("\n" + "=" * 40)
    print("评测完成")
    print("=" * 40)
    print(f"总耗时: {format_duration(total_duration)}")
    print(f"成功: {success_count}  超时: {timeout_count}  错误: {error_count}")
    if total_images > 0:
        print(f"总通过率: {pass_images}/{total_images} ({pass_images / total_images:.2%})")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Panel Label 并发评测脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--test-dir", default="demo/test_set", help="测试集根目录")
    parser.add_argument("--vis-dir", default="demo/test_set_vis_parallel", help="可视化输出根目录")
    parser.add_argument("--log-dir", default="demo/logs", help="日志输出目录")
    parser.add_argument("--workers", type=int, default=2, help="并发进程数")
    parser.add_argument("--timeout", type=int, default=1800, help="单型号超时秒数")
    parser.add_argument("--vis-mode", choices=["all", "failed", "none"], default="failed", help="可视化模式")
    parser.add_argument("--rule", default="all", choices=["front", "back", "all"], help="字符比较规则")
    parser.add_argument("--report", default="demo/parallel_eval_report.md", help="Markdown 报告输出路径")
    parser.add_argument("--summary", default="demo/parallel_eval_summary.json", help="JSON 汇总输出路径")

    args = parser.parse_args()

    # 创建输出目录
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    if args.vis_mode != "none":
        Path(args.vis_dir).mkdir(parents=True, exist_ok=True)

    # 扫描测试集目录
    test_dir = Path(args.test_dir)
    if not test_dir.exists():
        print(f"错误: 测试集目录不存在: {test_dir}", file=sys.stderr)
        sys.exit(1)

    product_types = sorted([
        d.name for d in test_dir.iterdir()
        if d.is_dir() and list(d.glob("*.jpg"))
    ])

    if not product_types:
        print(f"错误: 未在 {test_dir} 下发现包含 *.jpg 的型号子目录", file=sys.stderr)
        sys.exit(1)

    print(f"扫描到 {len(product_types)} 个型号")
    print(f"创建 {args.workers} 个并发进程")
    print()

    # 构造任务配置
    config = {
        "test_dir": args.test_dir,
        "vis_dir": args.vis_dir,
        "log_dir": args.log_dir,
        "vis_mode": args.vis_mode,
        "rule": args.rule,
        "timeout": args.timeout,
    }

    tasks = [(pt, config) for pt in product_types]

    # 并发执行
    start_time = time.time()
    with multiprocessing.Pool(args.workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(run_product_task, tasks),
            total=len(tasks),
            desc="评测进度",
            unit="型号"
        ))
    total_duration = time.time() - start_time

    # 生成报告
    generate_report(results, args, total_duration)
    generate_summary_json(results, args, total_duration)

    # 输出摘要
    print_summary(results, total_duration)

    print("报告已生成:")
    print(f"  - {args.report}")
    print(f"  - {args.summary}")


if __name__ == "__main__":
    main()
