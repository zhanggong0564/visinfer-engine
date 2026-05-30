'''
@Description : 线标检测接口的 HTTP 测试客户端
              对应路由：POST /api/v1/panel_label_detect
              使用：python scripts/panel_label_client.py --image <图片路径> --product-type 1017KM1_1
'''

import argparse
import json
import sys
import time
from pathlib import Path

import requests


def build_json_data(product: str, material_type: str, product_type: str, rule: str) -> str:
    """构造与 PanelLabelRequest 一致的 json_data 字符串"""
    payload = {
        "product": product,
        "type": material_type,
        "modelParams": {
            "product_type": product_type,
            "rule": rule,
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def post_panel_label_detect(
    url: str,
    image_path: Path,
    json_data: str,
    timeout: float,
) -> requests.Response:
    """以 multipart/form-data 发送线标检测请求"""
    with image_path.open("rb") as fp:
        files = {"file": (image_path.name, fp, "image/jpeg")}
        data = {"json_data": json_data}
        return requests.post(url, files=files, data=data, timeout=timeout)


def pretty_print(resp: requests.Response, elapsed: float) -> None:
    print(f"[HTTP {resp.status_code}]  耗时 {elapsed:.3f}s")
    try:
        body = resp.json()
        print(json.dumps(body, ensure_ascii=False, indent=2))
    except ValueError:
        print(resp.text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="线标检测接口客户端")
    parser.add_argument("--host", default="127.0.0.1", help="服务地址，默认 127.0.0.1")
    parser.add_argument("--port", type=int, default=3001, help="服务端口，默认 3001 (config.PORT)")
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="测试图片路径，例如 demo/data/test/IMG_20260311_190937_253.jpg",
    )
    parser.add_argument("--product", default="123", help="json_data.product")
    parser.add_argument("--type", dest="material_type", default="A0ST1919", help="json_data.type (物料号)")
    parser.add_argument("--product-type", default="1017KM1_1", help="modelParams.product_type")
    parser.add_argument(
        "--rule",
        default="all",
        choices=["front", "back", "all"],
        help="modelParams.rule，默认 all",
    )
    parser.add_argument("--timeout", type=float, default=60.0, help="请求超时秒数")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.image.is_file():
        print(f"图片不存在：{args.image}", file=sys.stderr)
        return 2

    url = f"http://{args.host}:{args.port}/api/v1/panel_label_detect"
    json_data = build_json_data(args.product, args.material_type, args.product_type, args.rule)

    print(f"POST  {url}")
    print(f"file       = {args.image}")
    print(f"json_data  = {json_data}")
    print("-" * 60)

    start = time.time()
    try:
        resp = post_panel_label_detect(url, args.image, json_data, args.timeout)
    except requests.RequestException as e:
        print(f"请求失败：{e}", file=sys.stderr)
        return 1

    pretty_print(resp, time.time() - start)
    return 0 if resp.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
