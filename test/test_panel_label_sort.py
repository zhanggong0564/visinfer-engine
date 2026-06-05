"""panel_label 线标排序（按型号固定 sort_mode）单元测试。

回归点：识别正确但顺序错（斜排被误并成一行、二维多列布局、列内方向反常）。
直接按文件加载 ordering/product_type 两个纯模块，避免触发插件包 __init__
（其会 import services 等重依赖）。
"""
import importlib.util
import os
import random

import pytest

_PLUGIN_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "plugins", "vie-plugin-panel-label", "vie_plugin_panel_label",
)


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_PLUGIN_DIR, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ordering = _load("pl_ordering", "ordering.py")
_pt = _load("pl_pt", "product_type.py")
compute_order = _ordering.compute_order
get_sort_mode = _pt.get_sort_mode


def _box(x, y, w=30, h=20):
    """以 (x,y) 为中心造一个展平的四点框 [x1,y1,...x4,y4]。"""
    return [x - w / 2, y - h / 2, x + w / 2, y - h / 2, x + w / 2, y + h / 2, x - w / 2, y + h / 2]


def _order_names(named_pts, sort_mode):
    """named_pts: [(name, x, y), ...]，返回按 sort_mode 排序后的名字序。"""
    names = [n for n, _, _ in named_pts]
    pts = [_box(x, y) for _, x, y in named_pts]
    perm = compute_order(pts, sort_mode)
    return [names[i] for i in perm]


def test_kvm_diagonal_real_coords_default_linear():
    """KVM 扇形斜排（log 真实坐标）：默认 linear 按上→下排成 [4,2,1]。"""
    pts = [
        [0.3273, 0.52275, 0.36433, 0.50475, 0.53533, 0.703, 0.498, 0.721],   # K2-1
        [0.35033, 0.51875, 0.388, 0.4955, 0.59866, 0.6865, 0.56066, 0.70975],  # K2-2
        [0.37066, 0.4975, 0.38666, 0.4645, 0.647, 0.536, 0.63066, 0.569],    # K2-4
    ]
    names = ["K2-1", "K2-2", "K2-4"]
    assert get_sort_mode("KVM") == "linear"
    perm = compute_order(pts, get_sort_mode("KVM"))
    assert [names[i] for i in perm] == ["K2-4", "K2-2", "K2-1"]


def test_qf2_columns_rowrev():
    """QF2：columns:2:rowrev —— 左列先、列内下→上；输入打乱也应稳定还原。"""
    items = [(f"{c}{y}", x, y) for c, x in [("L", 100), ("R", 400)] for y in range(100, 700, 100)]
    random.seed(1)
    random.shuffle(items)
    assert get_sort_mode("QF2") == "columns:2:rowrev"
    assert _order_names(items, get_sort_mode("QF2")) == [
        "L600", "L500", "L400", "L300", "L200", "L100",
        "R600", "R500", "R400", "R300", "R200", "R100",
    ]


def test_d1_columns_top_to_bottom():
    """D1 二维散布：columns:2 —— 左列(D1+/D1-) 再 中列(D1-1/D1-3)，列内上→下。"""
    d1 = [("D1-1", 0.48, 0.36), ("D1-3", 0.50, 0.78), ("D1+", 0.19, 0.55), ("D1-", 0.18, 0.66)]
    assert get_sort_mode("D1") == "columns:2"
    assert _order_names(d1, get_sort_mode("D1")) == ["D1+", "D1-", "D1-1", "D1-3"]


def test_columns_colrev_right_first():
    """columns:2:colrev —— 右列先、列内上→下。"""
    items = [(f"{c}{y}", x, y) for c, x in [("L", 100), ("R", 400)] for y in (100, 200, 300)]
    assert _order_names(items, "columns:2:colrev") == ["R100", "R200", "R300", "L100", "L200", "L300"]


def test_default_linear_single_row_left_to_right():
    """未登记型号走默认 linear：单行按主轴(偏水平)左→右。"""
    row = [(f"x{x}", x, 300) for x in (500, 100, 300, 200, 400)]
    assert get_sort_mode("UNKNOWN") == "linear"
    assert _order_names(row, get_sort_mode("UNKNOWN")) == ["x100", "x200", "x300", "x400", "x500"]


def test_linear_single_column_top_to_bottom():
    """竖直单列 linear：主轴偏竖直，按上→下。"""
    col = [(f"y{y}", 200, y) for y in (400, 100, 300, 200)]
    assert _order_names(col, "linear") == ["y100", "y200", "y300", "y400"]


def test_linear_rev():
    """linear:rev 整体反向。"""
    row = [(f"x{x}", x, 300) for x in (300, 100, 200)]
    assert _order_names(row, "linear:rev") == ["x300", "x200", "x100"]


@pytest.mark.parametrize("pts", [[], [_box(100, 100)]])
def test_degenerate_sizes(pts):
    """空 / 单元素不报错，原样返回。"""
    assert compute_order(pts, "columns:2:rowrev") == list(range(len(pts)))
