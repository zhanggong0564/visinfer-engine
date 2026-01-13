class BoundingBox:
    """表示一个边界框，支持多种格式、坐标系统和包含关系判断"""

    def __init__(self, x1, y1, x2, y2, format='xyxy', normalized=False, img_width=None, img_height=None):
        """
        初始化边界框

        参数:
        x1, y1, x2, y2: 边界框坐标，具体含义取决于format参数
        format: 坐标格式，支持 'xyxy' (左上右下)、'xywh' (左上宽高) 和 'cxcywh' (中心宽高)
        normalized: 是否为归一化坐标（0-1范围）
        img_width, img_height: 图像宽高，仅当normalized为True时需要提供
        """
        self.normalized = normalized

        # 如果是归一化坐标，转换为绝对坐标
        if normalized:
            if img_width is None or img_height is None:
                raise ValueError("当使用归一化坐标时，必须提供图像宽高")
            x1, x2 = x1 * img_width, x2 * img_width
            y1, y2 = y1 * img_height, y2 * img_height

        # 根据格式处理坐标
        if format == 'xyxy':
            self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        elif format == 'xywh':
            self.x1, self.y1 = x1, y1
            self.x2, self.y2 = x1 + x2, y1 + y2
        elif format == 'cxcywh':
            cx, cy, w, h = x1, y1, x2, y2
            self.x1, self.y1 = cx - w / 2, cy - h / 2
            self.x2, self.y2 = cx + w / 2, cy + h / 2
        else:
            raise ValueError(f"不支持的格式: {format}")

        # 确保坐标正确（x1 <= x2, y1 <= y2）
        self.x1, self.x2 = min(self.x1, self.x2), max(self.x1, self.x2)
        self.y1, self.y2 = min(self.y1, self.y2), max(self.y1, self.y2)

    def __repr__(self):
        return f"BoundingBox(x1={self.x1:.2f}, y1={self.y1:.2f}, x2={self.x2:.2f}, y2={self.y2:.2f})"

    def area(self):
        """计算边界框的面积"""
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def contains_strict(self, other_box):
        """
        判断当前边界框是否严格包含另一个边界框
        即另一个边界框的四个顶点都在当前边界框内
        """
        return (self.x1 <= other_box.x1 and
                self.y1 <= other_box.y1 and
                self.x2 >= other_box.x2 and
                self.y2 >= other_box.y2)

    def contains_loose(self, other_box, threshold=0.5):
        """
        判断当前边界框是否宽松包含另一个边界框
        即另一个边界框的面积的一定比例（threshold）在当前边界框内

        参数:
        threshold: 面积比例阈值，默认为0.5，表示至少50%的面积在内部
        """
        # 计算交集区域
        x_overlap = max(0, min(self.x2, other_box.x2) - max(self.x1, other_box.x1))
        y_overlap = max(0, min(self.y2, other_box.y2) - max(self.y1, other_box.y1))
        intersection_area = x_overlap * y_overlap

        # 计算交集面积占other_box面积的比例
        containment_ratio = intersection_area / other_box.area()

        return containment_ratio >= threshold

    def contains_center(self, other_box):
        """
        判断当前边界框是否包含另一个边界框的中心点
        """
        other_center_x = (other_box.x1 + other_box.x2) / 2
        other_center_y = (other_box.y1 + other_box.y2) / 2

        return (self.x1 <= other_center_x <= self.x2 and
                self.y1 <= other_center_y <= self.y2)

    def is_contained_by(self, other_box, method='strict', threshold=0.5):
        """
        判断当前边界框是否被另一个边界框包含

        参数:
        other_box: 另一个边界框对象
        method: 包含判断方法，可选 'strict'（严格包含）、'loose'（宽松包含）、'center'（中心点包含）
        threshold: 宽松包含的阈值，仅在method='loose'时有效
        """
        if method == 'strict':
            return other_box.contains_strict(self)
        elif method == 'loose':
            return other_box.contains_loose(self, threshold)
        elif method == 'center':
            return other_box.contains_center(self)
        else:
            raise ValueError(f"不支持的判断方法: {method}")


def check_box_containment(box1, box2, method='strict', threshold=0.5, img_width=None, img_height=None):
    """
    检查两个边界框之间的包含关系

    参数:
    box1, box2: 两个边界框，可以是四元组 (x1, y1, x2, y2) 或 BoundingBox 对象
    method: 包含判断方法，可选 'strict'（严格包含）、'loose'（宽松包含）、'center'（中心点包含）
    threshold: 宽松包含的阈值，仅在method='loose'时有效
    img_width, img_height: 图像宽高，如果边界框是归一化的，则必须提供

    返回:
    0: 无包含关系
    1: box1 包含 box2
    2: box2 包含 box1
    3: 互相包含（通常意味着两个框相同）
    """
    # 如果传入的是元组，转换为BoundingBox对象
    if not isinstance(box1, BoundingBox):
        # 尝试判断是否为归一化坐标
        normalized = all(0 <= coord <= 1 for coord in box1)
        box1 = BoundingBox(*box1, normalized=normalized, img_width=img_width, img_height=img_height)

    if not isinstance(box2, BoundingBox):
        normalized = all(0 <= coord <= 1 for coord in box2)
        box2 = BoundingBox(*box2, normalized=normalized, img_width=img_width, img_height=img_height)

    # 根据指定方法检查包含关系
    box1_contains_box2 = box1.is_contained_by(box2, method, threshold)
    box2_contains_box1 = box2.is_contained_by(box1, method, threshold)

    if box1_contains_box2 and box2_contains_box1:
        return 3  # 互相包含
    elif box1_contains_box2:
        return 2  # box2 包含 box1
    elif box2_contains_box1:
        return 1  # box1 包含 box2
    else:
        return 0  # 无包含关系


def deduplicate_lists(lst):
    """
    对包含子列表的列表进行去重，保留首次出现的子列表

    参数:
    lst: 包含子列表的列表（如 [[1,2], [3,4], [1,2]]）

    返回:
    去重后的新列表（如 [[1,2], [3,4]]）
    """
    # 1. 将每个子列表转换为元组（可哈希）
    tuple_list = [tuple(sublist) for sublist in lst]
    # 2. 利用集合去重（集合自动去重且保留插入顺序*）
    #    注意：Python 3.7+集合会保留插入顺序
    unique_tuples = set(tuple_list)
    # 3. 转换回列表，并保持原始顺序（需遍历原列表）
    #    方法：按原顺序保留首次出现的元组
    deduplicated = []
    seen = set()
    for tpl in tuple_list:
        if tpl not in seen:
            seen.add(tpl)
            deduplicated.append(list(tpl))  # 转换回列表
    return deduplicated

# 使用示例
if __name__ == "__main__":
    # 图像尺寸
    img_width, img_height = 800, 600

    # 创建归一化的边界框
    box1_normalized = [0.1, 0.1, 0.9, 0.9]  # 较大的框（归一化）
    box2_normalized = [0.2, 0.2, 0.5, 0.5]  # 完全在box1内部的框（归一化）
    box3_normalized = [0.8, 0.8, 1.0, 1.0]  # 部分在box1外部的框（归一化）

    # 测试不同的包含关系（使用归一化坐标）
    print(
        f"box1 和 box2: {check_box_containment(box1_normalized, box2_normalized, img_width=img_width, img_height=img_height)}")  # 1: box1 包含 box2
    print(
        f"box1 和 box3: {check_box_containment(box1_normalized, box3_normalized, img_width=img_width, img_height=img_height)}")  # 0: 无包含关系

    # 使用宽松包含判断
    print(
        f"宽松判断 box1 和 box3: {check_box_containment(box1_normalized, box3_normalized, method='loose', threshold=0.3, img_width=img_width, img_height=img_height)}")  # 1: box1 宽松包含 box3

    # 混合使用归一化和绝对坐标
    box4_absolute = (400, 300, 700, 500)  # 绝对坐标的框
    print(
        f"box1 和 box4: {check_box_containment(box1_normalized, box4_absolute, img_width=img_width, img_height=img_height)}")  # 0: 无包含关系