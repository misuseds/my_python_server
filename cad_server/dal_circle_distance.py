from pyautocad import Autocad, APoint
import math
from collections import defaultdict

# 初始化 AutoCAD 连接
acad = Autocad(create_if_not_exists=True)
acad.prompt("正在处理圆对象...\n")

# 获取所有圆
circles = []
for obj in acad.iter_objects("Circle"):
    center = obj.Center
    circles.append((center[0], center[1], obj))

if not circles:
    acad.prompt("未找到任何圆！\n")
    exit()

# 提取圆心坐标
centers = [(x, y) for x, y, _ in circles]

# 判断对称轴方向：比较 x 和 y 的分布离散程度
xs = [x for x, y in centers]
ys = [y for x, y in centers]

x_range = max(xs) - min(xs)
y_range = max(ys) - min(ys)

TOLERANCE = 1.0  # 距离容差，单位根据图纸调整

if x_range < y_range:
    # 圆主要沿 Y 方向分布 → 对称轴为水平线 y = const
    symmetry_axis_value = sum(ys) / len(ys)  # 或用 median
    axis_type = 'horizontal'  # 对称轴是 y = symmetry_axis_value
else:
    # 圆主要沿 X 方向分布 → 对称轴为竖直线 x = const
    symmetry_axis_value = sum(xs) / len(xs)
    axis_type = 'vertical'  # 对称轴是 x = symmetry_axis_value

acad.prompt(f"检测到对称轴类型: {axis_type}, 值: {symmetry_axis_value:.3f}\n")

# 构建圆心集合用于快速查找（四舍五入到容差）
def round_point(pt, tol=0.1):
    return (round(pt[0] / tol) * tol, round(pt[1] / tol) * tol)

center_set = set(round_point((x, y), TOLERANCE/2) for x, y in centers)

# 找出未对称的圆
asymmetric_circles = []

for x, y, obj in circles:
    if axis_type == 'vertical':
        mirror_x = 2 * symmetry_axis_value - x
        mirror_pt = (mirror_x, y)
    else:  # horizontal
        mirror_y = 2 * symmetry_axis_value - y
        mirror_pt = (x, mirror_y)

    if round_point(mirror_pt, TOLERANCE/2) not in center_set:
        asymmetric_circles.append((x, y, obj))

acad.prompt(f"发现 {len(asymmetric_circles)} 个未对称的圆。\n")

# 标注未对称圆之间的孔间距（简化：只标注每对一次，避免重复）
annotated_pairs = set()
for i in range(len(asymmetric_circles)):
    x1, y1, obj1 = asymmetric_circles[i]
    for j in range(i + 1, len(asymmetric_circles)):
        x2, y2, obj2 = asymmetric_circles[j]
        dist = math.hypot(x2 - x1, y2 - y1)
        if dist < 1e-6:
            continue
        # 避免重复标注
        pair_key = tuple(sorted([(x1, y1), (x2, y2)]))
        if pair_key in annotated_pairs:
            continue
        annotated_pairs.add(pair_key)

        # 添加对齐标注（两点之间）
        p1 = APoint(x1, y1)
        p2 = APoint(x2, y2)
        # 标注位置稍微偏移
        mid = APoint((x1 + x2) / 2, (y1 + y2) / 2)
        offset = APoint(-5, 5)  # 可根据需要调整
        text_pt = mid + offset
        try:
            acad.model.AddDimAligned(p1, p2, text_pt)
        except Exception as e:
            acad.prompt(f"标注失败: {e}\n")

acad.prompt("孔间距标注完成！\n")