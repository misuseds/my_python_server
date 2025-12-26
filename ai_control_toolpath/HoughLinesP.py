import cv2
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt

Tk().withdraw()

print("请选择一张刀路图（S形，从下往上走）...")
file_path = askopenfilename(
    title="选择刀路图像",
    filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
)

if not file_path:
    print("未选择图片，程序退出。")
    exit()

src = cv2.imread(file_path)
if src is None:
    print("无法读取图片。")
    exit()

height, width = src.shape[:2]

# === 1. 颜色二值化 ===
lower_cyan = np.array([200, 200, 0])
upper_cyan = np.array([255, 255, 50])
binary_lit = cv2.inRange(src, lower_cyan, upper_cyan)

lower_yellow = np.array([0, 200, 200])
upper_yellow = np.array([50, 255, 255])
binary_unlit = cv2.inRange(src, lower_yellow, upper_yellow)

# === 2. 合并所有刀路线（用于边缘检测）===
binary_all = cv2.bitwise_or(binary_lit, binary_unlit)

# 可选：轻微膨胀连接断点
kernel = np.ones((2, 2), np.uint8)
binary_all = cv2.dilate(binary_all, kernel, iterations=1)

# Canny 边缘检测（为霍夫变换准备）
edges = cv2.Canny(binary_all, 50, 150, apertureSize=3)

# === 3. 使用 HoughLinesP 检测线段（比 LSD 更可控）===
lines = cv2.HoughLinesP(
    edges,
    rho=1,
    theta=np.pi / 180,
    threshold=20,          # 累加器阈值，越低越敏感
    minLineLength=15,      # 最小线段长度（像素）
    maxLineGap=8           # 允许的最大间隙（用于连接短线）
)

if lines is None or len(lines) == 0:
    print("未检测到任何线段。")
    exit()

all_lines = lines  # shape: (N, 1, 4)

# === 4. 按行分组 ===
row_height = 25
rows = {}
for line in all_lines:
    x1, y1, x2, y2 = line[0]
    mid_y = (y1 + y2) / 2
    row_id = int(mid_y // row_height)
    if row_id not in rows:
        rows[row_id] = []
    rows[row_id].append((x1, y1, x2, y2))

# === 5-8. 保持原有逻辑不变（从下往上找当前行、分析、可视化）===
sorted_row_ids = sorted(rows.keys())
current_row_info = None

for row_id in sorted_row_ids:
    y_min = row_id * row_height
    y_max = (row_id + 1) * row_height

    has_lit = False
    has_unlit = False
    lit_side = None

    for (x1, y1, x2, y2) in rows[row_id]:
        seg_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.line(seg_mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, thickness=2)

        total = cv2.countNonZero(seg_mask)
        if total == 0:
            continue

        lit_px = cv2.countNonZero(cv2.bitwise_and(seg_mask, binary_lit))
        unlit_px = cv2.countNonZero(cv2.bitwise_and(seg_mask, binary_unlit))

        if lit_px / total > 0.3:
            has_lit = True
            lit_side = 'left' if (x1 + x2) / 2 < width / 2 else 'right'
        if unlit_px / total > 0.3:
            has_unlit = True

    if has_lit and has_unlit:
        current_row_info = {'row_id': row_id, 'y_range': (y_min, y_max), 'lit_side': lit_side}
        break

if current_row_info is None:
    total_lit = cv2.countNonZero(binary_lit)
    if total_lit / (height * width) > 0.9:
        print("\n✅ 刀路已完成。")
    elif sorted_row_ids:
        bottom_row_id = sorted_row_ids[0]
        y_min = bottom_row_id * row_height
        y_max = (bottom_row_id + 1) * row_height
        current_row_info = {'row_id': bottom_row_id, 'y_range': (y_min, y_max), 'lit_side': None}
        print("\nℹ️ 刀路尚未开始。")

# === 可视化 ===
fig, axes = plt.subplots(1, 3, figsize=(21, 6))

# 原图
axes[0].imshow(cv2.cvtColor(src, cv2.COLOR_BGR2RGB))
axes[0].set_title('原始刀路图')
axes[0].axis('off')

# 线段状态图
status_canvas = np.ones_like(src) * 255
for line in all_lines:
    x1, y1, x2, y2 = line[0]
    seg_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.line(seg_mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, 2)
    total = cv2.countNonZero(seg_mask)
    if total == 0:
        color = (150, 150, 150)
    else:
        lit_px = cv2.countNonZero(cv2.bitwise_and(seg_mask, binary_lit))
        unlit_px = cv2.countNonZero(cv2.bitwise_and(seg_mask, binary_unlit))
        if lit_px / total > 0.3:
            color = (0, 255, 0)
        elif unlit_px / total > 0.3:
            color = (0, 100, 255)
        else:
            color = (150, 150, 150)
    cv2.line(status_canvas, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

axes[1].imshow(cv2.cvtColor(status_canvas, cv2.COLOR_BGR2RGB))
axes[1].set_title('检测线段（绿=点亮，橙=未点亮）')
axes[1].axis('off')

# 分析图
result_img = src.copy()
if current_row_info:
    y_min, y_max = current_row_info['y_range']
    cv2.rectangle(result_img, (0, int(y_min)), (width, int(y_max)), (255, 0, 0), 3)
    row_id = current_row_info['row_id']
    if row_id in rows:
        for (x1, y1, x2, y2) in rows[row_id]:
            mid_y = (y1 + y2) / 2
            if y_min <= mid_y <= y_max:
                seg_mask = np.zeros((height, width), dtype=np.uint8)
                cv2.line(seg_mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, 2)
                lit_px = cv2.countNonZero(cv2.bitwise_and(seg_mask, binary_lit))
                unlit_px = cv2.countNonZero(cv2.bitwise_and(seg_mask, binary_unlit))
                color = (0, 255, 0) if lit_px > unlit_px else (0, 100, 255)
                cv2.line(result_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    text = f"当前行: {'左' if current_row_info['lit_side'] == 'left' else '右'}已点亮" \
           if current_row_info['lit_side'] else "起始行"
    cv2.putText(result_img, text, (10, int(y_min) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

axes[2].imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
axes[2].set_title('刀路状态分析（蓝框=当前行）')
axes[2].axis('off')

plt.tight_layout()
plt.show()

print(f"\n📊 调试信息:")
print(f"- 图像尺寸: {width} x {height}")
print(f"- 检测线段数: {len(all_lines)}")
print(f"- 分组行数: {len(rows)}")
print(f"- 行高度: {row_height} 像素")