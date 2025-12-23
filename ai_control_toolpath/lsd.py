import cv2
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os
import matplotlib.pyplot as plt

# 隐藏窗口
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

gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
height, width = gray.shape

# === 1. 二值化：区分"点亮"和"未点亮" ===
# 根据你的图：点亮=高亮（>200），未点亮=中等亮度（100~200）
_, binary_lit = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)      # 点亮区域
_, binary_unlit_full = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)  # <=100 的暗区
# 但"未点亮"实际是 100~200，所以：
binary_unlit = cv2.bitwise_and(cv2.bitwise_not(binary_lit), binary_unlit_full)

# === 2. LSD 检测线段 ===
# 注意：OpenCV 的 createLineSegmentDetector 参数是 _refine
lsd = cv2.createLineSegmentDetector(refine=cv2.LSD_REFINE_ADV)
all_lines, _, _, _ = lsd.detect(gray)

if all_lines is None or len(all_lines) == 0:
    print("未检测到任何线段。")
    exit()

# === 3. 按行分组线段 ===
row_height = 25  # 可调！根据你的图调整（你图中约 25px/行）
rows = {}
for line in all_lines:
    x1, y1, x2, y2 = line[0]
    mid_y = (y1 + y2) / 2
    row_id = int(mid_y // row_height)
    if row_id not in rows:
        rows[row_id] = []
    rows[row_id].append((x1, y1, x2, y2))

# === 4. 从下往上找"当前行" ===
# 获取所有行ID并按从小到大排序（从下往上）
sorted_row_ids = sorted(rows.keys())
current_row_info = None

for row_id in sorted_row_ids:
    y_min = row_id * row_height
    y_max = (row_id + 1) * row_height

    has_lit = False
    has_unlit = False
    lit_side = None

    for (x1, y1, x2, y2) in rows[row_id]:
        # 创建线段掩码
        seg_mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.line(seg_mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, thickness=2)

        # 计算该线段在点亮/未点亮区域中的覆盖率
        total_pixels = cv2.countNonZero(seg_mask)
        if total_pixels == 0:
            continue

        lit_pixels = cv2.countNonZero(cv2.bitwise_and(seg_mask, binary_lit))
        unlit_pixels = cv2.countNonZero(cv2.bitwise_and(seg_mask, binary_unlit))

        lit_ratio = lit_pixels / total_pixels
        unlit_ratio = unlit_pixels / total_pixels

        if lit_ratio > 0.3:   # 超过30%为点亮
            has_lit = True
            mid_x = (x1 + x2) / 2
            if mid_x < width / 2:
                lit_side = 'left'
            else:
                lit_side = 'right'
        if unlit_ratio > 0.3: # 超过30%为未点亮
            has_unlit = True

    if has_lit and has_unlit:
        current_row_info = {
            'row_id': row_id,
            'y_range': (y_min, y_max),
            'lit_side': lit_side
        }
        break

# === 5. 如果没找到"部分点亮"行，尝试找最下面的未点亮行（初始状态）===
if current_row_info is None:
    # 检查是否全完成
    total_lit_area = cv2.countNonZero(binary_lit)
    if total_lit_area / (height * width) > 0.9:
        print("\n✅ 刀路已完成：所有区域均已点亮。")
    else:
        # 找最下面的有线段的行（作为起始行）
        if sorted_row_ids:
            bottom_row_id = sorted_row_ids[0]  # 最小 row_id（最下方）
            y_min = bottom_row_id * row_height
            y_max = (bottom_row_id + 1) * row_height
            current_row_info = {
                'row_id': bottom_row_id,
                'y_range': (y_min, y_max),
                'lit_side': None  # 全未点亮，假设从左开始
            }
            print("\nℹ️  刀路尚未开始，将从最下行左侧开始。")

# === 6. 输出结果 ===
if current_row_info:
    y_min, y_max = current_row_info['y_range']
    lit_side = current_row_info['lit_side']

    if lit_side is None:
        # 全未点亮：默认从左往右
        print("\n✅ 当前刀路状态分析:")
        print(f"- 当前行 Y 范围: {y_min:.1f} ~ {y_max:.1f}")
        print("- 当前行状态: 全未点亮（起始状态）")
        print("- 下一刀方向: 向右切削（从左侧开始）")
    else:
        side_str = "左边" if lit_side == 'left' else "右边"
        print("\n✅ 当前刀路状态分析:")
        print(f"- 当前行 Y 范围: {y_min:.1f} ~ {y_max:.1f}")
        print(f"- 当前行状态: 既有已点亮线段，也有未点亮线段")
        print(f"- 已点亮侧: {side_str}")
        
        if lit_side == 'left':
            print("- 下一刀方向: 向右切削")
        else:
            print("- 下一刀方向: 向左切削")

        if y_min < height * 0.05:  # 修改条件：如果当前行靠近底部，则提示即将开始
            print("⚠️ 注意：当前行靠近图像底部，可能即将开始或需换向。")

# === 7. 可视化 - 显示带坐标的图像（当前行用蓝色框标注）===
def draw_result_image():
    img = src.copy()
    
    # 绘制所有检测线段（浅绿）
    for line in all_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 150), 1)
    
    if current_row_info:
        y_min, y_max = current_row_info['y_range']
        # 标记当前行区域（改为蓝色框，线宽为3）
        cv2.rectangle(img, (0, int(y_min)), (width, int(y_max)), (255, 0, 0), 3)  # 蓝色，更粗的线
        
        # 绘制当前行内线段状态
        row_id = current_row_info['row_id']
        if row_id in rows:
            for (x1, y1, x2, y2) in rows[row_id]:
                mid_y = (y1 + y2) / 2
                if y_min <= mid_y <= y_max:
                    seg_mask = np.zeros_like(gray, dtype=np.uint8)
                    cv2.line(seg_mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, 2)
                    lit_px = cv2.countNonZero(cv2.bitwise_and(seg_mask, binary_lit))
                    unlit_px = cv2.countNonZero(cv2.bitwise_and(seg_mask, binary_unlit))
                    
                    if lit_px > unlit_px:
                        color = (0, 255, 0)   # 绿：点亮
                    else:
                        color = (0, 100, 255) # 橙：未点亮
                    cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        # 标注文字
        text = f"当前行: {'左' if current_row_info['lit_side'] == 'left' else '右'}已点亮" if current_row_info['lit_side'] else "起始行"
        cv2.putText(img, text, (10, int(y_min) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return img

result_img = draw_result_image()

# 显示带正确坐标轴的图像（Y轴向下为正，与像素坐标一致）
fig, ax = plt.subplots(figsize=(14, 8))
ax.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), origin='upper')  # 关键：设置origin='upper'
ax.set_title('刀路状态分析结果')
ax.set_xlabel('X 坐标 (像素)')
ax.set_ylabel('Y 坐标 (像素)')

# 设置刻度标记，每50像素显示一个刻度
x_ticks = range(0, width + 1, max(1, width // 10))
y_ticks = range(0, height + 1, max(1, height // 10))
ax.set_xticks(x_ticks)
ax.set_yticks(y_ticks)

# 可选：添加网格线
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.show()

# === 8. 调试信息 ===
print(f"\n📊 调试信息:")
print(f"- 图像尺寸: {width} x {height}")
print(f"- 检测线段数: {len(all_lines)}")
print(f"- 分组行数: {len(rows)}")
print(f"- 行高度: {row_height} 像素")