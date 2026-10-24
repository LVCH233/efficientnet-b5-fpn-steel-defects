"""组内统一评价规则 —— 使用示例。只需 numpy、opencv-python(cv2)。
把本文件和 evaluate_v2.py 放在同一个文件夹,运行: python 评价示例.py
"""
import numpy as np
from evaluate_v2 import evaluate_v2

H, W = 256, 1600

def rle_decode(rle, h=H, w=W):
    """Kaggle Severstal 格式的 RLE(按列优先,1 起始)-> (h, w) 的 0/1 数组。空字符串返回全 0。"""
    m = np.zeros(h * w, np.uint8)
    if isinstance(rle, str) and rle.strip():
        s = list(map(int, rle.split()))
        for start, length in zip(s[0::2], s[1::2]):
            m[start - 1:start - 1 + length] = 1
    return m.reshape(w, h).T

def to_mask4(rles):
    """rles: {类别(1~4): RLE 字符串} -> (4, H, W) 的 0/1 数组"""
    m = np.zeros((4, H, W), np.uint8)
    for c, r in rles.items():
        m[c - 1] = rle_decode(r)
    return m

def score(pairs):
    """pairs: [(gt4, pred4), ...] -> 打印并返回三项指标(本组统一规则,覆盖 30%,评价时不删小块)"""
    r = evaluate_v2(pairs, covs=(0.3,), min_area=0)[0.3]
    print(f"人工缺陷 {r['n_gt']} 个 | 检出 {r['found']} 个 | 误报 {r['fp']} 块 | 分类正确 {r['correct_cls']} 个")
    print(f"检出率 {r['detect_rate']*100:.2f}%  误报率 {r['false_rate']*100:.2f}%  准确率 {r['accuracy']*100:.2f}%")
    return r

if __name__ == "__main__":
    z = np.zeros((4, H, W), np.uint8)
    g = z.copy(); g[0, 50:70, 100:180] = 1                      # 一个一类缺陷
    print("自检 1:完美预测(应为 100% / 0% / 100%)"); score([(g, g)])
    p = z.copy(); p[2, 50:70, 100:180] = 1
    print("自检 2:位置对、类别错(应检出 1、分类正确 0)"); score([(g, p)])
    print("自检 3:整张图四类报满(应检出 0、误报 4)"); score([(g, np.ones_like(z))])
    p = g.copy(); p[0, 50:70, 900:980] = 1
    print("自检 4:报对了,远处又多报一块(应检出 1、误报 1)"); score([(g, p)])
    # 实际使用:对每张图把标注和你的预测都转成 (4, 256, 1600) 的数组,放进列表后调用 score(pairs)
