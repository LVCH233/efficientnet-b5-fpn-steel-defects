"""组内统一评价规则的评价函数(规则说明见《组内统一评价规则.md》)。只依赖 numpy 和 opencv-python。

规则:
1. 人工缺陷:每类掩码的一个 8 连通区域算一个(不膨胀、不合并)。
2. 误报:预测连通块(每类分别取)中,落在"任一人工缺陷外扩 tol 像素(默认 15)"范围内的面积 < inside(默认 50%)的块。
3. 检出:人工缺陷的面积被非误报的预测(任意类别)覆盖 >= cov(默认 30%)。一个预测块可以同时让多个人工缺陷检出。
4. 分类正确:检出的人工缺陷上,覆盖像素最多的预测类别 = 人工类别。
5. 算法识别出的缺陷数 = 检出数 + 误报块数。
公式:检出率 = 检出数/人工缺陷数;误报率 = 误报块数/人工缺陷数;准确率 = 分类正确数/算法识别出的缺陷数。
评价时不额外删除小块(min_area = 0)。"""
import numpy as np, cv2

COV, INSIDE, TOL, MIN_AREA = 0.3, 0.5, 15, 0


def _gt_labels(gt4):
    """(4,H,W) -> 标签图(0=背景,1..N=人工缺陷)、每个缺陷的类别和面积。各类标注互不重叠(已核对)。"""
    lab = np.zeros(gt4.shape[1:], np.int32); cls, area = [0], [0]; nxt = 1
    for c in range(4):
        if not gt4[c].any(): continue
        n, l, st, _ = cv2.connectedComponentsWithStats(gt4[c].astype(np.uint8), connectivity=8)
        for j in range(1, n):
            lab[l == j] = nxt; cls.append(c + 1); area.append(int(st[j, 4])); nxt += 1
    return lab, np.array(cls), np.array(area)


def eval_image(gt4, pr4, cov=COV, inside=INSIDE, tol=TOL, min_area=MIN_AREA):
    """返回 dict:n_gt, found, correct, fp,以及按人工类别的 gt/found 计数、按预测类别的 fp 计数。"""
    lab, gcls, garea = _gt_labels(gt4); ng = len(gcls) - 1
    K = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    U = cv2.dilate((lab > 0).astype(np.uint8), K).astype(bool)
    cover = np.zeros((ng + 1, 5))          # 每个人工缺陷被各预测类别覆盖的像素数
    anyp = np.zeros(lab.shape, bool); fp = 0; fp_c = np.zeros(5, int)
    for c in range(4):
        m = pr4[c].astype(np.uint8)
        if not m.any(): continue
        n, l, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        keep = np.zeros(n, bool)
        for j in range(1, n):
            if st[j, 4] < min_area: continue
            comp = l == j
            if U[comp].mean() < inside: fp += 1; fp_c[c + 1] += 1
            else: keep[j] = True
        km = keep[l]
        if km.any():
            cover[:, c + 1] += np.bincount(lab[km], minlength=ng + 1)[:ng + 1]
            anyp |= km
    cov_any = np.bincount(lab[anyp], minlength=ng + 1)[:ng + 1]
    found = np.zeros(ng + 1, bool); found[1:] = cov_any[1:] >= cov * garea[1:]
    correct = int(sum(found[g] and cover[g].argmax() == gcls[g] for g in range(1, ng + 1)))
    pc_gt = np.bincount(gcls[1:], minlength=5); pc_found = np.bincount(gcls[1:][found[1:]], minlength=5)
    return dict(n_gt=ng, found=int(found.sum()), correct=correct, fp=fp, pc_gt=pc_gt, pc_found=pc_found, pc_fp=fp_c)


def evaluate_v2(pairs, covs=(0.1, 0.3, 0.5), **kw):
    """pairs: 迭代 (gt4, pred4)。返回 {cov: 指标}。cov 主口径 0.3,0.1/0.5 作敏感性对比。"""
    pairs = list(pairs) if not isinstance(pairs, list) else pairs
    res = {}
    for cv_ in covs:
        A = dict(n_gt=0, found=0, correct=0, fp=0, pc_gt=np.zeros(5, int), pc_found=np.zeros(5, int), pc_fp=np.zeros(5, int))
        for g, p in pairs:
            r = eval_image(g, p, cov=cv_, **kw)
            for k in A: A[k] = A[k] + r[k]
        ng, nid = A["n_gt"], A["found"] + A["fp"]
        res[cv_] = dict(n_images=len(pairs), n_gt=ng, n_identified=nid, found=A["found"], fp=A["fp"], correct_cls=A["correct"],
                        detect_rate=A["found"] / ng if ng else float("nan"), false_rate=A["fp"] / ng if ng else float("nan"),
                        accuracy=A["correct"] / nid if nid else float("nan"),
                        per_class={c: dict(n_gt=int(A["pc_gt"][c]), detect_rate=(A["pc_found"][c] / A["pc_gt"][c]) if A["pc_gt"][c] else float("nan"),
                                           fp=int(A["pc_fp"][c])) for c in range(1, 5)})
    return res


def format_report_v2(res):
    return "\n".join(f"新口径 覆盖≥{c}: 人工缺陷{r['n_gt']} 算法识别{r['n_identified']} | 检出率 {r['detect_rate']*100:.2f}% | "
                     f"误报率 {r['false_rate']*100:.2f}% | 准确率 {r['accuracy']*100:.2f}%" for c, r in res.items())
