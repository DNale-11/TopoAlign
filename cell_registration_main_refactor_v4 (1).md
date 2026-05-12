# `cell_registration/main.py` 最终重构说明 v4（供 Codex 直接实现）

## 1. 重构目标

请把 `cell_registration/main.py` 重构为一条清晰、单主线的 landmark-based 细胞配准流程。最终目标是：

1. 每次运行时都执行细胞分割
2. 从分割结果中提取细胞特征
3. 基于坐标、形态学特征和局部 topology 找到所有可稳定确认的 landmark
4. 用全部最终 landmark 估计配准变换
5. 导出 landmark、匹配可视化、配准可视化和调试结果

这次重构后的主方法仍然是：

- 先找 landmark
- 再做配准

不是：

- patch matching
- spatial cluster matching
- ICP
- RANSAC landmark 拟合
- 全量细胞 strict full assignment

---

## 2. 关键原则

### 2.1 不新增分割缓存
本版**不做 segmentation cache**，不增加任何缓存机制，不保存也不读取 segmentation cache，不新增任何相关参数和函数。

### 2.2 不要求所有细胞都进入最终 landmark
- fixed / moving 的细胞数可以不同
- 但 landmark 必须是**一对一**
- 主目标是保留尽可能多的、稳定成立的 landmark 对

### 2.3 `orientation` 不参与形态学匹配评分
`orientation` 字段可以继续保留在 feature CSV 中导出，但：

- 不参与 `feature_score`
- 不参与 `candidate_score`
- 不作为 Stage 2 的筛选依据

可用于调试观察，但不是匹配特征。

---

## 3. 明确删除与保留

### 3.1 从主流程删除
从 `cell_registration/main.py` 主流程中删除以下逻辑和分支：

- patch 相关逻辑
- spatial cluster 相关逻辑
- global coarse registration
- RANSAC 变换估计
- ICP 分支
- residual prune 分支
- segmentation cache 相关设计、参数、函数和元数据逻辑

### 3.2 保留
保留以下能力：

- `segmentation-only`
- 图像输入主入口
- `position-weight`
- `allow-scale`
- `save-match-table`
- `save-match-overlay`
- `save-match-plot`
- `save-registration-overlay`
- `save-features-dir`
- napari 调试能力

---

## 4. 新主流程

## Stage 0：输入、分割与特征准备

### 输入模式
首版仍然保留图像输入模式：

```powershell
python -m cell_registration.main fixed.tif moving.tif
```

### 行为要求
- 每次运行都执行分割
- 不读取任何 segmentation cache
- 不写入任何 segmentation cache
- 从分割结果中直接提取 feature table

### feature 表要求
feature 表至少包含以下列：

- `cell_id`
- `centroid_x`
- `centroid_y`
- `area`
- `perimeter`
- `major_axis_length`
- `minor_axis_length`
- `eccentricity`

并派生：

- `shape_ratio = major_axis_length / max(minor_axis_length, eps)`

### 说明
- `orientation` 只导出，不参与匹配评分
- fixed / moving 细胞数可以不同
- 不要求所有细胞都进入最终 landmark

---

## Stage 1：宽松坐标候选生成

### 目标
这里只做**宽松候选生成**，不在这里过早砍掉真实匹配。

### 方法
对每个 fixed cell，在 moving cell 的 KD-tree 上做半径查询：

- `candidate_radius_px` 默认建议设为 `25`
- 可允许用户调节
- 合理范围建议 `20 ~ 30 px`

### 规则
- 每个 fixed cell 可以对应多个 moving 候选
- 这里不只保留最近邻
- 这里不做 patch
- 这里不做 cluster
- 这里只负责生成候选边

### 输出
候选表至少包含：

- `fixed_idx`
- `moving_idx`
- `fixed_id`
- `moving_id`
- `centroid_distance_px`

### 建议函数
- `build_coordinate_candidates(...)`

---

## Stage 2：形态学特征 + 位置联合评分

### 2.1 参与评分的形态学特征
只使用以下特征：

- `area`
- `perimeter`
- `shape_ratio`
- `eccentricity`

**不使用：**
- `orientation`

### 2.2 特征归一化
对 `area / perimeter / shape_ratio / eccentricity` 使用 fixed + moving 合并后的稳健尺度：

- `median`
- `MAD` 或 `IQR`

归一化差异定义为：

```text
normalized_diff = abs(x_fixed - x_moving) / max(scale, eps)
```

### 2.3 位置项
保留位置约束，定义为：

```text
position_score = centroid_distance_px / candidate_radius_px
```

### 2.4 分数定义
先定义纯形态学原始分数：

```text
feature_score_raw =
    w_area * area_diff_norm +
    w_perimeter * perimeter_diff_norm +
    w_shape * shape_diff_norm +
    w_ecc * eccentricity_diff_norm
```

再做全候选集稳健归一化：

```text
feature_score = feature_score_raw / max(median(feature_score_raw over all candidates), eps)
```

然后与位置项合并：

```text
candidate_score_raw = feature_score + position_weight * position_score
candidate_score = candidate_score_raw / max(median(candidate_score_raw over all candidates), eps)
```

### 2.5 过滤规则
只保留：

- `feature_score <= feature_score_threshold`
- `candidate_score <= candidate_score_threshold`

### 2.6 设计原则
- Stage 1 用宽松半径保留真实候选
- Stage 2 再用 `position-weight` 精细控制
- 这样避免 Stage 1 过早砍掉真实匹配
- `feature_score`、`candidate_score` 都做稳健归一化，便于阈值迁移

### 建议函数
- `compute_feature_normalization_stats(...)`
- `score_feature_candidates(...)`
- `filter_feature_candidates(...)`

### 输出新增列
候选表增加：

- `feature_score_raw`
- `feature_score`
- `position_score`
- `candidate_score_raw`
- `candidate_score`

---

## Stage 3：局部 Topology 验证

这是确认 landmark 真伪的核心步骤。

### 3.1 方法
对每个通过 Stage 2 的候选 `(fixed_i, moving_j)`：

1. 取 fixed_i 周围 `topology_radius_px` 邻域
2. 取 moving_j 周围 `topology_radius_px` 邻域
3. 构建局部拓扑描述子
4. 比较两侧描述子
5. 只保留 topology 足够相似的 pair

推荐默认：

- `topology_radius_px = 20`
- `topology_k = 5`

### 3.2 描述子
首版实现以下三项：

1. `count_diff_norm`
2. `knn_distance_l2`
3. `angle_hist_distance`

### 3.3 邻居数量差异
定义：

```text
count_diff_norm = abs(n_fixed - n_moving) / max(max(n_fixed, n_moving), 1)
```

要求：

- `topo_count_weight` 默认应小于 `topo_dist_weight`
- 邻居数量项不能权重过高
- 因为边界和稀疏区域中邻居数量非常不稳定

### 3.4 kNN 距离描述子必须做尺度归一化
不能直接比较原始距离向量。

正确做法：

1. 在各自邻域内提取最近 `k` 个邻居距离
2. 升序排列
3. 若不足 `k`，用 `topology_radius_px` 填充
4. 分别做局部尺度归一化，例如：

```text
d_norm = d / max(median(d), eps)
```

5. 再比较：

```text
knn_distance_l2 = ||d_fixed_norm - d_moving_norm||_2
```

### 3.5 角度直方图必须 rotation-invariant
不能直接比较两个 angle histogram。

正确做法：

1. 计算 anchor 到邻居的相对角度
2. 构建 circular histogram
3. 对 moving histogram 做循环平移
4. 取最小距离：

```text
angle_hist_distance = min_shift_distance(hist_fixed, hist_moving)
```

这样对轻微整体旋转更稳定。

### 3.6 边界细胞特殊处理
如果 anchor 到图像边界的最小距离小于 `topology_radius_px`，说明邻域天然不完整。

要求：

- 标记 `is_border_cell`
- 对 border cell 放宽 topology 约束

首版建议：

```text
effective_threshold =
    topology_score_threshold * border_topology_relax_factor
```

触发条件：
- fixed anchor 或 moving anchor 任一为 border cell 时，使用放宽阈值

### 3.7 topology 总分
先算原始分数：

```text
topology_score_raw =
    topo_count_weight * count_diff_norm +
    topo_dist_weight * knn_distance_l2 +
    topo_angle_weight * angle_hist_distance
```

再做稳健归一化：

```text
topology_score = topology_score_raw / max(median(topology_score_raw over all stage-3 candidates), eps)
```

### 3.8 过滤规则
只保留：

- `topology_score <= effective_threshold`

### 3.9 实现硬约束
实现时必须保证：

- 邻域中不包含 anchor 自己
- 邻域不足 `k` 时 descriptor 仍然可计算
- 不允许出现 NaN / Inf
- histogram 必须做归一化，例如：
  `hist = hist / max(sum(hist), eps)`

### 建议函数
- `extract_local_neighbors(...)`
- `compute_scale_normalized_knn_descriptor(...)`
- `compute_rotation_invariant_angle_hist(...)`
- `score_topology_candidates(...)`
- `validate_candidates_by_topology(...)`

### 输出新增列
validated candidate 表至少包含：

- `fixed_id`
- `moving_id`
- `candidate_score`
- `feature_score`
- `topology_score_raw`
- `topology_score`
- `count_diff_norm`
- `knn_distance_l2`
- `angle_hist_distance`
- `is_border_cell`

---

## Stage 4：最大基数优先的 landmark 一对一选择

这是本版最关键的求解逻辑。

### 4.1 明确禁止
不能再使用：

- Hungarian + dummy
- full assignment
- 强制所有细胞都匹配

因为这里的目标不是 full assignment，而是：

- 在所有通过 topology 验证的候选中
- 先保留尽可能多的 landmark
- 再在该最大数量下最小化总代价

### 4.2 正确目标
Stage 4 的目标是：

1. 先最大化最终 landmark 数量
2. 再在该最大数量约束下最小化总匹配代价

即：

- Maximum Cardinality Matching
- then Minimum Cost under fixed cardinality

### 4.3 两阶段求解

#### Phase 1：最大基数匹配
- 左侧节点：fixed cells
- 右侧节点：moving cells
- 边：所有通过 Stage 3 的 validated candidate pairs
- 只看边是否存在，不看 cost

输出：
- 最大可保留 landmark 数 `M`

#### Phase 2：固定基数 `M` 下最小化总代价
在真实匹配数必须等于 `M` 的前提下，再最小化总代价。

推荐 cost：

```text
landmark_cost_raw = candidate_score + topology_weight * topology_score
landmark_cost = landmark_cost_raw / max(median(landmark_cost_raw over validated edges), eps)
```

最终 landmark 必须满足：

- 一对一
- 数量达到最大基数
- 在最大基数下总代价最小

### 4.4 实现建议
优先实现为：

- `solve_max_cardinality_matching(...)`
- `solve_min_cost_matching_with_fixed_cardinality(...)`

可以采用最小费用最大流或等价实现，但**不要**再回退到 Hungarian + dummy 的近似写法。

### 建议函数
- `build_validated_landmark_graph(...)`
- `solve_max_cardinality_matching(...)`
- `solve_min_cost_matching_with_fixed_cardinality(...)`
- `select_final_landmarks(...)`

### 输出
最终 landmark 表至少包含：

- `fixed_id`
- `moving_id`
- `candidate_score`
- `topology_score`
- `landmark_cost`

---

## Stage 5：基于全部最终 landmark 的配准估计

### 5.1 方法
使用 Stage 4 得到的全部最终 landmark 估计变换：

- 默认 rigid
- 若 `--allow-scale=True`，则估计 similarity transform

### 5.2 明确删除
这里不再使用：

- RANSAC
- ICP
- global coarse registration
- residual prune

### 5.3 保留
继续复用现有 `registration.py` 中的变换估计思路，但输入应改为：

- 全部最终 landmark 表

### 5.4 进入拟合前的稳定性约束
必须在拟合前检查：

```text
len(final_landmarks) >= min_landmark_pairs
spread_x_std_px >= min_landmark_spread_x_px
spread_y_std_px >= min_landmark_spread_y_px
```

其中：

- `spread_x_std_px = std(landmark_fixed_x)`
- `spread_y_std_px = std(landmark_fixed_y)`

如果 landmark 在 x 或 y 上过于集中，则直接报错，不估计 transform。

### 5.5 拟合后质量输出
虽然不做 residual prune，但必须输出拟合后的 residual 统计用于 QA：

- residual_mean
- residual_median
- residual_p90
- residual_p99

这些只做导出和调试，不做二次剔除。

### 建议函数
- `estimate_transform_from_landmarks(...)`
- `check_landmark_spread(...)`
- `compute_registration_residuals(...)`

---

## Stage 6：导出与调试输出

### 正式输出
保留并导出：

- `round1_cells.csv`
- `round2_cells.csv`
- `landmark_matches.csv`
- `match_overlay.tif`
- `match_plot.tif`
- `registration_overlay.tif`

### 调试输出
建议导出：

- `candidate_counts.csv`
- `feature_rejections.csv`
- `topology_rejections.csv`
- `validated_candidates.csv`
- `final_landmarks.csv`
- `transform_residuals.csv`
- `transform_summary.json`

---

## 5. CLI 参数调整

### 5.1 明确保留
- `--position-weight`
- `--allow-scale`
- `--save-registration-overlay`
- `--save-match-table`
- `--save-match-overlay`
- `--save-match-plot`
- `--save-features-dir`
- `--napari`
- `--segmentation-only`

### 5.2 明确删除
删除以下旧参数：

- `--top-k`
- `--top-per-patch`
- `--use-spatial-clusters`
- `--n-clusters`
- `--use-topology-filtering`
- `--k-pos-nei`
- `--k-neighbor`
- `--tau-pos`
- `--tau-nei`
- `--tau-map`
- `--use-ransac-transform`
- `--ransac-max-trials`
- `--ransac-residual-threshold`
- `--use-icp`
- 所有 `--icp-*`
- 所有 segmentation cache 相关参数

### 5.3 新增参数
新增以下参数：

- `--candidate-radius-px`
- `--feature-score-threshold`
- `--candidate-score-threshold`
- `--topology-radius-px`
- `--topology-k`
- `--topology-score-threshold`
- `--topology-weight`
- `--area-weight`
- `--perimeter-weight`
- `--shape-weight`
- `--eccentricity-weight`
- `--topo-count-weight`
- `--topo-dist-weight`
- `--topo-angle-weight`
- `--angle-bins`
- `--border-topology-relax-factor`
- `--min-landmark-pairs`
- `--min-landmark-spread-x-px`
- `--min-landmark-spread-y-px`
- `--save-debug-dir`

---

## 6. `main.py` 函数组织建议

首版仍然只改 `cell_registration/main.py`，不新增缓存模块。

### 输入与特征准备
- `prepare_features_from_images(...)`

### Stage 1
- `build_coordinate_candidates(...)`

### Stage 2
- `compute_feature_normalization_stats(...)`
- `score_feature_candidates(...)`
- `filter_feature_candidates(...)`

### Stage 3
- `score_topology_candidates(...)`
- `validate_candidates_by_topology(...)`

### Stage 4
- `solve_max_cardinality_matching(...)`
- `solve_min_cost_matching_with_fixed_cardinality(...)`
- `select_final_landmarks(...)`

### Stage 5
- `estimate_transform_from_landmarks(...)`
- `check_landmark_spread(...)`
- `compute_registration_residuals(...)`

### 导出
- `export_debug_tables(...)`

---

## 7. 明确不再坚持的旧约束

本版明确删除以下错误前提：

- `fixed / moving 的细胞数必须相同`
- `所有细胞必须进入最终 landmark`
- `主目标是 strict full assignment`
- `可以用 Hungarian + dummy 近似代替最大基数优先匹配`
- `orientation 应纳入形态学评分`
- `需要 segmentation cache`

---

## 8. 实现时的硬性要求

Codex 在实现时必须遵守以下要求：

1. 不要引入 segmentation cache 的任何代码、参数、JSON 元数据或路径处理逻辑
2. `orientation` 必须保留在导出特征表中，但不得参与 `feature_score` 或 `candidate_score`
3. Stage 1 必须使用宽松坐标半径，默认建议 `25 px`
4. Stage 2 必须做稳健归一化
5. Stage 3 的 kNN 距离必须做局部尺度归一化
6. Stage 3 的 angle histogram 必须做 rotation-invariant circular shift
7. Stage 3 必须包含 border cell 特殊处理
8. Stage 4 不能使用 Hungarian + dummy 作为主求解方式
9. Stage 4 必须体现“先最大基数，再最小代价”
10. Stage 5 必须在拟合前检查 landmark spread
11. Stage 5 必须导出 residual 统计，但不做 residual prune
12. 不要恢复 patch / cluster / RANSAC / ICP / global coarse registration 分支

---

## 9. 完成后的验收标准

实现完成后，应满足以下验收标准：

1. `main.py` 主流程已经收敛为：
   - segmentation
   - feature extraction
   - coordinate candidate generation
   - feature + position scoring
   - topology validation
   - max-cardinality-first landmark selection
   - transform estimation
   - export

2. 没有任何 segmentation cache 逻辑残留
3. 没有任何 patch / cluster / RANSAC / ICP / global coarse registration 逻辑残留在主流程中
4. `orientation` 不参与形态学评分
5. fixed / moving 细胞数不同也能运行
6. 最终 landmark 是部分一对一集合，而不是 full assignment
7. 变换估计使用全部最终 landmark
8. 结果能导出：
   - features
   - landmark 表
   - match overlay
   - registration overlay
   - debug tables
   - residual summary

---

## 10. 推荐执行顺序

请按以下顺序修改代码：

1. 清理 `main.py` 的 import 和旧参数
2. 删除 segmentation cache 相关内容
3. 删除 patch / cluster / RANSAC / ICP / global coarse registration 分支
4. 保留 segmentation-only 和原始图像输入主入口
5. 实现宽松坐标候选生成
6. 实现形态学 + 位置联合评分（不含 orientation）
7. 实现 topology 验证
8. 实现最大基数优先的 landmark 一对一选择
9. 用全部最终 landmark 估计 transform
10. 保留 overlay / plot / debug 导出
11. 更新 README 中与 CLI 和流程相关的说明
