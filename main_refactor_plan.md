# `cell_registration/main.py` 重构方案审阅稿 v3

## 1. 这次重构的真实目标

把 [`cell_registration/main.py`](f:\programme\nxy\Cell registration\cell_registration\main.py) 改成一条清晰的主流程：

1. 首次运行时做细胞分割
2. 保存 mask、feature CSV、分割参数和缓存元数据
3. 后续只调配准参数时，直接读取缓存 mask / features，不重复跑分割
4. 基于坐标、形态学特征、局部 topology 找到所有可稳定确认的 landmark
5. 用全部最终 landmark 做配准变换估计
6. 导出 landmark、overlay、registration overlay 和调试结果

这次重构后，主方法仍然是：

- 先找 landmark
- 再做配准

不是：

- patch matching
- ICP
- RANSAC landmark 拟合
- 全量细胞 strict full assignment

## 2. 明确删除和保留

### 删除

从 [`cell_registration/main.py`](f:\programme\nxy\Cell registration\cell_registration\main.py) 主流程删除：

- patch 相关逻辑
- spatial cluster 相关逻辑
- global coarse registration
- RANSAC 变换估计
- ICP 分支
- residual prune 分支

### 保留

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

## 3. 首次分割缓存机制

这是这次必须写进方案的部分。

### 设计目标

分割只做一次。之后如果只是改 landmark / topology / matching / transform 参数，就直接复用已有 mask 和 features。

### 缓存内容

对每张输入图像缓存：

- `mask.tif`
- `features.csv`
- `segmentation_meta.json`

`segmentation_meta.json` 至少记录：

- 原始图像路径
- 图像尺寸
- 图像最后修改时间
- Cellpose 参数
- 生成时间

### 缓存判定

默认行为：

- 如果缓存存在，且图像路径、尺寸、mtime、分割参数一致
- 则直接读取缓存，跳过分割

只有以下情况才重新分割：

- 没有缓存
- 显式指定 `--refresh-segmentation`
- 图像内容或尺寸变化
- 分割参数变化

### 建议新增参数

- `--segmentation-cache-dir`
- `--reuse-segmentation`
- `--refresh-segmentation`

### 建议新增函数

- `resolve_segmentation_cache_paths(...)`
- `load_cached_segmentation(...)`
- `save_cached_segmentation(...)`
- `prepare_features_from_images(...)`

## 4. 新主流程

---

## Stage 0: 输入和特征准备

### 输入

首版仍然保留“输入两张图像”的模式：

```powershell
python -m cell_registration.main fixed.tif moving.tif
```

内部流程：

- 先查缓存
- 缓存有效则直接读 `mask + features`
- 缓存无效则跑分割并存缓存

### 数据要求

feature 表至少包含：

- `cell_id`
- `centroid_x`
- `centroid_y`
- `area`
- `perimeter`
- `major_axis_length`
- `minor_axis_length`
- `eccentricity`
- `orientation`

并额外派生：

- `shape_ratio = major_axis_length / max(minor_axis_length, eps)`

### 注意

- fixed / moving 细胞数可以不同
- 不要求所有细胞都进入最终 landmark
- landmark 必须一对一，但 landmark 集合可以是部分匹配

---

## Stage 1: 宽松坐标候选生成

### 目标

只做宽松候选，不在这里过早砍死真实匹配。

### 方法

对每个 fixed cell，在 moving cell 的 KD-tree 上做半径查询：

- `candidate_radius_px` 默认建议放宽到 `20~30 px`
- 推荐默认值：`25 px`

### 为什么要改

当前如果 Stage 1 直接硬切 `<= 15 px`，而 Stage 2 又使用 `position-weight` 微调距离，会造成逻辑断层：

- Stage 1 先把稍远但真实的匹配彻底丢掉
- Stage 2 再对距离做软评分

这不一致。

### 新规则

- Stage 1 只负责“宽松候选进入图”
- Stage 2 再用 `position-weight` 进行精细控制

### 输出

候选边表至少包含：

- `fixed_idx`
- `moving_idx`
- `fixed_id`
- `moving_id`
- `centroid_distance_px`

### 建议新增函数

- `build_coordinate_candidates(...)`

---

## Stage 2: 形态学特征 + 位置联合评分

### 特征差异项

对每个候选 pair 计算：

- `area_diff_norm`
- `perimeter_diff_norm`
- `shape_diff_norm`
- `eccentricity_diff_norm`

### 归一化

对 `area / perimeter / shape_ratio / eccentricity` 使用固定 + moving 合并后的稳健尺度：

- `median`
- `MAD` 或 `IQR`

得到归一化差异：

- `normalized_diff = abs(x1 - x2) / max(scale, eps)`

### 位置项

位置项保留，而且必须保留：

- `position_score = centroid_distance_px / candidate_radius_px`

### 分数定义

先定义纯形态学分数：

`feature_score_raw = w_area * area_diff_norm + w_perimeter * perimeter_diff_norm + w_shape * shape_diff_norm + w_ecc * eccentricity_diff_norm`

再做稳健归一化：

`feature_score = feature_score_raw / max(median(feature_score_raw over candidates), eps)`

然后再和位置项合并：

`candidate_score_raw = feature_score + position_weight * position_score`

并再次归一化：

`candidate_score = candidate_score_raw / max(median(candidate_score_raw over candidates), eps)`

### 过滤规则

只保留：

- `feature_score <= feature_score_threshold`
- `candidate_score <= candidate_score_threshold`

### 这样改的原因

- 保留 `position-weight`
- 避免 feature 分数和 topology 分数量纲完全不同
- 为后续 Stage 3/4 的联合打分打基础

### 建议新增函数

- `compute_feature_normalization_stats(...)`
- `score_feature_candidates(...)`
- `filter_feature_candidates(...)`

---

## Stage 3: 局部 Topology 验证

这是确认 landmark 是否真实的核心。

### 方法

对每个通过 Stage 2 的候选 `(fixed_i, moving_j)`：

1. 取 fixed_i 周围 `topology_radius_px` 邻域
2. 取 moving_j 周围 `topology_radius_px` 邻域
3. 构建局部拓扑描述子
4. 比较描述子
5. 只保留 topology 足够相似的 pair

推荐默认：

- `topology_radius_px = 20`
- `topology_k = 5`

### 描述子

首版实现三项：

1. `count_diff_norm`
2. `knn_distance_l2`
3. `angle_hist_distance`

### 3.1 邻居数量差异

定义：

- `count_diff_norm = abs(n_fixed - n_moving) / max(max(n_fixed, n_moving), 1)`

### 重要约束

这个项权重不能过高。

建议默认：

- `topo_count_weight < topo_dist_weight`

因为邻居数量在边界和稀疏区域非常不稳定，容易误杀正确点。

### 3.2 kNN 距离向量必须做尺度归一化

这是必须修改项。

不能直接比较原始距离向量：

- 否则轻微 scale 就会把 `knn_distance_l2` 放大

正确做法：

先计算局部 kNN 距离向量，再对每个候选的两侧邻域分别做尺度归一化，例如：

- `d_norm = d / max(median(d), eps)`

如果邻域不足 `k`：

- 用 `topology_radius_px` 填充
- 再做归一化

然后比较：

- `knn_distance_l2 = ||d_fixed_norm - d_moving_norm||_2`

### 3.3 角度直方图必须 rotation-invariant

这也是必须修改项。

不能直接把两侧 angle histogram 生硬比较，因为轻微旋转会导致 histogram 平移。

正确做法：

- 先计算 anchor 到邻居的相对角度
- 分桶成 circular histogram
- 对 moving histogram 做循环平移
- 取所有 circular shift 中的最小距离

即：

- `angle_hist_distance = min_shift_distance(hist_fixed, hist_moving)`

这样 angle descriptor 对整体旋转是稳定的。

### 3.4 边界细胞特殊处理

这是需要显式写进方案的。

如果 anchor 距离图像边界小于 `topology_radius_px`，则该细胞邻域天然不完整。

建议：

- 标记 `is_border_cell`
- 对 border cell 降低 topology 约束强度

具体做法建议二选一：

1. 降低 topology 总权重
2. 放宽 topology 阈值

首版推荐：

- `border_topology_relax_factor`

例如：

- 非边界：`threshold = topology_score_threshold`
- 边界：`threshold = topology_score_threshold * border_topology_relax_factor`

### 3.5 topology 总分

先算原始 topology 分数：

`topology_score_raw = topo_count_weight * count_diff_norm + topo_dist_weight * knn_distance_l2 + topo_angle_weight * angle_hist_distance`

再做稳健归一化：

`topology_score = topology_score_raw / max(median(topology_score_raw over candidates), eps)`

### 3.6 过滤规则

只保留：

- `topology_score <= topology_score_threshold`

### 建议新增函数

- `extract_local_neighbors(...)`
- `compute_scale_normalized_knn_descriptor(...)`
- `compute_rotation_invariant_angle_hist(...)`
- `score_topology_candidates(...)`
- `validate_candidates_by_topology(...)`

---

## Stage 4: 最大基数优先的 landmark 一对一匹配

这是这次必须改掉的重点。

### 不能再写成 Hungarian + dummy

原因很明确：

- Hungarian 解决的是最小代价完全分配
- 你这里需要的是最大基数优先的部分匹配

这不是同一个问题。

### 正确目标

Stage 4 的目标是：

1. 先最大化最终 landmark 数量
2. 再在这个最大数量约束下最小化总匹配代价

即：

- Maximum Cardinality Matching
- then Minimum Cost on that cardinality

### 正确实现方向

文档里明确采用两阶段求解，不再使用 “Hungarian + dummy penalty” 作为主方案。

#### Phase 1: 最大基数匹配

只看 validated graph 是否有边，不看 cost：

- 节点左侧：fixed
- 节点右侧：moving
- 边：所有通过 topology 的候选

先求：

- 最大基数匹配

得到：

- 最大可保留 landmark 数 `M`

#### Phase 2: 在基数 `M` 下最小化总 cost

在保持真实匹配数必须等于 `M` 的前提下，再优化总代价。

精确实现可以采用：

- min-cost max-flow
- 或等价的“最大基数约束下最小权匹配”求解

不再把 dummy 当作主要机制。

### Landmark cost 定义

先做联合原始分数：

`landmark_cost_raw = candidate_score + topology_weight * topology_score`

再做稳健归一化：

`landmark_cost = landmark_cost_raw / max(median(landmark_cost_raw over validated edges), eps)`

### 输出

最终 landmark 必须满足：

- 一对一
- 数量达到最大基数
- 在最大基数下总代价最小

### 建议新增函数

- `build_validated_landmark_graph(...)`
- `solve_max_cardinality_matching(...)`
- `solve_min_cost_matching_with_fixed_cardinality(...)`
- `select_final_landmarks(...)`

---

## Stage 5: 用全部最终 landmark 做配准估计

### 方法

用 Stage 4 得到的全部最终 landmark 估计变换：

- 默认 rigid
- 可选 similarity，由 `--allow-scale` 控制

### 明确删除

- RANSAC
- ICP
- global coarse registration
- residual prune

### 保留

继续复用 [`estimate_rigid_transform_from_matches`](f:\programme\nxy\Cell registration\cell_registration\registration.py) 的思路，但输入改为：

- 全部最终 landmark 表

### 稳定性约束

这里必须明确定义 `min_landmark_spread`，否则 transform 可能集中在一个小区域上不稳定。

建议定义为：

- landmark 的二维凸包面积 / 图像面积 >= `min_landmark_spread_ratio`

如果不想引入凸包，也可以退化为：

- `std_x >= spread_x_min`
- `std_y >= spread_y_min`

首版推荐更简单明确的版本：

- `spread_x_std_px`
- `spread_y_std_px`

如果 landmark 在 x 或 y 上过于集中，则直接报错，不估计 transform。

### 额外约束

最终进入 transform 拟合前必须同时满足：

- `len(final_landmarks) >= min_landmark_pairs`
- `spread_x_std_px >= min_landmark_spread_x_px`
- `spread_y_std_px >= min_landmark_spread_y_px`

---

## Stage 6: 导出和调试输出

### 正式输出

- `round1_cells.csv`
- `round2_cells.csv`
- `landmark_matches.csv`
- `match_overlay.tif`
- `match_plot.tif`
- `registration_overlay.tif`

### 调试输出

- `candidate_counts.csv`
- `feature_rejections.csv`
- `topology_rejections.csv`
- `validated_candidates.csv`
- `final_landmarks.csv`
- `segmentation_meta_fixed.json`
- `segmentation_meta_moving.json`

## 5. `main.py` 参数调整

### 保留

- `--position-weight`
- `--allow-scale`
- `--save-registration-overlay`
- `--save-match-table`
- `--save-match-overlay`
- `--save-match-plot`
- `--save-features-dir`
- `--napari`
- `--segmentation-only`

### 删除

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

### 新增

- `--segmentation-cache-dir`
- `--reuse-segmentation`
- `--refresh-segmentation`
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

## 6. `main.py` 函数组织建议

首版仍然只改 [`cell_registration/main.py`](f:\programme\nxy\Cell registration\cell_registration\main.py)。

建议函数分层如下：

### 分割缓存

- `resolve_segmentation_cache_paths(...)`
- `load_cached_segmentation(...)`
- `save_cached_segmentation(...)`
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

### 导出

- `export_debug_tables(...)`

## 7. 这版方案和上一版相比的关键修正

这一版明确修正了以下问题：

1. Stage 4 不再写成 “Hungarian + dummy”
2. Stage 4 改成 “最大基数优先，再最小代价”
3. Stage 3 的 kNN 距离加入尺度归一化
4. Stage 3 的 angle histogram 加入 rotation-invariant circular shift
5. Stage 1 候选半径改成宽松半径，和 Stage 2 的 `position-weight` 逻辑统一
6. feature / topology / landmark cost 都加入稳健归一化
7. 邻居数量项明确降权
8. 边界细胞加入 topology 特殊处理
9. `min_landmark_spread` 给出明确可执行定义
10. 分割缓存写进主流程，不再重复跑 segmentation

## 8. 执行顺序

如果你批准这版方案，我按这个顺序做：

1. 清理 `main.py` 的 import 和旧参数
2. 加入分割缓存逻辑
3. 删掉 patch / cluster / RANSAC / ICP / global coarse registration
4. 实现宽松坐标候选
5. 实现 feature + position 评分
6. 实现 topology 验证
7. 实现最大基数优先的 landmark 选择
8. 用全部最终 landmark 估计 transform
9. 保留 overlay / plot / debug 导出
10. 更新 README

## 9. 等你确认的点

### 1. Stage 4 求解器

文档现在要求的是：

- 最大基数优先
- 再最小代价

如果你同意，我实现时会按这个约束做，不再走 Hungarian + dummy 的近似路子。

### 2. Stage 1 默认半径

我现在建议默认：

- `candidate_radius_px = 25`

如果你更想保守一点，也可以改成 `20`。

### 3. `min_landmark_spread` 的定义

首版文档现在采用更容易实现的版本：

- `x/y` 方向标准差阈值

如果你更偏好凸包面积定义，也可以换。
