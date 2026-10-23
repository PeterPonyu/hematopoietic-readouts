"""Bounded-memory refinement; full-output equivalence gate is mandatory.

Only membership lookup, finite-distance median storage and identity reindex
allocation change. Every selected cell and numerical median is retained.
"""
"""Retained FateID port with membership lookup caches only.

List order, distance calculations, forest parameters and predictions are unchanged.
The original source remains intact; full Dapp1 equivalence must pass before use.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import binomtest
from concurrent.futures import ThreadPoolExecutor
_median_pool = ThreadPoolExecutor(max_workers=4)

def _finite_block_medians(distance, assigned, unassigned):
    """Exact ordinary medians of finite distances, in bounded temporary blocks."""
    rows=distance.index.get_indexer(assigned)
    columns=distance.columns.get_indexer(unassigned)
    assert (rows>=0).all() and (columns>=0).all()
    matrix=distance.to_numpy(copy=False)
    medians=np.empty(len(columns),dtype=matrix.dtype)
    def calculate(start):
        values=matrix[np.ix_(rows,columns[start:start+256])].T.copy(order='C')
        if not np.isfinite(values).all():
            raise ValueError('The blocked-median adapter requires finite cell distances.')
        return start, np.median(values,axis=1,overwrite_input=True)
    # Independent columns have no shared writes or floating-point reduction.
    # Ordered map retains the original output axis while limiting temporaries.
    for start, values in _median_pool.map(calculate,range(0,len(columns),256)):
        medians[start:start+256]=values
    return pd.Series(medians,index=unassigned)


def bias(tvn):
    """
    计算偏向比和显著性p值（与R代码完全一致的实现）
    
    参数:
        tvn: DataFrame, 投票计数矩阵，行为细胞，列为轨迹
    
    返回:
        dict: 包含bias矩阵和p值向量
    """
    bias_matrix = tvn.copy()
    p_values = []
    
    # 对每一行（每个细胞）计算
    for idx, row in tvn.iterrows():
        votes = row.values
        # 按降序排列投票数
        sorted_votes = np.sort(votes)[::-1]
        second_highest = sorted_votes[1] if len(sorted_votes) > 1 else 0
        
        # 计算偏向比：每个轨迹的投票数除以第二高投票数（加0.001避免除零）
        denominator = second_highest + 0.001
        bias_row = votes / denominator
        bias_matrix.loc[idx] = bias_row
        
        # 计算p值：对最高和第二高投票数进行二项检验
        highest = sorted_votes[0] if len(sorted_votes) > 0 else 0
        second = sorted_votes[1] if len(sorted_votes) > 1 else 0
        
        if highest + second > 0:
            # 使用双侧二项检验（对应R中的alternative="t"）
            p_val = binomtest(int(highest), n=int(highest + second), p=0.5, alternative='two-sided').pvalue
        else:
            p_val = 1.0
        
        p_values.append(p_val)
    
    pv = pd.Series(p_values, index=tvn.index)
    
    return {"bias": bias_matrix, "pv": pv}

def fate_bias(x, y, tar, z=None, minnr=None, minnrh=None, adapt=True, 
              confidence=0.75, nbfactor=5, use_dist=False, seed=None, 
              nbtree=None, verbose=False, **kwargs):
    """
    计算单细胞的命运偏向（与R代码逻辑完全一致）
    """
    
    # 设置随机种子
    if seed is not None:
        np.random.seed(seed)
    
    # 确保y是Series并且索引与x.columns完全一致
    if not isinstance(y, pd.Series):
        y = pd.Series(y, index=x.columns)
    
    # 关键修复：确保y的索引与x.columns完全匹配
    valid_cells = list(x.columns)
    y = y.reindex(valid_cells)
    
    # 移除任何NaN值对应的细胞
    valid_mask = ~y.isna()
    valid_cells = [cell for cell in valid_cells if valid_mask[cell]]
    x = x.loc[:, valid_cells]
    y = y.loc[valid_cells]
    
    # 验证目标类别存在
    available_clusters = set(y.unique())
    missing_targets = set(tar) - available_clusters
    if missing_targets:
        raise ValueError(f"目标类别 {missing_targets} 不存在于聚类结果中")
    
    # 计算目标聚类的细胞数量统计
    target_counts = []
    for k in tar:
        count = sum(y == k)
        target_counts.append(count)
    
    median_count = np.median(target_counts)
    
    # 设置默认的minnr和minnrh值（与R代码逻辑一致）
    if minnrh is None:
        minnrh = max(round(median_count / 2), 20)
    if minnr is None:
        minnr = min(round(median_count / 2), 20)
    
    if verbose:
        print(f"minnr: {minnr}")
        print(f"minnrh: {minnrh}")
        print(f"有效细胞数: {len(valid_cells)}")
        print(f"目标类别细胞数: {dict(zip(tar, target_counts))}")
    
    # 创建轨迹名称到原始类别的映射
    trajectory_to_target = {}
    target_to_trajectory = {}
    for k in tar:
        trajectory_name = f"t{k}"
        trajectory_to_target[trajectory_name] = k
        target_to_trajectory[k] = trajectory_name
    
    # 初始化轨迹字典 - 关键修复：只使用有效的细胞索引
    tr = {}
    trall = []
    for k in tar:
        trajectory_name = f"t{k}"
        # 只获取在有效细胞中且属于目标类的细胞
        cells_in_cluster = [cell for cell in valid_cells if y[cell] == k]
        tr[trajectory_name] = cells_in_cluster.copy()
        trall.extend(cells_in_cluster)
    
    # 计算距离矩阵 - 确保索引与有效细胞一致
    if z is not None:
        if hasattr(z, 'values'):  # 如果是DataFrame
            di = pd.DataFrame(z.values, index=valid_cells, columns=valid_cells)
        else:  # 如果是numpy数组
            di = pd.DataFrame(z, index=valid_cells, columns=valid_cells)
    else:
        # 计算1-cor(x)距离矩阵
        correlation_matrix = x.T.corr()
        di = 1 - correlation_matrix
    
    # 确保距离矩阵的行列索引都是有效细胞
    if not (di.index.equals(pd.Index(valid_cells)) and di.columns.equals(pd.Index(valid_cells))):
        di = di.reindex(index=valid_cells, columns=valid_cells)
    
    # 选择特征矩阵
    if use_dist:
        d = di
    else:
        d = x
    
    # 初始化概率矩阵 - 使用有效细胞作为索引
    trajectory_names = [f"t{k}" for k in tar]
    num_trajectories = len(tar)
    
    probs = pd.DataFrame(
        np.full((len(valid_cells), num_trajectories), 1/num_trajectories),
        index=valid_cells,
        columns=trajectory_names
    )
    
    # 为目标聚类中的细胞设置初始概率
    for k in tar:
        trajectory_name = f"t{k}"
        cells_in_target = [cell for cell in valid_cells if y[cell] == k]
        
        # 将对应轨迹的概率设为1，其他设为0
        for traj in trajectory_names:
            if traj == trajectory_name:
                probs.loc[cells_in_target, traj] = 1
            else:
                probs.loc[cells_in_target, traj] = 0
    
    # 初始化投票矩阵
    votes = probs.copy()
    
    # 初始化迭代变量
    iteration = 0
    rfl = []
    weights = pd.Series(1.0, index=trajectory_names)
    
    # 主迭代循环
    while len(trall) < len(valid_cells):
        iteration += 1
        
        if verbose:
            weighted_test_sizes = weights * minnr
            print(f"测试集大小 第{iteration}轮: {weighted_test_sizes.to_dict()}")
        
        # 找出未分配的细胞 - 确保都在有效细胞范围内
        assigned_members = set(trall)
        unassigned = [cell for cell in valid_cells if cell not in assigned_members]
        
        if len(unassigned) == 0:
            if verbose:
                print("所有细胞已分配")
            break
        
        # 为每个目标轨迹选择测试细胞
        nl = {}
        n = []
        
        if len(unassigned) > 1:
            # 构建距离子矩阵：已分配细胞到未分配细胞
            # 关键修复：确保trall中的所有细胞都在di的索引中
            valid_trall = [cell for cell in trall if cell in di.index]
            # Distances are selected in bounded blocks below; no full u copy.
            
            for k in tar:
                trajectory_name = f"t{k}"
                # 找到属于当前轨迹的已分配细胞
                trajectory_members = set(tr[trajectory_name])
                trajectory_assigned = [cell for cell in valid_trall if cell in trajectory_members]
                
                if len(trajectory_assigned) > 0:
                    # 计算轨迹细胞到未分配细胞的中位数距离
                    median_distances = _finite_block_medians(di, trajectory_assigned, unassigned)
                    sorted_distances = median_distances.sort_values(ascending=True)
                    
                    # 根据权重确定选择数量
                    weighted_minnr = max(round(minnr * weights[trajectory_name]), 1)
                    selected = list(sorted_distances.index[:weighted_minnr])
                    nl[k] = selected
                    n.extend(selected)
        else:
            n = unassigned
            for k in tar:
                nl[k] = n.copy()
        
        # 去重
        n = list(set(n))
        
        if len(n) == 0:
            if verbose:
                print("没有可测试的细胞")
            break
        
        # 为每个目标轨迹选择训练细胞
        trt = []
        for k in tar:
            trajectory_name = f"t{k}"
            if k in nl and len(nl[k]) > 0:
                # 计算候选细胞到轨迹细胞的距离
                trajectory_cells = [cell for cell in tr[trajectory_name] if cell in di.columns]
                candidate_cells = [cell for cell in nl[k] if cell in di.index]
                
                if len(trajectory_cells) > 0 and len(candidate_cells) > 0:
                    distance_sub = di.loc[candidate_cells, trajectory_cells]
                    # 对每个轨迹细胞找最近的候选细胞
                    min_distances = distance_sub.min(axis=0)
                    sorted_min_dist = min_distances.sort_values(ascending=True)
                    training_selected = list(sorted_min_dist.index[:minnrh])
                    trt.extend(training_selected)
        
        trt = list(set(trt))  # 去重
        
        if len(trt) == 0:
            if verbose:
                print("没有可用的训练细胞")
            break
        
        if verbose:
            print(f"随机森林 第{iteration}轮: {len(n)}个细胞预测，{len(trt)}个训练细胞")
        
        # 准备随机森林的特征矩阵
        if use_dist:
            X_train = di.loc[trt, trt]
            X_test = di.loc[n, trt]
        else:
            X_train = d.loc[:, trt].T  # 转置使细胞为行
            X_test = d.loc[:, n].T
        
        # 准备训练标签：使用轨迹名称到原始类别的映射
        train_labels = []
        for cell in trt:
            max_prob_trajectory = probs.loc[cell].idxmax()
            # 使用映射字典获取原始类别
            original_class = trajectory_to_target[max_prob_trajectory]
            train_labels.append(original_class)
        
        # 设置随机森林参数
        if nbtree is None:
            nbtree = X_train.shape[1] * nbfactor
        
        # 训练随机森林
        rf = RandomForestClassifier(
            n_estimators=nbtree,
            random_state=seed,
            **kwargs
        )
        rf.fit(X_train, train_labels)
        rfl.append(rf)
        
        # 预测并获取投票
        test_predictions = rf.predict_proba(X_test)
        predicted_classes = rf.classes_
        
        # 构建投票矩阵，使用原始类别映射到轨迹名称
        tv_data = np.zeros((len(n), len(trajectory_names)))
        tv = pd.DataFrame(tv_data, index=X_test.index, columns=trajectory_names)
        
        # 填充预测结果到对应的轨迹列
        for i, original_class in enumerate(predicted_classes):
            trajectory_name = target_to_trajectory[original_class]
            if trajectory_name in trajectory_names:
                col_idx = trajectory_names.index(trajectory_name)
                tv.iloc[:, col_idx] = test_predictions[:, i]
        
        # 转换为投票计数（不是概率）
        tvn = (tv * nbtree).round().astype(int)
        
        # 计算归一化概率
        tv_normalized = tvn.div(tvn.sum(axis=1), axis=0)
        tv_normalized = tv_normalized.fillna(1/num_trajectories)  # 处理全零行
        
        # 更新概率矩阵
        probs.loc[n] = tv_normalized
        
        # 更新投票矩阵
        if iteration == 1:
            # 第一轮调整votes的量级
            max_total_votes = tvn.sum(axis=1).max()
            if max_total_votes > 0:
                votes = votes * max_total_votes
        votes.loc[n] = tvn
        
        # 自适应权重调整
        if adapt:
            for k in tar:
                trajectory_name = f"t{k}"
                # 统计概率超过confidence的细胞数
                successful_count = sum(probs.loc[n, trajectory_name] > confidence)
                weights[trajectory_name] = successful_count + 1
            # 归一化权重
            max_weight = weights.max()
            if max_weight > 0:
                weights = weights / max_weight
        
        # 计算偏向比并更新轨迹
        bias_result = bias(tvn)
        bias_matrix = bias_result["bias"]
        pv = bias_result["pv"]
        
        # 为每个轨迹添加符合条件的细胞
        for trajectory_name in trajectory_names:
            qualified_cells = []
            for cell in tvn.index:
                if (bias_matrix.loc[cell, trajectory_name] > 1 and 
                    pv.loc[cell] < 0.05):
                    qualified_cells.append(cell)
            
            # 添加新细胞到轨迹（避免重复）
            trajectory_members = set(tr[trajectory_name])
            for cell in qualified_cells:
                if cell not in trajectory_members:
                    tr[trajectory_name].append(cell)
                    trajectory_members.add(cell)
        
        # 更新已分配细胞列表
        trall.extend(n)
    
    if verbose:
        print(f"完成！共进行{iteration}轮迭代")
        print(f"最终分配细胞数: {len(trall)}/{len(valid_cells)}")
    
    return {
        "probs": probs,
        "votes": votes,
        "tr": tr,
        "rfl": rfl,
        "trall": trall
    }