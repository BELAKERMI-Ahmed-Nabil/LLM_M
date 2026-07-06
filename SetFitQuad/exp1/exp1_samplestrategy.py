import logging
import os
import json
import numpy as np
from datasets import load_dataset
from setfit import SetFitModel, Trainer
from setfit import TrainingArguments
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import KFold
import json
import os
import math
from functools import partial

logging.basicConfig(level=logging.INFO)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_DIR = "results"
TRAIN_SIZE = 50
VAL_SIZE = 10
N_FOLDS = 5 # 新增: 定義fold數量

#  Saves evaluation metrics to a JSON file (one per fold + a final average).
def save_results(output_dir, sampling_name, metrics, fold=None):
    """將結果保存為 JSON 檔案，支援fold的結果保存"""
    if fold is not None:
        filepath = f"{output_dir}/{sampling_name}_fold{fold}_results.json"
    else:
        filepath = f"{output_dir}/{sampling_name}_results.json"
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logging.info(f"Results saved to {filepath}")

# ════ (STRATEGY) ════
#  Random sampling: picks num_samples points at random (fixed seed 42).
def random_seed_sampling(embeddings, num_samples):
    """隨機種子抽樣"""
    np.random.seed(42)
    return np.random.choice(len(embeddings), num_samples, replace=False)

#  Wrapper returning the samples + the strategy proportion (100% — single strategy).
def random_seed_sampling_with_proportions(embeddings, num_samples):
    """隨機種子抽樣，比例設定為 100%。"""
    indices = random_seed_sampling(embeddings, num_samples)
    proportions = {"Random Seed (RS)": 1.0}  # 單一策略比例為 100%
    return indices, proportions


# ════ اSTRATEGY) ════
#  Grid sampling: picks evenly-spaced points along the index range (linspace).
def grid_sampling(embeddings, num_samples):
    """網格抽樣"""
    return np.linspace(0, len(embeddings) - 1, num_samples, dtype=int)

#  Wrapper returning the samples + strategy proportion (100%).
def grid_sampling_with_proportions(embeddings, num_samples):
    """網格抽樣，比例設定為 100%。"""
    indices = grid_sampling(embeddings, num_samples)
    proportions = {"Grid Sampling (GS)": 1.0}  # 單一策略比例為 100%
    return indices, proportions

# ════ (STRATEGY) ════
#  Max-Min distance: greedily picks the points farthest apart for wide coverage.
def max_min_distance_sampling(embeddings, num_samples):
    """最大-最小距離抽樣"""
    center = np.mean(embeddings, axis=0)
    distances = np.linalg.norm(embeddings - center, axis=1)
    selected = [np.argmax(distances)]
    for _ in range(1, num_samples):
        dist_to_selected = np.min(
            np.linalg.norm(embeddings[:, np.newaxis] - embeddings[selected], axis=2),
            axis=1
        )
        next_point = np.argmax(dist_to_selected)
        selected.append(next_point)
    return np.array(selected)
    
# Wrapper returning the samples + strategy proportion (100%).
def max_min_distance_sampling_with_proportions(embeddings, num_samples):
    indices = max_min_distance_sampling(embeddings, num_samples)
    proportions = {"Max-Min Distance (MMDS)": 1.0}
    return indices, proportions


# ════ (STRATEGY) ════
#  Density-based: picks points in dense regions (via nearest-neighbour distances).
def density_based_sampling(embeddings, num_samples):
    """密度基礎抽樣"""
    nn = NearestNeighbors(n_neighbors=min(5, len(embeddings) - 1)).fit(embeddings)
    distances, _ = nn.kneighbors(embeddings)
    density_scores = np.sum(distances, axis=1)
    return np.argsort(density_scores)[:num_samples]

#  Wrapper returning the samples + strategy proportion (100%).
def density_based_sampling_with_proportions(embeddings, num_samples):
    indices = density_based_sampling(embeddings, num_samples)
    proportions = {"Density-based (DBS)": 1.0}
    return indices, proportions


# ════ (STRATEGY) ════
#  Max-entropy: picks the points whose neighbourhood is most uncertain/diverse.
def max_entropy_sampling(embeddings, num_samples):
    """最大熵抽樣"""
    nn = NearestNeighbors(n_neighbors=min(5, len(embeddings) - 1))
    nn.fit(embeddings)
    distances, _ = nn.kneighbors(embeddings)
    entropies = -np.sum(np.log(distances + 1e-10) * distances, axis=1)
    return np.argsort(entropies)[-num_samples:]

# Wrapper returning the samples + strategy proportion (100%).
def max_entropy_sampling_with_proportions(embeddings, num_samples):
    indices = max_entropy_sampling(embeddings, num_samples)
    proportions = {"Max Entropy (MES)": 1.0}
    return indices, proportions


# ════ ا(STRATEGY) — الأفضل في التجربة ════
#  Cluster sampling: KMeans into num_samples clusters, takes one representative each.
def cluster_sampling(embeddings, num_samples):
    """聚類抽樣"""
    kmeans = KMeans(n_clusters=num_samples, random_state=42).fit(embeddings)
    return np.array([np.where(kmeans.labels_ == i)[0][0] for i in range(num_samples)])

#  Wrapper returning the samples + strategy proportion (100%).
def cluster_sampling_with_proportions(embeddings, num_samples):
    indices = cluster_sampling(embeddings, num_samples)
    proportions = {"Cluster Sampling (CS)": 1.0}
    return indices, proportions



#  Sets default proportions: if None, becomes 100% for the current strategy.
def set_default_proportions(self, proportions):
    """設置默認比例，若 proportions 為 None，設為 100%。"""
    if proportions is None:
        proportions = {self.sampling_name: 1.0}
    self.sample_proportions = proportions





# Builds a 6-feature matrix per point (min/mean/max distance + entropy
#     + diversity + coverage) — used as inputs to the sampler models (Lasso/Ridge/RF).
def create_enhanced_features(embeddings, indices):
    """計算更全面的特徵"""
    selected_embeddings = embeddings[indices]
    
    # 1. 基本距離特徵
    distances = np.linalg.norm(embeddings[:, np.newaxis] - selected_embeddings, axis=2)
    min_distances = np.min(distances, axis=1)
    mean_distances = np.mean(distances, axis=1)
    max_distances = np.max(distances, axis=1)
    
    # 2. 信息熵特徵
    distance_probs = distances / (np.sum(distances, axis=1, keepdims=True) + 1e-10)
    entropy = -np.sum(distance_probs * np.log(distance_probs + 1e-10), axis=1)
    
    # 3. 多樣性特徵
    selected_distances = np.linalg.norm(selected_embeddings[:, np.newaxis] - selected_embeddings, axis=2)
    diversity = np.mean(selected_distances[selected_distances > 0])
    diversity_scores = np.full_like(min_distances, diversity)
    
    # 4. 覆蓋度特徵
    coverage = np.mean(np.exp(-min_distances))
    coverage_scores = np.full_like(min_distances, coverage)
    
    # 組合所有特徵
    features = np.column_stack([
        min_distances,   # 最小距離
        mean_distances,  # 平均距離
        max_distances,   # 最大距離
        entropy,         # 熵
        diversity_scores,# 多樣性
        coverage_scores  # 覆蓋度
    ])
    
    return features

# ════(STRATEGY) ════
#  Random Forest combo: blends the 6 base strategies. Builds per-strategy features,
#     trains a RandomForest to learn each strategy's weight, then allocates samples by weight.
def random_forest_combination_with_proportions(embeddings, num_samples, base_prop=0.3):
    n = embeddings.shape[0]

   
    base_indices = {
        "Random Seed (RS)": random_seed_sampling(embeddings, int(n * base_prop)),
        "Grid Sampling (GS)": grid_sampling(embeddings, int(n * base_prop)),
        "Max-Min Distance (MMDS)": max_min_distance_sampling(embeddings, int(n * base_prop)),
        "Density-based (DBS)": density_based_sampling(embeddings, int(n * base_prop)),
        "Max Entropy (MES)": max_entropy_sampling(embeddings, int(n * base_prop)),
        "Cluster Sampling (CS)": cluster_sampling(embeddings, int(n * base_prop)),
    }


    strategy_features = np.zeros((n, len(base_indices) * 4))
    for i, (strategy_name, indices) in enumerate(base_indices.items()):
        selected_embeddings = embeddings[indices]
        distances = np.linalg.norm(embeddings[:, np.newaxis] - selected_embeddings, axis=2)
        min_distances = np.min(distances, axis=1)
        mean_distances = np.mean(distances, axis=1)
        std_distances = np.std(distances, axis=1)

        probs = distances / (np.sum(distances, axis=1, keepdims=True) + 1e-10)
        entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
        
   
        strategy_features[:, i*4]   = min_distances
        strategy_features[:, i*4+1] = mean_distances
        strategy_features[:, i*4+2] = std_distances
        strategy_features[:, i*4+3] = entropy


    scaler = StandardScaler()
    strategy_features = scaler.fit_transform(strategy_features)


    center = np.mean(embeddings, axis=0)
    distances_to_center = np.linalg.norm(embeddings - center, axis=1)
    
    nn = NearestNeighbors(n_neighbors=min(5, len(embeddings)))
    nn.fit(embeddings)
    distances, _ = nn.kneighbors(embeddings)
    

    density_scores = np.exp(-np.sum(distances, axis=1))
 
    diversity_scores = np.var(embeddings, axis=1)

    entropy_scores = -np.sum(distances * np.log(distances + 1e-10), axis=1)
    

    target = (distances_to_center * diversity_scores * (entropy_scores ** 2)) / (density_scores ** 0.5)


    rf_models = []
    importances = []
    for seed_offset in range(5):
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10],
            'min_samples_split': [2, 5]
        }
        rf = GridSearchCV(
            RandomForestRegressor(random_state=42 + seed_offset),
            param_grid,
            cv=5,
            scoring='neg_mean_squared_error'
        )
        rf.fit(strategy_features, target)
        rf_models.append(rf.best_estimator_)
        importances.append(rf.best_estimator_.feature_importances_)
    

    final_importances = np.mean(importances, axis=0)


    strategy_names = list(base_indices.keys())
    strategy_weights = np.zeros(len(strategy_names))
    for i in range(len(strategy_names)):
        strategy_weights[i] = np.mean(final_importances[i*4:(i+1)*4])


    mes_idx = strategy_names.index("Max Entropy (MES)")
    max_weight = np.max(strategy_weights)
    min_mes_weight = max_weight * 0.8  # MES 至少是 max_weight 的 80%
    if strategy_weights[mes_idx] < min_mes_weight:
        strategy_weights[mes_idx] = min_mes_weight
    

    strategy_weights /= np.sum(strategy_weights)  # 先歸一化
    importance_threshold = np.percentile(strategy_weights, 25)  # 第25百分位
    strategy_weights[strategy_weights < importance_threshold] = 0.0

    total_w = np.sum(strategy_weights)
    if total_w > 0:
        strategy_weights /= total_w
    else:


        strategy_weights = np.ones(len(strategy_names)) / len(strategy_names)



    strategy_samples = {}
    min_samples_per_strategy = max(1, int(num_samples * 0.1))


    valid_strategies = [
        s for s, w in zip(strategy_names, strategy_weights) if w > 0
    ]


    for s in valid_strategies:
        strategy_samples[s] = min_samples_per_strategy

    total_allocated = sum(strategy_samples.values())
    if total_allocated > num_samples:

        logging.warning("Min samples sum exceed total. Reducing to 1 per valid strategy.")
        strategy_samples = {s: 1 for s in valid_strategies}
        total_allocated = len(valid_strategies)

    remaining = num_samples - total_allocated


    if remaining > 0 and valid_strategies:

        valid_weights = np.array([
            strategy_weights[strategy_names.index(s)] for s in valid_strategies
        ])
        vw_sum = valid_weights.sum()
        if vw_sum > 0:
            valid_weights /= vw_sum
            distributed = 0
            for i in range(len(valid_strategies) - 1):
                add_num = int(remaining * valid_weights[i])
                strategy_samples[valid_strategies[i]] += add_num
                distributed += add_num

            leftover = remaining - distributed
            strategy_samples[valid_strategies[-1]] += leftover


    assert sum(strategy_samples.values()) == num_samples, "Sample allocation mismatch."



    all_indices = []
    for s, n_samples_s in strategy_samples.items():
        if n_samples_s > 0:
            sampling_function = SAMPLING_METHODS.get(s)
            if sampling_function is None:
                raise KeyError(f"Sampling method {s} not found.")
            sampled = sampling_function(embeddings, n_samples_s)
            if isinstance(sampled, tuple):
                sampled = sampled[0]
            all_indices.extend(sampled)


    all_indices = np.unique(all_indices)

    if len(all_indices) > num_samples:
        selected_indices = np.random.choice(all_indices, num_samples, replace=False)
    else:
        need_more = num_samples - len(all_indices)
        if need_more > 0:
            add_indices = random_seed_sampling(embeddings, need_more)
            selected_indices = np.concatenate([all_indices, add_indices])
        else:
            selected_indices = all_indices



    final_counts = {s: 0 for s in strategy_samples}
    for s, n_samples_s in strategy_samples.items():
        final_counts[s] = n_samples_s 



    proportions = {
        s: final_counts[s] / num_samples for s in final_counts
    }

    return selected_indices, proportions


# ════ (STRATEGY) ════

# Lasso (L1 regularization): learns sparse weights (zeros out weak strategies)
#     then allocates samples across the surviving strategies.
def lasso_sample_selection_with_proportions(embeddings, num_samples, base_prop=0.3):
    n = embeddings.shape[0]


    base_indices = {
        "Random Seed (RS)": random_seed_sampling(embeddings, int(n * base_prop)),
        "Grid Sampling (GS)": grid_sampling(embeddings, int(n * base_prop)),
        "Max-Min Distance (MMDS)": max_min_distance_sampling(embeddings, int(n * base_prop)),
        "Density-based (DBS)": density_based_sampling(embeddings, int(n * base_prop)),
        "Max Entropy (MES)": max_entropy_sampling(embeddings, int(n * base_prop)),
        "Cluster Sampling (CS)": cluster_sampling(embeddings, int(n * base_prop)),
    }


    strategy_features = np.zeros((n, len(base_indices) * 4))  # 每個策略4個特徵
    for i, (strategy_name, indices) in enumerate(base_indices.items()):
        selected_embeddings = embeddings[indices]
        

        distances = np.linalg.norm(embeddings[:, np.newaxis] - selected_embeddings, axis=2)
        

        min_distances = np.min(distances, axis=1)
        mean_distances = np.mean(distances, axis=1)
        
        distance_probs = distances / (np.sum(distances, axis=1, keepdims=True) + 1e-10)
        entropy = -np.sum(distance_probs * np.log(distance_probs + 1e-10), axis=1)
        
        diversity = np.std(distances, axis=1)
        
        
        strategy_features[:, i*4] = min_distances
        strategy_features[:, i*4+1] = mean_distances
        strategy_features[:, i*4+2] = entropy
        strategy_features[:, i*4+3] = diversity

    scaler = StandardScaler()
    strategy_features = scaler.fit_transform(strategy_features)

    center = np.mean(embeddings, axis=0)
    distances_to_center = np.linalg.norm(embeddings - center, axis=1)
    
    nn = NearestNeighbors(n_neighbors=min(5, len(embeddings)))
    nn.fit(embeddings)
    distances, _ = nn.kneighbors(embeddings)
    density_scores = np.exp(-np.sum(distances, axis=1))
    
    entropy_scores = -np.sum(distances * np.log(distances + 1e-10), axis=1)
    
    target = (distances_to_center * (entropy_scores ** 2)) / (density_scores ** 0.5)

    target = (target - np.mean(target)) / np.std(target)

    param_grid = {
        'alpha': [0.0001, 0.001, 0.01]  # 使用較小的 alpha 值以減少稀疏性
    }
    lasso_cv = GridSearchCV(
        Lasso(random_state=42, max_iter=10000, tol=1e-4),
        param_grid,
        cv=5,
        scoring='neg_mean_squared_error'
    )
    lasso_cv.fit(strategy_features, target)
    model = lasso_cv.best_estimator_

    strategy_weights = np.zeros(len(base_indices))
    for i in range(len(base_indices)):
        strategy_weights[i] = np.mean(np.abs(model.coef_[i*4:(i+1)*4]))
    
    mes_idx = list(base_indices.keys()).index("Max Entropy (MES)")
    min_mes_weight = np.max(strategy_weights) * 0.8  # MES 至少要有最大權重的 80%
    if strategy_weights[mes_idx] < min_mes_weight:
        strategy_weights[mes_idx] = min_mes_weight
    
    strategy_weights = strategy_weights / np.sum(strategy_weights)

    # Step 7: 分配樣本
    strategy_samples = {}
    total_allocated = 0
    min_samples_per_strategy = max(1, int(num_samples * 0.1))
    
    sorted_strategies = sorted(base_indices.keys(), 
                             key=lambda x: strategy_weights[list(base_indices.keys()).index(x)],
                             reverse=True)
    
    for strategy_name in sorted_strategies:
        strategy_idx = list(base_indices.keys()).index(strategy_name)
        if strategy_weights[strategy_idx] > 0:
            strategy_samples[strategy_name] = min_samples_per_strategy
            total_allocated += min_samples_per_strategy

    remaining_samples = num_samples - total_allocated
    if remaining_samples > 0 and strategy_samples:
        valid_strategies = list(strategy_samples.keys())
        valid_weights = np.array([strategy_weights[list(base_indices.keys()).index(s)] 
                                for s in valid_strategies])
        valid_weights = valid_weights / np.sum(valid_weights)
        
        for i in range(len(valid_strategies)-1):
            strategy_name = valid_strategies[i]
            additional_samples = int(remaining_samples * valid_weights[i])
            strategy_samples[strategy_name] += additional_samples
            total_allocated += additional_samples
        
        final_remaining = num_samples - total_allocated
        last_strategy = valid_strategies[-1]
        strategy_samples[last_strategy] += final_remaining

    all_indices = []
    proportions = {}
    for strategy_name, n_samples in strategy_samples.items():
        if n_samples > 0:
            sampling_function = SAMPLING_METHODS.get(strategy_name)
            if sampling_function is None:
                raise KeyError(f"Sampling method {strategy_name} not found.")
            indices = sampling_function(embeddings, n_samples)
            if isinstance(indices, tuple):
                indices = indices[0]
            all_indices.extend(indices)
            proportions[strategy_name] = n_samples / num_samples

    all_indices = np.unique(all_indices)
    if len(all_indices) > num_samples:
        selected_indices = np.random.choice(all_indices, num_samples, replace=False)
    else:
        remaining = num_samples - len(all_indices)
        if remaining > 0:
            additional_indices = random_seed_sampling(embeddings, remaining)
            selected_indices = np.concatenate([all_indices, additional_indices])
        else:
            selected_indices = all_indices

    return selected_indices, proportions

# ════ (STRATEGY) ════
#  Ridge (L2 regularization): like Lasso but spreads weights smoothly (no hard zeros),
#     with a special boost for the Max-Entropy (MES) strategy weight.
def ridge_sample_selection_with_proportions(embeddings, num_samples, base_prop=0.3):
    n = embeddings.shape[0]
    """
    加強版 Ridge 特徵選擇策略，強化 MES 特徵對最終取樣比例的影響。
    
    主要步驟：
    1. 使用每個基礎策略產生初始樣本（較大量），並對整個數據集計算「到該策略樣本最小距離」作為特徵。
    2. 對特徵做標準化（StandardScaler）。
    3. 定義複合目標值（距中心距離 * 負密度）。
    4. 使用 GridSearchCV 對 Ridge 進行交叉驗證，取得最佳 alpha。
    5. 根據模型係數計算策略權重，並強化 MES 係數。
    6. 根據最終權重，按兩輪分配樣本（先分配最小數量，再按權重分配剩餘）。
    7. 產生最終選取的索引與各策略所佔比例。
    """
    

    base_indices = {
        "Random Seed (RS)": random_seed_sampling(embeddings, int(n * base_prop)),
        "Grid Sampling (GS)": grid_sampling(embeddings, int(n * base_prop)),
        "Max-Min Distance (MMDS)": max_min_distance_sampling(embeddings, int(n * base_prop)),
        "Density-based (DBS)": density_based_sampling(embeddings, int(n * base_prop)),
        "Max Entropy (MES)": max_entropy_sampling(embeddings, int(n * base_prop)),
        "Cluster Sampling (CS)": cluster_sampling(embeddings, int(n * base_prop)),
    }

  
    strategy_features = np.zeros((n, len(base_indices)))
    for i, (strategy_name, indices) in enumerate(base_indices.items()):
        selected_embeddings = embeddings[indices]

        distances = np.min(
            np.linalg.norm(embeddings[:, np.newaxis] - selected_embeddings, axis=2),
            axis=1
        )
        strategy_features[:, i] = distances


    scaler = StandardScaler()
    strategy_features = scaler.fit_transform(strategy_features)


    center = np.mean(embeddings, axis=0)
    distances_to_center = np.linalg.norm(embeddings - center, axis=1)
    
    nn = NearestNeighbors(n_neighbors=min(5, len(embeddings)))
    nn.fit(embeddings)
    density_scores = -np.sum(nn.kneighbors(embeddings)[0], axis=1)
    
    target = distances_to_center * density_scores


    param_grid = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0]}
    ridge_cv = GridSearchCV(
        Ridge(random_state=42, max_iter=10000),
        param_grid,
        cv=5,
        scoring='neg_mean_squared_error'
    )
    ridge_cv.fit(strategy_features, target)
    model = ridge_cv.best_estimator_

    coef_threshold = np.std(model.coef_) * 0.1 
    strategy_weights = np.abs(model.coef_)
    strategy_weights[strategy_weights < coef_threshold] = 0.0

    if np.sum(strategy_weights) == 0:
        logging.warning("All Ridge coefficients are below threshold. Using uniform weights.")
        strategy_weights = np.ones(len(base_indices)) / len(base_indices)
    else:
        strategy_weights = strategy_weights / np.sum(strategy_weights)

    
    mes_name = "Max Entropy (MES)"
    mes_index = list(base_indices.keys()).index(mes_name)

    mes_boost_factor = 1.2
    strategy_weights[mes_index] *= mes_boost_factor
    strategy_weights /= np.sum(strategy_weights)  


    strategy_samples = {}
    total_allocated = 0
    min_samples_per_strategy = max(1, int(num_samples * 0.1))  # 最小樣本數（10%）
    

    sorted_strategies = sorted(
        base_indices.keys(),
        key=lambda x: strategy_weights[list(base_indices.keys()).index(x)],
        reverse=True
    )
    

    for strategy_name in sorted_strategies:
        strategy_idx = list(base_indices.keys()).index(strategy_name)
        if strategy_weights[strategy_idx] > 0:
            strategy_samples[strategy_name] = min_samples_per_strategy
            total_allocated += min_samples_per_strategy


    remaining_samples = num_samples - total_allocated
    

    if remaining_samples > 0 and strategy_samples:
        valid_strategies = list(strategy_samples.keys())
        valid_weights = np.array([
            strategy_weights[list(base_indices.keys()).index(s)] 
            for s in valid_strategies
        ])
        valid_weights = valid_weights / np.sum(valid_weights)
        
        for i in range(len(valid_strategies) - 1):
            strategy_name = valid_strategies[i]
            additional_samples = int(remaining_samples * valid_weights[i])
            strategy_samples[strategy_name] += additional_samples
            total_allocated += additional_samples
        

        final_remaining = num_samples - total_allocated
        last_strategy = valid_strategies[-1]
        strategy_samples[last_strategy] += final_remaining


    assert sum(strategy_samples.values()) == num_samples, "Sample allocation error"


    all_indices = []
    proportions = {}
    for strategy_name, n_samples in strategy_samples.items():
        if n_samples > 0:
            sampling_function = SAMPLING_METHODS.get(strategy_name)
            if sampling_function is None:
                raise KeyError(f"Sampling method {strategy_name} not found.")
            indices = sampling_function(embeddings, n_samples)
    

            if isinstance(indices, tuple):
                indices = indices[0]
            all_indices.extend(indices)
            proportions[strategy_name] = n_samples / num_samples


    if not math.isclose(sum(proportions.values()), 1.0, rel_tol=1e-9):
        logging.warning("Proportions do not sum to 1.0, got: %.5f", sum(proportions.values()))

 
    all_indices = np.unique(all_indices)
    if len(all_indices) > num_samples:
        selected_indices = np.random.choice(all_indices, num_samples, replace=False)
    else:
        remaining = num_samples - len(all_indices)
        if remaining > 0:
            additional_indices = random_seed_sampling(embeddings, remaining)
            selected_indices = np.concatenate([all_indices, additional_indices])
        else:
            selected_indices = all_indices

    return selected_indices, proportions


# ════  (STRATEGY) ════
#  Elastic Net: combines L1 + L2 (a mix of Lasso + Ridge) to distribute weights.
def elastic_net_sample_selection_with_proportions(embeddings, num_samples, base_prop=0.3):
    n = embeddings.shape[0]
    """ElasticNet 特徵選擇策略，基於完整數據的 L1 和 L2 正則化動態調整策略比例，並加強 MES 權重。"""

    base_indices = {
        "Random Seed (RS)": random_seed_sampling(embeddings, int(n * base_prop)),
        "Grid Sampling (GS)": grid_sampling(embeddings, int(n * base_prop)),
        "Max-Min Distance (MMDS)": max_min_distance_sampling(embeddings, int(n * base_prop)),
        "Density-based (DBS)": density_based_sampling(embeddings, int(n * base_prop)),
        "Max Entropy (MES)": max_entropy_sampling(embeddings, int(n * base_prop)),
        "Cluster Sampling (CS)": cluster_sampling(embeddings, int(n * base_prop)),
    }


    strategy_features = np.zeros((n, len(base_indices)))
    for i, (strategy_name, indices) in enumerate(base_indices.items()):
        selected_embeddings = embeddings[indices]

        distances = np.min(np.linalg.norm(embeddings[:, np.newaxis] - selected_embeddings, axis=2), axis=1)
        mean_distance = np.mean(np.linalg.norm(embeddings[:, np.newaxis] - selected_embeddings, axis=2), axis=1)
        max_distance = np.max(np.linalg.norm(embeddings[:, np.newaxis] - selected_embeddings, axis=2), axis=1)
       
        strategy_features[:, i] = 0.4 * distances + 0.3 * mean_distance + 0.3 * max_distance

   
    scaler = StandardScaler()
    strategy_features = scaler.fit_transform(strategy_features)

    
    center = np.mean(embeddings, axis=0)
    distances_to_center = np.linalg.norm(embeddings - center, axis=1)

    nn = NearestNeighbors(n_neighbors=min(5, len(embeddings)))
    nn.fit(embeddings)
    density_scores = -np.sum(nn.kneighbors(embeddings)[0], axis=1)

    variance = np.var(embeddings, axis=1)

    target = distances_to_center * density_scores * variance
    target = (target - np.mean(target)) / (np.std(target) + 1e-9)

    
    param_grid = {
        'alpha': [0.00001, 0.0001, 0.001],  # 更低的 alpha 範圍
        'l1_ratio': [0.1, 0.25, 0.5, 0.75, 0.9]
    }
    elastic_net_cv = GridSearchCV(
        ElasticNet(random_state=42, max_iter=10000, tol=1e-4),
        param_grid,
        cv=5,
        scoring='neg_mean_squared_error'
    )
    elastic_net_cv.fit(strategy_features, target)
    
    logging.info(f"Best ElasticNet parameters: {elastic_net_cv.best_params_}")
    model = elastic_net_cv.best_estimator_

   
    coef_abs = np.abs(model.coef_)
    coef_threshold = np.mean(coef_abs) - 0.5 * np.std(coef_abs)
    strategy_weights = coef_abs.copy()
    strategy_weights[strategy_weights < coef_threshold] = 0.0

    if np.sum(strategy_weights) == 0:

        strategy_weights = coef_abs.copy()

   
    mes_name = "Max Entropy (MES)"
    mes_index = list(base_indices.keys()).index(mes_name)
    mes_boost_factor = 1.2  
    strategy_weights[mes_index] *= mes_boost_factor

    
    strategy_weights_sum = np.sum(strategy_weights)
    if strategy_weights_sum > 0:
        strategy_weights /= strategy_weights_sum
    else:
        strategy_weights = np.ones(len(base_indices)) / len(base_indices)

    for name, weight in zip(base_indices.keys(), strategy_weights):
        logging.info(f"{name}: {weight:.4f}")

 

    strategy_samples = {}
    total_allocated = 0
    min_samples_per_strategy = max(1, int(num_samples * 0.1))

    # 依照權重排序
    sorted_strategies = sorted(
        base_indices.keys(),
        key=lambda x: strategy_weights[list(base_indices.keys()).index(x)],
        reverse=True
    )

    for strategy_name in sorted_strategies:
        strategy_idx = list(base_indices.keys()).index(strategy_name)
        if strategy_weights[strategy_idx] > 0:
            strategy_samples[strategy_name] = min_samples_per_strategy
            total_allocated += min_samples_per_strategy
            logging.info(f"Initial allocation for {strategy_name}: {min_samples_per_strategy}")

    remaining_samples = num_samples - total_allocated
    logging.info(f"Remaining samples after initial allocation: {remaining_samples}")

    if remaining_samples > 0 and strategy_samples:
        valid_strategies = list(strategy_samples.keys())
        valid_weights = np.array([
            strategy_weights[list(base_indices.keys()).index(s)]
            for s in valid_strategies
        ])
        valid_weights /= (np.sum(valid_weights) + 1e-9)

        for i in range(len(valid_strategies) - 1):
            strategy_name = valid_strategies[i]
            additional_samples = int(remaining_samples * valid_weights[i])
            strategy_samples[strategy_name] += additional_samples
            total_allocated += additional_samples
            logging.info(f"Additional allocation for {strategy_name}: {additional_samples}")

        final_remaining = num_samples - total_allocated
        last_strategy = valid_strategies[-1]
        strategy_samples[last_strategy] += final_remaining
        logging.info(f"Final allocation for {last_strategy}: {final_remaining}")

    total_samples = sum(strategy_samples.values())
    assert total_samples == num_samples, f"Sample allocation error: got {total_samples}, expected {num_samples}"

   
    all_indices = []
    proportions = {}
    for strategy_name, n_samples in strategy_samples.items():
        if n_samples > 0:
            sampling_function = SAMPLING_METHODS.get(strategy_name)
            if sampling_function is None:
                raise KeyError(f"Sampling method {strategy_name} not found.")
            indices = sampling_function(embeddings, n_samples)
            if isinstance(indices, tuple):
                indices = indices[0]
            all_indices.extend(indices)
            proportions[strategy_name] = n_samples / num_samples
            logging.info(f"Final proportion for {strategy_name}: {proportions[strategy_name]:.4f}")

    total_proportion = sum(proportions.values())
    assert math.isclose(total_proportion, 1.0, rel_tol=1e-9), \
        f"Proportion error: got {total_proportion}, expected 1.0"

    all_indices = np.unique(all_indices)
    if len(all_indices) > num_samples:
        selected_indices = np.random.choice(all_indices, num_samples, replace=False)
    else:
        remaining = num_samples - len(all_indices)
        if remaining > 0:
            additional_indices = random_seed_sampling(embeddings, remaining)
            selected_indices = np.concatenate([all_indices, additional_indices])
        else:
            selected_indices = all_indices

    return selected_indices, proportions


# ════   (STRATEGY) ════
#  Equal proportion: splits samples equally across the 6 base strategies (num/6 each).
def equal_proportion_sampling(embeddings, num_samples):
    """
    均等分配每個策略的樣本數量。
    """
    num_per_strategy = num_samples // 6  
    all_indices = []

    proportions = {}  

    for strategy_name in ["Random Seed (RS)", "Grid Sampling (GS)", 
                          "Max-Min Distance (MMDS)", "Density-based (DBS)", 
                          "Max Entropy (MES)", "Cluster Sampling (CS)"]:
        sampling_function = SAMPLING_METHODS.get(strategy_name)
        if sampling_function is None:
            raise KeyError(f"Sampling method {strategy_name} not found.")
        
        indices, _ = sampling_function(embeddings, num_per_strategy)
        all_indices.extend(indices)

        proportions[strategy_name] = num_per_strategy / num_samples

    all_indices = np.unique(all_indices)
    if len(all_indices) > num_samples:
        selected_indices = np.random.choice(all_indices, num_samples, replace=False)
    elif len(all_indices) < num_samples:
        additional_indices = random_seed_sampling(embeddings, num_samples - len(all_indices))
        selected_indices = np.concatenate([all_indices, additional_indices])
    else:
        selected_indices = all_indices

    total_allocated = sum(proportions.values())
    for strategy_name in proportions:
        proportions[strategy_name] /= total_allocated

    return selected_indices, proportions



SAMPLING_METHODS = {
    "Lasso Selection (LS)_20": partial(lasso_sample_selection_with_proportions, base_prop=0.2),
    "Lasso Selection (LS)_30": partial(lasso_sample_selection_with_proportions, base_prop=0.3),
    "Lasso Selection (LS)_40": partial(lasso_sample_selection_with_proportions, base_prop=0.4),

    "Ridge Selection (RidgeS)_20": partial(ridge_sample_selection_with_proportions, base_prop=0.2),
    "Ridge Selection (RidgeS)_30": partial(ridge_sample_selection_with_proportions, base_prop=0.3),
    "Ridge Selection (RidgeS)_40": partial(ridge_sample_selection_with_proportions, base_prop=0.4),

    "Elastic Net (EN)_20": partial(elastic_net_sample_selection_with_proportions, base_prop=0.2),
    "Elastic Net (EN)_30": partial(elastic_net_sample_selection_with_proportions, base_prop=0.3),
    "Elastic Net (EN)_40": partial(elastic_net_sample_selection_with_proportions, base_prop=0.4),

    "Random Forest (RF)_20": partial(random_forest_combination_with_proportions, base_prop=0.2),
    "Random Forest (RF)_30": partial(random_forest_combination_with_proportions, base_prop=0.3),
    "Random Forest (RF)_40": partial(random_forest_combination_with_proportions, base_prop=0.4),

    "Random Seed (RS)": random_seed_sampling_with_proportions,
    "Grid Sampling (GS)": grid_sampling_with_proportions,
    "Max-Min Distance (MMDS)": max_min_distance_sampling_with_proportions,
    "Density-based (DBS)": density_based_sampling_with_proportions,
    "Max Entropy (MES)": max_entropy_sampling_with_proportions,
    "Cluster Sampling (CS)": cluster_sampling_with_proportions,
    "Equal Proportion Sampling (EPS)": equal_proportion_sampling
}




# ════ تقييم: LCS ════
#  Computes the Longest Common Subsequence (LCS) length between two strings via DP.
def lcs_length(x, y):
    """計算字串 x, y 的最長共同子序列 (LCS) 長度。"""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


# ════ تقييم: LCS ════
#  Partial-match score = LCS length ÷ longer string length (a value in 0..1).
def compute_lcs_score(pred_term, ref_term):
    """計算 pred_term 與 ref_term 的 LCS 部分匹配分數。"""
    if not pred_term or not ref_term:
        return 0.0
    length = lcs_length(pred_term, ref_term)
    return length / max(len(pred_term), len(ref_term))

# ════ تقييم (HUNGARIAN) ════
#  Computes F1 for one field. Per review it builds a similarity matrix between
#     predictions and gold, then linear_sum_assignment (Hungarian algorithm) finds the
#     best one-to-one matching maximizing the total. use_lcs=True for text fields
#     (partial), False for exact match.
def compute_f1_score(predictions, references, task_key, use_lcs=False):
    """
    使用匈牙利演算法 (Hungarian algorithm) 計算多對多 partial match 的 F1 分數。
    
    predictions, references: List of List of dict
    task_key: str, 要比對的欄位，例如 "span", "polarity", "opinion_term"
    use_lcs: 是否使用 LCS 計算相似度（用於提取任務）
    """
    n = len(predictions)
    total_precision, total_recall = 0.0, 0.0

    for pred, ref in zip(predictions, references):
        pred_terms = [d[task_key] for d in pred]
        ref_terms = [d[task_key] for d in ref]

        if len(pred_terms) == 0 and len(ref_terms) == 0:
            total_precision += 1.0
            total_recall += 1.0
            continue

        if len(pred_terms) == 0 or len(ref_terms) == 0:
            continue

        score_matrix = np.zeros((len(pred_terms), len(ref_terms)), dtype=np.float32)
        for i, pterm in enumerate(pred_terms):
            for j, rterm in enumerate(ref_terms):
                if use_lcs:
                    score_matrix[i][j] = compute_lcs_score(pterm, rterm)
                else:
                    score_matrix[i][j] = 1.0 if pterm == rterm else 0.0

        cost_matrix = 1.0 - score_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        total_match_score = sum(score_matrix[row_ind[i], col_ind[i]] for i in range(len(row_ind)))

        precision = total_match_score / len(pred_terms)
        recall = total_match_score / len(ref_terms)

        total_precision += precision
        total_recall += recall

    avg_precision = total_precision / n
    avg_recall = total_recall / n

    if (avg_precision + avg_recall) > 0:
        final_f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)
    else:
        final_f1 = 0.0

    return {
        "precision": avg_precision,
        "recall": avg_recall,
        "f1": final_f1,
    }


#  Distributes num_samples across strategies by weight (Largest-Remainder method:
#     assign integer parts first, then give leftovers to the largest fractional parts).
def allocate_samples(strategy_weights, num_samples):
    """
    基於權重分配樣本數，確保總和為 num_samples
    
    Args:
        strategy_weights: 各策略的權重
        num_samples: 總樣本數
    """
    normalized_weights = strategy_weights / np.sum(strategy_weights)
    
    theoretical_samples = normalized_weights * num_samples
    
    allocated_samples = np.floor(theoretical_samples).astype(int)
    remaining_samples = num_samples - np.sum(allocated_samples)
    
    decimal_parts = theoretical_samples - allocated_samples
    if remaining_samples > 0:

        indices = np.argsort(decimal_parts)[-int(remaining_samples):]
        allocated_samples[indices] += 1
    
    return allocated_samples

# ════ (CHECKPOINT) ════
#  Saves/restores experiment progress so it can resume after any interruption.
class CheckpointManager:
    #  Constructor: sets the path of the checkpoint.json file.
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.checkpoint_file = os.path.join(output_dir, "checkpoint.json")
        
    #  Records the current fold of the running strategy into the checkpoint file.
    def save_checkpoint(self, sampling_method, current_fold):
        checkpoint = self.load_checkpoint() or {"completed_methods": {}}

        if sampling_method in checkpoint["completed_methods"]:
            completed_status = checkpoint["completed_methods"][sampling_method].get("completed", False)
        else:
            completed_status = False

        checkpoint["completed_methods"][sampling_method] = {
            "current_fold": current_fold,
            "completed": completed_status
        }

        with open(self.checkpoint_file, "w") as f:
            json.dump(checkpoint, f, indent=2)
        logging.info(f"Checkpoint saved: {self.checkpoint_file}")


            
    # Reads the checkpoint file if it exists, otherwise returns None.
    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, "r") as f:
                return json.load(f)
        return None
        
    #  Marks a strategy as "completed" once all its folds are done (to skip it later).
    def mark_completed(self, sampling_method):

        checkpoint = self.load_checkpoint() or {"completed_methods": {}}
        
        if sampling_method in checkpoint["completed_methods"]:
            checkpoint["completed_methods"][sampling_method]["completed"] = True
        else:
            checkpoint["completed_methods"][sampling_method] = {
                "current_fold": 5,
                "completed": True
            }
        
        with open(self.checkpoint_file, "w") as f:
            json.dump(checkpoint, f, indent=2)


# ════════════════════════════════════════════════════════════════════
# Main experiment class — runs the full lifecycle for ONE sampling strategy
# ════════════════════════════════════════════════════════════════════
class Experiment:
    
    #  Constructor: stores the strategy name+function and loads the sentence encoder
    #     all-MiniLM-L6-v2 (turns text into 384-dim vectors).
    def __init__(self, sampling_name="Random Seed"):
        self.sampling_name = sampling_name
        self.sampling_func = SAMPLING_METHODS[sampling_name]
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.models = {}
        self.fold_metrics = []
        # 初始化 SentenceTransformer
        self.model = SentenceTransformer(MODEL_NAME)

    # ──  البيانات ──
    #  Loads the JaquanTW/fewshot-absaquad dataset (full train + test); no sampling yet.
    def load_data(self):
        """只加載原始數據，不進行採樣"""
        logging.info(f"Loading datasets...")
        dataset = load_dataset("JaquanTW/fewshot-absaquad")
        self.full_dataset = dataset["train"]  # 保存完整訓練集
        self.test_dataset = dataset["test"]
        logging.info(f"Full training dataset size: {len(self.full_dataset)}")
        logging.info(f"Test dataset size: {len(self.test_dataset)}")

    # ──  أخذ العينات ──
    #  Encodes the fold's texts into vectors, then applies the chosen strategy to
    #     select 50 training examples (TRAIN_SIZE). Returns the picked data + proportions.
    def sample_fold_data(self, fold_data):
        """對每個 fold 的訓練數據進行採樣"""
        logging.info(f"Applying sampling strategy: {self.sampling_name}")
        
        texts = [x["text"] for x in fold_data]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        if self.sampling_name in ["Lasso Selection (LS)", "Elastic Net (EN)", 
                                "Random Forest (RF)", "Ridge Selection (RidgeS)"]:
            sampled_indices, proportions = self.sampling_func(embeddings, TRAIN_SIZE)
        else:
            result = self.sampling_func(embeddings, TRAIN_SIZE)
            if isinstance(result, tuple):
                sampled_indices, proportions = result
            else:
                sampled_indices = result
                proportions = {self.sampling_name: 1.0}

        if isinstance(sampled_indices, np.ndarray):
            sampled_indices = sampled_indices.flatten().astype(int).tolist()
        elif isinstance(sampled_indices, list):
            sampled_indices = [int(i) for i in sampled_indices]
        elif isinstance(sampled_indices, tuple):
            sampled_indices = [int(i) for i in np.ravel(sampled_indices)]

        sampled_data = fold_data.select(sampled_indices)
        logging.info(f"Sampled data size: {len(sampled_data)}")
        return sampled_data, proportions
    
    #  Helper: trains and evaluates the model on a single fold, returns its metrics.
    def train_and_evaluate_fold(self, train_data, val_data, fold_num):
        """在單個fold上訓練和評估模型"""
        logging.info(f"Training and evaluating fold {fold_num + 1}/{N_FOLDS}")
        
        self.train_dataset = train_data
        self.val_dataset = val_data
        
        self.train_quad_model(fold_num)

        metrics = self.evaluate(fold_num)
        return metrics

    #  التدريب (SetFit التبايني) ──
    #  Trains ONE SetFit model that predicts the whole quad as text "span|ac|ot|label".
    #     SetFit = contrastive fine-tuning of the sentence encoder + a classifier head.
    #     make_quad_label merges the four fields into a single label per example.
    def train_quad_model(self, fold_num):
        """Train unified quadruple model."""
        logging.info(f"Training Quad model for fold {fold_num + 1}...")

        def make_quad_label(x):
            span = (x["span"] or "").strip()
            ac   = (x["ac"]   or "").strip()
            ot   = x["ot"] if not isinstance(x["ot"], list) else " ".join(x["ot"])
            ot   = (ot or "").strip()
            lbl  = (x["label"] or "").strip()
            return {"text": x["text"], "label": f"{span}|{ac}|{ot}|{lbl}"}

        train_args = TrainingArguments(
            output_dir=f"{OUTPUT_DIR}/fold_{fold_num + 1}/quad",
            num_epochs=5,
            batch_size=8,
            body_learning_rate=2e-5,
            head_learning_rate=1e-3,
        )
        unique_labels = list(set(
            f"{r['span']}|{r['ac']}|{(r['ot'] if not isinstance(r['ot'], list) else ' '.join(r['ot']))}|{r['label']}"
            for r in self.train_dataset
        ))
        quad_model = SetFitModel.from_pretrained(MODEL_NAME, labels=unique_labels)
        trainer = Trainer(
            model=quad_model,
            args=train_args,
            train_dataset=self.train_dataset.map(make_quad_label),
            eval_dataset=self.val_dataset.map(make_quad_label),
        )
        trainer.train()
        self.models["quad"] = quad_model
        quad_model.save_pretrained(f"{OUTPUT_DIR}/fold_{fold_num + 1}/quad_model")

    # ──  التقييم الصارم ──
    # Exact quad match: score 1 only if all four fields match, then Hungarian
    #     finds the best assignment, and Precision/Recall/F1 are computed.
    def compute_quad_exact_match(self, gold_data, pred_data):
        n = len(gold_data)
        totP, totR = 0.0, 0.0
        for i in range(n):
            gold_t, pred_t = gold_data[i], pred_data[i]
            if len(gold_t) == 0 and len(pred_t) == 0:
                totP += 1; totR += 1; continue
            if len(gold_t) == 0 or len(pred_t) == 0:
                continue
            score_mat = np.zeros((len(gold_t), len(pred_t)), dtype=np.float32)
            for g_i, g4 in enumerate(gold_t):
                for p_i, p4 in enumerate(pred_t):
                    if g4[0]==p4[0] and g4[1]==p4[1] and g4[2]==p4[2] and g4[3]==p4[3]:
                        score_mat[g_i][p_i] = 1.0
            cost = 1 - score_mat
            row_ind, col_ind = linear_sum_assignment(cost)
            match = sum(score_mat[row_ind[k], col_ind[k]] for k in range(len(row_ind)))
            totP += match / len(pred_t)
            totR += match / len(gold_t)
        avgP, avgR = totP / n, totR / n
        f1 = 2 * avgP * avgR / (avgP + avgR) if (avgP + avgR) > 0 else 0.0
        return avgP, avgR, f1

    # ──  التقييم الجزئي ──
    #  Partial quad match: category & label exact, while span & ot use LCS;
    #     pair score = (ac + label + LCS(span) + LCS(ot)) / 4, then Hungarian + F1.
    def compute_quad_partial_match(self, gold_data, pred_data):
        n = len(gold_data)
        totP, totR = 0.0, 0.0
        for i in range(n):
            gold_t, pred_t = gold_data[i], pred_data[i]
            if len(gold_t) == 0 and len(pred_t) == 0:
                totP += 1; totR += 1; continue
            if len(gold_t) == 0 or len(pred_t) == 0:
                continue
            score_mat = np.zeros((len(gold_t), len(pred_t)), dtype=np.float32)
            for g_i, g4 in enumerate(gold_t):
                for p_i, p4 in enumerate(pred_t):
                    ac_s = 1.0 if g4[0] == p4[0] else 0.0
                    lb_s = 1.0 if g4[1] == p4[1] else 0.0
                    sp_s = compute_lcs_score(p4[2], g4[2])
                    ot_s = compute_lcs_score(p4[3], g4[3])
                    score_mat[g_i][p_i] = (ac_s + lb_s + sp_s + ot_s) / 4.0
            cost = 1 - score_mat
            row_ind, col_ind = linear_sum_assignment(cost)
            total = sum(score_mat[row_ind[k], col_ind[k]] for k in range(len(row_ind)))
            totP += total / len(pred_t)
            totR += total / len(gold_t)
        avgP, avgR = totP / n, totR / n
        f1 = 2 * avgP * avgR / (avgP + avgR) if (avgP + avgR) > 0 else 0.0
        return avgP, avgR, f1


    #  Runs the model on eval data (val during a fold, test at the end), splits the
    #     "span|ac|ot|label" output back into a quad, then computes Exact + Partial.
    def evaluate(self, fold_num=None):
        """Evaluate the quad model and return results."""
        logging.info(f"Evaluating models for fold {fold_num + 1 if fold_num is not None else 'final'}...")
        eval_dataset = self.val_dataset if fold_num is not None else self.test_dataset
        gold_data, pred_data = [], []

        all_texts = [example["text"] for example in eval_dataset]
        raw_preds = self.models["quad"].predict(all_texts)

        for example, raw_pred in zip(eval_dataset, raw_preds):
            parts = raw_pred.split("|")
            if len(parts) == 4:
                p_span, p_ac, p_ot, p_label = parts
            else:
                p_span, p_ac, p_ot, p_label = "", "", "", ""

            g_span  = example.get("span", "")
            g_ac    = example.get("ac", "")
            raw_ot  = example.get("ot", "")
            g_ot    = " ".join(raw_ot) if isinstance(raw_ot, list) else (raw_ot or "")
            g_label = example.get("label", "")

            gold_data.append([(g_span, g_ac, g_ot.strip(), g_label)])
            pred_data.append([(p_span.strip(), p_ac.strip(), p_ot.strip(), p_label.strip())])

        ex_p, ex_r, ex_f = self.compute_quad_exact_match(gold_data, pred_data)
        pm_p, pm_r, pm_f = self.compute_quad_partial_match(gold_data, pred_data)

        metrics = {
            "quad": {
                "exact_match":   {"precision": ex_p, "recall": ex_r, "f1": ex_f},
                "partial_match": {"precision": pm_p, "recall": pm_r, "f1": pm_f},
            }
        }
        if fold_num is not None:
            metrics["Fold"] = fold_num + 1
        metrics["Sample Proportions"] = self.sample_proportions

        logging.info(f"Quad Exact Match:   {metrics['quad']['exact_match']}")
        logging.info(f"Quad Partial Match: {metrics['quad']['partial_match']}")
        return metrics

    #  Averages the metrics across all 5 folds (the cross-validation result).
    def calculate_average_metrics(self):
        """Calculate average metrics across all folds."""
        avg_metrics = {
            "quad": {
                "exact_match":   {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                "partial_match": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            }
        }
        for metrics in self.fold_metrics:
            for match_type in ["exact_match", "partial_match"]:
                for key in ["precision", "recall", "f1"]:
                    avg_metrics["quad"][match_type][key] += metrics["quad"][match_type][key]
        n_folds = len(self.fold_metrics)
        for match_type in ["exact_match", "partial_match"]:
            for key in ["precision", "recall", "f1"]:
                avg_metrics["quad"][match_type][key] /= n_folds
        return avg_metrics

    # ════ المُشغّل الرئيسي (الانتقال من مرحلة لأخرى) ════
    #  Full flow: restore checkpoint → load data → 5-fold split →
    #     per fold: sample → train SetFit → evaluate → save → then average + final test.
    def run(self):
        """執行完整的實驗流程，包含斷點恢復機制"""
        checkpoint_manager = CheckpointManager(OUTPUT_DIR)
        checkpoint = checkpoint_manager.load_checkpoint()

        if (checkpoint and "completed_methods" in checkpoint and 
            self.sampling_name in checkpoint["completed_methods"] and
            checkpoint["completed_methods"][self.sampling_name].get("completed", False)):
            logging.info(f"{self.sampling_name} already completed. Skipping.")
            return
        
        self.load_data()
        
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        dataset_indices = list(range(len(self.full_dataset)))
        
        start_fold = 0
        if checkpoint and "completed_methods" in checkpoint and self.sampling_name in checkpoint["completed_methods"]:
            start_fold = checkpoint["completed_methods"][self.sampling_name]["current_fold"]
            logging.info(f"Resuming from fold {start_fold + 1}")
        
        for fold_num, (train_idx, val_idx) in enumerate(kf.split(dataset_indices)):
            if fold_num < start_fold:
                continue
                
            logging.info(f"\nStarting fold {fold_num + 1}/{N_FOLDS}")
            checkpoint_manager.save_checkpoint(self.sampling_name, fold_num)
            
            train_fold = self.full_dataset.select(train_idx.tolist())
            val_fold = self.full_dataset.select(val_idx.tolist())
            
            self.train_dataset, self.sample_proportions = self.sample_fold_data(train_fold)

            self.val_dataset = val_fold
            
            logging.info(f"Training set size after sampling: {len(self.train_dataset)}")
            logging.info(f"Validation set size: {len(self.val_dataset)}")
            
            try:
                self.train_quad_model(fold_num)
                
                fold_metrics = self.evaluate(fold_num)
                self.fold_metrics.append(fold_metrics)
                save_results(OUTPUT_DIR, self.sampling_name, fold_metrics, fold_num)
                
            except Exception as e:
                logging.error(f"Error in fold {fold_num + 1}: {str(e)}")
                raise e
            
            logging.info(f"Completed fold {fold_num + 1}/{N_FOLDS}")
        
        avg_metrics = self.calculate_average_metrics()
        avg_metrics["Sample_Proportions"] = self.sample_proportions
        save_results(OUTPUT_DIR, f"{self.sampling_name}_5fold_average", avg_metrics)
        
        test_metrics = self.evaluate()
        save_results(OUTPUT_DIR, f"{self.sampling_name}_test", test_metrics)
        
        checkpoint_manager.mark_completed(self.sampling_name)
        logging.info("Experiment completed successfully")

        
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    checkpoint_manager = CheckpointManager(OUTPUT_DIR)
    checkpoint = checkpoint_manager.load_checkpoint()
    
    if checkpoint and "completed_methods" in checkpoint:
        uncompleted_methods = [
            method for method, status in checkpoint["completed_methods"].items() 
            if not status["completed"]
        ]
        if uncompleted_methods:
            logging.info(f"Found uncompleted methods: {uncompleted_methods}")
            start_method = uncompleted_methods[0]
        else:
            logging.info("All methods completed. Exiting.")
            exit()
    else:
        start_method = None


    methods_to_run = list(SAMPLING_METHODS.keys())
    if start_method:

        start_idx = methods_to_run.index(start_method)
        methods_to_run = methods_to_run[start_idx:]
        
    for sampling_name in methods_to_run:
        logging.info(f"Running experiment with sampling method: {sampling_name}")
        experiment = Experiment(sampling_name=sampling_name)
        experiment.run()