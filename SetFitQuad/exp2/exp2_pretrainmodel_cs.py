import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from datasets import load_dataset
from setfit import SetFitModel, Trainer
from setfit import TrainingArguments
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from scipy.optimize import linear_sum_assignment
import os
import torch
from sklearn.neighbors import NearestNeighbors 

logging.basicConfig(level=logging.INFO)

# Define models configuration
MODELS_CONFIG = {
    "all-MiniLM-L6-v2": {
        "path": "sentence-transformers/all-MiniLM-L6-v2",
        "type": "general",
        "description": "Fast and good quality general purpose model"
    },
    "paraphrase-TinyBERT-L6-v2": {
        "path": "sentence-transformers/paraphrase-TinyBERT-L6-v2",
        "type": "lightweight",
        "description": "Lightweight model optimized for paraphrase tasks"
    },
    "all-mpnet-base-v2": {
        "path": "sentence-transformers/all-mpnet-base-v2",
        "type": "general",
        "description": "Best quality general purpose model"
    },
    "multi-qa-mpnet-base-cos-v1": {
        "path": "sentence-transformers/multi-qa-mpnet-base-cos-v1",
        "type": "task-specific",
        "description": "Optimized for QA and semantic search"
    },
    "paraphrase-multilingual-mpnet-base-v2": {
        "path": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "type": "multilingual",
        "description": "High quality multilingual model"
    }
}

OUTPUT_DIR = "results"
SAMPLE_SIZE = 50
N_FOLDS = 5
RANDOM_SEED = 42

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

def compute_lcs_score(pred_term, ref_term):
    """計算 pred_term 與 ref_term 的 LCS 部分匹配分數。"""
    if not pred_term or not ref_term:
        return 0.0
    length = lcs_length(pred_term, ref_term)
    return length / max(len(pred_term), len(ref_term))

def compute_f1_score(predictions, references, task_key, use_lcs=False):
    """
    使用匈牙利演算法計算多對多 partial match 的 F1，F1 直接使用 P 和 R 計算。
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

    # 正確的 F1 計算公式
    if (avg_precision + avg_recall) > 0:
        final_f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)
    else:
        final_f1 = 0.0

    return {
        "precision": avg_precision,
        "recall": avg_recall,
        "f1": final_f1,
    }


class CheckpointManager:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.checkpoint_file = os.path.join(output_dir, "model_checkpoint.json")
        
    def save_checkpoint(self, model_name, current_fold):
        checkpoint = self.load_checkpoint() or {"completed_models": {}}
        logging.debug(f"Before save: {json.dumps(checkpoint, indent=2)}")  # Debug檢查
        if model_name in checkpoint["completed_models"]:
            completed_status = checkpoint["completed_models"][model_name].get("completed", False)
        else:
            completed_status = False

        checkpoint["completed_models"][model_name] = {
            "current_fold": current_fold,
            "completed": completed_status
        }
        with open(self.checkpoint_file, "w") as f:
            json.dump(checkpoint, f, indent=2)
        logging.info(f"Checkpoint saved: {self.checkpoint_file}")
        logging.debug(f"After save: {json.dumps(checkpoint, indent=2)}")  # Debug檢查

                
    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, "r") as f:
                return json.load(f)
        return None

    def mark_completed(self, model_name):
        checkpoint = self.load_checkpoint() or {"completed_models": {}}
        logging.info(f"Before marking {model_name} completed: {json.dumps(checkpoint, indent=2)}")
        
        checkpoint["completed_models"][model_name] = {
            "current_fold": N_FOLDS,
            "completed": True
        }

        with open(self.checkpoint_file, "w") as f:
            json.dump(checkpoint, f, indent=2)

        logging.info(f"Marked {model_name} as completed. Updated checkpoint: {json.dumps(checkpoint, indent=2)}")

class EnhancedExperiment:
    def __init__(self, model_name: str):
        self.model_config = MODELS_CONFIG[model_name]
        self.model_name = model_name
        self.model_path = self.model_config["path"]
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.models = {}
        self.fold_metrics = []
        
        # Result dict
        self.results = {
            "model_info": self.model_config,
            "metrics": {},
            "timestamp": datetime.now().isoformat(),
            "training_params": {
                "train_size": SAMPLE_SIZE,
                "n_folds": N_FOLDS
            }
        }
        
        # 設置輸出目錄
        self.model_output_dir = Path(OUTPUT_DIR) / model_name
        self.model_output_dir.mkdir(parents=True, exist_ok=True)
        
        # GPU/CPU 設置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logging.info(f"Using GPU: {gpu_name} with {gpu_memory:.2f}GB memory")
        else:
            logging.info("Using CPU for training")
        
        # 初始化 SentenceTransformer
        self.st_model = SentenceTransformer(self.model_path)
        self.st_model = self.st_model.to(self.device)

    def load_data(self):
        """加載數據集"""
        logging.info(f"Loading datasets for model: {self.model_name}...")
        dataset = load_dataset("JaquanTW/fewshot-absaquad")
        self.full_dataset = dataset["train"]
        self.test_dataset = dataset["test"]
        logging.info(f"Full dataset size: {len(self.full_dataset)}")
        logging.info(f"Test dataset size: {len(self.test_dataset)}")

    def cluster_sampling(self, embeddings, num_samples):
        """使用 KMeans 進行 cluster sampling"""
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=num_samples, random_state=RANDOM_SEED).fit(embeddings)
        
        cluster_indices = []
        for c in range(num_samples):
            members = np.where(kmeans.labels_ == c)[0]
            if len(members) > 0:
                cluster_indices.append(members[0])
                
        if len(cluster_indices) < num_samples:
            needed = num_samples - len(cluster_indices)
            all_indices = set(range(len(embeddings)))
            used = set(cluster_indices)
            remain_indices = list(all_indices - used)
            np.random.shuffle(remain_indices)
            cluster_indices.extend(remain_indices[:needed])
        
        return np.array(cluster_indices)


    def process_fold(self, train_fold, val_fold, fold_num):
        """處理每個 fold 的數據，只對訓練集採樣，驗證集保持完整"""
        logging.info(f"Processing fold {fold_num + 1} data...")

        # 只對訓練集進行採樣
        train_sampled, proportions = self.sample_data(train_fold, SAMPLE_SIZE)

        # 設置 self.sample_proportions，避免 AttributeError
        self.sample_proportions = proportions

        # 驗證集保持完整
        val_dataset = val_fold

        logging.info(f"Fold {fold_num + 1} - Train size: {len(train_sampled)}, Val size: {len(val_dataset)}")
        return train_sampled, val_dataset



    def max_entropy_sampling(self, embeddings, num_samples):
        """最大熵抽樣（Max Entropy Sampling, MES）"""
        nn = NearestNeighbors(n_neighbors=min(5, len(embeddings) - 1))
        nn.fit(embeddings)
        distances, _ = nn.kneighbors(embeddings)
        entropies = -np.sum(np.log(distances + 1e-10) * distances, axis=1)
        return np.argsort(entropies)[-num_samples:]

    def max_entropy_sampling_with_proportions(self, embeddings, num_samples):
        """最大熵抽樣 + 產生 proportions"""
        indices = self.max_entropy_sampling(embeddings, num_samples)  # 加上 self.
        proportions = {"Max Entropy (MES)": 1.0}
        return indices, proportions

    def sample_data(self, data, size, use_mes=False):
        """對數據進行抽樣，可選擇使用最大熵抽樣 (MES) 或 Cluster Sampling (CS)"""
        if len(data) <= size:
            logging.info(f"Data size {len(data)} is smaller than required size {size}, using full dataset")
            return data, {"full_dataset": 1.0}  

        logging.info(f"Generating embeddings for sampling...")
        texts = [x["text"] for x in data]
        embeddings = self.st_model.encode(texts, batch_size=32, show_progress_bar=True)

        if use_mes:
            sampled_indices, proportions = self.max_entropy_sampling_with_proportions(embeddings, size)  # 加上 self.
        else:
            sampled_indices = self.cluster_sampling(embeddings, size)
            proportions = {"Cluster Sampling (CS)": 1.0}

        sampled_data = data.select(sampled_indices)
        logging.info(f"Sampled size: {len(sampled_data)} (from {len(data)}) using {'MES' if use_mes else 'CS'}")

        return sampled_data, proportions
    
    def train_quad_model(self):
        """Train unified quadruple model."""
        logging.info("Training unified quadruple model...")

        def make_quad_label(x):
            span = (x["span"] or "").strip()
            ac   = (x["ac"]   or "").strip()
            ot   = x["ot"] if not isinstance(x["ot"], list) else " ".join(x["ot"])
            ot   = (ot or "").strip()
            lbl  = (x["label"] or "").strip()
            return {"text": x["text"], "label": f"{span}|{ac}|{ot}|{lbl}"}

        train_args = TrainingArguments(
            output_dir=str(self.model_output_dir / "quad"),
            num_epochs=5,
            batch_size=8,
            body_learning_rate=2e-5,
            head_learning_rate=1e-3,
        )
        unique_labels = list(set(
            f"{r['span']}|{r['ac']}|{(r['ot'] if not isinstance(r['ot'], list) else ' '.join(r['ot']))}|{r['label']}"
            for r in self.train_dataset
        ))
        quad_model = SetFitModel.from_pretrained(self.model_path, labels=unique_labels)
        trainer = Trainer(
            model=quad_model,
            args=train_args,
            train_dataset=self.train_dataset.map(make_quad_label),
            eval_dataset=self.val_dataset.map(make_quad_label),
        )
        trainer.train()
        self.models["quad"] = quad_model

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

    def _log_average_metrics(self):
        """記錄平均指標"""
        metrics = self.results["average_metrics"]
        logging.info(f"\nAverage Results for model: {self.model_name}")
        logging.info("=" * 50)
        for task, task_metrics in metrics.items():
            logging.info(f"\n{task.title()}:")
            for metric, value in task_metrics.items():
                logging.info(f"{metric}: {value:.4f}")

    def run(self):
        """執行完整的實驗流程"""
        checkpoint_manager = CheckpointManager(OUTPUT_DIR)
        checkpoint = checkpoint_manager.load_checkpoint()

        # 檢查是否已完成
        if (checkpoint and "completed_models" in checkpoint and 
            self.model_name in checkpoint["completed_models"] and
            checkpoint["completed_models"][self.model_name].get("completed", False)):
            logging.info(f"{self.model_name} 已完成，直接跳過。")
            return

        self.load_data()

        # 創建 KFold 分割器
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        dataset_indices = list(range(len(self.full_dataset)))

        # 確定起始 fold
        start_fold = 0
        if checkpoint and "completed_models" in checkpoint and self.model_name in checkpoint["completed_models"]:
            start_fold = checkpoint["completed_models"][self.model_name]["current_fold"]
            logging.info(f"Resuming {self.model_name} from fold {start_fold + 1}")

        # 執行每個 fold
        for fold_num, (train_idx, val_idx) in enumerate(kf.split(dataset_indices)):
            if fold_num < start_fold:
                continue

            logging.info(f"\nProcessing fold {fold_num + 1}/{N_FOLDS} for {self.model_name}")
            checkpoint_manager.save_checkpoint(self.model_name, fold_num)

            # 準備當前 fold 的數據
            train_fold = self.full_dataset.select(train_idx.tolist())
            val_fold = self.full_dataset.select(val_idx.tolist())
            self.train_dataset, self.val_dataset = self.process_fold(train_fold, val_fold, fold_num)

            try:
                # 訓練模型
                self.train_quad_model()

                # 評估和保存結果
                fold_metrics = self.evaluate(fold_num)
                self.fold_metrics.append(fold_metrics)
                results_file = os.path.join(OUTPUT_DIR, f"{self.model_name}_fold{fold_num + 1}_results.json")
                with open(results_file, 'w', encoding='utf-8') as f:
                    json.dump(fold_metrics, f, ensure_ascii=False, indent=2)
                    
            except Exception as e:
                logging.error(f"Error in fold {fold_num + 1} for {self.model_name}: {str(e)}")
                raise e

            logging.info(f"Completed fold {fold_num + 1}/{N_FOLDS}")

        # 完成所有fold後的處理
        self._save_final_results()
        checkpoint_manager.mark_completed(self.model_name)
        logging.info(f"Experiment completed for {self.model_name}")

def run_all_experiments():
    checkpoint_manager = CheckpointManager(OUTPUT_DIR)
    checkpoint = checkpoint_manager.load_checkpoint()

    if checkpoint and "completed_models" in checkpoint:
        logging.info("Loaded checkpoint with completed model statuses.")
    else:
        logging.info("No checkpoint found. Starting fresh.")
        checkpoint = {"completed_models": {}}

    all_results = {}
    for model_name in MODELS_CONFIG.keys():
        # 加強檢查已完成模型的邏輯
        completed = checkpoint["completed_models"].get(model_name, {}).get("completed", False)
        current_fold = checkpoint["completed_models"].get(model_name, {}).get("current_fold", 0)
        logging.info(f"Checking status for {model_name}: completed={completed}, current_fold={current_fold}")

        if completed:
            logging.info(f"Skipping {model_name}: already completed.")
            continue

        # 未完成模型執行實驗
        logging.info(f"Starting experiments with {model_name}")
        try:
            experiment = EnhancedExperiment(model_name)
            experiment.run()
            all_results[model_name] = experiment.results
            checkpoint_manager.mark_completed(model_name)  # 標記完成
        except Exception as e:
            logging.error(f"Error with {model_name}: {e}")
            continue  # 不影響其他模型

    # 保存比較結果
    comparative_results_file = Path(OUTPUT_DIR) / "comparative_results.json"
    with open(comparative_results_file, 'w', encoding='utf-8') as f:
        json.dump({"model_results": all_results}, f, ensure_ascii=False, indent=2)
    logging.info(f"Comparative results saved to {comparative_results_file}")


def generate_summary_report():
    comparative_file = Path(OUTPUT_DIR) / "comparative_results.json"
    if not comparative_file.exists():
        logging.error("No comparative results found. Please run experiments first.")
        return
    
    with open(comparative_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    from datetime import datetime
    report = {
        "timestamp": datetime.now().isoformat(),
        # 這裡可以再擴充對 results 做排名、比較等
        "message": "Summary not implemented yet"
    }
    
    report_file = Path(OUTPUT_DIR) / "summary_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    logging.info(f"\nSummary report saved to {report_file}")

if __name__ == "__main__":
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    logging.info("Starting all experiments...")
    
    checkpoint_manager = CheckpointManager(OUTPUT_DIR)
    checkpoint = checkpoint_manager.load_checkpoint()

    if checkpoint and "completed_models" in checkpoint:
        uncompleted_models = [
            model for model, status in checkpoint["completed_models"].items()
            if not status["completed"]
        ]
        if uncompleted_models:
            start_model = uncompleted_models[0]
        else:
            logging.info("All models completed.")
            exit()
    else:
        start_model = list(MODELS_CONFIG.keys())[0]

    for model_name in MODELS_CONFIG.keys():
        if model_name < start_model:
            continue
        logging.info(f"Starting experiment with model: {model_name}")
        experiment = EnhancedExperiment(model_name)
        experiment.run()
        torch.cuda.empty_cache()