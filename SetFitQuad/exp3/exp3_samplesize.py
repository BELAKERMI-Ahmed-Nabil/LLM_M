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
from scipy.optimize import linear_sum_assignment
import os
import torch

logging.basicConfig(level=logging.INFO)

MODEL_CONFIG = {
    "paraphrase-multilingual-mpnetbase-v2": {
        "path": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "type": "multilingual",
        "description": "High quality multilingual model"
    }
}

OUTPUT_DIR = "results"
SAMPLE_SIZES = [1, 10, 50, 100, 150, 200]
N_FOLDS = 5
RANDOM_SEED = 42

def lcs_length(x, y):
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
    def __init__(self, output_dir, sample_size):
        self.output_dir = output_dir
        self.sample_size = sample_size
        self.checkpoint_file = os.path.join(output_dir, f"model_checkpoint_{sample_size}.json")
        
    def save_checkpoint(self, model_name, current_fold):
        checkpoint = self.load_checkpoint() or {"completed_models": {}}
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

    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, "r") as f:
                return json.load(f)
        return None

    def mark_completed(self, model_name):
        checkpoint = self.load_checkpoint() or {"completed_models": {}}
        checkpoint["completed_models"][model_name] = {
            "current_fold": N_FOLDS,
            "completed": True
        }
        with open(self.checkpoint_file, "w") as f:
            json.dump(checkpoint, f, indent=2)

class EnhancedExperiment:
    def __init__(self, sample_size: int):
        self.model_name = "paraphrase-multilingual-mpnetbase-v2"
        self.sample_proportions = {"Cluster Sampling": 1.0} 
        self.model_config = MODEL_CONFIG[self.model_name]
        self.model_path = self.model_config["path"]
        self.sample_size = sample_size
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.models = {}
        self.fold_metrics = []
        
        self.results = {
            "model_info": self.model_config,
            "metrics": {},
            "timestamp": datetime.now().isoformat(),
            "training_params": {
                "train_size": sample_size,
                "n_folds": N_FOLDS
            }
        }
        
        self.model_output_dir = Path(OUTPUT_DIR) / f"{self.model_name}_{sample_size}"
        self.model_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logging.info(f"Using GPU: {gpu_name} with {gpu_memory:.2f}GB memory")
        else:
            logging.info("Using CPU for training")
        
        self.st_model = SentenceTransformer(self.model_path)
        self.st_model = self.st_model.to(self.device)

    def load_data(self):
        logging.info(f"Loading datasets...")
        dataset = load_dataset("JaquanTW/fewshot-absaquad")
        self.full_dataset = dataset["train"]
        self.test_dataset = dataset["test"]
        logging.info(f"Full dataset size: {len(self.full_dataset)}")
        logging.info(f"Test dataset size: {len(self.test_dataset)}")

    def cluster_sampling(self, embeddings, num_samples):
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

    def sample_data(self, data, size):
        if len(data) <= size:
            logging.info(f"Data size {len(data)} is smaller than required size {size}, using full dataset")
            return data
            
        logging.info(f"Generating embeddings for sampling...")
        texts = [x["text"] for x in data]
        embeddings = self.st_model.encode(texts, batch_size=32, show_progress_bar=True)
        
        sampled_indices = self.cluster_sampling(embeddings, size)
        sampled_data = data.select(sampled_indices)
        
        logging.info(f"Sampled size: {len(sampled_data)} (from {len(data)})")
        return sampled_data

    def process_fold(self, train_fold, val_fold, fold_num):
        logging.info(f"Processing fold {fold_num + 1} data...")
        train_sampled = self.sample_data(train_fold, self.sample_size)
        val_dataset = val_fold
        logging.info(f"Fold {fold_num + 1} - Train size: {len(train_sampled)}, Val size: {len(val_dataset)}")
        return train_sampled, val_dataset

    def process_fold_cached(self, train_fold, val_fold, fold_num, train_embeddings):
        """مثل process_fold لكن يستخدم embeddings محسوبة مسبقاً بدل إعادة حسابها."""
        logging.info(f"Processing fold {fold_num + 1} data (cached embeddings)...")
        size = self.sample_size
        if len(train_fold) <= size:
            train_sampled = train_fold
        else:
            sampled_indices = self.cluster_sampling(train_embeddings, size)
            train_sampled = train_fold.select(sampled_indices.tolist())
        logging.info(f"Fold {fold_num + 1} - Train size: {len(train_sampled)}, Val size: {len(val_fold)}")
        return train_sampled, val_fold
    
    def train_quad_model(self):
        logging.info("Training unified quadruple model...")

        # SetFit يحتاج على الأقل عينتين — نتعامل مع sample_size صغير جداً
        if len(self.train_dataset) < 2:
            logging.warning(
                f"Only {len(self.train_dataset)} sample(s) in training set. "
                "Using majority-label dummy model."
            )
            row = self.train_dataset[0]
            ot_val = row["ot"] if not isinstance(row["ot"], list) else " ".join(row["ot"])
            only_label = f"{(row['span'] or '').strip()}|{(row['ac'] or '').strip()}|{(ot_val or '').strip()}|{(row['label'] or '').strip()}"

            class _DummyQuadModel:
                def predict(self_, texts, **kwargs):
                    return [only_label] * len(texts)

            self.models["quad"] = _DummyQuadModel()
            return


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
        totP, totR = 0, 0
        for i in range(n):
            gold_t = gold_data[i]
            pred_t = pred_data[i]
            if len(gold_t) == 0 and len(pred_t) == 0:
                totP += 1; totR += 1; continue
            if len(gold_t) == 0 or len(pred_t) == 0:
                continue
            score_mat = np.zeros((len(gold_t), len(pred_t)), dtype=np.float32)
            for g_i, g4 in enumerate(gold_t):
                for p_i, p4 in enumerate(pred_t):
                    if g4 == p4:
                        score_mat[g_i][p_i] = 1.0
            cost = 1 - score_mat
            row_ind, col_ind = linear_sum_assignment(cost)
            match = sum(score_mat[row_ind[k], col_ind[k]] for k in range(len(row_ind)))
            totP += match / len(pred_t)
            totR += match / len(gold_t)
        avgP = totP / n
        avgR = totR / n
        f1 = 2 * avgP * avgR / (avgP + avgR) if (avgP + avgR) > 0 else 0
        return avgP, avgR, f1

    def compute_quad_partial_match(self, gold_data, pred_data):
        n = len(gold_data)
        totP, totR = 0, 0
        for i in range(n):
            gold_t = gold_data[i]
            pred_t = pred_data[i]
            if len(gold_t) == 0 and len(pred_t) == 0:
                totP += 1; totR += 1; continue
            if len(gold_t) == 0 or len(pred_t) == 0:
                continue
            score_mat = np.zeros((len(gold_t), len(pred_t)), dtype=np.float32)
            for g_i, g4 in enumerate(gold_t):
                for p_i, p4 in enumerate(pred_t):
                    ac_score = 1 if g4[0] == p4[0] else 0
                    lb_score = 1 if g4[1] == p4[1] else 0
                    sp_lcs = compute_lcs_score(p4[2], g4[2])
                    ot_lcs = compute_lcs_score(p4[3], g4[3])
                    score_mat[g_i][p_i] = (ac_score + lb_score + sp_lcs + ot_lcs) / 4
            cost = 1 - score_mat
            row_ind, col_ind = linear_sum_assignment(cost)
            s = sum(score_mat[row_ind[k], col_ind[k]] for k in range(len(row_ind)))
            totP += s / len(pred_t)
            totR += s / len(gold_t)
        avgP = totP / n
        avgR = totR / n
        f1 = 2 * avgP * avgR / (avgP + avgR) if (avgP + avgR) > 0 else 0
        return avgP, avgR, f1

    def evaluate(self, fold_num=None):
        logging.info(f"Evaluating for fold {fold_num + 1 if fold_num is not None else 'final'}...")
        eval_dataset = self.val_dataset if fold_num is not None else self.test_dataset

        # ── batch predict: جمع كل النصوص ثم دفعة واحدة ──
        all_texts = [example["text"] for example in eval_dataset]
        all_raw_preds = self.models["quad"].predict(all_texts, batch_size=32)

        gold_data, pred_data = [], []
        for example, raw_pred in zip(eval_dataset, all_raw_preds):
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
            metrics["fold"] = fold_num + 1

        logging.info(f"Metrics: {json.dumps(metrics, indent=2)}")
        return metrics

    def calculate_average_metrics(self):
        avg_metrics = {
            "quad": {
                "exact_match":   {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                "partial_match": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            }
        }
        for fm in self.fold_metrics:
            for match_type in ("exact_match", "partial_match"):
                for metric in ("precision", "recall", "f1"):
                    avg_metrics["quad"][match_type][metric] += fm["quad"][match_type][metric]
        n = len(self.fold_metrics)
        for match_type in ("exact_match", "partial_match"):
            for metric in ("precision", "recall", "f1"):
                avg_metrics["quad"][match_type][metric] /= n
        self.results["average_metrics"] = avg_metrics
        logging.info(f"\nAverage metrics for sample_size={self.sample_size}: {json.dumps(avg_metrics, indent=2)}")
        return avg_metrics

    def _save_final_results(self):
        avg_metrics = self.calculate_average_metrics()
        final_metrics = {"average_metrics": avg_metrics, "sample_size": self.sample_size}
        results_file = os.path.join(OUTPUT_DIR, f"{self.model_name}_{self.sample_size}_average_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(final_metrics, f, ensure_ascii=False, indent=2)

    def run(self):
        pass


def run_experiments():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    logging.info("Starting experiments with different sample sizes...")

    # ── تحميل البيانات مرة واحدة لجميع التجارب ──
    logging.info("Loading dataset once for all experiments...")
    dataset = load_dataset("JaquanTW/fewshot-absaquad")
    full_dataset = dataset["train"]
    test_dataset = dataset["test"]
    logging.info(f"Full dataset: {len(full_dataset)}, Test: {len(test_dataset)}")

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    dataset_indices = list(range(len(full_dataset)))
    all_fold_splits = list(kf.split(dataset_indices))

    for sample_size in SAMPLE_SIZES:
        logging.info(f"\n{'='*50}")
        logging.info(f"Starting sample_size={sample_size}")
        logging.info(f"{'='*50}")

        model_name = "paraphrase-multilingual-mpnetbase-v2"
        ckpt_manager = CheckpointManager(OUTPUT_DIR, sample_size)
        ckpt = ckpt_manager.load_checkpoint()

        if (ckpt and "completed_models" in ckpt
                and model_name in ckpt["completed_models"]
                and ckpt["completed_models"][model_name].get("completed", False)):
            logging.info(f"sample_size={sample_size} already completed, skipping.")
            continue

        # تحديد من أين نكمل
        start_fold = 0
        ckpt = ckpt_manager.load_checkpoint()
        if ckpt and "completed_models" in ckpt and model_name in ckpt["completed_models"]:
            start_fold = ckpt["completed_models"][model_name].get("current_fold", 0)
            if start_fold > 0:
                logging.info(f"Resuming from fold {start_fold + 1}")

        # إنشاء كائن التجربة مرة واحدة لكل sample_size
        experiment = EnhancedExperiment(sample_size=sample_size)
        experiment.full_dataset = full_dataset
        experiment.test_dataset = test_dataset

        # ── حساب الـ embeddings مرة واحدة للـ fold splits ──
        logging.info("Computing embeddings once for sampling (cached)...")
        all_texts = [x["text"] for x in full_dataset]
        cached_embeddings = experiment.st_model.encode(
            all_texts, batch_size=32, show_progress_bar=True
        )

        for fold_num, (train_idx, val_idx) in enumerate(all_fold_splits):
            if fold_num < start_fold:
                logging.info(f"Fold {fold_num+1} already done, skipping.")
                continue

            logging.info(f"\nFold {fold_num+1}/{N_FOLDS} — sample_size={sample_size}")
            ckpt_manager.save_checkpoint(model_name, fold_num)

            train_fold = full_dataset.select(train_idx.tolist())
            val_fold   = full_dataset.select(val_idx.tolist())

            # استخدام الـ embeddings المحسوبة مسبقاً للـ sampling
            train_embeddings = cached_embeddings[train_idx]
            experiment.train_dataset, experiment.val_dataset = experiment.process_fold_cached(
                train_fold, val_fold, fold_num, train_embeddings
            )

            experiment.train_quad_model()
            fold_metrics = experiment.evaluate(fold_num)
            experiment.fold_metrics.append(fold_metrics)

            results_file = os.path.join(
                OUTPUT_DIR,
                f"{model_name}_{sample_size}_fold{fold_num+1}_results.json"
            )
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(fold_metrics, f, ensure_ascii=False, indent=2)

            ckpt_manager.save_checkpoint(model_name, fold_num + 1)
            logging.info(f"Fold {fold_num+1} completed and saved.")
            torch.cuda.empty_cache()

        ckpt_manager.mark_completed(model_name)
        logging.info(f"All folds done for sample_size={sample_size}")


if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    run_experiments()