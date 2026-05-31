import logging
import json
import time  # 紀錄實驗時間
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from datasets import load_dataset
from setfit import SetFitModel, Trainer
from setfit import TrainingArguments
from sentence_transformers import SentenceTransformer
import numpy as np
from scipy.optimize import linear_sum_assignment
import os
import torch

logging.basicConfig(level=logging.INFO)

# -----------------------
# 使用多語系 mpnet 模型
# -----------------------
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"

# -----------------------
# 主要參數設定
# -----------------------
OUTPUT_DIR = "results"
TRAIN_SIZE = 50  # cluster sampling
RANDOM_SEED = 42

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# -----------------------
# LCS & partial match 函式
# -----------------------
def lcs_length(x, y):
    m, n= len(x), len(y)
    dp=[[0]*(n+1) for _ in range(m+1)]
    for i in range(1,m+1):
        for j in range(1,n+1):
            if x[i-1]==y[j-1]:
                dp[i][j]=dp[i-1][j-1]+1
            else:
                dp[i][j]=max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]

def compute_lcs_score(pred_term, ref_term):
    if not pred_term or not ref_term:
        return 0.0
    l= lcs_length(pred_term, ref_term)
    return l/ max(len(pred_term), len(ref_term))

def compute_f1_score(predictions, references, task_key, use_lcs=False):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    n = len(predictions)
    total_precision, total_recall = 0.0, 0.0

    for pred, ref in zip(predictions, references):
        pred_list = [d[task_key] for d in pred]
        ref_list = [d[task_key] for d in ref]
        
        if len(pred_list) == 0 and len(ref_list) == 0:
            total_precision += 1.0
            total_recall += 1.0
            continue
            
        if len(pred_list) == 0 or len(ref_list) == 0:
            continue

        score_mat = np.zeros((len(pred_list), len(ref_list)), dtype=np.float32)
        for i, pterm in enumerate(pred_list):
            for j, rterm in enumerate(ref_list):
                if use_lcs:
                    score_mat[i][j] = compute_lcs_score(pterm, rterm)
                else:
                    score_mat[i][j] = 1.0 if pterm == rterm else 0.0
                    
        cost = 1 - score_mat
        row_ind, col_ind = linear_sum_assignment(cost)
        match_score = sum(score_mat[row_ind[k], col_ind[k]] for k in range(len(row_ind)))

        precision = match_score / len(pred_list)
        recall = match_score / len(ref_list)
        
        total_precision += precision
        total_recall += recall

    avg_precision = total_precision / n
    avg_recall = total_recall / n
    
    # Calculate F1 using averaged precision and recall
    if (avg_precision + avg_recall) > 0:
        final_f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)
    else:
        final_f1 = 0.0

    return {
        "precision": avg_precision,
        "recall": avg_recall,
        "f1": final_f1,
    }

class SingleRunExperiment:
    def __init__(self):
        self.model_name= MODEL_NAME
        self.device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.st_model= SentenceTransformer(self.model_name).to(self.device)

        self.model_output_dir= Path(OUTPUT_DIR)/self.model_name
        self.model_output_dir.mkdir(parents=True, exist_ok=True)
        self.results={"model": self.model_name, "train_size": TRAIN_SIZE}
        self.models={}

    def load_data(self):
        logging.info("Loading dataset: JaquanTW/fewshot-absaquad")
        dataset= load_dataset("JaquanTW/fewshot-absaquad")
        self.full_train_dataset= dataset["train"]
        self.val_dataset       = dataset["validation"]
        self.test_dataset      = dataset["test"]
        logging.info(f"Train={len(self.full_train_dataset)}, Val={len(self.val_dataset)}, Test={len(self.test_dataset)}")

    def cluster_sampling(self, data, num_samples):
        from sklearn.cluster import KMeans
        texts=[x["text"] for x in data]
        logging.info(f"Generating embeddings for cluster sampling: {len(texts)} samples...")
        emb= self.st_model.encode(texts, batch_size=32, show_progress_bar=True)
        kmeans= KMeans(n_clusters=num_samples, random_state=RANDOM_SEED).fit(emb)
        cluster_indices=[]
        for c in range(num_samples):
            members=np.where(kmeans.labels_==c)[0]
            if len(members)>0:
                cluster_indices.append(members[0])
        if len(cluster_indices)< num_samples:
            needed= num_samples- len(cluster_indices)
            all_inds= set(range(len(data)))
            used= set(cluster_indices)
            remain=list(all_inds- used)
            np.random.shuffle(remain)
            cluster_indices.extend(remain[:needed])
        subset= data.select(cluster_indices)
        return subset

    def train_quad_model(self):
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

        quad_model = SetFitModel.from_pretrained(self.model_name, labels=unique_labels)

        trainer = Trainer(
            model=quad_model,
            args=train_args,
            train_dataset=self.train_dataset.map(make_quad_label),
            eval_dataset=self.val_dataset.map(make_quad_label),
        )
        trainer.train()
        self.models["quad"] = quad_model

    def evaluate(self, dataset, desc="Validation"):
        logging.info(f"Evaluating on {desc} set...")

        # ── batch predict ──
        all_texts = [example["text"] for example in dataset]
        all_raw_preds = self.models["quad"].predict(all_texts, batch_size=32)

        gold_data, pred_data = [], []
        for example, raw_pred in zip(dataset, all_raw_preds):
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
        logging.info(f"{desc} metrics: {json.dumps(metrics, indent=2)}")
        return metrics

    def compute_quad_exact_match(self, gold_data, pred_data):
        """
        exact => (ac,label,span,ot)全部相同 => score=1 else 0
        """
        import numpy as np
        n = len(gold_data)
        totP, totR = 0, 0
        for i in range(n):
            gold_t = gold_data[i]
            pred_t = pred_data[i]
            if len(gold_t)==0 and len(pred_t)==0:
                totP += 1
                totR += 1
                continue
            if len(gold_t)==0 or len(pred_t)==0:
                continue
                
            score_mat = np.zeros((len(gold_t), len(pred_t)), dtype=np.float32)
            for g_i, g4 in enumerate(gold_t):
                for p_i, p4 in enumerate(pred_t):
                    if g4[0]==p4[0] and g4[1]==p4[1] and g4[2]==p4[2] and g4[3]==p4[3]:
                        score_mat[g_i][p_i] = 1.0
                        
            cost = 1-score_mat
            row_ind, col_ind = linear_sum_assignment(cost)
            match_count = sum(score_mat[row_ind[k], col_ind[k]] for k in range(len(row_ind)))
            precision = match_count/len(pred_t)
            recall = match_count/len(gold_t)
            totP += precision
            totR += recall
            
        avgP = totP/n
        avgR = totR/n
        f1 = 2*avgP*avgR/(avgP+avgR) if (avgP+avgR)>0 else 0
        return avgP, avgR, f1

    def compute_quad_partial_match(self, gold_data, pred_data):
        """
        partial => ac,label exact(0or1), span,ot => lcs(0~1)
        quad_score = (ac_s + lb_s + span_lcs + ot_lcs)/4
        """
        import numpy as np
        n = len(gold_data)
        totP, totR = 0, 0
        for i in range(n):
            gold_t = gold_data[i]
            pred_t = pred_data[i]
            if len(gold_t)==0 and len(pred_t)==0:
                totP += 1
                totR += 1
                continue
            if len(gold_t)==0 or len(pred_t)==0:
                continue
                
            score_mat = np.zeros((len(gold_t), len(pred_t)), dtype=np.float32)
            for g_i,g4 in enumerate(gold_t):
                for p_i,p4 in enumerate(pred_t):
                    ac_score = 1 if (g4[0]==p4[0]) else 0
                    lb_score = 1 if (g4[1]==p4[1]) else 0
                    sp_lcs = compute_lcs_score(p4[2], g4[2])
                    ot_lcs = compute_lcs_score(p4[3], g4[3])
                    quad_s = (ac_score + lb_score + sp_lcs + ot_lcs)/4
                    score_mat[g_i][p_i] = quad_s
                    
            cost = 1-score_mat
            row_ind, col_ind = linear_sum_assignment(cost)
            sum_score = sum(score_mat[row_ind[k], col_ind[k]] for k in range(len(row_ind)))
            precision = sum_score/len(pred_t)
            recall = sum_score/len(gold_t)
            totP += precision
            totR += recall
            
        avgP = totP/n
        avgR = totR/n
        f1 = 2*avgP*avgR/(avgP+avgR) if (avgP+avgR)>0 else 0
        return avgP, avgR, f1

    def run(self):
        start_t= time.time()
        self.load_data()
        logging.info(f"Sampling {TRAIN_SIZE} from train set...")
        self.train_dataset= self.cluster_sampling(self.full_train_dataset, TRAIN_SIZE)
        logging.info("Using official val/test")

        self.train_quad_model()

        val_metrics= self.evaluate(self.val_dataset, "Validation")
        self.results["val_metrics"]= val_metrics

        test_metrics= self.evaluate(self.test_dataset, "Test")
        self.results["test_metrics"]= test_metrics

        end_t= time.time()
        self.results["total_experiment_time_sec"]= (end_t- start_t)
        logging.info(f"Total experiment time= {end_t- start_t:.2f}s")

        stamp= datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file= Path(OUTPUT_DIR)/ f"{self.model_name}_results_{stamp}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        logging.info(f"Results saved to: {out_file}")

if __name__=="__main__":
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
        gname= torch.cuda.get_device_name(0)
        gmem = torch.cuda.get_device_properties(0).total_memory/(1024**3)
        logging.info(f"Using GPU: {gname} ({gmem:.2f} GB)")
    else:
        logging.info("Using CPU")

    experiment= SingleRunExperiment()
    experiment.run()
