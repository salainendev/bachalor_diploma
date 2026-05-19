import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score

from dev.utils import save_model_with_architecture, make_hard_negative_pairs
from dev.augmentations import augment_scaled_parameters

# ------ evaluate multi head regression ------

def testing_model(model, loader, y_scaler, DEVICE, target_names, logging=True, use_oracle_metrics=True, save = False, **kwargs):
    model.eval()
    all_targets_scaled = []
    all_preds_heads = []      # для oracle
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.cpu().numpy()

            all_preds = model(x_batch)   # [batch, K, out_dim], [batch, K]

            all_targets_scaled.append(y_batch)

            all_preds_heads.append(all_preds.cpu().numpy())

    y_true_scaled = np.concatenate(all_targets_scaled, axis=0)
    y_true_orig = y_scaler.inverse_transform(y_true_scaled)

    if logging:
        print("Метрики по выбранному предсказанию (max confidence):\n")

    mape_oracle = {}
    if use_oracle_metrics:
        all_heads = np.concatenate(all_preds_heads, axis=0)
        
       
        mse_per_head = np.mean((all_heads - y_true_scaled[:, np.newaxis, :])**2, axis=2)
        best_head_idx = np.argmin(mse_per_head, axis=1)          
        best_preds = all_heads[np.arange(all_heads.shape[0]), best_head_idx]
        y_pred_oracle_orig = y_scaler.inverse_transform(best_preds)

        if logging:
            print("\nМетрики по лучшей голове (oracle):\n")

    # Вычисляем метрики для каждой переменной
    for i, name in enumerate(target_names):
        true_vals = y_true_orig[:, i]

        # Метрики для выбранного предсказания


        if use_oracle_metrics:
            pred_oracle = y_pred_oracle_orig[:, i]
            mape_ora = np.mean(np.abs((true_vals - pred_oracle) / (np.abs(true_vals) + 1e-7))) * 100
            if logging:
                print(f"{name} (oracle): MAPE = {mape_ora:.4f}%")
                mape_oracle[name] = mape_ora
            # Можно также сохранить oracle метрики для анализа (например, в глобальные списки)

        if logging:
            print("-" * 30)

    # Сохранение модели по лучшему MAPE для каждой переменной (по выбранному предсказанию)
    # Используем глобальные списки metrics_xuv, metrics_masslos, metrics_helium
    # (они должны быть определены в области видимости функции)
    metrics_xuv = kwargs.get('metrics_xuv', [])
    metrics_masslos = kwargs.get('metrics_masslos', [])
    metrics_helium = kwargs.get('metrics_helium', [])

    if save:
        for i, name in enumerate(target_names):
            mape_val = mape_oracle[name]
            if name == "(XUVInt)":
                metrics_xuv.append(mape_val)
                if mape_val <= min(metrics_xuv):
                    if save:
                        save_model_with_architecture(model, "best_xuv.pkl")
            elif name == "log(Msw)":
                metrics_masslos.append(mape_val)
                if mape_val <= min(metrics_masslos):
                    if save:
                        save_model_with_architecture(model, "best_msw.pkl")
            elif name == "(Helium)":
                metrics_helium.append(mape_val)
                if mape_val <= min(metrics_helium):
                    if save:
                        save_model_with_architecture(model, "best_helium.pkl")

       
            

    # Возвращаем метрики для дальнейшего анализа (опционально)
    return {
        "oracle_mape": mape_oracle,
        "oracle_head":best_head_idx.numpy()
    }

# --------- evaluate metric learning models -------


@torch.no_grad()
def validate_with_negatives(model, loader, device, _scaler, neg_type='shuffle', margin=None):
    model.eval()
    cos = nn.CosineSimilarity(dim=-1)
    all_sim_pos = []
    all_sim_neg = []
    
    for S, a, b in loader:
        S, a, b = S.to(device), a.to(device), b.to(device)
        a, b = augment_scaled_parameters(a, b, _scaler, 
                                                        rel_noise=0.1, mode='gaussian')
        B = S.size(0)
        
        # Позитивные эмбеддинги
        z_spec, z_param_pos,_,_  = model(S, a, b)   # (B, D) each
        
        # Генерация негативных параметров
        if neg_type == 'shuffle':
            a_neg, b_neg = make_hard_negative_pairs(a, b)
        else:
            raise ValueError(f"Unknown neg_type: {neg_type}")
        
        # Эмбеддинги негативных параметров (спектр тот же)
        _, z_param_neg,_,_  = model(S, a_neg, b_neg)   # (B, D)
        
        # Косинусные сходства
        sim_pos = cos(z_spec, z_param_pos)   # (B,)
        sim_neg = cos(z_spec, z_param_neg)   # (B,)
        
        all_sim_pos.append(sim_pos.cpu())
        all_sim_neg.append(sim_neg.cpu())
    
    all_sim_pos = torch.cat(all_sim_pos)   # (N,)
    all_sim_neg = torch.cat(all_sim_neg)   # (N,)
    
    # Метрики
    sim_pos_mean = all_sim_pos.mean().item()
    sim_neg_mean = all_sim_neg.mean().item()
    gap = sim_pos_mean - sim_neg_mean
    
    # Доля отвергнутых негативов (по порогу)
    threshold = 0.75   # можно передавать параметром
    neg_rejected = (all_sim_neg < threshold).float().mean().item()
    
    # AUC: позитивы vs негативы
    scores = torch.cat([all_sim_pos, all_sim_neg]).numpy()
    labels = np.concatenate([np.ones(len(all_sim_pos)), np.zeros(len(all_sim_neg))])
    auc = roc_auc_score(labels, scores)
    
    return {
        'sim_pos': sim_pos_mean,
        'sim_neg': sim_neg_mean,
        'gap': gap,
        'neg_rejected': neg_rejected,
        'auc': auc
    }