import torch
import dill

def save_model_with_architecture(model, filepath='complete_model.pkl'):
    """
    Сохраняет модель ПОЛНОСТЬЮ: архитектура + веса + код класса
    """
    # Используем dill для сохранения всего, включая определение класса
    with open(filepath, 'wb') as f:
        torch.save(model, f, pickle_module=dill)
    print(f"✅ Модель со всей архитектурой сохранена в {filepath}")

# ===== ЗАГРУЗКА =====
def load_model_with_architecture(filepath='complete_model.pkl', device='cuda'):
    """
    Загружает модель ПОЛНОСТЬЮ, даже если класс не определён в текущем скоупе
    """
    with open(filepath, 'rb') as f:
        model = torch.load(f, map_location=device, pickle_module=dill)
    model.eval()
    print(f"✅ Модель со всей архитектурой загружена из {filepath}")
    return model

def make_hard_negative_pairs(a, b, n_bins=5):
    """
    Делит батч на бины по параметрам и берёт негативы из других бинов.
    Гарантирует, что негатив имеет существенно другие параметры.
    """
    B = a.size(0)
    device = a.device
    
    # Нормализуем для бинирования
    a_norm = (a - a.min()) / (a.max() - a.min() + 1e-8)
    b_norm = (b - b.min()) / (b.max() - b.min() + 1e-8)
    
    # Создаём 2D бины (n_bins × n_bins)
    a_bin = (a_norm * (n_bins - 1)).long().clamp(0, n_bins - 1)
    b_bin = (b_norm * (n_bins - 1)).long().clamp(0, n_bins - 1)
    bin_id = a_bin * n_bins + b_bin  # уникальный ID бина
    
    perm_indices = torch.zeros(B, dtype=torch.long, device=device)
    
    for i in range(B):
        my_bin = bin_id[i]
        # Кандидаты: все из ДРУГИХ бинов
        other_bins = bin_id != my_bin
        # И не сам себя
        other_bins[i] = False
        
        if other_bins.sum() > 0:
            # Случайный пример из другого бина
            candidates = torch.where(other_bins)[0]
            perm_indices[i] = candidates[torch.randint(0, len(candidates), (1,), device=device)]
        else:
            # Все в одном бине — берём farthest по расстоянию
            a_dist = (a_norm[i] - a_norm).abs()
            b_dist = (b_norm[i] - b_norm).abs()
            total_dist = a_dist + b_dist
            total_dist[i] = -1  # исключаем себя
            perm_indices[i] = total_dist.argmax()
    
    return a[perm_indices], b[perm_indices]

def make_hard_negative_pairs_soft_tradeoff(a, b, tradeoff_strength=0.3, n_bins=5):
    """
    Мягкий учет tradeoff: не жесткое правило, а взвешенная метрика.
    
    Args:
        tradeoff_strength: 0-1, насколько сильно учитывать корреляцию
        0 =完全不 учитывать (стандартное бинирование)
        1 = полностью учитывать (a*b ≈ const)
    """
    B = a.size(0)
    device = a.device
    
    # Нормализуем
    a_norm = (a - a.min()) / (a.max() - a.min() + 1e-8)
    b_norm = (b - b.min()) / (b.max() - b.min() + 1e-8)
    
    # 1. Стандартное расстояние в пространстве параметров
    param_dist = torch.abs(a_norm.unsqueeze(1) - a_norm.unsqueeze(0)) + \
                 torch.abs(b_norm.unsqueeze(1) - b_norm.unsqueeze(0))
    
    # 2. Штраф за "подозрительные" пары (которые могли бы быть эквивалентны из-за tradeoff)
    # Чем ближе a1*b1 к a2*b2, тем больше штраф
    product1 = a_norm * b_norm
    product2 = product1.unsqueeze(1)
    product_similarity = 1 - torch.abs(product1.unsqueeze(1) - product2) / (product1.max() - product1.min() + 1e-8)
    
    # 3. Комбинированная метрика
    # Для пар с похожим произведением УМЕНЬШАЕМ расстояние (они менее "hard")
    tradeoff_penalty = 1 - tradeoff_strength * product_similarity
    effective_dist = param_dist * tradeoff_penalty
    
    # 4. Исключаем слишком похожие (другой параметрический бин - опционально)
    if n_bins > 0:
        a_bin = (a_norm * (n_bins - 1)).long().clamp(0, n_bins - 1)
        b_bin = (b_norm * (n_bins - 1)).long().clamp(0, n_bins - 1)
        same_bin = (a_bin.unsqueeze(1) == a_bin.unsqueeze(0)) & \
                   (b_bin.unsqueeze(1) == b_bin.unsqueeze(0))
        same_bin.fill_diagonal_(False)
        effective_dist[same_bin] = 0  # штрафуем одинаковые бины
    
    # 5. Выбираем farthest с учетом tradeoff
    effective_dist.fill_diagonal_(-1)
    hardest_idx = effective_dist.argmax(dim=1)
    
    return a[hardest_idx], b[hardest_idx]