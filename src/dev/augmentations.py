import torch

def sharpen_spectrum(signal, amount=0.5, kernel_size=3):
    """
    Заострение спектра (unsharp mask) для 1D сигнала.
    Делает пики более выраженными и узкими.
    
    Args:
        signal: (B, L) или (L,)
        amount: сила заострения (0.2-0.8 оптимально)
        kernel_size: размер ядра сглаживания (3 или 5)
    """
    original_shape = signal.shape
    if signal.dim() == 1:
        signal = signal.unsqueeze(0)  # (1, L)
    
    B, L = signal.shape
    signal_unsqueezed = signal.unsqueeze(1)  # (B, 1, L)
    
    # Сглаживание
    blur = torch.nn.functional.avg_pool1d(
        signal_unsqueezed, kernel_size, stride=1, padding=kernel_size//2
    ).squeeze(1)  # (B, L)
    
    # Unsharp mask: sharpened = original + amount * (original - blur)
    sharpened = signal + amount * (signal - blur)
    
    if len(original_shape) == 1:
        sharpened = sharpened.squeeze(0)
    
    return sharpened

def sharpen_spectrum_adaptive(signal, percentile=90, max_amount=0.8):
    """
    Адаптивное заострение: сильнее для сигналов с острыми пиками.
    
    Args:
        signal: (B, L)
        percentile: процентиль для оценки "остроконечности"
        max_amount: максимальная сила заострения
    """
    # Оценка остроты пиков (отношение максимума к среднему)
    max_val = signal.max(dim=-1, keepdim=True)[0]
    mean_val = signal.abs().mean(dim=-1, keepdim=True)
    sharpness_ratio = max_val / (mean_val + 1e-8)
    
    # Нормируем силу заострения
    amount = (sharpness_ratio / sharpness_ratio.max()) * max_amount
    amount = amount.clamp(0.1, max_amount)
    
    return sharpen_spectrum(signal, amount=amount.mean().item(), kernel_size=3)

def add_realistic_noise(signal, snr_db=40):
    """Добавляет белый шум с заданным SNR."""
    signal_power = (signal ** 2).mean(dim=-1, keepdim=True)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise_std = torch.sqrt(noise_power)
    return signal + noise_std * torch.randn_like(signal)


def augment_spectrum_noise_and_sharpen(S, 
                                       snr_db=40,
                                       sharpen_prob=0.5,
                                       sharpen_amount=0.5,
                                       scale_range=(0.95, 1.05)):
    """
    Аугментация: шум (SNR) + заострение + масштабирование.
    """
    B, L = S.shape
    device = S.device
    S_aug = S.clone()
    
    # 1. Белый шум
    S_aug = add_realistic_noise(S_aug, snr_db=snr_db)
    
    # 2. Заострение
    if torch.rand(1) < sharpen_prob:
        S_aug = sharpen_spectrum_adaptive(S_aug, max_amount=sharpen_amount)
    
    # 3. Масштабирование
    scale = scale_range[0] + torch.rand(B, 1, device=device) * (scale_range[1] - scale_range[0])
    S_aug = S_aug * scale
    
    return S_aug

def augment_parameters(A, B, rel_noise=0.2, mode='uniform', eps=1e-6):
    """
    Добавляет относительный шум ±rel_noise к параметрам.
    mode: 'uniform' (равномерный в [-20%, +20%]) или 'gaussian' (~95% в пределах ±20%)
    eps: защита от деления на ноль и отрицательных значений (физические параметры > 0)
    """
    A, B = A.float().view(-1), B.float().view(-1)
    
    # Физические параметры не могут быть ≤ 0
    A_safe = torch.clamp(A, min=eps)
    B_safe = torch.clamp(B, min=eps)
    
    if mode == 'uniform':
        noise_A = torch.rand_like(A_safe) * 2 * rel_noise - rel_noise
        noise_B = torch.rand_like(B_safe) * 2 * rel_noise - rel_noise
    elif mode == 'gaussian':
        # std = rel_noise / 2 → ~95% значений попадут в ±20%
        noise_A = torch.randn_like(A_safe) * (rel_noise / 2.0)
        noise_B = torch.randn_like(B_safe) * (rel_noise / 2.0)
    else:
        raise ValueError("mode должен быть 'uniform' или 'gaussian'")
        
    return A_safe * (1 + noise_A), B_safe * (1 + noise_B)

def augment_scaled_parameters(A_scaled, B_scaled, scaler_ab, rel_noise=0.2, mode='uniform'):
    """
    Корректно добавляет ±20% шум к параметрам, уже отмасштабированным StandardScaler.
    """
    device = A_scaled.device
    
    # 1. Обратное преобразование в физические значения
    AB_scaled = torch.stack([A_scaled, B_scaled], dim=1).cpu().numpy()
    AB_orig = scaler_ab.inverse_transform(AB_scaled)
    
    # 2. Аугментация в физическом пространстве
    A_orig, B_orig = torch.from_numpy(AB_orig[:, 0]), torch.from_numpy(AB_orig[:, 1])
    A_aug, B_aug = augment_parameters(A_orig, B_orig, rel_noise, mode)
    
    # 3. Снова масштабируем и возвращаем на GPU
    AB_aug = torch.stack([A_aug, B_aug], dim=1).numpy()
    AB_aug_scaled = scaler_ab.transform(AB_aug)
    
    return (torch.tensor(AB_aug_scaled[:, 0], device=device, dtype=torch.float32),
            torch.tensor(AB_aug_scaled[:, 1], device=device, dtype=torch.float32))