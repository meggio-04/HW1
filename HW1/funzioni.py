import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

""" 

    DEGRADATION 

"""
# Degradazione sintetica: riduce il contrasto e aggiunge rumore gaussiano.
# alpha: fattore di contrazione (0-1, piu' basso = meno contrasto)
# center: livello attorno a cui si contrae la dinamica (128 = grigio medio)
# noise_sigma: deviazione standard del rumore gaussiano (media zero)
# seed: seme del generatore casuale, rende il rumore riproducibile
def degrade_image(img, alpha, center, noise_sigma, seed):
    rng = np.random.default_rng(seed)
    # g = alpha * (f - center) + center, calcolato in float32 per evitare overflow
    degraded = alpha * (img.astype(np.float32) - center) + center
    noise= rng.normal(0, noise_sigma, img.shape).astype(np.float32)
    # Arrotonda, satura in [0, 255] e torna a uint8
    return np.clip(np.rint(degraded + noise), 0, 255).astype(np.uint8)

# Variante della degradazione con trasformazione lineare g = gain * f + bias.
# gain/bias: coefficienti della trasformazione; noise_sigma e seed come sopra.
# Non e' usata nella pipeline principale.
def degrade_image_alternative(img, gain, bias, noise_sigma, seed):
    rng = np.random.default_rng(seed)
    # 1. Riduzione del contrasto 
    low_contrast = cv2.convertScaleAbs(img, alpha=gain, beta=bias)
    # 2. Rumore gaussiano
    noise = rng.normal(0, noise_sigma, img.shape).astype(np.float32)
    # Il rumore si somma in float32 per non andare in overflow, poi si satura in [0, 255]
    noisy_img = np.clip(low_contrast.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy_img

""" 
    
    PIPELINE ENHANCEMENT METHODS 
    
"""

""" 1. BASELINE """

# Riferimento: la degradata senza alcuna elaborazione
def method_baseline(img):
    return img.copy()

""" 2. CONTRAST STRETCHING """

# Stretching lineare sulla sola luminanza L di CIE-Lab.
# low_perc/high_perc: percentili che diventano 0 e 255 (ignorano i pixel estremi)
# a e b non vengono toccati, quindi la tinta non cambia.
def method_stretching_LAB(img, low_perc, high_perc):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = l.astype(np.float32)

    p_low = np.percentile(l, low_perc)
    p_high = np.percentile(l, high_perc)
    # max(...) evita la divisione per zero su immagini quasi piatte
    l_stretched = np.clip((l - p_low) * (255.0 / max(p_high - p_low, 1e-5)), 0, 255).astype(np.uint8)

    return cv2.cvtColor(cv2.merge((l_stretched, a, b)), cv2.COLOR_LAB2BGR)

# Stretching lineare sul solo canale V di HSV (parametri come sopra).
# H e S restano invariati, ma la saturazione percepita puo' cambiare.
def method_stretching_HSV(img, low_perc, high_perc):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # float32 per evitare overflow/underflow di uint8
    v = v.astype(np.float32)

    p_low = np.percentile(v, low_perc)
    p_high = np.percentile(v, high_perc)

    v_stretched = np.clip((v - p_low) * (255.0 / max((p_high - p_low), 1e-5)), 0, 255).astype(np.uint8)

    return cv2.cvtColor(cv2.merge((h, s, v_stretched)), cv2.COLOR_HSV2BGR)


# Stretching indipendente su ciascun canale B, G, R (parametri come sopra).
# Ogni canale ha i propri percentili, quindi cambiano i rapporti B:G:R
def method_stretching_RGB(img, low_perc, high_perc):
    out = np.zeros_like(img)
    for i in range(3):
        channel = img[:, :, i].astype(np.float32)
        p_low, p_high = np.percentile(channel, (low_perc, high_perc))
        out[:, :, i] = np.clip((channel - p_low) * (255.0 / max(p_high - p_low, 1e-5)), 0, 255).astype(np.uint8)
    return out

""" 3. HISTOGRAM EQUALIZATION """

# Equalizzazione dell'istogramma sulla sola L di CIE-Lab: i colori (a, b) non cambiano.
# Distribuisce i livelli in modo non lineare, quindi puo' amplificare il rumore.
def method_histogramEqualization_LAB(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = cv2.equalizeHist(l)
    return cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2BGR)

# Equalizzazione sul solo canale V di HSV: H e S restano invariati,
# ma i colori possono apparire piu' slavati o piu' saturi.
def method_histogramEqualization_HSV(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v_eq = cv2.equalizeHist(v)
    return cv2.cvtColor(cv2.merge((h, s, v_eq)), cv2.COLOR_HSV2BGR)

""" 4. DENOISING + CLAHE + BOOST """

# Sfocatura gaussiana (kernel 5x5, sigma 2) dei canali cromatici a e b:
def smoothing(a, b):
    a_smooth = cv2.GaussianBlur(a, (5, 5), 2)
    b_smooth = cv2.GaussianBlur(b, (5, 5), 2)
    return a_smooth, b_smooth

# Guadagno cromatico: scala a e b attorno al neutro (128).
# chroma_gain > 1 aumenta la saturazione, < 1 la riduce.
def boost(a, b, chroma_gain):
    a = np.clip((a.astype(np.float32) - 128) * chroma_gain + 128, 0, 255).astype(np.uint8)
    b = np.clip((b.astype(np.float32) - 128) * chroma_gain + 128, 0, 255).astype(np.uint8)
    return a, b

# Metodo combinato: denoising NLM -> stretching di L -> CLAHE -> guadagno cromatico.
# h, h_color: forza del denoising su luminanza e colore (piu' alto = piu' rumore tolto,
#             ma anche piu' dettagli smussati)
# clip_limit: limite di contrasto di CLAHE (basso = meno amplificazione del rumore)
# tile_grid: numero di blocchi in cui CLAHE divide l'immagine
# chroma_gain: guadagno cromatico applicato da boost
def method_denoise_LAB(img, h, h_color, clip_limit, tile_grid, chroma_gain):
    # 1. Non-local means sul colore: media patch simili in tutta l'immagine
    denoised = cv2.fastNlMeansDenoisingColored(img, None, h, h_color, 7, 21)

    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 2. Stretching lineare della luminanza sui percentili 1-99
    l_float = l.astype(np.float32)
    p_low, p_high = np.percentile(l_float, (1, 99))
    l_stretched = np.clip((l_float - p_low) * (255.0 / max(p_high - p_low, 1e-5)), 0, 255).astype(np.uint8)

    # 3. CLAHE: equalizzazione adattiva locale con clipLimit contenuto
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_clahe = clahe.apply(l_stretched)

    # 4. Filtro bilaterale su L (leviga le zone piatte preservando i bordi), non è stato usato nella pipeline finale
    #l_smooth = cv2.bilateralFilter(l_clahe, d=7, sigmaColor=20, sigmaSpace=10)

    # 5. Smoothing dei canali cromatici e guadagno.
    a_smooth, b_smooth = smoothing(a, b)
    a_boost, b_boost = boost(a_smooth, b_smooth, chroma_gain)

    # Ricomposizione: L da CLAHE, a e b con guadagno; conversione di nuovo in BGR
    return cv2.cvtColor(cv2.merge((l_clahe, a_boost, b_boost)), cv2.COLOR_LAB2BGR)
""" 

    METRICS
    
"""

# PSNR in dB rispetto all'originale: piu' alto e' meglio. Vale inf se le immagini coincidono.
def psnr(orig, res):
    mse = np.mean((orig.astype(np.float32) - res.astype(np.float32)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)

# Errore assoluto medio rispetto all'originale: piu' basso e' meglio
def mae(orig, res):
    return float(np.mean(np.abs(orig.astype(np.float32) - res.astype(np.float32))))

# SSIM (scikit-image) sui 3 canali; data_range=255 perche' le immagini sono uint8
def Ssim(orig, res):
    return ssim(orig, res, channel_axis=2, data_range=255)