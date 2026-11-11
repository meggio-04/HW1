import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import funzioni as fz
from skimage.metrics import structural_similarity as ssim

# Parametri della degradazione, fissi per tutte le immagini 
ALPHA = 0.5        # fattore di contrazione del contrasto
CENTER = 128       # livello attorno a cui si contrae la dinamica
NOISE_SIGMA = 10   # deviazione standard del rumore gaussiano
SEED = 42          # seme per rendere il rumore riproducibile

# cartelle di output 
DEG_DIR = "results/degraded"
FIG_DIR = "results/figures"
PROC_DIR = "results/processed"
CSV_PATH = "results/risultati_metriche.csv"

# Crea e salva un'unica figura con originale, degradata e risultati dei metodi affiancati.
# img_orig/img_degraded: immagini BGR; results: dizionario {nome metodo: immagine};
# image_name: nome del file, usato nel titolo e nel nome del file di output.
def save_pipeline(img_orig, img_degraded, results, image_name):
    # Lista di (titolo, immagine): prima originale e degradata, poi i metodi
    pipeline_images = [
        ("Originale", img_orig),
        ("Degradata", img_degraded)
    ]
    pipeline_images += list(results.items())

    n_images = len(pipeline_images)
    # Una sola riga con un pannello per immagine
    fig, axes = plt.subplots(1, n_images, figsize=(4 * n_images, 4))

    for ax, (name, img) in zip(axes, pipeline_images):
        # OpenCV usa BGR, matplotlib vuole RGB
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(name, fontsize=10)
        ax.axis("off")

    fig.suptitle(f"Pipeline - {image_name}", fontsize=14)
    plt.tight_layout()

    # Salva la figura nella cartella figures: pipeline_<nome>.png
    output_path = f"{FIG_DIR}/pipeline_{image_name}.png"
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    
    # Visualizzazione a schermo disattivata: 
    # plt.show()
    
    # Chiude la figura per liberare memoria
    plt.close(fig)


# Immagini da elaborare
image_paths = ["image/img.png", "image/img2.png", "image/img3A.png", "image/img4.png", "image/img5.png"]
# Una riga per ogni coppia (immagine, metodo), raccolte per la tabella finale
rows = []

for image in image_paths:
    img = cv2.imread(image)

    # cv2.imread restituisce None se il file non esiste o non e' leggibile
    if img is None:
        print(f"Impossibile leggere {image}")
        continue

    # Nome del file senza estensione (es. "image/img.jpg" -> "img")
    stem = image.split("/")[-1].rsplit(".", 1)[0]

    # Degradazione con parametri fissi: i metodi vedono solo questa versione
    img_degraded = fz.degrade_image(img, alpha=ALPHA, center=CENTER, noise_sigma=NOISE_SIGMA, seed=SEED)

    # Salva la degradata (PNG, senza perdita)
    cv2.imwrite(f"{DEG_DIR}/{stem}.png", img_degraded)

    # Metodi di elaborazione applicati alla degradata, con parametri identici per ogni immagine
    pipeline = {
        "Baseline": fz.method_baseline(img_degraded),
        "Contrast Stretching LAB": fz.method_stretching_LAB(img_degraded, 2, 98),
        "Histogram Equalization LAB": fz.method_histogramEqualization_LAB(img_degraded),
        "Denoise + CLAHE": fz.method_denoise_LAB(img_degraded, h=20, h_color=20, clip_limit=1.5, tile_grid=(8, 8), chroma_gain=1.5)
    }

    # Metriche calcolate rispetto all'originale (non alla degradata)
    for name, method in pipeline.items():
        # Salva il risultato del metodo come immagine singola
        fname = name.replace(" + ", "_").replace(" ", "_")
        cv2.imwrite(f"{PROC_DIR}/{stem}_{fname}.png", method)
        rows.append([
            image.split("/")[-1],
            name,
            round(fz.psnr(img, method), 2),
            round(fz.Ssim(img, method), 3),
            round(fz.mae(img, method), 2)
        ])

    # Salvataggio della figura su file invece del plot a schermo
    save_pipeline(img, img_degraded, pipeline, stem)


# Tabella finale con tutte le metriche
df = pd.DataFrame(
    rows,
    columns=["Immagine", "Metodo", "PSNR (dB)", "SSIM", "MAE"]
)

print(df.to_string(index=False))

# Esporta in CSV (separatore ';' e virgola decimale per aprirlo in Excel italiano)
df.to_csv(CSV_PATH, index=False, sep=";", decimal=",", encoding="utf-8-sig")

# Tabella comparativa del PSNR: una riga per metodo, una colonna per immagine
print("\nPSNR:")
print(df.pivot(index="Metodo", columns="Immagine", values="PSNR (dB)"))