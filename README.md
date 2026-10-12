# Tugas 05 — Transfer Learning untuk Klasifikasi Image dengan Data Terbatas

**Nama:** Kevin Sukias K  
**NIM:** 2802526416  
**Mata kuliah:** Deep Learning, Sesi 05 (Advanced CNNs and Transfer Learning)

Studi perbandingan strategi training ResNet-18 pada Oxford-IIIT Pet (subset 10 kelas, ±2.000 image): from scratch, feature extraction, progressive fine-tuning, uji kontrol data augmentation, dan arsitektur pembanding (EfficientNet-B0, MobileNetV3-Small).

## Hasil utama

| Kode | Strategi | Arsitektur | Augmentasi | Trainable params | Val Acc | Test Macro F1 | Train time (s) |
|---|---|---|---|---|---|---|---|
| B1 | scratch | ResNet-18 | ya | 11,181,642 | 0.3758 | 0.4241 | 95.5 |
| B2 | feature extraction | ResNet-18 | ya | 5,130 | 0.9329 | 0.9366 | 208.9 |
| B3 | fine-tuning (`layer4` + `fc`) | ResNet-18 | ya | 8,398,858 | 0.9262 | 0.9361 | 67.8 |
| B4 | fine-tuning, tanpa augmentasi | ResNet-18 | tidak | 8,398,858 | 0.9362 | 0.9425 | 38.0 |
| B5 | fine-tuning, tanpa augmentasi | EfficientNet-B0 | tidak | 1,142,202 | **0.9497** | **0.9631** | 41.0 |
| B5 | fine-tuning, tanpa augmentasi | MobileNetV3-Small | tidak | 1,245,738 | 0.9396 | 0.9126 | 38.7 |

Sumber angka: [`outputs/results_summary.csv`](outputs/results_summary.csv). Konfigurasi B5 dipilih berdasarkan validation accuracy, bukan test set.

## Isi repository

```
.
├── KevinSukias_Transfer_Learning.ipynb   # notebook lengkap, sudah dieksekusi (Colab, Tesla T4)
├── requirements.txt
├── docs/
│   ├── KevinSukias_BagianA.pdf           # Bagian A: analisis konseptual
│   └── KevinSukias_Laporan.pdf           # Bagian C: laporan C1–C5 + AI disclosure
└── outputs/                              # log mentah dan grafik hasil eksekusi notebook
```

### `outputs/`
| File | Isi |
|---|---|
| `config.json` | konfigurasi eksperimen + environment (versi library, GPU, seed) |
| `split_manifest.csv` | daftar file per split train/val/test (bukti tidak ada data leakage) |
| `logs/<run>_epochs.csv`, `all_epoch_logs.csv` | log per epoch: loss, accuracy, waktu, learning rate per param group |
| `results_summary.csv` / `.json` | metrik akhir tiap run |
| `predictions/<run>_test_predictions.csv` | prediksi + confidence untuk tiap image test |
| `C1_tabel_konsolidasi.csv` / `.md` | tabel konsolidasi C1 |
| `C3_misclassified.csv` | kasus misklasifikasi untuk failure analysis (C3) |
| `C5_gradcam_correlation.csv` | korelasi Grad-CAM `layer4` sebelum vs. sesudah fine-tuning (C5) |
| `figures/` | semua grafik yang dipakai di laporan |

## Cara menjalankan
1. Buka notebook di Google Colab, pilih runtime **T4 GPU**.
2. Runtime → Run all (±10 menit). Dataset diunduh otomatis lewat `torchvision.datasets.OxfordIIITPet`.
3. Semua output tersimpan ke folder `outputs/` dan di-zip di cell terakhir.

Lokal: `pip install -r requirements.txt`, lalu jalankan notebook.

## Protokol eksperimen
- Seed 42 untuk Python, NumPy, PyTorch; cuDNN deterministic.
- Batch size 32, AdamW (weight decay 0.01), CrossEntropyLoss.
- Early stopping pada validation loss, patience 5, maksimal 40 epoch; weights dari epoch terbaik dipulihkan.
- Input 224×224, normalisasi ImageNet. Data augmentation hanya pada training set.
- Split stratified 70/15/15 dari gabungan split `trainval` + `test` bawaan dataset.
- Kelas: Bengal, Egyptian Mau, British Shorthair, Russian Blue, Persian, Beagle, Basset Hound, Boxer, Pug, Shiba Inu. Pasangan Bengal–Egyptian Mau, British Shorthair–Russian Blue, dan Beagle–Basset Hound sengaja dipilih karena mirip secara visual.

**Catatan efisiensi:** image di-cache di RAM setelah resize (decode JPEG hanya sekali), training memakai mixed precision (AMP) dan `channels_last`. Ini mempercepat setiap epoch tanpa mengubah data, transformasi, maupun hyperparameter, dan dipakai sama di semua kondisi B1–B5.

## Referensi
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR*.
- Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *ICML*.
- Howard, A., et al. (2019). Searching for MobileNetV3. *ICCV*.
- Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *ICCV*.
- Parkhi, O. M., Vedaldi, A., Zisserman, A., & Jawahar, C. V. (2012). Cats and dogs. *CVPR*. (Oxford-IIIT Pet dataset)
- Model dan pretrained weights: `torchvision.models`.

## Penggunaan AI
Penggunaan AI pada tugas ini dijelaskan di bagian *Pernyataan Penggunaan AI* pada akhir `docs/KevinSukias_Laporan.pdf`.
