# PANDUAN PROYEK — Earthquake Classification & False-Trigger Reduction

**Versi:** 1.0 — 2026-05-06
**Pembimbing:** Pak Indra
**Lokasi proyek:** `/home/indra/eq_team/earthquake-classification/`
**Bahasa instruksi:** Bahasa Indonesia

> **⚠ REVISI 2026-06-10:** Audit kode menemukan beberapa masalah validitas
> yang harus diperbaiki dengan prioritas. Baca **[PANDUAN_new.md](PANDUAN_new.md)**
> (v2.0 — Revisi Kritis & Prioritas Perbaikan) **sebelum menjalankan training
> baru**. Aturan v1.0 di file ini tetap berlaku kecuali yang direvisi di v2.0.

> Baca panduan ini **sampai habis** sebelum mulai ngoding. Kalau ada yang
> belum jelas, tanya dulu sebelum eksekusi — terutama untuk hal-hal yang
> menyangkut GPU, dataset, atau git.

---

## 1. Latar Belakang

### 1.1 Konteks Ilmiah

Sistem peringatan dini gempa (Earthquake Early Warning, EEW) dan jaringan
monitoring seismik bergantung pada **detektor gempa otomatis** yang
membaca rekaman kontinu dari stasiun seismograf dan mengeluarkan trigger
ketika diduga ada gempa. Detektor klasik seperti **STA/LTA**
(Short-Term Average / Long-Term Average) sederhana, cepat, dan murah —
tetapi memiliki **false-trigger rate** yang tinggi: getaran dari truk,
kereta, kegiatan industri, badai, atau noise instrumen sering ikut
ter-trigger sebagai "gempa".

False trigger ini bermasalah karena:
- Membuat operator membuang waktu memverifikasi event palsu.
- Mengganggu alarm publik (false alarm) → menurunkan kepercayaan
  masyarakat pada sistem.
- Membebani pipeline analisis hilir (lokalisasi, magnitudo, dst.).

**Tujuan proyek ini:** membangun classifier biner (gempa vs bukan-gempa)
yang berjalan setelah detektor STA/LTA atau detektor real-time lainnya,
untuk **memfilter trigger palsu** sebelum di-broadcast/diteruskan ke
modul berikutnya.

### 1.2 Tujuan Konkret

1. Membangun pipeline data dari STEAD untuk klasifikasi biner pada
   window pendek (4–10 detik).
2. Membangun baseline klasik (STA/LTA + fitur statistik + RandomForest)
   sebagai pembanding "minimum acceptable".
3. Mengimplementasikan minimal **3 arsitektur deep-learning SOTA**
   (lihat §5) dan membandingkan kinerjanya secara adil.
4. Menyediakan analisis: precision/recall trade-off, ROC/PR curve,
   inference latency, robustness terhadap SNR rendah.
5. Menulis laporan/skripsi yang reproducible (config, seed, checkpoint
   tersimpan).

### 1.3 Hubungan dengan Proyek Tetangga

Proyek `earthquake-azimuth-pytorch` (di luar folder ini) mengerjakan
estimasi back-azimuth. Ada banyak komponen yang bisa **dipinjam**
(loss function, evaluation utility, augmentation, training loop) tetapi
**jangan langsung di-fork**. Cukup baca kode di sana sebagai referensi
dan tulis ulang yang relevan dengan tugas klasifikasi.

---

## 2. Tinjauan SOTA (per 2025–2026)

Sebelum menentukan model, pelajari pekerjaan-pekerjaan kunci berikut.
Tugas pertama Anda: **baca abstract + arsitektur** dari minimal 5 paper
di tabel ini.

| Tahun | Nama | Window | Arsitektur | Paper |
|-------|------|--------|------------|-------|
| 2018 | **GPD** (Generalized Phase Detection) | 4 s @ 100 Hz (400 samples) | CNN-1D, 3-class (P/S/Noise) | Ross et al. 2018, BSSA |
| 2018 | **ConvNetQuake** | 10 s @ 100 Hz (1000) | CNN-1D | Perol et al. 2018, Sci.Adv. |
| 2019 | **CRED** | 60 s | CNN+RNN | Mousavi et al. 2019 |
| 2019 | **PhaseNet** | 30 s @ 100 Hz (3000) | U-Net 1D | Zhu & Beroza 2019, GJI |
| 2020 | **EQTransformer** | 60 s @ 100 Hz (6000) | CNN + LSTM + multi-head attention | Mousavi et al. 2020, Nat.Commun. |
| 2024 | **SeisT** | 60 s | Multi-Scale Mixed Conv + Multi-Path Transformer | Li et al. 2024, IEEE TGRS |
| 2024 | **PhaseNO** | 30 s | Graph Neural Operator | 2024 |
| 2025 | **SeisMoLLM** | 60 s | GPT-2 backbone (frozen) + LoRA + conv embedder | Wang et al. 2025 |
| 2025 | **SeismicXM** | variable | Cross-task foundation model | SRL 2025 |

### 2.1 Pelajaran Kunci dari SOTA

- **Window pendek (4–10 s) cukup** untuk klasifikasi biner. EQTransformer
  pakai 60 s karena tugasnya lebih kompleks (deteksi + phase picking
  bersamaan). Untuk **false-trigger reduction**, GPD-style 4 s dan
  ConvNetQuake-style 10 s adalah baseline yang masuk akal.
- **Centering pada P-arrival** lebih efektif daripada window random.
- Augmentasi yang penting: **noise injection (variasi SNR)**,
  **amplitude scaling**, **time shift kecil (±0.5 s)**, **channel
  dropout**. **JANGAN** flip waktu / flip vertikal — itu melanggar fisika
  gelombang seismik.
- Sin/cos / circular tricks **tidak relevan** di sini (tidak ada label
  sirkular). Output cukup **1 logit (BCE)** atau **2 logit
  (CE 2-class)** dengan softmax.
- **Class imbalance** STEAD: ~1.03 M gempa vs ~0.235 M noise (≈4.4:1).
  Tangani dengan: (a) subsampling kelas mayoritas, (b) `pos_weight`
  pada `BCEWithLogitsLoss`, atau (c) **focal loss**.
- Foundation models (SeisMoLLM, SeismicXM) saat ini belum tentu lebih
  baik dari EQTransformer-tuned untuk task spesifik ini — tapi menarik
  untuk dicoba di akhir.

### 2.2 Rekomendasi Window

Setelah membaca SOTA, kita pilih:

**Window utama: 6 detik @ 100 Hz = 600 sampel**
- 1 detik sebelum P-arrival + 5 detik sesudah
- Untuk noise: window random 6 detik dari rekaman noise 60 detik
- Cukup pendek untuk inference real-time (< 5 ms / sample di RTX 5090)
- Cukup panjang untuk menangkap onset P + sebagian S (untuk gempa lokal)

**Eksperimen tambahan (opsional, di Phase 4):** ulangi dengan window 4 s
dan 10 s untuk membandingkan trade-off latency vs akurasi.

---

## 3. Setup Lingkungan — WAJIB Diikuti

### 3.1 tmux — SELALU Pakai untuk Training

Sebelum bikin session baru, cek dulu yang sudah berjalan:

```bash
tmux ls
```

Kalau muncul session bernama `train` (punya Pak Indra), **JANGAN
diutak-atik**. Bikin session sendiri dengan nama yang jelas:

```bash
tmux new -s eqclass        # session baru
# atau attach ke session sendiri yang sudah ada:
tmux attach -t eqclass
```

Konvensi nama tmux untuk proyek ini:
- `eqclass`        → session utama (training, evaluasi)
- `eqclass-data`   → session khusus preprocessing dataset
- `eqclass-jupyter`→ session untuk jupyter notebook eksplorasi

Keluar dari session **tanpa membunuh proses**: `Ctrl-b` lalu `d` (detach).
Kill session hanya kalau yakin tidak ada job penting: `tmux kill-session -t eqclass`.

### 3.2 Conda Environment — JANGAN Bikin Baru

Pakai environment yang sudah ada:

```bash
conda activate pytorch_Py12
```

Untuk running suatu program python (tanpa) notebook, SETELAH melakukan perintah di atas,

```bash
python nama_file.py
```


Versi yang sudah terinstal (jangan diubah tanpa konfirmasi):
- Python 3.12.12
- PyTorch 2.10.0+cu130
- CUDA 13.0
- numpy, scipy, h5py, scikit-learn, xgboost, optuna, transformers, peft

Kalau butuh library tambahan (mis. `seisbench`, `obspy`, `wandb`):
**konfirmasi dulu ke pembimbing**, lalu pasang dengan
`conda run -n pytorch_Py12 pip install <pkg>`.

### 3.3 GPU Etiquette — RTX 5090 Dipakai Bersama

```bash
nvidia-smi
```

- Kalau VRAM terpakai > 20 GB oleh proses orang lain, **tunggu** atau
  pakai batch lebih kecil.
- Set `CUDA_VISIBLE_DEVICES=0` secara eksplisit di script.
- Selalu kosongkan VRAM setelah selesai: `del model; torch.cuda.empty_cache()`.
- Jangan jalankan training paralel dari banyak terminal — gunakan
  satu tmux + satu proses utama.

---

## 4. Struktur Folder yang Akan Anda Buat

```
/home/indra/eq_team/earthquake-classification/
├── PANDUAN.md                      ← file ini (jangan diubah tanpa konfirmasi)
├── README.md                       ← ringkasan proyek (Anda yang tulis)
├── EXPERIMENTS.md                  ← log eksperimen (Anda update tiap run)
├── .gitignore                      ← pastikan abaikan data/, results/, __pycache__/
├── requirements.txt                ← daftar pip package (di luar yang sudah di env)
├── pyproject.toml                  ← (opsional) metadata proyek
│
├── data/                           ← TIDAK di-commit; subset hasil preprocessing
│   ├── windows_6s_train.npy        ← memmap (N, 3, 600) float32
│   ├── windows_6s_train_labels.npy ← (N,) uint8 (0=noise, 1=eq)
│   ├── windows_6s_val.npy
│   ├── windows_6s_val_labels.npy
│   ├── windows_6s_test.npy
│   ├── windows_6s_test_labels.npy
│   └── metadata_splits.csv         ← trace_name, p_arrival_sample, snr, dst
│
├── notebooks/
│   ├── 00_data_exploration.ipynb   ← cek STEAD, distribusi SNR, magnitudo
│   ├── 01_window_strategy.ipynb    ← visualisasi window 4/6/10 s
│   ├── 02_baseline_stalta.ipynb    ← baseline STA/LTA + fitur klasik
│   ├── 03_cnn_baseline.ipynb       ← CNN-1D simple (eksplorasi arsitektur)
│   ├── 04_eqtransformer_lite.ipynb ← variant lebih ringan dari EQTransformer
│   └── 05_evaluation.ipynb         ← perbandingan akhir, ROC/PR, error analysis
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── stead_subset.py         ← fungsi extract window dari merge.hdf5
│   │   ├── dataset.py              ← torch.utils.data.Dataset (baca memmap)
│   │   └── augmentations.py        ← GaussianNoise, AmplitudeScaling, TimeShift, ChannelDropout
│   ├── models/
│   │   ├── __init__.py
│   │   ├── stalta.py               ← STA/LTA classical detector + threshold
│   │   ├── feature_baseline.py     ← STA/LTA fitur + RandomForest/XGBoost
│   │   ├── cnn_1d.py               ← GPD-style + ResNet-1D
│   │   ├── eqtransformer_lite.py   ← versi ringan EQTransformer
│   │   └── seist_classifier.py     ← (opsional, Phase 4)
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py              ← loop training (BF16 AMP, callbacks)
│   │   └── losses.py               ← BCEWithLogits + pos_weight, FocalLoss
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py              ← acc, prec, rec, F1, ROC-AUC, PR-AUC
│   │   └── visualization.py        ← confusion matrix, ROC/PR plot
│   └── utils/
│       ├── __init__.py
│       └── seed.py                 ← set_seed(42) untuk reproducibility
│
├── configs/
│   ├── data_config.yaml            ← path STEAD, window length, split ratio
│   ├── stalta.yaml
│   ├── cnn_1d.yaml
│   └── eqtransformer_lite.yaml
│
├── scripts/
│   ├── 01_build_subset.py          ← extract window dari STEAD → memmap
│   ├── 02_train_baseline.py        ← train STA/LTA + RF/XGB
│   ├── 03_train_cnn.py             ← train CNN-1D
│   ├── 04_train_eqt_lite.py        ← train EQTransformer-lite
│   └── 05_evaluate_all.py          ← evaluasi semua model di test set
│
├── results/                        ← TIDAK di-commit (pakai .gitignore)
│   ├── checkpoints/                ← *.pt
│   ├── logs/                       ← tensorboard / wandb
│   ├── figures/                    ← *.png, *.pdf untuk laporan
│   └── tables/                     ← metric per model (CSV/JSON)
│
└── tests/
    ├── test_dataset.py
    ├── test_metrics.py
    └── test_models.py
```

---

## 5. Rencana Model — Roadmap 5 Fase

Lakukan secara berurutan. Jangan lompat ke fase berikutnya sebelum fase
saat ini selesai dan hasilnya tertulis di `EXPERIMENTS.md`.

### Fase 1 — Setup & Eksplorasi (Minggu 1)

- [ ] Setup folder sesuai §4
- [ ] `git init`, bikin `.gitignore`, commit pertama
- [ ] `notebooks/00_data_exploration.ipynb`:
  - Buka `merge.csv`, hitung jumlah trace EQ vs noise
  - Distribusi SNR, magnitudo, jarak
  - Cek: berapa banyak yang punya `p_arrival_sample` valid
  - Cek class imbalance ratio
- [ ] `notebooks/01_window_strategy.ipynb`:
  - Visualisasi 5–10 contoh EQ di window 4 s, 6 s, 10 s
  - Visualisasi 5–10 contoh noise
  - Diskusi: window mana yang dipilih (default: **6 s**)

### Fase 2 — Dataset Building (Minggu 2)

- [ ] Implementasi `scripts/01_build_subset.py`:
  - Baca `merge.csv`, filter EQ trace dengan `p_arrival_sample` valid
  - Filter noise trace
  - Stratified split 70/15/15 berdasarkan **event_id** (BUKAN trace_id)
    untuk menghindari data leakage. Untuk noise, split per
    `network_code` agar stasiun-noise yang sama tidak bocor antara
    train/val/test.
  - Untuk tiap trace EQ: extract `(3, 600)` window centered di
    `p_arrival_sample - 100` (1 s sebelum P).
  - Untuk tiap trace noise: extract `(3, 600)` window dari posisi random.
  - Bandpass filter 1–45 Hz, demean, normalize per-trace (peak amplitude
    atau std).
  - Simpan sebagai memmap `.npy`:
    `data/windows_6s_train.npy` shape `(N_train, 3, 600)` float32
    `data/windows_6s_train_labels.npy` shape `(N_train,)` uint8
  - Estimasi ukuran: 600 × 3 × 4 B × 1.26 M trace ≈ 9 GB total.
    Kalau host RAM tipis, simpan dalam 3 file (train/val/test) — lebih
    fleksibel.
- [ ] Pakai multiprocessing (`mp.Pool`) dengan per-worker HDF5 handle
  untuk speed. Lihat referensi: `earthquake-azimuth-pytorch/scripts/run_traditional.py`.
- [ ] Simpan `data/metadata_splits.csv` (trace_name, label, snr,
  magnitude, distance, p_arrival_sample, split).
- [ ] Tulis `tests/test_dataset.py`: shape, range nilai, label
  consistency, no-NaN.
- [ ] Update `EXPERIMENTS.md` dengan jumlah sampel akhir tiap split.

### Fase 3 — Baseline (Minggu 3)

#### Fase 3a: STA/LTA Klasik

- [ ] `src/models/stalta.py`: implementasi STA/LTA characteristic
  function (mis. STA = 0.5 s, LTA = 5 s, threshold = 3.0).
  Output: trigger atau tidak.
- [ ] Evaluasi langsung di test set sebagai *no-ML baseline*.
- [ ] Catat: precision, recall, F1, false-trigger rate.

#### Fase 3b: STA/LTA Feature + Classifier Klasik

- [ ] `src/models/feature_baseline.py`: ekstrak ~30 fitur per window
  (STA/LTA peak, rms, kurtosis, dominant freq, spectral centroid,
  P/S amplitude ratio, ZCR, dst.)
- [ ] Train RandomForest dan XGBoost.
- [ ] Bandingkan dengan STA/LTA murni.

> **Target Fase 3:** F1 > 0.85, ROC-AUC > 0.92. Kalau tidak tercapai,
> ada bug di pipeline data.

### Fase 4 — Deep Learning (Minggu 4–6)

#### Fase 4a: GPD-style CNN-1D (Wajib)

- [ ] `src/models/cnn_1d.py`:
  - `GPDClassifier`: 4 conv layer + 2 FC, output 1 logit
  - `ResNet1D`: 8 residual block, output 1 logit
- [ ] `scripts/03_train_cnn.py` dengan:
  - Loss: `BCEWithLogitsLoss(pos_weight=1/4.4)` ATAU `FocalLoss(γ=2)`
  - Optimizer: AdamW, lr=3e-3, weight_decay=0.01
  - Scheduler: OneCycleLR
  - Augmentasi: HorizontalRotation **jangan dipakai** (tidak relevan
    untuk binary class). Pakai: GaussianNoise, AmplitudeScaling,
    TimeShift, ChannelDropout(p=0.05).
  - Batch size: 512, BF16 AMP, `torch.compile`
  - Epoch: 30–50, early stopping patience=10
- [ ] **Target:** F1 > 0.95, ROC-AUC > 0.98 di test set.

#### Fase 4b: EQTransformer-lite (Wajib)

- [ ] `src/models/eqtransformer_lite.py`: versi simplified — hilangkan
  jalur phase-picking, sisakan encoder CNN+BiLSTM+attention untuk
  klasifikasi biner saja. Target ~1–3 M parameter.
- [ ] Train dengan recipe mirip Fase 4a tapi lr lebih kecil (1e-3) dan
  scheduler CosineAnnealing dengan warmup 3 epoch.
- [ ] **Target:** F1 ≥ Fase 4a, dengan trade-off latency lebih besar.

#### Fase 4c: SeisT Classifier (Opsional, Stretch)

- [ ] Adaptasi backbone SeisT-S dari proyek tetangga
  (`earthquake-azimuth-pytorch/src/models/transformer_models.py`)
  dengan mengganti head: GAP → Linear(2). Frozen pretrained
  weights kalau tersedia, fine-tune head + last 2 block.
- [ ] **Target:** F1 setara atau lebih baik dari 4b.

### Fase 5 — Evaluasi, Analisis, Laporan (Minggu 7–8)

- [ ] `scripts/05_evaluate_all.py`: jalankan semua model di test set,
  hasilkan tabel ringkas.
- [ ] `notebooks/05_evaluation.ipynb`:
  - Confusion matrix per model
  - ROC dan PR curve overlay
  - Error breakdown by SNR bin (0–10 dB, 10–20 dB, > 20 dB)
  - Error breakdown by distance bin (lokal/regional)
  - Inference latency: ms/sample di batch 1, 32, 256
  - Diskusi: model mana yang dipilih untuk deployment dan kenapa
- [ ] Tulis `README.md` proyek (ringkasan eksperimen, cara reproduce).
- [ ] Tulis draft bab/skripsi.

---

## 6. Tabel Target Performa

| Model | Akurasi | F1 | ROC-AUC | PR-AUC | Latency (batch=1) | Catatan |
|-------|---------|-----|---------|--------|-------------------|---------|
| STA/LTA murni | 0.80–0.85 | 0.78–0.85 | 0.88–0.92 | — | < 0.1 ms | Baseline non-ML |
| STA/LTA + RF | 0.88–0.92 | 0.87–0.91 | 0.93–0.96 | 0.85–0.90 | < 1 ms | CPU OK |
| GPD-style CNN | 0.95–0.97 | 0.95–0.97 | 0.98–0.99 | 0.96–0.98 | < 2 ms | Baseline DL |
| EQTransformer-lite | **≥ 0.97** | **≥ 0.97** | **≥ 0.99** | **≥ 0.98** | 5–15 ms | Target utama |
| SeisT-classifier | ≥ EQT-lite | ≥ EQT-lite | ≥ EQT-lite | ≥ EQT-lite | 10–25 ms | Opsional |

> Angka-angka di atas adalah ekspektasi berdasarkan literatur. Kalau
> hasil Anda jauh di bawah, **ada bug** — jangan disembunyikan, lapor.

---

## 7. Metrik Evaluasi — Wajib Dilaporkan

Untuk setiap model, hitung dan laporkan:

1. **Accuracy** — sebagai sanity check
2. **Precision, Recall, F1** — pada threshold 0.5 dan threshold optimal
   (dari Youden's J atau F1-max di validation set)
3. **ROC-AUC** dan **PR-AUC** — independen dari threshold
4. **False Trigger Rate (FTR)** = FP / (FP + TN) — kunci untuk EEW
5. **Detection Rate (sensitivity)** = TP / (TP + FN)
6. **Confusion matrix** lengkap
7. **Latency**: ms per sample untuk batch 1, 32, 256 di RTX 5090
8. **Per-SNR breakdown**: F1 untuk SNR < 10 dB, 10–20 dB, > 20 dB
9. **Calibration**: reliability diagram (apakah probabilitas keluaran
   well-calibrated?)

Format pelaporan: simpan sebagai `results/tables/metrics_<model>.json`
dengan struktur baku, lalu agregat ke `results/tables/summary.csv`.

---

## 8. Tips, Pitfall, & Peringatan

### 8.1 Data Leakage — Jangan Salah Split

- Split berdasarkan **event_id**, bukan trace. Satu gempa bisa direkam
  banyak stasiun → kalau di-split per trace, model bisa "mengenali"
  event yang sama di train dan test → metric melambung tapi tidak
  realistis.
- Untuk noise, split berdasarkan **stasiun** (network_code +
  receiver_code) supaya stasiun yang sama tidak muncul di train dan
  test.

### 8.2 Class Imbalance

- Rasio EQ:noise ≈ 4.4:1. Pilihan:
  1. **Subsample EQ** ke ~235 K (balance) — paling sederhana, kehilangan
     data
  2. **`pos_weight=1/4.4`** di `BCEWithLogitsLoss` — pakai semua data
  3. **FocalLoss(γ=2)** — fokus ke hard examples
- Untuk false-trigger reduction, **prioritas adalah recall noise tinggi**
  (jangan banyak gempa dianggap noise) — pilih threshold yang
  memaksimalkan recall di kelas EQ pada constraint FTR < 5%.

### 8.3 Preprocessing — Konsisten Train/Test

- Bandpass 1–45 Hz dengan Butterworth order 4, zero-phase (`filtfilt`).
- Demean per trace.
- Normalize per trace: `x = x / (max(|x|) + 1e-6)` — robust ke amplitudo
  absolut yang bervariasi 6 orde magnitude.
- **JANGAN** normalize global (mean/std seluruh dataset) — hilangkan
  variasi amplitudo absolut yang penting.

### 8.4 Augmentasi yang Sah & Tidak Sah

| Augmentasi | Sah? | Catatan |
|------------|------|---------|
| GaussianNoise (variasi SNR) | ✅ | Kunci untuk robustness |
| AmplitudeScaling (0.5–2.0×) | ✅ | Simulasi variasi gain |
| TimeShift (±0.5 s) | ✅ | Simulasi P-pick error |
| ChannelDropout (p ≤ 0.05) | ✅ | Hanya pada E/N, jangan Z |
| HorizontalRotation | ⚠️ | Sah secara fisika, tapi tidak relevan untuk binary class — boleh dipakai tapi efeknya kecil |
| TimeReversal | ❌ | Melanggar fisika gelombang |
| Vertical flip (Z → -Z) | ❌ | Melanggar fisika |
| SpectrogramFlip | ❌ | Melanggar fisika |

### 8.5 Reproducibility — Wajib

- Set seed: `numpy`, `random`, `torch`, `torch.cuda` — semua di awal
  script.
- Simpan config sebagai YAML di `configs/`, **commit** ke git.
- Simpan checkpoint `best_<metric>.pt` dengan info epoch + metric.
- Catat versi git commit di nama folder hasil:
  `results/runs/2026-05-10_cnn1d_<commit_hash>/`.

### 8.6 Jangan Lakukan Ini

- ❌ Loading seluruh `merge.hdf5` (92 GB) ke RAM. Selalu pakai memmap
  atau index-on-demand.
- ❌ `git add data/` atau `git add results/`. Ukuran besar, masuk
  `.gitignore`.
- ❌ `git push --force` ke main / master.
- ❌ Membuat conda env baru tanpa konfirmasi.
- ❌ Membunuh tmux session orang lain (`tmux kill-session -t train`).
- ❌ Training tanpa tmux — kalau ssh putus, training hilang.
- ❌ Mengubah `PANDUAN.md` ini tanpa konfirmasi pembimbing.

### 8.7 Lakukan Ini

- ✅ Commit setiap fase selesai, dengan pesan jelas:
  `feat(data): build 6s window subset (train=720k, val=160k, test=155k)`
- ✅ Update `EXPERIMENTS.md` setiap kali run training penting.
- ✅ Cek `nvidia-smi` sebelum training.
- ✅ Diskusi mingguan dengan pembimbing — bawa angka, bukan opini.
- ✅ Baca paper rujukan sebelum implementasi, jangan cuma copy kode.

---

## 9. Tools & Resource

### 9.1 Library Utama (sudah di `pytorch_Py12`)

```python
torch              # 2.10.0+cu130
numpy, scipy, h5py
sklearn, xgboost
matplotlib, seaborn
optuna             # (kalau perlu HPO di Fase 4)
transformers, peft # (untuk Fase 4c saja)
```

### 9.2 Library Tambahan yang Mungkin Diperlukan

```bash
# Konfirmasi dulu sebelum install:
conda run -n pytorch_Py12 pip install obspy seisbench wandb tensorboard
```

`obspy` dipakai kalau perlu STA/LTA built-in dan filter Butterworth yang
matang. `seisbench` dipakai kalau ingin alternatif loader STEAD.

### 9.3 Referensi Lokal

- Proyek tetangga (referensi kode, **bukan** untuk di-fork):
  `/home/indra/indra/code/earthquake-azimuth-pytorch/`
- STEAD raw: `/home/indra/indra/STEAD/merged/`
  - `merge.hdf5` (92 GB) — waveform `f['data/<trace_name>']` shape
    `(6000, 3)`, transpose ke `(3, 6000)`
  - `merge.csv` (354 MB) — 35 kolom metadata per trace
- Kolom CSV penting:
  - `trace_name` (key untuk HDF5)
  - `trace_category` (`earthquake_local` / `noise`)
  - `p_arrival_sample` (sample index, 100 Hz)
  - `s_arrival_sample`
  - `source_magnitude`, `source_distance_km`
  - `snr_db` (3-component SNR)
  - `network_code`, `receiver_code`
  - `source_id` ← **kunci untuk split per-event**

### 9.4 Daftar Bacaan Wajib

1. Ross et al. 2018 — *Generalized Seismic Phase Detection with Deep
   Learning* (BSSA)
2. Mousavi et al. 2020 — *Earthquake transformer—an attentive
   deep-learning model for simultaneous earthquake detection and phase
   picking* (Nat. Commun.)
3. Zhu & Beroza 2019 — *PhaseNet: a deep-neural-network-based
   seismic arrival-time picking method* (GJI)
4. Mousavi et al. 2019 — *STanford EArthquake Dataset (STEAD): A Global
   Data Set of Seismic Signals for AI* (IEEE Access)
5. Li et al. 2024 — *SeisT: A Foundational Deep-Learning Model for
   Earthquake Monitoring* (IEEE TGRS)

---

## 10. Cara Memulai (Quickstart)

```bash
# 1. Cek tmux
tmux ls

# 2. Buat session sendiri
tmux new -s eqclass

# 3. Activate env
conda activate pytorch_Py12

# 4. Cek GPU
nvidia-smi

# 5. Masuk ke folder proyek
cd /home/indra/eq_team/earthquake-classification

# 6. Init git
git init
git add PANDUAN.md
git commit -m "chore: add project guide"

# 7. Bikin folder skeleton
mkdir -p data notebooks src/{data,models,training,evaluation,utils} configs scripts results/{checkpoints,logs,figures,tables} tests
touch .gitignore README.md EXPERIMENTS.md requirements.txt

# 8. Edit .gitignore — minimal:
cat > .gitignore <<'EOF'
__pycache__/
*.pyc
.ipynb_checkpoints/
data/
results/
*.npy
*.pt
*.h5
*.hdf5
.env
.vscode/
.idea/
EOF

# 9. Mulai notebook eksplorasi
jupyter lab notebooks/00_data_exploration.ipynb
```

Lalu lanjut ke Fase 1 di §5.

---

## 11. Checklist Mingguan

Setiap akhir minggu, kirim ringkasan ke pembimbing dengan format:

```
Minggu ke-N (tanggal):
- Sudah dilakukan: [...]
- Hasil/metric: [...]
- Hambatan: [...]
- Rencana minggu depan: [...]
- Pertanyaan: [...]
```

Lampirkan plot/tabel kalau ada.

---

## 12. Bantuan & Eskalasi

- **Bug kode / arsitektur**: tanya di forum tim, sertakan minimal
  reproducible example.
- **Kebingungan teori seismologi**: konsultasi langsung dengan
  pembimbing.
- **Akses data / GPU bermasalah**: lapor segera, jangan kerja sendirian
  > 1 hari kalau blocker.
- **Hasil aneh** (akurasi 100% atau 50%): hampir pasti ada bug —
  prioritas debugging, jangan lanjut ke fase berikutnya.

---

**Selamat bekerja. Pelan-pelan saja, yang penting hasilnya benar dan
reproducible.** — Pak Indra
