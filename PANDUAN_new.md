# PANDUAN PROYEK v2.0 — Revisi Kritis & Prioritas Perbaikan

**Versi:** 2.0 — 2026-06-10
**Pembimbing:** Pak Indra
**Status:** WAJIB dibaca sebelum menjalankan training berikutnya
**Hubungan dengan v1.0:** Dokumen ini **melengkapi** `PANDUAN.md` (v1.0), bukan
menggantikannya. Semua aturan v1.0 (tmux, conda, GPU etiquette, git, dst.)
**tetap berlaku** kecuali yang secara eksplisit direvisi di sini.

> Dokumen ini lahir dari audit kode & data tanggal 2026-06-10. Auditnya
> membaca semua script preprocessing + training, memindai seluruh
> `combined_5s.npy` (1.488.570 trace), dan membandingkan semua file metrik
> di `results/`. Temuannya diurutkan menjadi prioritas **P0–P3**.
> Kerjakan **berurutan per prioritas** — jangan loncat ke P2 sebelum P0
> dan P1 selesai dan tercatat di `EXPERIMENTS.md`.

---

## 1. Status Proyek per 2026-06-10

### 1.1 Perubahan dari rencana v1.0 yang DISAHKAN

Implementasi sudah berkembang dari rencana awal v1.0. Perubahan berikut
**disetujui pembimbing** dan menjadi baseline resmi:

| Aspek | Rencana v1.0 | Kondisi sekarang (disahkan) |
|-------|--------------|------------------------------|
| Task | Biner (EQ vs noise) | **Multi-class** sumber sinyal |
| Data | STEAD saja | **STEAD + PNW** (gabungan) |
| Window | 6 s | 3 s / 5 s / 10 s (default eksperimen: **5 s**) |
| Format | subset per-split | memmap gabungan di root tim (`combined_*.npy`) |

Catatan: tujuan akhir proyek **tidak berubah** — false-trigger reduction
untuk EEW. Formulasi multi-class adalah jalan ke sana, dan evaluasi biner
turunan (gempa vs bukan-gempa) tetap wajib dilaporkan (lihat §4.4).

### 1.2 Isi dataset gabungan (window 5 s)

Hasil pemindaian penuh 2026-06-10 — **kualitas data BERSIH**: 0 baris
all-zero, 0 NaN dari 1.488.570 trace (bug zero-fill explosion sudah
diperbaiki sebelum penggabungan). Komposisi kelas:

| Label folder (sekarang) | Asal | Jumlah | Catatan |
|--------------------------|------|--------:|---------|
| `memmap` | STEAD | 1.030.231 | gempa STEAD |
| `memmap_noise` | STEAD | 235.426 | noise |
| `memmap_earthquake` | PNW | 163.064 | gempa PNW |
| `memmap_surface_event` | PNW | 27.174 | longsor/lahar/avalanche |
| `memmap_no_event` | STEAD | 16.448 | window pra-P |
| `memmap_explosion` | PNW | 15.875 | ledakan tambang |
| `memmap_sonic` | PNW | 206 | sonic boom — **sangat kecil** |
| `memmap_thunder` | PNW | 146 | petir — **sangat kecil** |

Rasio kelas terbesar : terkecil ≈ **7.000 : 1**.

### 1.3 Mengapa angka 98% saat ini BELUM bisa dipakai di laporan

Lima model sudah terlatih (CNN, MobileNetV2/V3, TCN, Transformer), tetapi
angka-angkanya belum layak masuk skripsi/paper karena:

1. **Augmentasi terlarang** dipakai di 4 dari 5 script (time-reversal,
   melanggar §8.4 v1.0) → model dilatih dengan data yang melanggar fisika.
2. **Ruang label tidak seragam**: TCN dilatih 8 kelas (gempa STEAD dan PNW
   dipisah), model lain 7 kelas (digabung) → vektor stratifikasi berbeda →
   **test set tiap model berbeda** → tabel perbandingan tidak sah.
3. **Split acak per-trace** → potensi leakage antar event/stasiun → semua
   angka optimistis (lihat §5 — diperbaiki di P2).
4. Dua kelas (sonic, thunder) **tidak pernah terdeteksi** (precision =
   recall = 0,0) di semua model — imbalance tidak ditangani.
5. Tabel `README.md` mengutip run transformer **terlama** (V1) dan run TCN
   yang file metriknya **sudah tertimpa** — angkanya tidak cocok dengan
   file mana pun di `results/`.

**Aturan pelaporan sementara** (berlaku sampai P2 selesai): semua angka
wajib diberi keterangan *"split acak per-trace — berpotensi optimistis
karena event leakage"*. Jangan masukkan angka apa pun sebagai hasil final.

---

## 2. Ringkasan Temuan & Prioritas

Prinsip penentuan prioritas: **prioritas ≈ (dampak ilmiah × urgensi) /
usaha**. Bug yang merusak *setiap run baru* dan bisa diperbaiki dalam
hitungan menit (augmentasi) naik ke P0. Perbaikan yang mengubah *cara
mengukur* (split bebas-leakage) penting, tetapi butuh pembangunan ulang
metadata + koordinasi data bersama — masuk P2, dan sementara itu hasil
tetap bisa dipakai **dengan caveat** §1.3.

| # | Temuan | Dampak | Usaha | Prioritas |
|---|--------|--------|-------|-----------|
| 1 | Augmentasi `flip(-1)` (time-reversal) & `roll` sirkular | melanggar fisika, kontaminasi tiap run | menit | **P0** |
| 2 | Ruang label beda antar model (7 vs 8 kelas) | perbandingan model tidak sah | menit–jam | **P0** |
| 3 | Tiap script bikin split sendiri | test set beda antar model | jam | **P0** |
| 4 | Tanpa git, tanpa `EXPERIMENTS.md`, file metrik tertimpa | hasil tidak terlacak | jam | **P0** |
| 5 | Output TCN ditulis ke root tim | melanggar konvensi, artefak nyasar | menit | **P0** |
| 6 | 5 script ~90% copy-paste, sudah saling menyimpang | sumber bug #2, #5 | hari | **P1** |
| 7 | Z-score per-kanal, tanpa bandpass | buang rasio amplitudo antar-kanal (fisika!) | jam + retrain | **P1** |
| 8 | Imbalance tak ditangani; sonic/thunder mati | 2 kelas tidak pernah diprediksi | hari | **P1** |
| 9 | Evaluasi belum menjawab pertanyaan EEW (FTR dkk.) | metrik §7 v1.0 belum ada | hari | **P1** |
| 10 | Split per-trace → event/station leakage | semua angka optimistis | hari–minggu | **P2** |
| 11 | Metadata gabungan tanpa `source_id`/stasiun/`trace_name` | prasyarat #10 | hari | **P2** |
| 12 | Confound dataset (kelas ↔ asal data berkorelasi 100%) | "noise vs explosion" mungkin = "STEAD vs PNW" | analisis | **P2** |
| 13 | Polish (backend plot, bf16, logging, pickle, dsb.) | kenyamanan & kebersihan | menit–jam | **P3** |

---

## 3. P0 — Perbaiki SEKARANG (sebelum run training berikutnya)

Semua item P0 selesai dulu, **baru boleh ada run training baru**. Total
usaha: ± 1 hari kerja.

### P0-1. Hapus augmentasi terlarang

Lokasi pelanggaran (semua di `__getitem__` dataset):

| File | Baris ± | Masalah |
|------|---------|---------|
| `scripts/training/script_transformer.py` | 254 | `x = x.flip(-1)` — time-reversal |
| `scripts/training/script_CNN.py` | 196 | `x = x.flip(-1)` |
| `scripts/training/script_MobileNetV2.py` | 197 | `x = x.flip(-1)` |
| `scripts/training/script_mobilenetV3.py` | 194 | `x = x.flip(-1)` |
| `scripts/training/script_TCN.py` | 197–198 | `torch.roll` — shift **sirkular** |

Kenapa salah (§8.4 v1.0 sudah melarang, tapi diulang di sini):

- `flip(-1)` membalik arah waktu. Gelombang seismik **kausal**: P datang
  sebelum S, onset sebelum koda, envelope membesar lalu meluruh. Sinyal
  terbalik tidak pernah ada di alam — model belajar dari sampel mustahil.
  Flip juga memindahkan onset P dari sampel 100 ke sampel ~400, merusak
  konvensi window P-centered.
- `torch.roll` itu shift **melingkar**: 20 sampel terakhir (ekor koda)
  pindah ke *depan* P — artefak non-kausal juga, hanya lebih kecil.

Pengganti yang sah (urutan kanal dataset: **E, N, Z** → Z = indeks 2):

```python
if self.augment:
    # 1) Noise injection — variasi SNR (sah, kunci robustness)
    if torch.rand(1) < 0.5:
        sigma = 0.01 + 0.04 * torch.rand(1).item()
        x = x + sigma * torch.randn_like(x)

    # 2) Time shift NON-sirkular ±0.2 s (geser + zero-pad, JANGAN roll)
    if torch.rand(1) < 0.5:
        s = int(torch.randint(-20, 21, (1,)))
        if s > 0:
            x = torch.cat([torch.zeros_like(x[:, :s]), x[:, :-s]], dim=-1)
        elif s < 0:
            x = torch.cat([x[:, -s:], torch.zeros_like(x[:, :s])], dim=-1)

    # 3) Channel dropout — hanya horizontal (E/N), JANGAN Z
    if torch.rand(1) < 0.05:
        x[int(torch.randint(0, 2, (1,)))] = 0.0
```

> Catatan ekspektasi: menghapus flip kemungkinan **tidak banyak mengubah**
> akurasi headline. Alasannya bukan mengejar metrik, tapi validitas fisika
> — penguji skripsi akan menanyakan ini.

### P0-2. Satu ruang label kanonik untuk SEMUA model

Gempa STEAD dan gempa PNW adalah **kelas fisik yang sama** — memisahkannya
(seperti di `script_TCN.py` sekarang) berarti menyuruh model menebak *asal
dataset*, bukan *jenis sumber*. Definisikan SATU pemetaan, dipakai semua
script (nanti dipindah ke `src/data.py` saat P1):

```python
LABEL_MAP = {
    "memmap":               "earthquake",      # STEAD EQ
    "memmap_earthquake":    "earthquake",      # PNW EQ  → digabung
    "memmap_noise":         "noise",
    "memmap_no_event":      "no_event",
    "memmap_explosion":     "explosion",
    "memmap_sonic":         "sonic",
    "memmap_thunder":       "thunder",
    "memmap_surface_event": "surface_event",
}   # 7 kelas; nama folder "memmap*" tidak boleh muncul di figur/laporan
```

Konsekuensi: checkpoint TCN 8-kelas yang ada menjadi **tidak sebanding**
— TCN ikut dilatih ulang pada gelombang retraining P1. Tabel hasil lama
diarsipkan (jangan dihapus), beri nama `results/tables/_archive_pre_v2/`.

### P0-3. Satu file split bersama untuk SEMUA model

Saat ini tiap script memanggil `train_test_split` sendiri; karena vektor
label TCN berbeda, **test set-nya pun berbeda**. Buat split SEKALI, simpan,
dan semua script hanya me-load:

```python
# scripts/make_splits.py  — jalankan SEKALI, commit script-nya
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

meta = np.load("/home/indra/eq_team/metadata_5s.npy", allow_pickle=True).item()
labels_raw = np.array([LABEL_MAP.get(str(l), str(l)) for l in meta["label"]])
le = LabelEncoder(); labels = le.fit_transform(labels_raw)

idx = np.arange(len(labels))
train_idx, test_idx = train_test_split(
    idx, test_size=0.20, random_state=42, stratify=labels)
train_idx, val_idx = train_test_split(
    train_idx, test_size=0.125, random_state=42, stratify=labels[train_idx])

np.savez("data/splits_5s.npz",
         train=train_idx, val=val_idx, test=test_idx,
         classes=le.classes_,
         split_type="random_per_trace_v2")   # ← v2 = ruang label kanonik
```

- `data/` tidak di-commit (`.gitignore`) — commit **script-nya**, dan catat
  SHA256 file split di `EXPERIMENTS.md` (`sha256sum data/splits_5s.npz`).
- **PENTING — jangan salah paham:** langkah ini hanya menjamin
  *komparabilitas antar model*. Ini **belum** memperbaiki event leakage —
  itu pekerjaan P2 (§5). Field `split_type` ada justru supaya setiap hasil
  bisa ditelusuri dipakai split yang mana.

### P0-4. Git + `EXPERIMENTS.md` + hasil ber-stempel run

Wajib v1.0 yang belum jalan sama sekali. Urutannya:

```bash
cd /home/indra/eq_team/earthquake-classification
git init
git add PANDUAN.md PANDUAN_new.md README.md .gitignore scripts/
git commit -m "chore: snapshot kondisi proyek sebelum revisi v2.0"
touch EXPERIMENTS.md
```

Aturan output mulai sekarang:

- **Dilarang menimpa** file metrik/figur. Bukti kerusakannya sudah ada:
  angka TCN di README tidak cocok dengan file mana pun karena
  `tcn_test_metrics.txt` ditimpa run berikutnya, dan versi `V2`/`V3`
  dinamai manual.
- Setiap run training menulis ke folder sendiri:
  `results/runs/<YYYY-MM-DD>_<model>_<git-hash-pendek>/`
  berisi checkpoint, log JSON, figur, metrik, dan salinan config.
- Setiap run penting dicatat di `EXPERIMENTS.md`: tanggal, model, config,
  split (`split_type` + SHA), metrik utama, observasi.

### P0-5. Tertibkan jalur output TCN

`script_TCN.py` masih menulis `best_tcn_5s.pt`, log, figur, dan metrik ke
**root tim** `/home/indra/eq_team/` (baris ±397–398, 600, 608) — melanggar
konvensi README. Arahkan ke `results/` seperti script lain (atau langsung
ke skema `results/runs/` di atas). Artefak TCN yang sekarang nyasar di root
tim dipindah/diarsipkan setelah dicek tidak dipakai proses lain.

### Checklist P0 (tick di EXPERIMENTS.md)

- [ ] flip/roll dihapus dari 5 script, diganti augmentasi §P0-1
- [ ] `LABEL_MAP` kanonik dipakai semua script; tidak ada string `memmap*` di output
- [ ] `data/splits_5s.npz` dibuat; semua script load split yang sama
- [ ] `git init` + commit awal; `EXPERIMENTS.md` ada
- [ ] skema `results/runs/` jalan; tidak ada lagi penimpaan file
- [ ] output TCN tidak lagi ke root tim

---

## 4. P1 — Konsolidasi Pipeline & Imbalance (1–2 minggu)

Target P1: **satu kali gelombang retraining** kelima model dengan protokol
identik, di atas kode yang sudah dirapikan. Jangan retrain sebelum P1-1
dan P1-2 selesai — supaya tidak melatih dua kali.

### P1-1. Refactor: hentikan copy-paste 5 × 600 baris

Lima script training ~90% identik; penyimpangannya sudah terbukti
menghasilkan bug nyata (ruang label beda, output nyasar). Pindah ke
struktur §4 v1.0:

```
src/
├── data.py          ← LABEL_MAP, EqDataset, load split, normalisasi
├── augmentations.py ← augmentasi sah §P0-1 (satu-satunya implementasi)
├── models/          ← cnn1d.py, mobilenet1d.py, tcn.py, transformer.py
├── engine.py        ← train loop, early stopping, AMP, checkpoint
└── evaluation.py    ← metrik lengkap §P1-4
scripts/
└── train.py         ← python scripts/train.py --config configs/tcn.yaml
configs/
└── {cnn,mobilenetv2,mobilenetv3,tcn,transformer}.yaml
```

Checkpoint wajib **self-describing** (state_dict telanjang itu jebakan —
urutan kelas LabelEncoder tidak tersimpan di mana pun):

```python
torch.save({
    "state_dict":  model.state_dict(),
    "classes":     list(le.classes_),      # urutan label!
    "config":      cfg,                    # dict YAML utuh
    "norm":        "zscore_per_trace_joint",
    "split_file":  "splits_5s.npz", "split_sha256": split_sha,
    "git_commit":  commit_hash,
    "epoch":       epoch, "val_loss": best_val_loss,
}, ckpt_path)
```

### P1-2. Normalisasi & preprocessing yang konsisten secara fisika

- **Ganti z-score per-kanal → z-score per-trace (3 kanal bersama).**
  Z-score per-kanal menyamakan energi E, N, Z — padahal **rasio amplitudo
  antar-kanal itu fitur fisik** (surface event kaya gelombang permukaan di
  horizontal; rasio Z/horizontal membedakan tipe sumber). Normalisasi per
  trace (mean & std dihitung dari ketiga kanal sekaligus) mempertahankan
  rasio itu.
- **Putuskan bandpass.** v1.0 §8.3 mensyaratkan Butterworth 1–45 Hz
  zero-phase. Saat ini tidak ada bandpass sama sekali — perbedaan respons
  instrumen STEAD vs PNW ikut masuk model (memperparah confound §P2-3).
  Keputusan (pakai/tidak + parameternya) ditulis di config dan di
  checkpoint, dan **sama untuk semua kelas dan semua model**.
- **Hapus `F.interpolate(size=500)`** di semua `__getitem__` — data 5 s
  memang sudah 500 sampel (no-op buang CPU). Ganti dengan
  `assert x.shape[-1] == SEQ_LEN`, supaya kalau script diarahkan ke
  `combined_3s.npy`/`combined_10s.npy` dia **gagal keras**, bukan diam-diam
  meregangkan data (= mengubah sampling rate efektif).

### P1-3. Tangani imbalance + keputusan taksonomi sonic/thunder

Fakta sekarang: CE polos di atas rasio 7.000:1 → sonic (206 trace) dan
thunder (146 trace) **mati total** (P = R = 0,0 di kelima model), dan
macro-metric diam-diam menyerap dua kelas mati itu.

Rancang sebagai **ablation study** (bagus untuk bab skripsi):

1. Baseline: CE polos (sudah ada).
2. CE berbobot kelas (mis. bobot ∝ `1/√count`).
3. `WeightedRandomSampler` (oversampling minoritas).
4. Focal loss (γ = 2).

Keputusan taksonomi (pilih satu, konsultasikan):

- **Opsi A (disarankan):** gabung `sonic` + `thunder` → `acoustic`
  (keduanya sumber akustik-atmosferik; 352 trace gabungan — tetap kecil
  tapi tidak hopeless).
- **Opsi B:** keluarkan keduanya dari training, laporkan kualitatif saja.

Apa pun pilihannya, laporkan **interval kepercayaan** untuk kelas kecil
(support test sonic = 41, thunder = 29 → metrik per-kelas punya
ketidakpastian ±~15%; pakai Wilson CI).

### P1-4. Suite evaluasi yang menjawab pertanyaan EEW

§7 v1.0 BERLAKU dan belum diimplementasikan. Tambahan untuk konteks
multi-class:

- **Pandangan biner turunan** — wajib: `earthquake` vs gabungan semua
  kelas lain. Dari sini hitung **FTR** = FP/(FP+TN), dan **recall gempa
  pada constraint FTR ≤ 5%** (sapu threshold di validation set). Ini
  metrik yang menjawab tujuan proyek; akurasi 8-kelas tidak.
- PR-AUC per kelas (lebih informatif dari ROC-AUC saat imbalance ekstrem).
- Breakdown per-SNR (< 10 / 10–20 / > 20 dB) — `snr_db` **sudah ada** di
  `metadata_5s.npy` (1.155.262 trace terisi), tinggal dipakai.
- Kalibrasi (reliability diagram + ECE) dan latency (batch 1/32/256).
- `scripts/evaluate_all.py`: load semua checkpoint → evaluasi di test set
  yang sama → tulis `results/tables/summary.csv`. **Tabel README
  digenerate dari file ini**, tidak pernah disalin tangan lagi.

### P1-5. Gelombang retraining v2

Setelah P1-1…P1-3 selesai: latih ulang kelima model, protokol identik
(split sama, label sama, augmentasi sama, normalisasi sama), catat semua
di `EXPERIMENTS.md`. Inilah tabel perbandingan pertama yang **sah** —
masih dengan caveat split acak (§1.3) sampai P2.

---

## 5. P2 — Validitas Ilmiah Penuh: Split Bebas-Leakage (2–4 minggu)

### P2-0. Kenapa ini P2, bukan P0?

Pertanyaan yang wajar — leakage kan masalah serius? Betul, dan justru
karena seriusnya, perbaikannya tidak boleh setengah-setengah:

1. Augmentasi terlarang **merusak model** pada setiap run baru — tiap hari
   ditunda = run rusak bertambah; perbaikannya 5 menit. Leakage tidak
   merusak model, ia **menggelembungkan angka pengukuran** — selama semua
   hasil diberi caveat (§1.3), riset bisa jalan terus dengan jujur.
2. Perbaikan split butuh **prasyarat infrastruktur**: `metadata_*.npy`
   gabungan belum punya `source_id`/stasiun/`trace_name` sama sekali, jadi
   split per-event *belum mungkin dilakukan* sebelum metadata dibangun
   ulang (§P2-1) — dan itu menyentuh file bersama di root tim
   (**koordinasi dengan pembimbing sebelum menulis apa pun ke sana**).
3. Prinsip §2: dampak × urgensi / usaha. P0 = usaha menit, mencegah
   kerusakan berjalan. P2 = usaha minggu, memperbaiki pengukuran.

### P2-1. Perluas metadata gabungan (prasyarat)

`combine_by_seconds_npy.py` saat ini hanya membawa 5 kunci numerik
(`META_KEYS`, baris ±34–40). Tambahkan per trace:

- `trace_name` (folder per-kelas sudah punya `trace_names.npy`),
- `event_id` / `source_id` (STEAD: kolom `source_id` di `merge.csv`;
  PNW: kolom event di CSV comcat — join via `trace_name`),
- `network_code` + `station_code` (STEAD: kolom CSV / parse dari
  `trace_name`; PNW: kolom `station_*` di CSV),
- `dataset_origin` (`stead` / `pnw`) dan `label_original`.

Catatan teknis penting:

- **Waveform TIDAK perlu ditulis ulang** — urutan baris tidak berubah.
  Cukup pass kedua yang hanya menghasilkan `metadata_*_v2.npy`. Tulis dulu
  ke nama baru, validasi, baru gantikan yang lama (file bersama!).
- Validasi wajib: panjang tiap array == 1.488.570, dan urutan per-blok
  cocok dengan `index.json` tiap folder sumber.
- Simpan label sebagai **kode integer + daftar kelas**, bukan array object
  (menghindari `allow_pickle=True` di semua loader).

### P2-2. Split per-event & per-stasiun

Aturan (sesuai §8.1 v1.0, sekarang baru bisa dieksekusi):

- Kelas bersumber event (earthquake, explosion, sonic, thunder,
  surface_event): grup berdasarkan **event_id** — satu event utuh masuk
  satu split saja.
- Kelas tanpa event (noise, no_event): grup berdasarkan
  **network+station** — satu stasiun tidak boleh muncul di dua split.

Implementasi: `sklearn.model_selection.GroupShuffleSplit` (atau
`StratifiedGroupKFold`) per kelompok kelas, lalu gabungkan indeksnya.
Simpan sebagai `data/splits_5s_eventaware.npz`
(`split_type="event_station_aware_v1"`).

Lalu: **latih ulang kelima model** pada split ini (pipeline P1 membuat ini
murah — hanya ganti satu path di config), dan laporkan **dua tabel
berdampingan**: split acak vs split event-aware. Selisihnya ("leakage
gap") adalah hasil riset yang menarik dan layak satu sub-bab skripsi.
**Ekspektasi: angka AKAN turun. Itu bukan kemunduran — angka jujur 95%
mengalahkan angka tak-terpertahankan 98% di sidang.**

### P2-3. Eksperimen confound dataset (STEAD vs PNW)

Fakta yang harus dihadapi di skripsi: kelas berkorelasi sempurna dengan
asal data (noise & no_event hanya dari STEAD; explosion, sonic, thunder,
surface_event hanya dari PNW). Run TCN 8-kelas lama bahkan membuktikan
*fingerprint* dataset bisa dipelajari: gempa STEAD vs gempa PNW terpisahkan
dengan F1 ~0,92. Artinya sebagian performa "noise vs explosion" bisa jadi
cuma "STEAD vs PNW".

Wajib dikerjakan:

1. **Uji lintas-dataset** untuk kelas yang ada di dua sumber: latih gempa
   hanya STEAD → uji di gempa PNW (dan sebaliknya). Selisihnya = ukuran
   domain gap.
2. Preprocessing identik antar sumber (bandpass §P1-2 membantu meredam
   perbedaan instrumen).
3. Satu sub-bab diskusi keterbatasan di skripsi. Kalau memungkinkan, cari
   trace noise dari PNW untuk menyeimbangkan asal kelas (diskusikan dulu).

---

## 6. P3 — Polish Engineering (boleh dicicil kapan saja)

Tidak menghalangi sains, tapi rapikan saat menyentuh file terkait:

- `matplotlib.use("Agg")` + `plt.close(fig)`; **hapus `plt.show()`** dari
  script headless (di tmux bisa menggantung tergantung backend).
- AMP: pakai **bf16** (`torch.amp.autocast(dtype=torch.bfloat16)`, tanpa
  `GradScaler`) — lebih stabil dari fp16 untuk transformer, didukung penuh
  RTX 5090.
- Ganti `TeeLogger` (membajak `sys.stdout`/`stderr` global) dengan modul
  `logging`, atau cukup `python train.py 2>&1 | tee run.log`.
- Seed worker DataLoader (`worker_init_fn` + `generator`) supaya augmentasi
  reproducible; pisahkan seed split vs seed training di config.
- Bersihkan: folder `code/` kosong (sudah ada `MOVED.md`), artefak TCN di
  root tim (setelah P0-5), `.vscode/` di root tim.
- `roc_auc_score` yang `except ValueError → nan`: log kelas mana yang
  gagal, jangan diam-diam.

---

## 7. Urutan Kerja & Pembagian (saran)

```
Minggu 1     : P0 lengkap (semua item) + mulai P1-1 (refactor)
Minggu 2     : P1-1 selesai, P1-2 (normalisasi), P1-4 (evaluate_all)
Minggu 3     : P1-3 (ablation imbalance) + P1-5 (retraining v2)
Minggu 4–5   : P2-1 (metadata) + P2-2 (split event-aware) + retraining v3
Minggu 6     : P2-3 (cross-dataset) + penulisan analisis
P3           : dicicil sepanjang jalan
```

Checklist progres dipelihara di `EXPERIMENTS.md`, ditick lewat commit.
Setiap akhir minggu tetap kirim ringkasan format §11 v1.0.

---

## 8. Yang TIDAK Berubah dari v1.0

Supaya tidak ada keraguan — aturan berikut tetap berlaku penuh:

- §3 v1.0: tmux selalu, conda `pytorch_Py12` (jangan bikin env baru),
  GPU etiquette (`nvidia-smi` dulu, jangan ganggu session `train`).
- §8.4 v1.0: tabel augmentasi sah/tidak sah (P0-1 adalah penegakan
  aturan ini, bukan aturan baru).
- §8.6 v1.0: larangan-larangan (jangan load 92 GB ke RAM, jangan
  `git add data/`, jangan force-push, dst.).
- §9 v1.0: referensi data & daftar bacaan.
- Dataset memmap gabungan **tetap di root tim** `/home/indra/eq_team/` —
  jangan dipindah ke folder proyek.

---

**Pesan penutup tetap sama dengan v1.0, dengan satu tambahan: kalau
perbaikan ini membuat angkamu turun, itu tandanya perbaikannya bekerja.
Pelan-pelan saja, yang penting hasilnya benar dan reproducible.**
— Pak Indra
