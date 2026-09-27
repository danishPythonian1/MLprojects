# Framework: EDA & Data Cleaning
## Kompetisi Kaggle — Predicting Electric Vehicle Purchases

---

## 1. Tujuan
Menyiapkan dataset `train.csv` (668.665 baris, 15 kolom) agar siap dipakai untuk pemodelan klasifikasi target `Will_Buy_EV` (Yes/No).

---

## 2. Struktur Kolom

| Tipe | Kolom |
|---|---|
| ID | `id` |
| Numerik kontinu | `Annual_Income_USD`, `Daily_Commute_km` |
| Numerik diskrit | `Age`, `Number_of_Cars_Owned`, `Charging_Stations_Near_Home`, `Charging_Stations_Near_Work`, `Environmental_Concern_Level` |
| Kategorikal nominal | `Gender`, `City_Type`, `Current_Car_Type` |
| Kategorikal biner | `Home_Charging_Possible`, `Subsidy_Available` |
| Kategorikal ordinal | `Range_Anxiety_Level` (Low/Medium/High) |
| Target | `Will_Buy_EV` (No/Yes) — biner, imbalance ±82.5% : 17.5% |

---

## 3. Tahapan Framework

### A. Data Understanding
1. Cek dimensi, tipe data, dan contoh baris.
2. Cek nilai unik tiap kolom kategorikal (validasi kategori valid, tidak ada typo/inkonsistensi kapitalisasi).
3. Statistik deskriptif kolom numerik (min, max, mean, quartile).

### B. Data Quality Check
1. **Missing values** — cek per kolom.
2. **Duplicate rows** — cek baris duplikat penuh, dan duplikasi `id`.
3. **Outlier detection** — metode IQR pada kolom numerik kontinu/diskrit, dievaluasi apakah outlier tersebut error input atau variasi alami (mis. income tinggi tetap valid).
4. **Konsistensi kategorikal** — whitespace, casing, kategori tak terduga.
5. **Validitas rentang nilai** — usia, income, commute km tidak boleh negatif atau di luar rentang wajar.

### C. Exploratory Data Analysis (EDA)
1. Distribusi target (`Will_Buy_EV`) — cek class imbalance.
2. Distribusi tiap fitur numerik (histogram) & kategorikal (bar chart).
3. Hubungan tiap fitur terhadap target:
   - Numerik → boxplot / mean per kelas target.
   - Kategorikal → crosstab proporsi target per kategori.
4. Korelasi antar fitur numerik (heatmap) untuk deteksi multikolinearitas.
5. Insight awal yang relevan untuk feature engineering (mis. `Range_Anxiety_Level` rendah cenderung lebih mau beli EV, dsb.).

### D. Data Cleaning
1. Standardisasi string (`strip`, konsistensi kapital) pada kolom kategorikal.
2. Encoding awal yang aman disimpan di dataset "cleaned" (tanpa one-hot dulu, supaya fleksibel untuk tahap modeling):
   - Kolom biner `Yes/No` → tetap disimpan sebagai string, siap di-encode saat modeling.
3. Penanganan outlier: **tidak dihapus** karena tervalidasi sebagai variasi alami (bukan error), namun didokumentasikan agar tim modeling sadar (mis. Annual_Income_USD & Number_of_Cars_Owned punya outlier di sisi atas).
4. Cek ulang tipe data akhir tiap kolom.
5. Simpan dataset hasil cleaning → `train_cleaned.csv`.

### E. Output
1. `framework_EDA_cleaning.md` — dokumen ini.
2. `eda_cleaning_pipeline.py` — kode Python end-to-end (EDA + cleaning) yang dapat dijalankan ulang.
3. `train_cleaned.csv` — dataset yang sudah melalui proses cleaning & validasi.

---

## 4. Temuan Utama EDA (ringkasan)
- **Tidak ada missing value** dan **tidak ada baris duplikat** di seluruh 668.665 baris.
- `id` unik untuk setiap baris (0–668664), aman dijadikan index.
- Semua kategori kolom kategorikal sudah bersih (tidak ada typo/whitespace), sehingga tidak perlu perbaikan besar.
- Target `Will_Buy_EV` **imbalanced**: ±82.5% "No" vs 17.5% "Yes" → perlu diperhatikan saat modeling (mis. stratified split, class weighting, atau resampling).
- Outlier terdeteksi (IQR) pada `Annual_Income_USD` (~3.678 baris) dan `Number_of_Cars_Owned` (~13.489 baris di batas atas), tetapi nilai-nilai ini masih dalam rentang wajar (bukan kesalahan input) sehingga dipertahankan.
- Rentang nilai numerik semuanya masuk akal secara domain (Age 25–69, Income $30rb–$188rb, dsb.) — tidak ditemukan nilai negatif atau anomali ekstrem.

Detail visual & analisis lebih lanjut ada di script `eda_cleaning_pipeline.py` dan grafik yang dihasilkan.
