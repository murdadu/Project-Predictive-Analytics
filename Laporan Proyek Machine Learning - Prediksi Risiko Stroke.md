# Laporan Proyek Machine Learning - Javier Elian Putra Karjadi

## Kesehatan - Prediksi Risiko Stroke

### Latar Belakang
Stroke merupakan kondisi medis serius yang menempati urutan kedua sebagai penyebab kematian tertinggi di dunia menurut World Health Organization (WHO). Selain itu, stroke juga menjadi penyebab utama kecacatan jangka panjang, yang berdampak signifikan terhadap kualitas hidup penderitanya serta membebani sistem kesehatan secara ekonomi dan sosial. Faktor risiko stroke sangat beragam, mulai dari faktor yang tidak dapat diubah seperti usia dan riwayat keluarga, hingga faktor yang dapat dimodifikasi seperti hipertensi, penyakit jantung, diabetes, dan gaya hidup (misalnya merokok dan obesitas).

Deteksi dini individu yang berisiko tinggi sangat krusial dalam pencegahan stroke. Intervensi medis dan perubahan gaya hidup yang dilakukan lebih awal dapat secara drastis mengurangi kemungkinan terjadinya stroke. Namun, penilaian risiko secara manual seringkali tidak mampu menangkap hubungan yang kompleks antara berbagai faktor risiko. Oleh karena itu, pemanfaatan *machine learning* untuk menganalisis data kesehatan pasien menjadi solusi yang menjanjikan. Dengan kemampuannya mengenali pola-pola tersembunyi, *machine learning* dapat membantu tenaga medis dalam mengidentifikasi individu berisiko tinggi secara lebih akurat dan efisien, sehingga tindakan preventif dapat diberikan secara tepat sasaran.

## Business Understanding

### Problem Statement
Berdasarkan latar belakang tersebut, permasalahan yang akan diselesaikan dalam proyek ini adalah:
* Bagaimana cara membangun sebuah model klasifikasi yang dapat memprediksi dengan andal apakah seorang pasien berisiko mengalami stroke berdasarkan data demografis, riwayat medis, dan gaya hidup mereka?
* Bagaimana cara mengatasi tantangan dataset yang sangat tidak seimbang (*imbalanced*), di mana jumlah pasien stroke jauh lebih sedikit dibandingkan pasien non-stroke, agar model yang dihasilkan tidak bias dan efektif dalam mendeteksi kasus stroke?

### Goals
Tujuan dari proyek ini adalah sebagai berikut:
* Menganalisis dan memahami data pasien untuk mengidentifikasi fitur-fitur yang paling berpengaruh terhadap risiko stroke.
* Melakukan pra-pemrosesan data untuk menangani masalah seperti nilai yang hilang (*missing values*) dan ketidakseimbangan kelas.
* Mengembangkan beberapa model *machine learning* dan membandingkan kinerjanya untuk menemukan model yang paling sesuai untuk masalah ini.
* Mengevaluasi model terbaik menggunakan metrik yang relevan dengan konteks medis, terutama kemampuan model untuk meminimalkan kasus stroke yang tidak terdeteksi (*false negatives*).

### Solution Approach
Untuk mencapai tujuan yang telah ditetapkan, pendekatan solusi yang akan dilakukan adalah sebagai berikut:
1.  **Membangun Beberapa Model Klasifikasi:** Mengembangkan dan membandingkan performa dari tiga algoritma klasifikasi yang berbeda:
    * **Logistic Regression:** Sebagai model dasar (*baseline*) yang sederhana dan mudah diinterpretasikan.
    * **Random Forest:** Sebagai model *ensemble* yang lebih kompleks dan kuat.
    * **XGBoost:** Sebagai model *gradient boosting* yang canggih untuk performa tinggi.
    Tujuan perbandingan ini adalah untuk memilih algoritma dengan kinerja terbaik, terutama pada metrik `Recall`.
2.  **Melakukan Optimisasi pada Model Terbaik:** Setelah model terbaik dipilih, dilakukan proses *hyperparameter tuning* menggunakan `GridSearchCV`. Proses ini bertujuan untuk mencari kombinasi parameter terbaik dari model tersebut agar kinerjanya, terutama dalam mendeteksi kasus stroke, menjadi lebih optimal.

Keberhasilan pendekatan ini akan diukur menggunakan metrik evaluasi seperti `Recall`, `Precision`, `F1-Score`, dan `ROC-AUC`.

## Data Understanding
Dataset yang digunakan dalam proyek ini adalah **"Stroke Prediction Dataset"** yang diperoleh dari platform Kaggle. Dataset ini berisi 5110 catatan pasien dengan 12 fitur, termasuk informasi demografis, riwayat medis, dan variabel target yang menunjukkan apakah pasien mengalami stroke.

* **Sumber Dataset:** [Kaggle: Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

Kondisi data menunjukkan adanya 201 nilai yang hilang pada fitur `bmi` dan distribusi kelas target yang sangat tidak seimbang.

### Variabel-variabel pada Dataset
Berikut adalah daftar variabel beserta deskripsinya:
| Variabel | Tipe Data | Deskripsi |
|---|---|---|
| **id** | Numerik | ID unik untuk setiap pasien |
| **gender** | Kategorikal | Jenis kelamin pasien ('Male', 'Female', 'Other') |
| **age** | Numerik | Usia pasien |
| **hypertension** | Biner | 0 = tidak hipertensi, 1 = menderita hipertensi |
| **heart_disease** | Biner | 0 = tidak punya penyakit jantung, 1 = punya penyakit jantung |
| **ever_married** | Kategorikal | Status pernikahan ('Yes' atau 'No') |
| **work_type** | Kategorikal | Tipe pekerjaan pasien |
| **Residence_type** | Kategorikal | Tipe area tempat tinggal ('Urban' atau 'Rural') |
| **avg_glucose_level**| Numerik | Rata-rata kadar glukosa dalam darah |
| **bmi** | Numerik | Indeks Massa Tubuh (Body Mass Index) |
| **smoking_status** | Kategorikal | Status merokok pasien |
| **stroke** | Biner | **Variabel Target**: 1 = pasien mengalami stroke, 0 = tidak |

# Data Preparation

Pada tahap ini, dilakukan serangkaian proses untuk membersihkan, mentransformasi, dan menyiapkan data agar siap digunakan untuk membangun model klasifikasi. Teknik yang digunakan diterapkan secara berurutan untuk memastikan kualitas dan konsistensi data.

## 1. Menangani Data Anomali dan Tidak Relevan

**Proses yang Dilakukan:**
- Menghapus satu baris data di mana nilai gender adalah `'Other'`.
- Menghapus seluruh kolom `id`.

**Alasan:**
- Data `'Other'` hanya berjumlah satu sampel sehingga tidak representatif dan dapat dianggap sebagai anomali.
- Kolom `id` merupakan pengenal unik yang tidak memiliki nilai prediktif atau korelasi dengan target risiko stroke.

## 2. Imputasi Nilai Hilang (Missing Values)

**Proses yang Dilakukan:**  
Terdapat 201 nilai yang hilang pada fitur `bmi`. Nilai-nilai ini diisi (imputed) menggunakan nilai median dari kolom `bmi`.

**Alasan:**  
Median dipilih karena lebih tahan terhadap nilai ekstrem (outlier) dibandingkan rata-rata (mean). Hal ini memastikan bahwa nilai yang diimputasi tidak terlalu terpengaruh oleh beberapa pasien dengan BMI yang sangat tinggi atau rendah.

## 3. Pembagian Dataset

**Proses yang Dilakukan:**  
Dataset dibagi menjadi data latih (80%) dan data uji (20%) menggunakan fungsi `train_test_split` dari library Scikit-learn.

**Alasan:**  
Pembagian ini penting untuk mengevaluasi kinerja model secara objektif. Parameter `stratify=y` digunakan agar proporsi kelas stroke tetap seimbang di kedua subset data.

## 4. Transformasi Fitur (Encoding dan Scaling)

**Proses yang Dilakukan:**
- **One-Hot Encoding:** Diterapkan pada semua fitur kategorikal (`gender`, `work_type`, dll.).
- **Standardisasi:** Diterapkan pada semua fitur numerik (`age`, `avg_glucose_level`, `bmi`) menggunakan `StandardScaler`.

**Alasan:**  
Model machine learning memerlukan input numerik, sehingga encoding diperlukan. Standardisasi penting agar fitur berada dalam skala yang sama, terutama untuk model seperti Logistic Regression.

## 5. Penanganan Ketidakseimbangan Data (Oversampling)

**Proses yang Dilakukan:**  
Teknik SMOTE (Synthetic Minority Over-sampling Technique) diterapkan pada data latih.

**Alasan:**  
Karena kelas stroke sangat minoritas (~5%), SMOTE membantu menyeimbangkan distribusi kelas tanpa diterapkan pada data uji agar evaluasi tetap realistis.

---

## Modeling

Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan klasifikasi risiko stroke. Pendekatan yang digunakan adalah membandingkan beberapa algoritma kemudian melakukan optimisasi pada model yang paling sesuai dengan tujuan proyek.



### Penjelasan Algoritma yang Digunakan

Tiga algoritma klasifikasi dengan karakteristik berbeda dipilih untuk dievaluasi:

#### 1. Logistic Regression
- **Cara Kerja**:  
  Logistic Regression adalah algoritma klasifikasi yang bekerja dengan memprediksi probabilitas suatu data masuk ke dalam kategori tertentu. Algoritma ini menggunakan fungsi logistik (sigmoid) untuk "menekan" output dari persamaan linear menjadi nilai antara 0 dan 1. Nilai ini kemudian diinterpretasikan sebagai probabilitas. Jika probabilitasnya melebihi ambang batas tertentu (biasanya 0.5), data diklasifikasikan sebagai kelas positif (dalam kasus ini, *stroke*).  
  Model ini mencari sebuah *decision boundary* (garis atau bidang batas) terbaik yang memisahkan dua kelas.

- **Parameter Awal**:  
  - `random_state=42` untuk memastikan hasil yang dapat direproduksi  
  - `max_iter=1000` agar model memiliki iterasi cukup untuk mencapai konvergensi



#### 2. Random Forest
- **Cara Kerja**:  
  Random Forest adalah model ensemble yang membangun banyak *Decision Trees* pada sampel data acak. Setiap pohon memberikan suara, dan hasil akhir ditentukan oleh suara mayoritas (*majority voting*).  
  Dengan menggabungkan banyak model yang berbeda-beda, Random Forest mampu mengurangi overfitting dan meningkatkan stabilitas serta akurasi.

- **Parameter Awal**:  
  - `random_state=42` untuk menjaga konsistensi hasil



#### 3. XGBoost (Extreme Gradient Boosting)
- **Cara Kerja**:  
  XGBoost adalah algoritma boosting yang membangun pohon secara berurutan (sequentially), di mana setiap pohon baru berfokus pada memperbaiki kesalahan prediksi pohon sebelumnya. Data yang sulit diprediksi akan mendapat bobot lebih besar, sehingga model lebih fokus belajar dari kesalahan.  
  Proses *gradient boosting* ini membuat XGBoost sangat kuat dalam menghasilkan model yang presisi dan efisien.

- **Parameter Awal**:  
  - `random_state=42` untuk hasil yang konsisten  
  - `use_label_encoder=False` untuk menghindari warning  
  - `eval_metric='logloss'` sebagai metrik selama pelatihan



### Proses Pemilihan dan Improvement Model

#### Perbandingan Model Awal
Ketiga model dilatih menggunakan data latih, lalu diuji menggunakan data uji. Tujuannya adalah memilih model dengan kinerja terbaik sesuai fokus proyek.

#### Pemilihan Model Terbaik
Setelah evaluasi, **Logistic Regression** dipilih sebagai model terbaik.

#### Alasan Pemilihan
Meskipun Random Forest dan XGBoost menunjukkan akurasi lebih tinggi, Logistic Regression memberikan nilai **Recall** yang jauh lebih tinggi (`0.80`).  
Dalam konteks medis, **Recall lebih penting** karena kegagalan mendeteksi pasien yang sebenarnya stroke (False Negative) bisa sangat berbahaya. Oleh karena itu, prioritas utama adalah memaksimalkan Recall.

---

### Proses Improvement (Hyperparameter Tuning)

#### Metode: GridSearchCV
Model Logistic Regression yang dipilih kemudian dioptimasi menggunakan **GridSearchCV**, sebuah teknik untuk mencari kombinasi hyperparameter terbaik.

- **Penjelasan Proses**:  
  GridSearchCV secara sistematis menguji kombinasi parameter seperti:
  - `C` (regularisasi),
  - `solver` (algoritma optimasi),
  - `penalty` (tipe regularisasi),
  
  Teknik ini menggunakan *cross-validation* untuk memilih kombinasi yang menghasilkan nilai **Recall tertinggi**, agar model semakin optimal dalam mendeteksi pasien berisiko stroke.


---

## Evaluation

Pada bagian ini, dilakukan evaluasi terhadap kinerja model menggunakan metrik yang relevan dengan konteks masalah klasifikasi pada data medis yang tidak seimbang.

### Metrik Evaluasi

Metrik yang digunakan adalah sebagai berikut:

#### Confusion Matrix
Sebuah tabel yang merangkum hasil prediksi dengan membandingkannya dengan kelas aktual. Tabel ini membantu kita melihat secara detail di mana model berhasil dan di mana ia membuat kesalahan. Isinya adalah:

- **True Positive (TP)**: Pasien stroke yang diprediksi dengan benar sebagai stroke.
- **True Negative (TN)**: Pasien non-stroke yang diprediksi dengan benar sebagai non-stroke.
- **False Positive (FP)**: Pasien non-stroke yang salah diprediksi sebagai stroke. Ini bisa dianggap sebagai "alarm palsu".
- **False Negative (FN)**: Pasien stroke yang salah diprediksi sebagai non-stroke. Ini adalah tipe kesalahan paling fatal dalam konteks medis karena pasien berisiko tidak mendapatkan penanganan.

#### Accuracy
Metrik ini mengukur seberapa sering model membuat prediksi yang benar secara keseluruhan (baik prediksi stroke maupun non-stroke).  
Meskipun umum digunakan, metrik ini bisa sangat menyesatkan pada kasus data tidak seimbang karena model bisa saja hanya menebak kelas mayoritas untuk mencapai akurasi tinggi.

#### Precision
Metrik ini mengukur tingkat keakuratan dari prediksi positif.  
**Pertanyaan yang dijawab:** *Dari semua pasien yang oleh model diprediksi akan stroke, berapa persen yang benar-benar stroke?*  
Presisi tinggi berarti model jarang memberikan alarm palsu.

#### Recall (Sensitivity)
Metrik ini mengukur kemampuan model untuk *menemukan* semua kasus positif yang sebenarnya.  
**Pertanyaan yang dijawab:** *Dari semua pasien yang benar-benar menderita stroke, berapa persen yang berhasil dideteksi oleh model?*  
Ini adalah metrik paling krusial untuk proyek ini.

#### F1-Score
Metrik ini memberikan skor tunggal yang menyeimbangkan antara Precision dan Recall.  
F1-Score sangat berguna ketika kita ingin menjaga keseimbangan antara tidak membuat terlalu banyak alarm palsu dan tidak melewatkan kasus nyata.

---

### Hasil Proyek Berdasarkan Metrik Evaluasi

Tabel berikut merangkum hasil perbandingan akhir dari semua model pada data uji:

| Model                     | Accuracy  | Precision | Recall | F1-Score | ROC-AUC  |
|--------------------------|-----------|-----------|--------|----------|----------|
| Logistic Regression Tuned | 0.709393 | 0.126888  | 0.84   | 0.220472 | 0.771337 |
| Logistic Regression       | 0.733855 | 0.132450  | 0.80   | 0.227273 | 0.765226 |
| XGBoost                   | 0.933464 | 0.275000  | 0.22   | 0.244444 | 0.595082 |
| Random Forest             | 0.929550 | 0.133333  | 0.08   | 0.100000 | 0.526626 |

---

### Kesimpulan

Berdasarkan hasil evaluasi, dapat disimpulkan bahwa **Logistic Regression Tuned** adalah solusi terbaik untuk masalah ini.  
Proses **hyperparameter tuning** berhasil meningkatkan metrik paling krusial, yaitu **Recall**, dari `0.80` menjadi `0.84`.  

Artinya, model final mampu mengidentifikasi 84% dari total pasien stroke yang sebenarnya dalam data uji.
Meskipun akurasi dan presisinya lebih rendah dibanding model lain, kemampuan model ini dalam meminimalkan kasus yang terlewat (False Negatives) menjadikannya **pilihan yang paling bertanggung jawab** dan **sesuai dengan tujuan proyek**.

