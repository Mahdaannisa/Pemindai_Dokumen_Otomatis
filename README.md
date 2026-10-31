# Pemindaian Dokumen Otomatis

## 1. Nama Anggota

| No. | Nama Anggota | NPM |
|---|---|---|
| 1 | Mahda Annisa | 2408107010036 |
| 2 | M. Rayan Zishan | 2408107010063 |
| 3 | Zanna Zikraana | 2408107010094 |
| 4 | Cut Farah Salsabila | 2408107010100 |


---

## 2. Deskripsi

Pemindaian Dokumen Otomatis merupakan proyek pengolahan citra digital yang bertujuan untuk mendeteksi batas dokumen dari sebuah foto, memperbaiki perspektif, dan menghasilkan gambar hasil pemindaian dalam format hitam-putih.

Foto dokumen sering kali memiliki masalah seperti posisi kamera yang miring, bayangan, pencahayaan yang tidak merata, dan latar belakang yang menyerupai warna kertas. Permasalahan tersebut dapat memengaruhi kualitas hasil pemindaian.

Proyek ini menggunakan Python dan OpenCV untuk melakukan deteksi tepi, pencarian kontur, deteksi empat sudut dokumen, transformasi perspektif, dan thresholding.

### Tujuan

1. Mendeteksi empat sudut dokumen secara otomatis.
2. Meluruskan perspektif dokumen yang difoto dari sudut tertentu.
3. Menghasilkan gambar dengan rasio mendekati ukuran kertas A4.
4. Mengubah gambar menjadi citra biner hitam-putih agar tulisan lebih mudah dibaca.
5. Membandingkan beberapa metode thresholding untuk melihat perbedaan hasil pemrosesan.

### Teknologi yang Digunakan

- Python
- OpenCV
- NumPy
- Jupyter Notebook
- Streamlit untuk antarmuka website

---

## 3. Pipeline

Pipeline merupakan tahapan pemrosesan citra dari foto dokumen asli hingga menghasilkan gambar hasil pemindaian.

### Diagram Pipeline

```text
Foto Dokumen
     |
     v
Membaca dan Mengubah Ukuran Citra
     |
     v
Grayscale dan CLAHE
     |
     v
Gaussian Blur
     |
     v
Deteksi Tepi Canny
     |
     v
Pencarian Kontur (findContours)
     |
     v
Penyederhanaan Kontur (approxPolyDP)
     |
     v
Pemilihan dan Pengurutan Empat Sudut
     |
     v
Transformasi Perspektif
(getPerspectiveTransform dan warpPerspective)
     |
     v
Grayscale, CLAHE, dan Gaussian Blur
     |
     v
Thresholding
     |
     v
Pembersihan Noise dengan Morfologi Opening
     |
     v
Hasil Pemindaian Hitam-Putih
```

### Penjelasan Pipeline

**1. Membaca dan Mengubah Ukuran Citra**

Foto dokumen dibaca menggunakan OpenCV dan ukurannya disesuaikan untuk membantu mengurangi beban pemrosesan.

**2. Grayscale dan CLAHE**

Citra diubah menjadi grayscale agar pemrosesan lebih sederhana. CLAHE digunakan untuk meningkatkan kontras lokal sehingga batas dokumen lebih mudah dikenali.

**3. Gaussian Blur dan Canny**

Gaussian Blur membantu mengurangi noise. Selanjutnya, Canny digunakan untuk mendeteksi tepi dokumen.

**4. Pencarian Kontur dan approxPolyDP**

Kontur dicari menggunakan `findContours`, kemudian disederhanakan menggunakan `approxPolyDP` untuk menemukan kandidat bentuk dokumen dengan empat titik sudut.

**5. Pengurutan Empat Sudut**

Empat titik sudut diurutkan agar transformasi perspektif dapat dilakukan dengan susunan titik yang konsisten.

**6. Transformasi Perspektif**

Fungsi `getPerspectiveTransform` dan `warpPerspective` digunakan untuk meluruskan dokumen sehingga hasilnya menyerupai dokumen yang dipindai secara langsung.

Ukuran keluaran pada notebook adalah 840 × 1188 piksel, dengan rasio mendekati ukuran kertas A4 dalam orientasi potret.

**7. Thresholding**

Citra yang telah diluruskan diproses menggunakan metode thresholding untuk memisahkan tulisan dari latar belakang.

**8. Pembersihan Noise**

Operasi morfologi opening digunakan untuk membantu mengurangi bintik-bintik kecil pada citra biner.

---

## 4. Dataset

Dataset yang digunakan terdiri dari **29 foto dokumen** yang dikumpulkan untuk pengujian sistem pemindaian dokumen otomatis.

Jenis dokumen meliputi halaman buku, struk, tulisan tangan, fotokopi, dan brosur.

### Kategori Dataset

| No. | Kategori | Jumlah Foto |
|---|---|---:|
| 1 | Ideal | 5 |
| 2 | Cahaya dan bayangan | 4 |
| 3 | Sudut miring atau rotasi | 12 |
| 4 | Kondisi sulit | 8 |
| | **Total** | **29** |

Dataset juga menggunakan file `ground_truth.csv` yang berisi nama file, kategori foto, jenis dokumen, dan informasi apakah keempat sudut dokumen terlihat penuh atau tidak.

Dataset tersebut digunakan untuk membantu evaluasi hasil pemrosesan pada berbagai kondisi foto.

---

## 5. Hasil

Berdasarkan hasil pengujian yang dicatat dalam laporan, sistem berhasil memproses seluruh foto dalam dataset.

### Ringkasan Hasil Pengujian

| No. | Parameter | Hasil |
|---|---|---|
| 1 | Jumlah foto dataset | 29 |
| 2 | Foto berhasil dibaca | 29 |
| 3 | Foto berstatus pemrosesan berhasil | 29 |
| 4 | Foto berstatus pemrosesan gagal | 0 |
| 5 | Ukuran keluaran | 840 × 1188 piksel |
| 6 | Format hasil | PNG |
| 7 | Metode thresholding utama | Adaptive Gaussian + morfologi opening |

### Tingkat Keberhasilan Pemrosesan

Tingkat keberhasilan berdasarkan status pemrosesan program dihitung sebagai berikut:

**Keberhasilan = (29 / 29) × 100% = 100%**

Hasil tersebut menunjukkan bahwa seluruh foto menghasilkan keluaran berdasarkan status program.

Namun, tingkat keberhasilan pemrosesan tidak sama dengan akurasi deteksi sudut sebesar 100%. Ketepatan empat sudut dokumen masih perlu diperiksa secara visual atau dibandingkan dengan koordinat sudut acuan.

---

## 6. Eksperimen Pembanding

Eksperimen dilakukan untuk membandingkan metode thresholding dan pengaruh nilai epsilon terhadap penyederhanaan kontur.

### A. Perbandingan Metode Thresholding

| Metode | Cara Kerja | Hasil Pengamatan |
|---|---|---|
| Otsu | Menentukan ambang global secara otomatis berdasarkan distribusi intensitas citra. | Cocok pada pencahayaan yang merata, tetapi dapat kurang optimal ketika terdapat bayangan. |
| Adaptive Mean | Menentukan ambang berdasarkan rata-rata intensitas lingkungan piksel. | Dapat menyesuaikan perubahan pencahayaan, tetapi terkadang menghasilkan lebih banyak noise. |
| Adaptive Gaussian | Menentukan ambang lokal menggunakan pembobotan Gaussian. | Memberikan hasil visual yang cukup baik dalam menyeimbangkan tulisan dan latar belakang pada contoh yang diamati. |
| Adaptive Gaussian + Opening | Menggabungkan Adaptive Gaussian dengan operasi morfologi opening. | Digunakan sebagai keluaran utama untuk membantu mengurangi bintik-bintik kecil. |

Berdasarkan pengamatan dalam laporan, Adaptive Gaussian memberikan hasil visual yang cukup baik sehingga dipilih sebagai metode keluaran utama. Operasi morfologi opening digunakan untuk membantu mengurangi noise kecil.

### B. Eksperimen Nilai Epsilon

Eksperimen juga dilakukan menggunakan beberapa nilai epsilon pada fungsi `approxPolyDP`.

| Nilai Epsilon | Tujuan Pengujian |
|---|---|
| 0,01 | Mengamati penyederhanaan kontur dengan toleransi kecil. |
| 0,015 | Mengamati perubahan jumlah titik kontur. |
| 0,02 | Menguji penyederhanaan dengan toleransi menengah. |
| 0,03 | Memeriksa apakah bentuk dokumen masih terwakili dengan baik. |
| 0,04 | Mengamati kemungkinan hilangnya detail sudut akibat penyederhanaan. |

Nilai epsilon memengaruhi tingkat penyederhanaan kontur. Nilai yang lebih kecil cenderung mempertahankan lebih banyak detail, sedangkan nilai yang lebih besar memberikan toleransi penyederhanaan yang lebih tinggi.

Hasil eksperimen digunakan untuk membantu memahami pemilihan kandidat kontur dalam proses deteksi sudut dokumen.

---

## 7. Cara Instalasi di Laptop Lokal

### Persyaratan

Pastikan Python dan Git sudah terpasang pada laptop.

### Langkah 1: Clone Repository

Buka terminal, kemudian jalankan:

```bash
git clone https://github.com/Mahdaannisa/Pemindai_Dokumen_Otomatis.git
```

Masuk ke folder proyek:

```bash
cd NAMA-REPOSITORY
```

### Langkah 2: Membuat Virtual Environment

```bash
python -m venv venv
```

Aktifkan virtual environment di Windows:

```bash
venv\Scripts\activate
```

Untuk macOS atau Linux:

```bash
source venv/bin/activate
```

### Langkah 3: Instal Dependensi

Jika repository sudah memiliki file `requirements.txt`, jalankan:

```bash
pip install -r requirements.txt
```

Jika file tersebut belum tersedia, instal pustaka dasar yang dibutuhkan:

```bash
pip install numpy opencv-python-headless matplotlib pandas streamlit
```

Untuk menjalankan notebook Jupyter, instal:

```bash
pip install notebook ipykernel
```

### Langkah 4: Menjalankan Notebook

Jalankan perintah:

```bash
jupyter notebook
```

Buka file `UTS_Kelompok03.ipynb` dan jalankan sel secara berurutan.

Pastikan dataset tersedia dan lokasi file sesuai dengan kode yang digunakan.

### Langkah 5: Menjalankan Website Streamlit

Jika aplikasi Streamlit sudah tersedia dalam file `app.py`, jalankan:

```bash
streamlit run app.py
```

Buka alamat lokal yang ditampilkan oleh Streamlit pada terminal untuk menggunakan aplikasi.

Pengguna dapat mengunggah foto dokumen dan melihat hasil pemrosesan sesuai fitur yang tersedia pada aplikasi.

---

## 8. Link Proyek

**Repository GitHub:**  
[https://github.com/Mahdaannisa/Pemindai_Dokumen_Otomatis]

**Website Streamlit:**  
[https://pemindaidokumenotomatis-qd2ga3bxuruastsmxmq3j3.streamlit.app/]

---

## Kesimpulan

Proyek Pemindaian Dokumen Otomatis menggunakan pengolahan citra digital untuk mendeteksi empat sudut dokumen, memperbaiki perspektif, dan menghasilkan citra hitam-putih.

Berdasarkan pengujian yang dicatat dalam laporan, sebanyak 29 dari 29 foto berhasil diproses menurut status program. Eksperimen thresholding menunjukkan bahwa Adaptive Gaussian yang diikuti morfologi opening memberikan hasil visual yang cukup baik pada contoh yang diamati.

Meskipun demikian, ketepatan deteksi sudut masih perlu dievaluasi lebih lanjut agar sistem dapat menghasilkan pemindaian dokumen yang lebih akurat pada berbagai kondisi pencahayaan dan sudut pengambilan gambar.
