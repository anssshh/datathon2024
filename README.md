# RISTEK Datathon 2024
## About the Competition
### Tujuan Kompetisi
Objektif dari kompetisi ini adalah untuk mengembangkan model machine learning untuk deteksi fraud pada pengguna platform fintech. Segala metode machine learning, matematis, dan statistik dapat digunakan untuk meningkatkan performa dari model yang dipakai. Selain mengembangkan model klasifikasi, kemampuan analisis mengenai pola pengguna yang teridentifikasi sebagai fraud juga diperlukan untuk menjelaskan cara kerja model yang dipakai.
## Apa itu fraud detection?
Fraud detection adalah proses identifikasi tindakan pengguna pada suatu skenario termasuk sebagai tindakan penipuan atau bukan. Dalam konteks kompetisi ini, tindakan penipuan didefinisikan sebagai pengguna platform yang telah meminjam produk keuangan tetapi tercatat belum melakukan pembayaran sampai tenggat waktu yang telah ditentukan.

### Evaluation
#### Penilaian Keseluruhan

![image](https://github.com/user-attachments/assets/3cabc5fa-b0d9-4766-b2cc-bd3f8153de6d)

#### Detail Kriteria
- Skor Private Leaderboard: skor yang muncul setelah tahap kompetisi berakhir, yaitu 50% dari data test berdasarkan kriteria yang sudah ditentukan panitia.
- Analisis: pemahaman format data, interpretasi data berdasarkan pemahaman peserta, dan temuan-temuan penting yang dapat membantu pemrosesan data.
- Pemrosesan Data: transformasi dari data mentah menjadi bentuk data yang dapat diterima oleh model. Termasuk proses rekayasa fitur.
- Modeling: perancangan arsitektur model beserta intuisi arsitektur tersebut, evaluasi dari model yang dibentuk, dan analisis hasil prediksi model.
- Sistematika Notebook: mencakup bagian pendahuluan, kesimpulan, dan tata tulis serta kelengkapan isi notebook.
Bagian analisis, pemrosesan data, dan modeling wajib disertai dengan penjelasan yang lengkap dan detail, tidak berupa kode saja karena penjelasan tersebut akan masuk ke dalam penilaian.

#### Metrik Leaderboard
Performa model dievaluasi menggunakan metrik Average Precision dengan average='macro'. Secara formal, metrik Average Precision dirumuskan dalam bentuk:

![image](https://github.com/user-attachments/assets/c4277c63-987c-4fd3-ae65-7a8ab0e08b46)

Implementasi metrik tersebut dalam Python dengan menggunakan library Scikit-Learn adalah sebagai berikut.

```
from sklearn.metrics import average_precision_score

score = average_precision_score(y_true, y_pred)

```
Penggunaan metrik Average Precision didasarkan pada penekanan fokus untuk merefleksikan kemampuan model dalam mendeteksi pengguna dengan label fraud sebagai fraud, bukan sebagai pengguna biasa (non-fraud).

---

### Dataset
Peserta dapat mengunduh dataset kompetisi dengan kode sebagai berikut.
```
!pip install gdown

import os
import gdown
import zipfile
import logging
from genericpath import isdir

def download_data(url, filename, dir_name: str = "data") -> None:
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    os.chdir(dir_name)
    logging.info("Downloading data....")
    gdown.download(
        url, quiet=False
    )
    logging.info("Extracting zip file....")
    with zipfile.ZipFile(f"{filename}.zip", 'r') as zip_ref:
        zip_ref.extractall(filename)
    os.remove(f"{filename}.zip")
    os.chdir("..")

download_data(url="XXX",
              filename="ristek-datathon-2024",
              dir_name="datathon-2024")

```

## Hasil dan Pembahasan
### EDA
Dataset yang dianalisis memiliki ukuran sangat besar, dengan hampir satu juta observasi dan sejumlah besar fitur. Distribusi label pada data menunjukkan ketidakseimbangan yang ekstrem, di mana sebesar 98,74% transaksi tergolong non-penipuan (non-fraud). Ketimpangan ini menyulitkan proses identifikasi aktivitas penipuan. Selain itu, tidak ditemukan nilai kosong (missing value) dalam dataset. Namun, ditemukan sekitar 400 ribu observasi duplikat—yakni baris dengan fitur identik namun label berbeda—yang menimbulkan dilema dalam pengambilan keputusan apakah data tersebut perlu dipertahankan atau dihapus.

Seluruh fitur numerik telah dinormalisasi, dengan variabel pc0 hingga pc16 memiliki rentang nilai sangat kecil dan median mendekati nol. Visualisasi box plot memperlihatkan kotak yang sangat kecil, menandakan distribusi sempit dan variasi antar nilai yang minimal. Hal ini menyebabkan data bersifat abstrak dan menyulitkan dalam penarikan pola atau karakteristik yang jelas.

Analisis korelasi menunjukkan adanya multikolinearitas tinggi antar fitur, seperti antara pc4 dengan pc9, dan pc8 dengan pc6, yang memiliki korelasi sempurna (nilai korelasi = 1). Korelasi tinggi antar fitur menyebabkan redundansi informasi yang tidak menambah nilai prediktif tetapi justru memperberat proses pelatihan. Namun, karena fitur telah dianonimkan demi menjaga kerahasiaan data, penghapusan fitur tidak dilakukan secara agresif.

### Preprocessing
Tahap prapemrosesan dilakukan dengan memeriksa keberadaan data null dan duplikat. Label 1 (fraud) ditemukan sebanyak 7.109 data duplikat, sedangkan label 0 (non-fraud) sebanyak 303.395 data duplikat. Mengingat ketidakseimbangan distribusi label (data label 0 berjumlah sekitar 847.042, atau 78 kali lebih banyak daripada label 1), diputuskan untuk menghapus data duplikat berlabel 0 sebagai salah satu langkah mengatasi class imbalance.

Selain itu, ditemukan pola fitur tertentu yang secara konsisten muncul pada dataset non-borrower user, seperti pc0 = 0 dan pc1–pc16 = -1 (kecuali pc10 = 0.0) serta pc0 = 1 dengan pola serupa. Karena dataset non-borrower user tidak digunakan dalam pelatihan model, maka data dengan pola identitas tersebut juga dihapus dari keseluruhan dataset.

### Modeling
Model Logistic Regression (LR) dipilih karena sifatnya yang mudah diimplementasikan dan dapat diinterpretasikan dengan baik—karakteristik yang sangat penting dalam konteks deteksi penipuan. Selain itu, LR memungkinkan penerapan regularisasi (L1/L2) untuk mengurangi dampak multikolinearitas, serta menghasilkan probabilitas prediksi yang dapat digunakan untuk pengambilan keputusan berbasis ambang risiko.

Model LR juga dinilai efisien dalam mengklasifikasikan data dengan label tidak diketahui atau ganda. Dalam pengujian, LR memberikan performa yang kompetitif dibandingkan model lain seperti LightGBM, FA-CNN, dan RLS. Skor rata-rata precision yang diperoleh berturut-turut adalah:
- LightGBM: (nilai tidak disebutkan)
- FA-CNN: (nilai tidak disebutkan)
- RLS: 0.6774
- LR: 0.8138
Hasil ini menunjukkan bahwa LR memiliki performa prediktif lebih tinggi berdasarkan average precision (AP) score.

## Competition Link
https://www.kaggle.com/competitions/ristek-datathon-2024
## Citation
Alwin Djuliansah, Anders Willard Leo, Belati Jagad Bintang Syuhada, Darren Aldrich, Ghana Ahmada Yudistira. (2024). RISTEK Datathon 2024. Kaggle. https://kaggle.com/competitions/ristek-datathon-2024

Huang, X., Yang, Y., Wang, Y., Wang, C., Zhang, Z., Xu, J., Chen, L., & Vazirgiannis, M. (2023). DGraph: A Large-Scale Financial Dataset for Graph Anomaly Detection. arXiv. https://arxiv.org/abs/2207.03579
