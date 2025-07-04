# RISTEK Datathon 2024
## Tujuan Kompetisi
Objektif dari kompetisi ini adalah untuk mengembangkan model machine learning untuk deteksi fraud pada pengguna platform fintech. Segala metode machine learning, matematis, dan statistik dapat digunakan untuk meningkatkan performa dari model yang dipakai. Selain mengembangkan model klasifikasi, kemampuan analisis mengenai pola pengguna yang teridentifikasi sebagai fraud juga diperlukan untuk menjelaskan cara kerja model yang dipakai.
## Apa itu fraud detection?
Fraud detection adalah proses identifikasi tindakan pengguna pada suatu skenario termasuk sebagai tindakan penipuan atau bukan. Dalam konteks kompetisi ini, tindakan penipuan didefinisikan sebagai pengguna platform yang telah meminjam produk keuangan tetapi tercatat belum melakukan pembayaran sampai tenggat waktu yang telah ditentukan.

---

## Evaluation
### Penilaian Keseluruhan

![image](https://github.com/user-attachments/assets/3cabc5fa-b0d9-4766-b2cc-bd3f8153de6d)

### Detail Kriteria
- Skor Private Leaderboard: skor yang muncul setelah tahap kompetisi berakhir, yaitu 50% dari data test berdasarkan kriteria yang sudah ditentukan panitia.
- Analisis: pemahaman format data, interpretasi data berdasarkan pemahaman peserta, dan temuan-temuan penting yang dapat membantu pemrosesan data.
- Pemrosesan Data: transformasi dari data mentah menjadi bentuk data yang dapat diterima oleh model. Termasuk proses rekayasa fitur.
- Modeling: perancangan arsitektur model beserta intuisi arsitektur tersebut, evaluasi dari model yang dibentuk, dan analisis hasil prediksi model.
- Sistematika Notebook: mencakup bagian pendahuluan, kesimpulan, dan tata tulis serta kelengkapan isi notebook.
Bagian analisis, pemrosesan data, dan modeling wajib disertai dengan penjelasan yang lengkap dan detail, tidak berupa kode saja karena penjelasan tersebut akan masuk ke dalam penilaian.

### Metrik Leaderboard
Performa model dievaluasi menggunakan metrik Average Precision dengan average='macro'. Secara formal, metrik Average Precision dirumuskan dalam bentuk:

![image](https://github.com/user-attachments/assets/c4277c63-987c-4fd3-ae65-7a8ab0e08b46)

Implementasi metrik tersebut dalam Python dengan menggunakan library Scikit-Learn adalah sebagai berikut.

```
from sklearn.metrics import average_precision_score

score = average_precision_score(y_true, y_pred)

```
Penggunaan metrik Average Precision didasarkan pada penekanan fokus untuk merefleksikan kemampuan model dalam mendeteksi pengguna dengan label fraud sebagai fraud, bukan sebagai pengguna biasa (non-fraud).

---

## Dataset
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

---

## Competition Link
https://www.kaggle.com/competitions/ristek-datathon-2024
## Citation
Alwin Djuliansah, Anders Willard Leo, Belati Jagad Bintang Syuhada, Darren Aldrich, Ghana Ahmada Yudistira. (2024). RISTEK Datathon 2024. Kaggle. https://kaggle.com/competitions/ristek-datathon-2024

Huang, X., Yang, Y., Wang, Y., Wang, C., Zhang, Z., Xu, J., Chen, L., & Vazirgiannis, M. (2023). DGraph: A Large-Scale Financial Dataset for Graph Anomaly Detection. arXiv. https://arxiv.org/abs/2207.03579
