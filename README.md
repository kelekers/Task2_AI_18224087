# Task 2 - Seleksi AI

Repositori ini berisi implementasi model klasifikasi berbasis machine learning from scratch serta eksperimen lokal search. Fokus utama proyek ini adalah membangun, mengevaluasi, dan membandingkan model Decision Tree, Logistic Regression, dan SVM untuk tugas klasifikasi loan approval, dengan pendekatan yang dapat dijelaskan secara eksplisit dan diperiksa secara manual.

## Ringkasan proyek

Bagian yang paling relevan untuk tugas ini adalah:

- ML source code: [src/dtl_lr_svm](src/dtl_lr_svm)
- Notebook eksperimen: [notebooks/dtl_lr_svm](notebooks/dtl_lr_svm)
- Data pelatihan dan validasi: [train.csv](train.csv) dan [test.csv](test.csv)
- Dependensi Python: [requirements.txt](requirements.txt)

Struktur utama proyek mengikuti alur berikut:

- Decision Tree: model CART, split kategorik, gain ratio, chi-square, dan cost-complexity pruning
- Logistic Regression: implementasi dari scratch dengan Newton-Raphson / IRLS
- SVM: Linear SVM dan RBF approximation via Random Fourier Features
- Notebook eksplorasi: berisi evaluasi, visualisasi, threshold tuning, serta pembuatan submission

## Struktur folder

```text
.
├── README.md
├── requirements.txt
├── train.csv
├── test.csv
├── src/
│   └── dtl_lr_svm/
│       ├── decision_tree.py
│       ├── dtl.py
│       ├── lr.py
│       ├── svm.py
├── notebooks/
│   └── dtl_lr_svm/
│       ├── dtl.ipynb
│       ├── lr.ipynb
│       ├── svm.ipynb
│       └── utils.py
└── docs/
    └── Task2_AI_18224087.pdf
```

## Persiapan lingkungan

1. Buat environment Python baru bila diperlukan.
2. Install dependency dari file [requirements.txt](requirements.txt).

```bash
pip install -r requirements.txt
```

## Cara menjalankan

### 1. Menjalankan notebook eksperimen
Buka notebook di folder [notebooks/dtl_lr_svm](notebooks/dtl_lr_svm):

- [notebooks/dtl_lr_svm/dtl.ipynb](notebooks/dtl_lr_svm/dtl.ipynb)
- [notebooks/dtl_lr_svm/lr.ipynb](notebooks/dtl_lr_svm/lr.ipynb)
- [notebooks/dtl_lr_svm/svm.ipynb](notebooks/dtl_lr_svm/svm.ipynb)

Pastikan Anda menjalankan notebook dari direktori yang benar agar path menuju data dan source code dapat dikenali. Notebook sudah dibuat agar menambahkan path ke folder source yang relevan.

### 2. Menjalankan modul langsung dari source
Modul yang relevan ada di [src/dtl_lr_svm](src/dtl_lr_svm). Jika ingin dipanggil dari Python langsung, gunakan struktur import yang sesuai, misalnya:

```python
import sys
sys.path.insert(0, "src/dtl_lr_svm")

from dtl import DecisionTreeCART
from lr import LogisticRegressionScratch
from svm import LinearSVMScratch
```

## Model yang diimplementasikan

### Decision Tree
- CART dan variasi lanjutan
- split kategorik berbasis subset
- gain ratio dan chi-square
- pruning berbasis cost complexity
- evaluasi performa dengan macro F1

### Logistic Regression
- implementasi dari scratch
- regularisasi L2
- Newton-Raphson / IRLS
- threshold tuning
- preprocessing numerik dan kategorik

### SVM
- Linear SVM soft-margin
- RBF approximation via random Fourier features
- tuning hyperparameter C dan gamma
- threshold tuning pada decision score

## Data

Dataset utama yang digunakan adalah:

- [train.csv](train.csv)
- [test.csv](test.csv)

Data ini dipakai untuk preprocessing, training, validasi, dan pembuatan submission. Notebook juga menggunakan utility di [notebooks/dtl_lr_svm/utils.py](notebooks/dtl_lr_svm/utils.py) untuk split stratifikasi, encoding kategori, feature engineering, dan metrik seperti macro F1.

## Evaluasi

Metode evaluasi yang digunakan meliputi:

- train/validation/test split yang terstratifikasi
- cross-validation untuk pemilihan hyperparameter
- threshold tuning untuk mengoptimalkan macro F1
- confusion matrix dan classification report

## Catatan penting

- Fokus repositori ini adalah eksperimen machine learning dan implementasi from scratch.
- Folder local search tidak dibahas di README ini karena sudah dipisahkan dan telah dipelihara dalam commit sebelumnya.
- File yang relevan untuk area tugas ini adalah folder ML dan notebook eksperimen, bukan file pendukung yang tidak dibutuhkan.

## Status proyek

Area yang relevan pada repositori ini sudah dicek dan dinilai dalam kondisi aman untuk penggunaan umum dalam konteks eksperimen/modeling:

- import source module sudah valid
- data utama dapat dibaca dengan benar
- notebook utama dapat diakses dan dijalankan dengan environment yang sesuai
- dependency utama telah ditulis pada [requirements.txt](requirements.txt)

## Lisensi dan penggunaan

Proyek ini dibuat untuk kebutuhan tugas dan eksperimen akademik. Penggunaan lebih lanjut dapat disesuaikan dengan kebutuhan kerjasama atau pengembangan lanjutan.