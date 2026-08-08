"""
Utilitas: preprocessing, metrics, dan stratified cross-validation.
Hanya menggunakan numpy/pandas (pandas untuk I/O tabel, bukan komputasi model).
"""

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Metrics (from scratch)
# ----------------------------------------------------------------------
def confusion_counts(y_true, y_pred, positive_label=1):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = np.sum((y_true == positive_label) & (y_pred == positive_label))
    tn = np.sum((y_true != positive_label) & (y_pred != positive_label))
    fp = np.sum((y_true != positive_label) & (y_pred == positive_label))
    fn = np.sum((y_true == positive_label) & (y_pred != positive_label))
    return tp, tn, fp, fn


def precision_recall_f1(y_true, y_pred, positive_label=1):
    tp, tn, fp, fn = confusion_counts(y_true, y_pred, positive_label)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def macro_f1_score(y_true, y_pred, labels=(0, 1)):
    """Macro F1: rata-rata F1 tiap kelas tanpa memperhitungkan proporsi kelas."""
    f1_scores = []
    for label in labels:
        _, _, f1 = precision_recall_f1(y_true, y_pred, positive_label=label)
        f1_scores.append(f1)
    return float(np.mean(f1_scores))


def accuracy_score(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def classification_report_dict(y_true, y_pred, labels=(0, 1)):
    report = {}
    for label in labels:
        p, r, f1 = precision_recall_f1(y_true, y_pred, positive_label=label)
        report[f"class_{label}"] = {"precision": p, "recall": r, "f1": f1}
    report["accuracy"] = accuracy_score(y_true, y_pred)
    report["macro_f1"] = macro_f1_score(y_true, y_pred, labels)
    return report


# ----------------------------------------------------------------------
# Stratified split & K-Fold (from scratch, tanpa sklearn)
# ----------------------------------------------------------------------
def stratified_train_val_test_split(X, y, val_size=0.15, test_size=0.15, random_state=42):
    """
    Membagi data menjadi train/val/test dengan stratifikasi berdasarkan y,
    supaya proporsi kelas konsisten di ketiga split (penting karena data
    imbalance 78:22).
    """
    rng = np.random.RandomState(random_state)
    y = np.asarray(y)
    n = len(y)

    train_idx, val_idx, test_idx = [], [], []

    for label in np.unique(y):
        idx_label = np.where(y == label)[0]
        rng.shuffle(idx_label)

        n_label = len(idx_label)
        n_test = int(round(n_label * test_size))
        n_val = int(round(n_label * val_size))

        test_idx.extend(idx_label[:n_test])
        val_idx.extend(idx_label[n_test:n_test + n_val])
        train_idx.extend(idx_label[n_test + n_val:])

    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    test_idx = np.array(test_idx)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return train_idx, val_idx, test_idx


def stratified_k_fold(y, n_splits=5, random_state=42, n_repeats=1):
    """
    Generator (train_idx, val_idx) dengan stratifikasi kelas per fold.
    Mendukung repeated stratified k-fold via n_repeats.
    """
    y = np.asarray(y)
    n = len(y)

    for repeat in range(n_repeats):
        rng = np.random.RandomState(random_state + repeat)

        # Buat fold assignment per kelas agar proporsi tiap fold seimbang
        fold_assignment = np.zeros(n, dtype=int)
        for label in np.unique(y):
            idx_label = np.where(y == label)[0]
            rng.shuffle(idx_label)
            folds = np.array_split(idx_label, n_splits)
            for fold_i, fold_indices in enumerate(folds):
                fold_assignment[fold_indices] = fold_i

        for fold_i in range(n_splits):
            val_idx = np.where(fold_assignment == fold_i)[0]
            train_idx = np.where(fold_assignment != fold_i)[0]
            yield train_idx, val_idx


# ----------------------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------------------
def encode_categoricals(df, categorical_cols, mapping=None):
    """
    Label/ordinal encoding sederhana untuk fitur kategorik (cocok untuk tree,
    yang tidak butuh one-hot karena split biner tree sudah menangani ini).
    Jika mapping diberikan (dict kolom -> dict kategori->kode), gunakan itu
    (penting agar encoding test set konsisten dengan train set).
    Kategori yang muncul di test tapi tidak ada di train (tidak seharusnya
    terjadi jika train cukup representatif, tapi dijaga agar tidak error)
    dipetakan ke kode -1.
    Mengembalikan (df_encoded, mapping_terpakai).
    """
    df = df.copy()
    if mapping is None:
        mapping = {}
        for col in categorical_cols:
            categories = sorted(df[col].unique())
            mapping[col] = {cat: i for i, cat in enumerate(categories)}

    for col in categorical_cols:
        df[col] = df[col].map(mapping[col]).fillna(-1).astype(int)

    return df, mapping


def cap_outliers(df, column, lower=None, upper=None):
    """Clip nilai kolom ke batas [lower, upper] jika diberikan (winsorizing sederhana)."""
    df = df.copy()
    if lower is not None:
        df[column] = df[column].clip(lower=lower)
    if upper is not None:
        df[column] = df[column].clip(upper=upper)
    return df


def add_engineered_features(df):
    """
    Menambahkan fitur hasil interaksi/rasio finansial yang relevan
    secara domain untuk kasus loan approval:
      - income_per_exp_year: income dibagi (1 + pengalaman kerja)
      - credit_hist_ratio: rasio panjang riwayat kredit terhadap usia
      - loan_to_income_recompute: loan_amnt / (income + 1), versi re-derivasi
        dari loan_percent_income yang terbukti (via feature importance)
        memberi sinyal tambahan kecil namun konsisten pada tree

    CATATAN PENTING (revisi setelah analisis lebih teliti):
    Sempat diuji fitur `home_ownership_x_lpi_bucket` -- interaction term
    eksplisit antara person_home_ownership dan loan_percent_income (bucket),
    dirancang dari analisis subgrup previous_loan_defaults_on_file == 'No'
    yang menunjukkan pola interaksi tajam (RENT+high-lpi ~100% approval,
    OWN+low-lpi ~13% approval).

    Uji awal pada SATU titik hyperparameter (depth=6) sempat menunjukkan
    kenaikan macro F1. Namun ketika dibandingkan secara adil lewat grid
    search penuh (membandingkan titik optimal masing-masing, bukan
    hyperparameter yang sama dipaksakan ke keduanya), fitur ini TIDAK
    terbukti membantu -- bahkan performa titik optimalnya sedikit lebih
    rendah dari tanpa fitur ini (0.8660 vs 0.8684 macro F1 CV; selisih
    dalam rentang std sehingga tidak signifikan secara statistik).

    Penjelasan: person_home_ownership dan loan_percent_income sudah
    tersedia sebagai fitur individual, dan tree bisa menemukan interaksi
    yang sama lewat kombinasi 2 split berurutan pada kedua fitur asli
    tersebut -- sehingga fitur gabungan eksplisit menjadi redundan,
    sekaligus menambah kardinalitas kategorik (12 kategori) yang justru
    sedikit merugikan kriteria gain_ratio (bias terhadap split information
    yang lebih kecil). Fitur ini TIDAK disertakan pada versi final.
    Didokumentasikan sebagai bagian dari "Percobaan yang Gagal" -- termasuk
    revisi kesimpulan awal yang ternyata terburu-buru karena baru diuji
    pada satu titik hyperparameter, bukan perbandingan yang adil.
    """
    df = df.copy()
    df["income_per_exp_year"] = df["person_income"] / (df["person_emp_exp"] + 1)
    df["credit_hist_ratio"] = df["cb_person_cred_hist_length"] / (df["person_age"] + 1)
    df["loan_to_income_recompute"] = df["loan_amnt"] / (df["person_income"] + 1)
    return df
