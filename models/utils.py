from scipy.signal import butter, filtfilt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, average_precision_score, jaccard_score
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split


def perform_patient_split(
    df,
    patient_id="RN181281",
    file_col="file",
    label_col="Label"
):
    files = df[file_col].astype(str).str.replace("\\", "/", regex=False)

    test_mask = files.str.contains(patient_id)
    train_mask = ~test_mask

    if test_mask.sum() == 0:
        raise ValueError(f"No samples found for patient_id={patient_id}")

    y_train = df.loc[train_mask, label_col].values
    y_test  = df.loc[test_mask, label_col].values

    return train_mask.values, test_mask.values, y_train, y_test


def perform_random_split(
    df,
    label_col="Label",
    test_size=0.2,
    random_state=42,
    stratify=True
):
    indices = df.index.values
    labels = df[label_col].values

    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=labels if stratify else None
    )

    train_mask = df.index.isin(train_idx)
    test_mask = df.index.isin(test_idx)

    y_train = df.loc[train_mask, label_col].values
    y_test = df.loc[test_mask, label_col].values

    return train_mask, test_mask, y_train, y_test


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(signal_uV, fs):
    b, a = butter_bandpass(20, 500, fs, order=4)
    return filtfilt(b, a, signal_uV)

def encode_labels(y_train, y_test):
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)  
    labels = le.classes_
    return y_train, y_test, labels

def get_labels(df):
    labels = {}
    for index, row in df.iterrows():
        if str(row["Label"]).lower() in ["fibrillation", "psw"]:
            key = row["file"] 
            item = {"label": row["Label"], "start": row["Start"], "end": row["End"]}

            if key not in labels:
                labels[key] = []

            labels[key].append(item)
    return labels

def get_monitor():
    monitor = EarlyStopping(monitor='val_loss',
                        mode='min', 
                        restore_best_weights=True,
                        patience=3)
    return monitor

def get_class_weights(y_train):
    cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(cw))
    return class_weights


def get_optimizer():
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    return optimizer

def plot_training_history(history):
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(history.history['accuracy'], label='train acc')
    plt.plot(history.history['val_accuracy'], label='val acc')
    plt.legend()

    plt.show()

def plot_confusion_matrix(y_true, y_pred, labels):
    conf = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf, display_labels=labels)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=labels, zero_division=0))

def get_extended_report(
    y_true,
    y_pred,
    y_score=None,      
    average="macro"      
):
    print("\nAdditional Metrics:")

    iou = jaccard_score(y_true, y_pred, average=average)
    print(f"IoU (Jaccard, {average}): {iou:.4f}")

    if y_score is None:
        print("ROC-AUC: skipped (no y_score provided)")
        print("PR-AUC: skipped (no y_score provided)")
        return

    try:
        roc_auc = roc_auc_score(
            y_true,
            y_score,
            multi_class="ovr",
            average=average
        )
        print(f"ROC-AUC ({average}): {roc_auc:.4f}")
    except ValueError:
        print("ROC-AUC: not available (check y_score shape)")

    try:
        pr_auc = average_precision_score(
            y_true,
            y_score,
            average=average
        )
        print(f"PR-AUC / Avg Precision ({average}): {pr_auc:.4f}")
    except ValueError:
        print("PR-AUC: not available (check y_score shape)")

def find_optimal_threshold(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    return thresholds[np.argmax(f1_scores)]

def to_binary(y_test, X_test, model):
    if len(y_test.shape) > 1:  
        y_true_classes = np.argmax(y_test, axis=1)
    else:
        y_true_classes = y_test

    y_pred = model.predict(X_test)

    if len(np.array(y_pred).shape) > 1:
        y_pred_classes = np.argmax(y_pred, axis=1)
    else:
        y_pred_classes = y_pred

    y_true_grouped = np.where(y_true_classes == 1, 0, 1)
    y_pred_grouped = np.where(y_pred_classes == 1, 0, 1)

    grouped_labels = ['Normal', 'Spontaneous activity']

    plot_confusion_matrix(y_true_grouped, y_pred_grouped, grouped_labels)

import numpy as np
from sklearn.metrics import fbeta_score, f1_score

def best_threshold_binary(y_true_bin, y_score, metric="f2", n_steps=1000, beta=2.0):
    thresholds = np.linspace(0.0, 1.0, n_steps + 1)
    best_t, best_s = 0.5, -1.0

    for t in thresholds:
        y_pred_bin = (y_score >= t).astype(np.int32)

        if metric == "f1":
            s = f1_score(y_true_bin, y_pred_bin, zero_division=0)
        elif metric == "f2":
            s = fbeta_score(y_true_bin, y_pred_bin, beta=beta, zero_division=0)
        else:
            raise ValueError("metric must be 'f1' or 'f2'")

        if s > best_s:
            best_s, best_t = s, t

    return float(best_t), float(best_s)


def fit_thresholds_one_vs_rest(y_val, y_prob_val, metric="f2", n_steps=1000):
    y_flat = y_val.reshape(-1)

    thresholds = {}
    for c in [1, 2]:
        y_true_bin = (y_flat == c).astype(np.int32)
        y_score = y_prob_val[..., c].reshape(-1)

        beta = 3.0 if c == 1 else 2.0 
        t, s = best_threshold_binary(y_true_bin, y_score, metric=metric, n_steps=n_steps, beta=beta)

        thresholds[c] = {"t": t, "best_score": s, "beta": beta}
    return thresholds

def predict_with_two_thresholds(y_prob, t_fib, t_psw):
    p1 = y_prob[..., 1]
    p2 = y_prob[..., 2]

    fib_ok = (p1 >= t_fib)
    psw_ok = (p2 >= t_psw)

    y_pred = np.zeros(p1.shape, dtype=np.int32)

    y_pred[fib_ok & ~psw_ok] = 1
    y_pred[psw_ok & ~fib_ok] = 2

    both = fib_ok & psw_ok
    y_pred[both] = np.where(p1[both] >= p2[both], 1, 2)

    return y_pred
