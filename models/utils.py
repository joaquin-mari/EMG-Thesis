from scipy.signal import butter, filtfilt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, average_precision_score, jaccard_score
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder


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

    # IoU (Jaccard Index)
    iou = jaccard_score(y_true, y_pred, average=average)
    print(f"IoU (Jaccard, {average}): {iou:.4f}")

    if y_score is None:
        print("ROC-AUC: skipped (no y_score provided)")
        print("PR-AUC: skipped (no y_score provided)")
        return

    # ROC-AUC
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

    # Precision-Recall AUC / Average Precision
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


