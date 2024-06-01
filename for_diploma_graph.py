import pandas
import os
import json
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score

import pandas as pd

x_phd = []
x_mle = []
y_phd = []
y_mle = []
phd = []
mle = []

folder_path = "/Users/tatanastelmah/PycharmProjects/diploma/dim"
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    if filename == ".DS_Store":
        continue
    if os.path.isfile(file_path):
        with open(file_path, 'r', encoding="utf-8") as file:
            try:
                data = json.load(file)
            except:
                continue
            for elem in data['texts']:
                # text_gen = elem['generated']['text']
                phd_gen = elem['generated']['dim_PHdim']
                if phd_gen < 2 or not phd_gen or phd_gen > 17:
                    continue
                mle_gen = elem['generated']['dim_MLE']
                # text_real = elem['real']['text']
                phd_real = elem['real']['dim_PHdim']
                mle_real = elem['real']['dim_MLE']

                if np.isnan(phd_gen) or np.isnan(not phd_real):
                    continue

                # gen
                x_phd.append(phd_gen)
                y_phd.append(1)
                x_mle.append(mle_gen)
                y_mle.append(1)
                phd.append((phd_gen, 1))
                mle.append((mle_gen, 1))

                # real
                x_phd.append(phd_real)
                y_phd.append(0)
                x_mle.append(mle_real)
                y_mle.append(0)
                phd.append((phd_real, 0))
                mle.append((mle_real, 0))

x_phd = pd.DataFrame(x_phd, columns=['Dim_PHD'])
x_mle = pd.DataFrame(x_mle, columns=['Dim_MLE'])
phd = pd.DataFrame(phd, columns=['Dim_PHD', 'is_gen'])
print(phd.isnull().sum())
mle = pd.DataFrame(mle, columns=['Dim_MLE', 'is_gen'])
phd = phd.sort_values(by='Dim_PHD')
mle = mle.sort_values(by='Dim_MLE')


# корреляция

fig = plt.figure(figsize=(10, 6))
plt.scatter(x_phd['Dim_PHD'], x_mle['Dim_MLE'], color='m', label='Данные')
plt.title('Корреляция между PHdim и MLE')
plt.xlabel('PHdim')
plt.ylabel('MLE')


corr_pearson = x_phd['Dim_PHD'].corr(x_mle['Dim_MLE'])
corr_spearman = x_phd['Dim_PHD'].corr(x_mle['Dim_MLE'], method='spearman')
# Вычисление корреляции с использованием метода Кендалла (pandas)
corr_kendall = x_phd['Dim_PHD'].corr(x_mle['Dim_MLE'], method='kendall')
print(corr_pearson, corr_spearman, corr_kendall)
correlation_text = (f'Корреляция (Пирсон): {corr_pearson:.2f}\n'
                    f'Корреляция (Спирмен): {corr_spearman:.2f}\n'
                    f'Корреляция (Кендалл): {corr_kendall:.2f}')

# Выведем текст на графике в одном квадратике
plt.text(0.05, 0.95, correlation_text,
         transform=plt.gca().transAxes, fontsize=12, fontweight='bold',
         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'),
         verticalalignment='top')


plt.legend()
plt.grid(True)
plt.savefig("correlation.png", dpi=300, bbox_inches='tight')
# plt.show()

# размерность
#
# phd['is_gen'] = phd['is_gen'].replace({0: 'Real text', 1: 'Generated text'})
# mle['is_gen'] = mle['is_gen'].replace({0: 'Real text', 1: 'Generated text'})

fig, axs = plt.subplots(1, 2, figsize=(12, 8))

# Построение boxplot для Phd
boxplot_phd = phd.boxplot(column='Dim_PHD', by='is_gen', grid=False, patch_artist=True, ax=axs[0],
                          medianprops=dict(color='blue'), flierprops=dict(marker='x', color='grey', markersize=6))
colors = ['#98FF98', '#C8A2C8']  # мятный и лиловый
for patch, color in zip(boxplot_phd.findobj(matplotlib.patches.PathPatch), colors):
    patch.set_facecolor(color)
axs[0].set_title('Dimension Phd')
axs[0].set_xlabel('')
axs[0].set_ylabel('Dimension')
axs[0].grid(True)

# Построение boxplot для MLE
boxplot_mle = mle.boxplot(column='Dim_MLE', by='is_gen', grid=False, patch_artist=True, ax=axs[1],
                          medianprops=dict(color='blue'), flierprops=dict(marker='x', color='grey', markersize=6))
for patch, color in zip(boxplot_mle.findobj(matplotlib.patches.PathPatch), colors):
    patch.set_facecolor(color)
axs[1].set_title('Dimension MLE')
axs[1].set_xlabel('')
axs[1].set_ylabel('Dimension')
axs[1].grid(True)

# Общий заголовок
plt.suptitle('Dimension Phd and MLE for Generated and Human Texts', fontsize=14, fontweight='bold')
# plt.legend()
plt.grid(True)
plt.savefig("dimensions.png", dpi=300, bbox_inches='tight')
plt.show()

# тренируем и считаем метрики

train_val, test = train_test_split(phd, test_size=0.1, stratify=phd['is_gen'], random_state=42)
train, val = train_test_split(train_val, test_size=0.25, stratify=train_val['is_gen'], random_state=42)

# Вывод размеров полученных наборов данных
print(f"Train size: {len(train)}")
print(f"Validation size: {len(val)}")
print(f"Test size: {len(test)}")

# Просмотр первых строк каждого набора данных
print("Train set:")
print(train.head())
print("\nValidation set:")
print(val.head())
print("\nTest set:")
print(test.head())

X_train = train[['Dim_PHD']]
print(X_train.isnull().sum())
y_train = train['is_gen']
X_val = val[['Dim_PHD']]
y_val = val['is_gen']
X_test = test[['Dim_PHD']]
y_test = test['is_gen']

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
import math
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

svm_classifier = SVC()
gradient_boosting_classifier = GradientBoostingClassifier()
svm_classifier.fit(X_train, y_train)
gradient_boosting_classifier.fit(X_train, y_train)

svm_predictions = svm_classifier.predict(X_test)
gb_predictions = gradient_boosting_classifier.predict(X_test)

report_svm = classification_report(y_test, svm_predictions)
accuracy_svm = accuracy_score(y_test, svm_predictions)
auc_roc_svm = roc_auc_score(y_test, svm_predictions)

print("Accuracy:", accuracy_svm)
print("Classification SVM Report:\n", report_svm)
validation_accuracy = svm_classifier.score(X_val, y_val)
print("Validation Accuracy:", validation_accuracy)
print("ROC-AUC SVM:", auc_roc_svm)
print("-------------------------------------------------------------")


report_gb = classification_report(y_test, gb_predictions)
accuracy_gb = accuracy_score(y_test, gb_predictions)
auc_roc_gb = roc_auc_score(y_test, gb_predictions)
print("ROC-AUC GB:", auc_roc_gb)
print("-------------------------------------------------------------")

print("Accuracy:", accuracy_gb)
print("Classification GB Report:\n", report_gb)
validation_accuracy = gradient_boosting_classifier.score(X_val, y_val)
print("Validation Accuracy:", validation_accuracy)


def find_optimal_threshold(feature, target):
    thresholds = np.sort(feature)
    best_threshold = thresholds[0]
    best_accuracy = 0

    for threshold in thresholds:
        predictions = feature >= threshold
        accuracy = accuracy_score(target, predictions)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    return best_threshold, best_accuracy


# Найдём пороги для каждого признака
thresholds = []
accuracies = []

threshold, accuracy = find_optimal_threshold(X_train['Dim_PHD'].values, y_train.values)

print(f"Optimal Threshold for Dim_PHD: {threshold}")
print(f"Training Accuracy: {accuracy}")

# Применим порог к тестовым данным
predictions = X_test['Dim_PHD'].values >= threshold
test_accuracy = accuracy_score(y_test, predictions)

print(f"Test Accuracy: {test_accuracy}")
