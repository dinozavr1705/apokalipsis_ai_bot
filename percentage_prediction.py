import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    classification_report, roc_curve, precision_recall_curve, auc
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
import warnings

warnings.filterwarnings('ignore')

np.random.seed(42)


def generate_data(n_samples=20000):
    print(f"Генерация {n_samples} записей...")

    age = np.clip(np.random.normal(45, 20, n_samples), 0, 100).astype(int)
    physical = np.random.normal(7, 2, n_samples)
    physical = np.clip(physical, 1, 10).astype(int)
    iq = np.clip(np.random.normal(105, 20, n_samples), 40, 160).astype(int)
    vision = np.clip(100 - (age / 3) + np.random.normal(0, 15, n_samples), 0, 100).astype(int)
    parents = np.random.choice([0, 1, 2], n_samples, p=[0.1, 0.2, 0.7])
    movies = np.random.poisson(8, n_samples) + np.random.randint(0, 15, n_samples)
    movies = np.clip(movies, 0, 50).astype(int)
    autism = np.random.choice([0, 1], n_samples, p=[0.92, 0.08])

    age_factor = np.exp(-age / 40)
    physical_factor = physical / 10
    iq_factor = np.where(iq > 115, 0.9, np.where(iq > 100, 0.7, np.where(iq > 85, 0.5, 0.3)))
    vision_factor = vision / 100
    social_factor = parents / 2
    prep_factor = np.log1p(movies) / np.log1p(50)
    autism_penalty = np.where(autism == 1, -0.25, 0.0)

    young_strong = ((100 - age) / 100) * (physical / 10) * 0.3
    smart_prepared = (iq > 110).astype(float) * np.minimum(movies / 15, 1.0) * 0.25
    old_alone = (age > 70).astype(float) * (parents == 0).astype(float) * -0.35
    blind_old = (age > 60).astype(float) * ((100 - vision) / 100) * -0.2

    base_score = (
            age_factor * 0.25 +
            physical_factor * 0.20 +
            iq_factor * 0.15 +
            vision_factor * 0.10 +
            social_factor * 0.10 +
            prep_factor * 0.08 +
            autism_penalty
    )

    total_score = base_score + young_strong + smart_prepared + old_alone + blind_old
    survival_prob = 1 / (1 + np.exp(-12 * (total_score - 0.5)))

    uncertainty = np.abs(survival_prob - 0.5) / 0.5
    noise = np.random.normal(0, 0.08, n_samples) * uncertainty
    survival_prob += noise
    survival_prob = np.clip(survival_prob, 0.01, 0.99)

    survived = (survival_prob > 0.5).astype(int)

    df = pd.DataFrame({
        'age': age,
        'physical_ability': physical,
        'autism': autism,
        'parents_count': parents,
        'iq': iq,
        'vision': vision,
        'apocalypse_movies': movies,
        'survival_probability': survival_prob,
        'survived': survived
    })

    n_survived = df['survived'].sum()
    n_died = len(df) - n_survived
    n_target = min(n_survived, n_died)

    survived_indices = df[df['survived'] == 1].index[:n_target]
    died_indices = df[df['survived'] == 0].index[:n_target]
    balanced_indices = np.concatenate([survived_indices, died_indices])
    df = df.loc[balanced_indices].sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Данные: {len(df)} записей")
    print(f"Баланс: {df['survived'].mean():.2%} выживших")

    return df


class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes=[64, 32], output_size=1, dropout_rate=0.2):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.training = True

        self.weights = []
        self.biases = []

        layer_sizes = [input_size] + hidden_sizes + [output_size]

        for i in range(len(layer_sizes) - 1):
            limit = np.sqrt(2.0 / layer_sizes[i])
            W = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * limit
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(W)
            self.biases.append(b)

        self.m_w = [np.zeros_like(w) for w in self.weights]
        self.v_w = [np.zeros_like(w) for w in self.weights]
        self.m_b = [np.zeros_like(b) for b in self.biases]
        self.v_b = [np.zeros_like(b) for b in self.biases]
        self.t = 0

        # Для анализа
        self.gradient_norms = []
        self.weight_norms = []
        self.activation_stats = []

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def sigmoid(self, x):
        return np.where(x >= 0,
                        1 / (1 + np.exp(-x)),
                        np.exp(x) / (1 + np.exp(x)))

    def dropout(self, x, rate):
        if not self.training or rate == 0:
            return x
        mask = np.random.binomial(1, 1 - rate, size=x.shape) / (1 - rate)
        return x * mask

    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        self.dropout_masks = []
        self.layer_outputs = []

        for i in range(len(self.hidden_sizes)):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            a = self.relu(z)

            if i < len(self.hidden_sizes) - 1:
                a = self.dropout(a, self.dropout_rate)
                self.dropout_masks.append(a > 0 if self.training else None)

            self.activations.append(a)
            self.layer_outputs.append(a)

        z_out = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        self.z_values.append(z_out)
        output = self.sigmoid(z_out)
        self.activations.append(output)
        self.layer_outputs.append(output)

        return output

    def backward(self, X, y, output, learning_rate):
        m = X.shape[0]
        dZ_out = output - y.reshape(-1, 1)

        gradients_w = []
        gradients_b = []

        # Сохраняем нормы градиентов для анализа
        grad_norms = []

        dW_out = np.dot(self.activations[-2].T, dZ_out) / m
        db_out = np.sum(dZ_out, axis=0, keepdims=True) / m
        gradients_w.append(dW_out)
        gradients_b.append(db_out)
        grad_norms.append(np.linalg.norm(dW_out))

        dA = np.dot(dZ_out, self.weights[-1].T)

        for i in range(len(self.hidden_sizes) - 1, -1, -1):
            dZ = dA * self.relu_derivative(self.z_values[i])
            dW = np.dot(self.activations[i].T, dZ) / m
            db = np.sum(dZ, axis=0, keepdims=True) / m

            gradients_w.insert(0, dW)
            gradients_b.insert(0, db)
            grad_norms.insert(0, np.linalg.norm(dW))

            if i > 0:
                dA = np.dot(dZ, self.weights[i].T)

        self.gradient_norms.append(grad_norms)

        self.t += 1
        beta1, beta2 = 0.9, 0.999
        epsilon = 1e-8

        for i in range(len(self.weights)):
            self.m_w[i] = beta1 * self.m_w[i] + (1 - beta1) * gradients_w[i]
            self.v_w[i] = beta2 * self.v_w[i] + (1 - beta2) * (gradients_w[i] ** 2)
            m_w_hat = self.m_w[i] / (1 - beta1 ** self.t)
            v_w_hat = self.v_w[i] / (1 - beta2 ** self.t)

            self.m_b[i] = beta1 * self.m_b[i] + (1 - beta1) * gradients_b[i]
            self.v_b[i] = beta2 * self.v_b[i] + (1 - beta2) * (gradients_b[i] ** 2)
            m_b_hat = self.m_b[i] / (1 - beta1 ** self.t)
            v_b_hat = self.v_b[i] / (1 - beta2 ** self.t)

            self.weights[i] -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
            self.biases[i] -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + epsilon)

        # Сохраняем нормы весов
        weight_norms = [np.linalg.norm(w) for w in self.weights]
        self.weight_norms.append(weight_norms)

    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        cross_entropy = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return cross_entropy

    def train(self, X, y, X_val, y_val, epochs=300, learning_rate=0.001, batch_size=32):
        self.training = True
        n_samples = X.shape[0]
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        # Новые метрики для анализа
        train_precisions = []
        train_recalls = []
        train_f1_scores = []
        val_precisions = []
        val_recalls = []
        val_f1_scores = []

        best_val_acc = 0
        best_weights = None
        best_biases = None
        patience_counter = 0
        patience_limit = 20

        l2_lambda = 0.001
        start_time = time.time()

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0
            correct = 0
            n_batches = 0

            # Для расчета precision, recall, f1
            all_preds = []
            all_labels = []

            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                output = self.forward(X_batch)
                batch_loss = self.compute_loss(y_batch, output)

                l2_penalty = 0
                for w in self.weights:
                    l2_penalty += np.sum(w ** 2)
                batch_loss += l2_lambda * l2_penalty / (2 * batch_size)

                epoch_loss += batch_loss * len(X_batch)
                n_batches += 1

                preds = (output > 0.5).astype(int)
                correct += np.sum(preds.flatten() == y_batch)
                all_preds.extend(preds.flatten())
                all_labels.extend(y_batch)

                current_lr = learning_rate
                if epoch > 100:
                    current_lr = learning_rate * 0.5
                if epoch > 200:
                    current_lr = learning_rate * 0.1

                self.backward(X_batch, y_batch, output, current_lr)

            avg_loss = epoch_loss / n_samples
            train_acc = correct / n_samples

            # Расчет дополнительных метрик
            from sklearn.metrics import precision_score, recall_score, f1_score
            train_precision = precision_score(all_labels, all_preds, zero_division=0)
            train_recall = recall_score(all_labels, all_preds, zero_division=0)
            train_f1 = f1_score(all_labels, all_preds, zero_division=0)

            train_losses.append(avg_loss)
            train_accuracies.append(train_acc)
            train_precisions.append(train_precision)
            train_recalls.append(train_recall)
            train_f1_scores.append(train_f1)

            self.training = False
            val_output = self.forward(X_val)
            self.training = True

            val_loss = self.compute_loss(y_val, val_output)
            val_preds = (val_output > 0.5).astype(int)
            val_acc = np.mean(val_preds.flatten() == y_val)

            val_precision = precision_score(y_val, val_preds.flatten(), zero_division=0)
            val_recall = recall_score(y_val, val_preds.flatten(), zero_division=0)
            val_f1 = f1_score(y_val, val_preds.flatten(), zero_division=0)

            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            val_precisions.append(val_precision)
            val_recalls.append(val_recall)
            val_f1_scores.append(val_f1)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_weights = [w.copy() for w in self.weights]
                best_biases = [b.copy() for b in self.biases]
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience_limit:
                print(f"\nРанняя остановка на эпохе {epoch + 1}: нет улучшений в течение {patience_limit} эпох")
                break

            if (epoch + 1) % 30 == 0 or epoch == 0 or epoch == epochs - 1:
                elapsed = time.time() - start_time
                progress = (epoch + 1) / epochs * 100
                print(f"Эпоха {epoch + 1:3d}/{epochs} ({progress:5.1f}%) | "
                      f"Loss: {avg_loss:.4f}→{val_loss:.4f} | "
                      f"Acc: {train_acc:.4f}→{val_acc:.4f} | "
                      f"F1: {train_f1:.4f}→{val_f1:.4f} | "
                      f"Лучшая val: {best_val_acc:.4f}")

        if best_weights is not None:
            self.weights = best_weights
            self.biases = best_biases

        self.training = False

        total_time = time.time() - start_time
        print(f"\nОбучение завершено за {total_time / 60:.1f} минут")
        print(f"Лучшая валидационная точность: {best_val_acc:.4f}")
        print(f"Финальный train accuracy: {train_accuracies[-1]:.4f}")
        print(f"Финальный val accuracy: {val_accuracies[-1]:.4f}")
        print(f"Loss уменьшился: {train_losses[0]:.4f} → {train_losses[-1]:.4f}")

        gap = train_accuracies[-1] - val_accuracies[-1]
        if gap > 0.1:
            print(f"⚠️  Большой разрыв train-val: {gap:.4f} (возможно переобучение)")
        elif gap > 0.05:
            print(f"⚠️  Умеренный разрыв train-val: {gap:.4f}")
        else:
            print(f"✅  Хороший разрыв train-val: {gap:.4f}")

        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
            'train_precisions': train_precisions,
            'train_recalls': train_recalls,
            'train_f1_scores': train_f1_scores,
            'val_precisions': val_precisions,
            'val_recalls': val_recalls,
            'val_f1_scores': val_f1_scores,
            'gradient_norms': self.gradient_norms,
            'weight_norms': self.weight_norms
        }

    def predict_proba(self, X):
        self.training = False
        return self.forward(X)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) > threshold).astype(int)

    def evaluate(self, X, y):
        y_pred_proba = self.predict_proba(X).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)

        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1_score': f1_score(y, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y, y_pred_proba),
            'confusion_matrix': confusion_matrix(y, y_pred),
            'y_pred_proba': y_pred_proba,
            'y_true': y
        }


def plot_advanced_visualizations(df, nn, results, train_results, X_test_scaled, y_test):
    """Расширенные визуализации для анализа модели"""

    # Фигура 1: Основные метрики обучения
    fig1, axes1 = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Loss
    axes1[0, 0].plot(train_results['train_losses'], label='Train', linewidth=2, alpha=0.8)
    axes1[0, 0].plot(train_results['val_losses'], label='Val', linewidth=2, alpha=0.8)
    axes1[0, 0].set_xlabel('Эпоха')
    axes1[0, 0].set_ylabel('Loss')
    axes1[0, 0].set_title('Loss во время обучения')
    axes1[0, 0].legend()
    axes1[0, 0].grid(True, alpha=0.3)

    # 2. Accuracy
    axes1[0, 1].plot(train_results['train_accuracies'], label='Train', linewidth=2, alpha=0.8)
    axes1[0, 1].plot(train_results['val_accuracies'], label='Val', linewidth=2, alpha=0.8)
    axes1[0, 1].axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='Цель 0.9')
    axes1[0, 1].set_xlabel('Эпоха')
    axes1[0, 1].set_ylabel('Accuracy')
    axes1[0, 1].set_title('Accuracy во время обучения')
    axes1[0, 1].legend()
    axes1[0, 1].grid(True, alpha=0.3)

    # 3. F1 Score
    axes1[0, 2].plot(train_results['train_f1_scores'], label='Train F1', linewidth=2, alpha=0.8)
    axes1[0, 2].plot(train_results['val_f1_scores'], label='Val F1', linewidth=2, alpha=0.8)
    axes1[0, 2].set_xlabel('Эпоха')
    axes1[0, 2].set_ylabel('F1 Score')
    axes1[0, 2].set_title('F1 Score во время обучения')
    axes1[0, 2].legend()
    axes1[0, 2].grid(True, alpha=0.3)

    # 4. Confusion Matrix
    cm = results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Не выжил', 'Выжил'],
                yticklabels=['Не выжил', 'Выжил'],
                ax=axes1[1, 0])
    axes1[1, 0].set_title(f'Confusion Matrix (Accuracy: {results["accuracy"]:.2%})')
    axes1[1, 0].set_ylabel('Истинный класс')
    axes1[1, 0].set_xlabel('Предсказанный класс')

    # 5. Распределение вероятностей
    y_test_proba = results['y_pred_proba']
    axes1[1, 1].hist(y_test_proba, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes1[1, 1].axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Порог 0.5')
    axes1[1, 1].set_xlabel('Вероятность выживания')
    axes1[1, 1].set_ylabel('Количество примеров')
    axes1[1, 1].set_title('Распределение вероятностей предсказаний')
    axes1[1, 1].legend()
    axes1[1, 1].grid(True, alpha=0.3)

    # 6. Precision-Recall
    axes1[1, 2].plot(train_results['train_recalls'], train_results['train_precisions'],
                     label='Train', linewidth=2, alpha=0.7)
    axes1[1, 2].plot(train_results['val_recalls'], train_results['val_precisions'],
                     label='Val', linewidth=2, alpha=0.7)
    axes1[1, 2].set_xlabel('Recall')
    axes1[1, 2].set_ylabel('Precision')
    axes1[1, 2].set_title('Precision-Recall во время обучения')
    axes1[1, 2].legend()
    axes1[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_metrics_comprehensive.png', dpi=100, bbox_inches='tight')
    plt.show()

    # Фигура 2: ROC и Precision-Recall кривые
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))

    # ROC кривая
    fpr, tpr, thresholds = roc_curve(results['y_true'], results['y_pred_proba'])
    roc_auc = auc(fpr, tpr)

    axes2[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC кривая (AUC = {roc_auc:.4f})')
    axes2[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes2[0].set_xlim([0.0, 1.0])
    axes2[0].set_ylim([0.0, 1.05])
    axes2[0].set_xlabel('False Positive Rate')
    axes2[0].set_ylabel('True Positive Rate')
    axes2[0].set_title('ROC кривая')
    axes2[0].legend(loc="lower right")
    axes2[0].grid(True, alpha=0.3)

    # Precision-Recall кривая
    precision, recall, thresholds = precision_recall_curve(results['y_true'], results['y_pred_proba'])
    pr_auc = auc(recall, precision)

    axes2[1].plot(recall, precision, color='green', lw=2, label=f'PR кривая (AUC = {pr_auc:.4f})')
    axes2[1].set_xlim([0.0, 1.0])
    axes2[1].set_ylim([0.0, 1.05])
    axes2[1].set_xlabel('Recall')
    axes2[1].set_ylabel('Precision')
    axes2[1].set_title('Precision-Recall кривая')
    axes2[1].legend(loc="lower left")
    axes2[1].grid(True, alpha=0.3)

    # Распределение ошибок по вероятностям
    errors = np.abs(results['y_pred_proba'] - results['y_true'])
    axes2[2].scatter(results['y_pred_proba'], errors, alpha=0.5, s=10)
    axes2[2].set_xlabel('Предсказанная вероятность')
    axes2[2].set_ylabel('Абсолютная ошибка')
    axes2[2].set_title('Распределение ошибок')
    axes2[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('roc_pr_curves.png', dpi=100, bbox_inches='tight')
    plt.show()

    # Фигура 3: Анализ градиентов и весов
    if hasattr(nn, 'gradient_norms') and nn.gradient_norms:
        fig3, axes3 = plt.subplots(2, 2, figsize=(15, 12))

        # Нормы градиентов по слоям
        gradient_norms = np.array(nn.gradient_norms)
        for i in range(gradient_norms.shape[1]):
            axes3[0, 0].plot(gradient_norms[:, i], label=f'Слой {i + 1}', alpha=0.7)
        axes3[0, 0].set_xlabel('Эпоха (мини-батчи)')
        axes3[0, 0].set_ylabel('Норма градиента')
        axes3[0, 0].set_title('Нормы градиентов по слоям')
        axes3[0, 0].legend()
        axes3[0, 0].grid(True, alpha=0.3)

        # Нормы весов по слоям
        weight_norms = np.array(nn.weight_norms)
        for i in range(weight_norms.shape[1]):
            axes3[0, 1].plot(weight_norms[:, i], label=f'Слой {i + 1}', alpha=0.7)
        axes3[0, 1].set_xlabel('Эпоха')
        axes3[0, 1].set_ylabel('Норма весов')
        axes3[0, 1].set_title('Нормы весов по слоям')
        axes3[0, 1].legend()
        axes3[0, 1].grid(True, alpha=0.3)

        # Распределение весов в первом слое
        if len(nn.weights) > 0:
            weights_first = nn.weights[0].flatten()
            axes3[1, 0].hist(weights_first, bins=50, alpha=0.7, color='purple', edgecolor='black')
            axes3[1, 0].set_xlabel('Значение веса')
            axes3[1, 0].set_ylabel('Частота')
            axes3[1, 0].set_title('Распределение весов в первом слое')
            axes3[1, 0].grid(True, alpha=0.3)

            # Важность признаков (на основе абсолютных значений весов)
            feature_importance = np.abs(nn.weights[0]).mean(axis=1)
            features = ['age', 'physical', 'autism', 'parents', 'iq', 'vision', 'movies']
            axes3[1, 1].barh(features, feature_importance, color='teal')
            axes3[1, 1].set_xlabel('Среднее абсолютное значение веса')
            axes3[1, 1].set_title('Важность признаков (первый слой)')
            axes3[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('gradients_weights_analysis.png', dpi=100, bbox_inches='tight')
        plt.show()

    # Фигура 4: Анализ данных
    fig4, axes4 = plt.subplots(3, 3, figsize=(18, 15))

    # Распределение возраста
    axes4[0, 0].hist(df[df['survived'] == 0]['age'], alpha=0.5, label='Не выжил', bins=20)
    axes4[0, 0].hist(df[df['survived'] == 1]['age'], alpha=0.5, label='Выжил', bins=20)
    axes4[0, 0].set_xlabel('Возраст')
    axes4[0, 0].set_ylabel('Частота')
    axes4[0, 0].set_title('Распределение возраста')
    axes4[0, 0].legend()
    axes4[0, 0].grid(True, alpha=0.3)

    # Распределение физических способностей
    axes4[0, 1].hist(df[df['survived'] == 0]['physical_ability'], alpha=0.5, label='Не выжил', bins=10)
    axes4[0, 1].hist(df[df['survived'] == 1]['physical_ability'], alpha=0.5, label='Выжил', bins=10)
    axes4[0, 1].set_xlabel('Физические способности')
    axes4[0, 1].set_ylabel('Частота')
    axes4[0, 1].set_title('Распределение физических способностей')
    axes4[0, 1].legend()
    axes4[0, 1].grid(True, alpha=0.3)

    # Распределение IQ
    axes4[0, 2].hist(df[df['survived'] == 0]['iq'], alpha=0.5, label='Не выжил', bins=20)
    axes4[0, 2].hist(df[df['survived'] == 1]['iq'], alpha=0.5, label='Выжил', bins=20)
    axes4[0, 2].set_xlabel('IQ')
    axes4[0, 2].set_ylabel('Частота')
    axes4[0, 2].set_title('Распределение IQ')
    axes4[0, 2].legend()
    axes4[0, 2].grid(True, alpha=0.3)

    # Корреляция признаков
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, ax=axes4[1, 0])
    axes4[1, 0].set_title('Матрица корреляций признаков')

    # Зависимость выживания от возраста и физической формы
    scatter = axes4[1, 1].scatter(df['age'], df['physical_ability'],
                                  c=df['survived'], cmap='RdYlGn', alpha=0.6)
    axes4[1, 1].set_xlabel('Возраст')
    axes4[1, 1].set_ylabel('Физические способности')
    axes4[1, 1].set_title('Выживание: возраст vs физическая форма')
    axes4[1, 1].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes4[1, 1], label='Выжил (1) / Не выжил (0)')

    # Распределение фильмов
    axes4[1, 2].hist(df[df['survived'] == 0]['apocalypse_movies'], alpha=0.5, label='Не выжил', bins=15)
    axes4[1, 2].hist(df[df['survived'] == 1]['apocalypse_movies'], alpha=0.5, label='Выжил', bins=15)
    axes4[1, 2].set_xlabel('Количество фильмов')
    axes4[1, 2].set_ylabel('Частота')
    axes4[1, 2].set_title('Распределение просмотренных фильмов')
    axes4[1, 2].legend()
    axes4[1, 2].grid(True, alpha=0.3)

    # Визуализация зрение vs IQ
    scatter = axes4[2, 0].scatter(df['vision'], df['iq'],
                                  c=df['survived'], cmap='RdYlGn', alpha=0.6)
    axes4[2, 0].set_xlabel('Зрение (%)')
    axes4[2, 0].set_ylabel('IQ')
    axes4[2, 0].set_title('Выживание: зрение vs IQ')
    axes4[2, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes4[2, 0], label='Выжил (1) / Не выжил (0)')

    # Распределение выживания по родителям
    parent_survival = df.groupby('parents_count')['survived'].mean()
    axes4[2, 1].bar(parent_survival.index, parent_survival.values, color='orange')
    axes4[2, 1].set_xlabel('Количество живых родителей')
    axes4[2, 1].set_ylabel('Доля выживших')
    axes4[2, 1].set_title('Выживаемость по количеству родителей')
    axes4[2, 1].grid(True, alpha=0.3)

    # Распределение аутизма
    autism_survival = df.groupby('autism')['survived'].mean()
    axes4[2, 2].bar(autism_survival.index, autism_survival.values, color='purple')
    axes4[2, 2].set_xlabel('Аутизм (0=нет, 1=да)')
    axes4[2, 2].set_ylabel('Доля выживших')
    axes4[2, 2].set_title('Выживаемость при аутизме')
    axes4[2, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('data_analysis_comprehensive.png', dpi=100, bbox_inches='tight')
    plt.show()

    # Фигура 5: Детальный анализ предсказаний
    fig5, axes5 = plt.subplots(2, 2, figsize=(15, 12))

    # Калибровочная кривая
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(results['y_true'], results['y_pred_proba'], n_bins=10)

    axes5[0, 0].plot([0, 1], [0, 1], "k:", label="Идеально откалибровано")
    axes5[0, 0].plot(prob_pred, prob_true, "s-", label="Наша модель")
    axes5[0, 0].set_xlabel('Средняя предсказанная вероятность')
    axes5[0, 0].set_ylabel('Доля положительных')
    axes5[0, 0].set_title('Калибровочная кривая')
    axes5[0, 0].legend()
    axes5[0, 0].grid(True, alpha=0.3)

    # Анализ порогов
    thresholds = np.linspace(0.1, 0.9, 9)
    accuracies = []
    f1_scores = []

    for threshold in thresholds:
        y_pred_threshold = (results['y_pred_proba'] > threshold).astype(int)
        accuracies.append(accuracy_score(results['y_true'], y_pred_threshold))
        f1_scores.append(f1_score(results['y_true'], y_pred_threshold, zero_division=0))

    axes5[0, 1].plot(thresholds, accuracies, 'o-', label='Accuracy')
    axes5[0, 1].plot(thresholds, f1_scores, 's-', label='F1 Score')
    axes5[0, 1].axvline(x=0.5, color='r', linestyle='--', alpha=0.5, label='Порог 0.5')
    axes5[0, 1].set_xlabel('Порог классификации')
    axes5[0, 1].set_ylabel('Метрика')
    axes5[0, 1].set_title('Зависимость метрик от порога')
    axes5[0, 1].legend()
    axes5[0, 1].grid(True, alpha=0.3)

    # Распределение ошибок по классам
    y_pred = (results['y_pred_proba'] > 0.5).astype(int)
    errors = y_pred != results['y_true']

    axes5[1, 0].hist(results['y_pred_proba'][errors], bins=20, alpha=0.7, color='red',
                     label='Ошибки', edgecolor='black')
    axes5[1, 0].hist(results['y_pred_proba'][~errors], bins=20, alpha=0.7, color='green',
                     label='Правильные', edgecolor='black')
    axes5[1, 0].set_xlabel('Предсказанная вероятность')
    axes5[1, 0].set_ylabel('Количество')
    axes5[1, 0].set_title('Распределение вероятностей для ошибок и правильных предсказаний')
    axes5[1, 0].legend()
    axes5[1, 0].grid(True, alpha=0.3)

    # Матрица ошибок в процентах
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    sns.heatmap(cm_percentage, annot=True, fmt='.1f', cmap='Reds',
                xticklabels=['Не выжил', 'Выжил'],
                yticklabels=['Не выжил', 'Выжил'],
                ax=axes5[1, 1])
    axes5[1, 1].set_title('Confusion Matrix (%)')
    axes5[1, 1].set_ylabel('Истинный класс')
    axes5[1, 1].set_xlabel('Предсказанный класс')

    plt.tight_layout()
    plt.savefig('predictions_detailed_analysis.png', dpi=100, bbox_inches='tight')
    plt.show()


def main():
    print("=" * 80)
    print("РАСШИРЕННАЯ НЕЙРОННАЯ СЕТЬ ДЛЯ ПРЕДСКАЗАНИЯ ВЫЖИВАНИЯ")
    print("С КОМПЛЕКСНОЙ ВИЗУАЛИЗАЦИЕЙ")
    print("=" * 80)

    # Генерируем данные
    df = generate_data(20000)

    features = ['age', 'physical_ability', 'autism', 'parents_count',
                'iq', 'vision', 'apocalypse_movies']
    target = 'survived'

    X = df[features].values
    y = df[target].values

    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # Масштабирование признаков
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Создаем модель
    input_size = X_train.shape[1]
    nn = NeuralNetwork(input_size=input_size, hidden_sizes=[32, 16], dropout_rate=0.3)

    print(f"\nАрхитектура сети: {input_size} → [32, 16] → 1 (Dropout: 0.3)")
    print(f"Данные: train={X_train.shape[0]}, val={X_val.shape[0]}, test={X_test.shape[0]}")
    print(f"Баланс классов: {y_train.mean():.2%} выживших")

    print(f"\nНачинаем обучение...")
    train_results = nn.train(
        X_train_scaled, y_train,
        X_val_scaled, y_val,
        epochs=200,
        learning_rate=0.0008,
        batch_size=64
    )

    # Оценка на тестовой выборке
    results = nn.evaluate(X_test_scaled, y_test)

    print(f"\n" + "=" * 80)
    print("РЕЗУЛЬТАТЫ НА ТЕСТЕ:")
    print("=" * 80)
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1-score:  {results['f1_score']:.4f}")
    print(f"ROC-AUC:   {results['roc_auc']:.4f}")

    print(f"\n" + "=" * 80)
    print("ДИАГНОСТИКА МОДЕЛИ:")
    print("=" * 80)

    y_test_proba = results['y_pred_proba']
    confidence_high = np.mean((y_test_proba > 0.8) | (y_test_proba < 0.2))
    confidence_medium = np.mean(((y_test_proba > 0.7) & (y_test_proba <= 0.8)) |
                                ((y_test_proba >= 0.2) & (y_test_proba < 0.3)))
    confidence_low = np.mean((y_test_proba >= 0.3) & (y_test_proba <= 0.7))

    print(f"Уверенность предсказаний:")
    print(f"  Высокая (>0.8 или <0.2): {confidence_high:.1%}")
    print(f"  Средняя (0.7-0.8 или 0.2-0.3): {confidence_medium:.1%}")
    print(f"  Низкая (0.3-0.7): {confidence_low:.1%}")

    if results['accuracy'] >= 0.9:
        print("\n🎯 ЦЕЛЬ ДОСТИГНУТА! Accuracy > 0.9")
    elif results['accuracy'] >= 0.85:
        print("\n⚠️  Хороший результат, но можно лучше")
    else:
        print("\n🔴 Нужно улучшить модель")

    # Расширенная визуализация
    print(f"\n" + "=" * 80)
    print("СОЗДАНИЕ КОМПЛЕКСНЫХ ВИЗУАЛИЗАЦИЙ...")
    print("=" * 80)

    plot_advanced_visualizations(df, nn, results, train_results, X_test_scaled, y_test)

    # Примеры предсказаний
    test_cases = [
        [25, 9, 0, 2, 130, 100, 25],
        [80, 3, 0, 0, 90, 60, 5],
        [35, 8, 0, 2, 115, 90, 15],
        [95, 2, 0, 1, 85, 30, 1],
        [45, 7, 0, 2, 110, 80, 30],
        [30, 6, 1, 1, 120, 85, 10],
    ]

    test_cases_scaled = scaler.transform(test_cases)
    predictions = nn.predict_proba(test_cases_scaled)

    print(f"\n" + "=" * 80)
    print("ПРИМЕРЫ ПРЕДСКАЗАНИЙ:")
    print("=" * 80)

    for i, (case, pred) in enumerate(zip(test_cases, predictions), 1):
        prob = pred[0]
        if prob > 0.8 or prob < 0.2:
            confidence = "очень уверенно"
        elif prob > 0.7 or prob < 0.3:
            confidence = "уверенно"
        else:
            confidence = "неуверенно"
        binary = "ВЫЖИВЕТ" if prob > 0.5 else "НЕ ВЫЖИВЕТ"
        print(f"\n{i}. Возраст={case[0]}, способности={case[1]}/10, "
              f"аутизм={'да' if case[2] == 1 else 'нет'}, родители={case[3]}, "
              f"iq={case[4]}, зрение={case[5]}%, фильмы={case[6]}")
        print(f"   Вероятность: {prob:.1%} ({confidence})")
        print(f"   Предсказание: {binary}")

    # Сохранение модели
    model_data = {
        'weights_0': nn.weights[0],
        'weights_1': nn.weights[1],
        'weights_2': nn.weights[2],
        'biases_0': nn.biases[0],
        'biases_1': nn.biases[1],
        'biases_2': nn.biases[2],
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
        'features': features
    }

    np.savez('final_model_advanced.npz', **model_data)
    print(f"\nМодель сохранена в 'final_model_advanced.npz'")

    # Сохранение полной информации
    model_info = {
        'accuracy': float(results['accuracy']),
        'precision': float(results['precision']),
        'recall': float(results['recall']),
        'f1_score': float(results['f1_score']),
        'roc_auc': float(results['roc_auc']),
        'architecture': [input_size, 32, 16, 1],
        'features': features,
        'epochs_trained': len(train_results['train_losses']),
        'best_val_accuracy': float(max(train_results['val_accuracies'])),
        'final_val_accuracy': float(train_results['val_accuracies'][-1]),
        'final_train_loss': float(train_results['train_losses'][-1]),
        'final_val_loss': float(train_results['val_losses'][-1]),
        'train_val_gap': float(train_results['train_accuracies'][-1] - train_results['val_accuracies'][-1]),
        'dropout_rate': 0.3,
        'learning_rate': 0.0008,
        'batch_size': 64,
        'confidence_distribution': {
            'high': float(confidence_high),
            'medium': float(confidence_medium),
            'low': float(confidence_low)
        },
        'notes': 'Расширенная модель с комплексной визуализацией'
    }

    with open('model_info_advanced.json', 'w') as f:
        json.dump(model_info, f, indent=2)

    print(f"Информация сохранена в 'model_info_advanced.json'")

    print(f"\n" + "=" * 80)
    print("СОЗДАНО 5 КОМПЛЕКСНЫХ ВИЗУАЛИЗАЦИЙ:")
    print("1. training_metrics_comprehensive.png - основные метрики обучения")
    print("2. roc_pr_curves.png - ROC и Precision-Recall кривые")
    print("3. gradients_weights_analysis.png - анализ градиентов и весов")
    print("4. data_analysis_comprehensive.png - анализ данных")
    print("5. predictions_detailed_analysis.png - детальный анализ предсказаний")
    print("=" * 80)


if __name__ == "__main__":
    main()