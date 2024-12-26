import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


class FraudDetectionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class FraudDetectionNN(nn.Module):
    def __init__(self, input_size):
        super(FraudDetectionNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


def load_and_preprocess_data():
    # Генерація синтетичних даних для прикладу
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=5, n_classes=2, random_state=42)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None, epochs=50):
    train_losses = []
    val_losses = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = criterion.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch.view(-1, 1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                y_pred = model(X_val)
                val_loss += criterion(y_pred, y_val.view(-1, 1)).item()

        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))

        if scheduler:
            scheduler.step()

        if (epoch + 1) % 10 == 0:
            logging.info(
                f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')

    return train_losses, val_losses


def evaluate_model(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test, y_test = X_test.to(device), y_test.to(device)
            y_pred = model(X_test)
            predictions.extend(y_pred.cpu().numpy())
            actuals.extend(y_test.cpu().numpy())

    auc_score = roc_auc_score(actuals, predictions)
    logging.info(f"AUC-ROC: {auc_score:.4f}")
    print(classification_report(actuals, (np.array(predictions) > 0.5)))
    return np.array(predictions), np.array(actuals), auc_score


def plot_results(train_losses, val_losses, predictions, actuals, auc_score):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    fpr, tpr, _ = roc_curve(actuals, predictions)
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.4f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()

    plt.tight_layout()
    plt.show()


def fuzzy_surface_view(model, feature1, feature2, resolution=50):
    """
    Візуалізація поверхні моделі для двох входів.
    """
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x, y)

    Z = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            inputs = np.zeros(model.network[0].in_features)  # Кількість вхідних параметрів
            inputs[feature1] = X[i, j]
            inputs[feature2] = Y[i, j]
            Z[i, j] = model(torch.FloatTensor(inputs).unsqueeze(0)).item()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    ax.set_title('Поверхня моделі')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Output')
    plt.show()


def main():
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    train_dataset = FraudDetectionDataset(X_train, y_train)
    test_dataset = FraudDetectionDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    model = FraudDetectionNN(input_size=X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train_losses, val_losses = train_model(model, train_loader, test_loader, criterion, optimizer, scheduler)
    predictions, actuals, auc_score = evaluate_model(model, test_loader)

    plot_results(train_losses, val_losses, predictions, actuals, auc_score)

    fuzzy_surface_view(model, feature1=0, feature2=1)  # Вибір перших двох ознак для візуалізації


if __name__ == "__main__":
    main()
