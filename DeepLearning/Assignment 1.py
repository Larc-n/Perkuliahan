import numpy as np
import matplotlib.pyplot as plt

# 1. Dataset setup (Iris Setosa = 0, Versicolor = 1)
# Training Data (80 samples)
X_train = np.array([
    [5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2], [4.7, 3.2, 1.3, 0.2], [4.6, 3.1, 1.5, 0.2],
    [5.0, 3.6, 1.4, 0.2], [5.4, 3.9, 1.7, 0.4], [4.6, 3.4, 1.4, 0.3], [5.0, 3.4, 1.5, 0.2],
    [4.4, 2.9, 1.4, 0.2], [4.9, 3.1, 1.5, 0.1], [5.4, 3.7, 1.5, 0.2], [4.8, 3.4, 1.6, 0.2],
    [4.8, 3.0, 1.4, 0.1], [4.3, 3.0, 1.1, 0.1], [5.8, 4.0, 1.2, 0.2], [5.7, 4.4, 1.5, 0.4],
    [5.4, 3.9, 1.3, 0.4], [5.1, 3.5, 1.4, 0.3], [5.7, 3.8, 1.7, 0.3], [5.1, 3.8, 1.5, 0.3],
    [5.4, 3.4, 1.7, 0.2], [5.1, 3.7, 1.5, 0.4], [4.6, 3.6, 1.0, 0.2], [5.1, 3.3, 1.7, 0.5],
    [4.8, 3.4, 1.9, 0.2], [5.0, 3.0, 1.6, 0.2], [5.0, 3.4, 1.6, 0.4], [5.2, 3.5, 1.5, 0.2],
    [5.2, 3.4, 1.4, 0.2], [4.7, 3.2, 1.6, 0.2], [4.8, 3.1, 1.6, 0.2], [5.4, 3.4, 1.5, 0.4],
    [5.2, 4.1, 1.5, 0.1], [5.5, 4.2, 1.4, 0.2], [4.9, 3.1, 1.5, 0.1], [5.0, 3.2, 1.2, 0.2],
    [5.5, 3.5, 1.3, 0.2], [4.9, 3.1, 1.5, 0.1], [4.4, 3.0, 1.3, 0.2], [5.1, 3.4, 1.5, 0.2],
    [7.0, 3.2, 4.7, 1.4], [6.4, 3.2, 4.5, 1.5], [6.9, 3.1, 4.9, 1.5], [5.5, 2.3, 4.0, 1.3],
    [6.5, 2.8, 4.6, 1.5], [5.7, 2.8, 4.5, 1.3], [6.3, 3.3, 4.7, 1.6], [4.9, 2.4, 3.3, 1.0],
    [6.6, 2.9, 4.6, 1.3], [5.2, 2.7, 3.9, 1.4], [5.0, 2.0, 3.5, 1.0], [5.9, 3.0, 4.2, 1.5],
    [6.0, 2.2, 4.0, 1.0], [6.1, 2.9, 4.7, 1.4], [5.6, 2.9, 3.6, 1.3], [6.7, 3.1, 4.4, 1.4],
    [5.6, 3.0, 4.5, 1.5], [5.8, 2.7, 4.1, 1.0], [6.2, 2.2, 4.5, 1.5], [5.6, 2.5, 3.9, 1.1],
    [5.9, 3.2, 4.8, 1.8], [6.1, 2.8, 4.0, 1.3], [6.3, 2.5, 4.9, 1.5], [6.1, 2.8, 4.7, 1.2],
    [6.4, 2.9, 4.3, 1.3], [6.6, 3.0, 4.4, 1.4], [6.8, 2.8, 4.8, 1.4], [6.7, 3.0, 5.0, 1.7],
    [6.0, 2.9, 4.5, 1.5], [5.7, 2.6, 3.5, 1.0], [5.5, 2.4, 3.8, 1.1], [5.5, 2.4, 3.7, 1.0],
    [5.8, 2.7, 3.9, 1.2], [6.0, 2.7, 5.1, 1.6], [5.4, 3.0, 4.5, 1.5], [6.0, 3.4, 4.5, 1.6],
    [6.7, 3.1, 4.7, 1.5], [6.3, 2.3, 4.4, 1.3], [5.6, 3.0, 4.1, 1.3], [5.5, 2.5, 4.0, 1.3]
])
y_train = np.array([0]*40 + [1]*40)

# Validation Data (20 samples)
X_val = np.array([
    [5.0, 3.5, 1.3, 0.3], [4.5, 2.3, 1.3, 0.3], [4.4, 3.2, 1.3, 0.2], [5.0, 3.5, 1.6, 0.6],
    [5.1, 3.8, 1.9, 0.4], [4.8, 3.0, 1.4, 0.3], [5.1, 3.8, 1.6, 0.2], [4.6, 3.2, 1.4, 0.2],
    [5.3, 3.7, 1.5, 0.2], [5.0, 3.3, 1.4, 0.2], [5.5, 2.6, 4.4, 1.2], [6.1, 3.0, 4.6, 1.4],
    [5.8, 2.6, 4.0, 1.2], [5.0, 2.3, 3.3, 1.0], [5.6, 2.7, 4.2, 1.3], [5.7, 3.0, 4.2, 1.2],
    [5.7, 2.9, 4.2, 1.3], [6.2, 2.9, 4.3, 1.3], [5.1, 2.5, 3.0, 1.1], [5.7, 2.8, 4.1, 1.3]
])
y_val = np.array([0]*10 + [1]*10)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Hyperparameters
lr = 0.1
epochs = 5
bias = 0.5
weights = np.array([0.5, 0.5, 0.5, 0.5])

train_loss_hist, val_loss_hist = [], []
train_acc_hist, val_acc_hist = [], []

for epoch in range(epochs):
    # Training Phase
    train_sse, train_correct = 0, 0
    for i in range(len(X_train)):
        x_i, target = X_train[i], y_train[i]
        z = bias + np.dot(weights, x_i)
        gz = sigmoid(z)
        pred = 1 if gz >= 0.5 else 0
        
        err = gz - target
        train_sse += err**2
        if pred == target:
            train_correct += 1
            
        # Derivative calculations matching GSheet
        dbias = 2 * err * gz * (1 - gz)
        dweights = dbias * x_i
        
        # Update weights & bias
        bias -= lr * dbias
        weights -= lr * dweights
        
    train_loss_hist.append(train_sse / len(X_train))
    train_acc_hist.append(train_correct / len(X_train))
    
    # Validation Phase
    val_sse, val_correct = 0, 0
    for i in range(len(X_val)):
        x_i, target = X_val[i], y_val[i]
        z = bias + np.dot(weights, x_i)
        gz = sigmoid(z)
        pred = 1 if gz >= 0.5 else 0
        
        val_sse += (gz - target)**2
        if pred == target:
            val_correct += 1
            
    val_loss_hist.append(val_sse / len(X_val))
    val_acc_hist.append(val_correct / len(X_val))

for i in range(len(train_acc_hist)):
    print(f"Training Accuracy: {train_acc_hist[i]:.4f}, Validation Accuracy: {val_acc_hist[i]:.4f}")
    print(f"Training Loss: {train_loss_hist[i]:.4f}, Validation Loss: {val_loss_hist[i]:.4f}")

# Plot Accuracy Chart
plt.figure(figsize=(6, 4))
plt.plot(range(1, epochs + 1), train_acc_hist, label='Training Accuracy', marker='o')
plt.plot(range(1, epochs + 1), val_acc_hist, label='Validation Accuracy', marker='s')
plt.title('Accuracy Chart (Python)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('accuracy_chart_python.png')

# Plot Loss Chart
plt.figure(figsize=(6, 4))
plt.plot(range(1, epochs + 1), train_loss_hist, label='Training Loss', marker='o')
plt.plot(range(1, epochs + 1), val_loss_hist, label='Validation Loss', marker='s')
plt.title('Loss Chart (Spreadsheet)')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.savefig('loss_chart_python.png')
