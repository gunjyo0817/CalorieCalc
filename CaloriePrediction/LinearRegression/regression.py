import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
import matplotlib.pyplot as plt

food_labels = [
    "cucumber", "chicken", "shrimp", "brocolli", "bean", "salmon", "pork", "rice",
    "pumpkin", "egg_dish", "nut", "carrot", "king_oyster", "vegetable",
    "winter_melon", "mango", "avocado", "baby_corn", "beef", "sprout",
    "eggplant", "mushroom", "potato", "brocoli", "corn", "toast", "onion",
    "cabbage", "egg", "green_pepper", "green_bean", "tofu", "fish", "tomato",
    "lemon", "yam", "high", "protein", "fat", "carbs", "GMM"
]

def augment_data(x, y, noise_level=0.01, augment_size=100):
    indices = torch.randint(0, x.size(0), (augment_size,))
    x_sample = x[indices]
    y_sample = y[indices]

    x_augmented = x_sample + noise_level * torch.randn_like(x_sample)
    y_augmented = y_sample + noise_level * torch.randn_like(y_sample)

    x_combined = torch.cat([x, x_augmented], dim=0)
    y_combined = torch.cat([y, y_augmented], dim=0)

    return x_combined, y_combined

def select_important_features(x, y, num_features=10):
    x_np, y_np = x.numpy(), y.numpy().flatten()
    selector = SelectKBest(score_func=f_regression, k=num_features)
    x_selected = selector.fit_transform(x_np, y_np)
    return torch.tensor(x_selected, dtype=torch.float32), selector.get_support(indices=True)

def generate_interaction_features(x, food_labels):
    n_features = x.shape[1]
    interaction_terms = []
    interaction_labels = []

    for i in range(n_features):
        for j in range(i + 1, n_features): 
            interaction_terms.append(x[:, i] * x[:, j])
            interaction_labels.append(f"{food_labels[i]}*{food_labels[j]}")

    interaction_tensor = torch.stack(interaction_terms, dim=1)
    return interaction_tensor, interaction_labels


csv_file_path = 'train.csv'
df = pd.read_csv(csv_file_path)

features = df.drop(columns=['image_name', 'calories']).values
target = df['calories'].values

x_train = torch.tensor(features, dtype=torch.float32)
y_train = torch.tensor(target, dtype=torch.float32).unsqueeze(1)

remove_features = ["salmon", "pumpkin", "rice", "king_oyster", "potato", "brocolli"]
remove_indices = [food_labels.index(feature) for feature in remove_features]
x_train = torch.tensor(np.delete(features, remove_indices, axis=1), dtype=torch.float32)
food_labels = [label for i, label in enumerate(food_labels) if i not in remove_indices]

x_train[x_train <= 0] = 1e-5 
x_train_log = np.log(x_train + 1e-5);
x_train = torch.cat([x_train, x_train_log], dim=1)

food_labels_log = [f"log({label})" for label in food_labels]
food_labels += food_labels_log

scaler = StandardScaler()
x_train = torch.tensor(scaler.fit_transform(x_train.numpy()), dtype=torch.float32)

num_selected_features = len(food_labels)
x_train, selected_indices = select_important_features(x_train, y_train, num_features=num_selected_features)

x_train, y_train = augment_data(x_train, y_train, noise_level=0.05, augment_size=100)

class LinearRegressionModel(nn.Module):
    def __init__(self, input_features):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_features, 1)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)

model = LinearRegressionModel(num_selected_features)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

def custom_loss(output, target, model, penalty_factor=1000):
    mse_loss = criterion(output, target)
    penalty = 0;
    return mse_loss + penalty

epochs = 70000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    predictions = model(x_train)
    loss = custom_loss(predictions, y_train, model)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")


test_csv_file_path = 'test.csv' 
df_test = pd.read_csv(test_csv_file_path)

features_test = df_test.drop(columns=['image_name', 'calories']).values
target_test = df_test['calories'].values

x_test = torch.tensor(np.delete(features_test, remove_indices, axis=1), dtype=torch.float32)
x_test_log = np.log(x_test + 1e-5);
x_test = torch.cat([x_test, x_test_log], dim=1)
x_test = torch.tensor(scaler.transform(x_test.numpy()), dtype=torch.float32)
x_test_selected = x_test[:, selected_indices]

y_test = torch.tensor(target_test, dtype=torch.float32).unsqueeze(1)


model.eval()
with torch.no_grad():
    predictions = model(x_test_selected)

image_names = df_test['image_name'].values
for i in range(len(predictions)):
    print(f"Image Name = {image_names[i]}: True Value = {(y_test[i]).item():.2f}, Predicted Value = {(predictions[i]).item():.2f}")


def print_regression_equation(model):
    weights = model.linear.weight.detach().numpy()
    bias = model.linear.bias.detach().numpy()

    equation = "y = "
    for i in range(weights.shape[1]):
        equation += f"{weights[0, i]:.4f} * {food_labels[i]} + "
    equation += f"{bias[0]:.4f}"
    print("回歸方程:", equation)

print_regression_equation(model)

y_test_np = y_test.numpy().flatten()
predictions_np = predictions.numpy().flatten()

mse = mean_squared_error(y_test_np, predictions_np)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_np, predictions_np)
r2 = r2_score(y_test_np, predictions_np)


plt.figure(figsize=(12, 6))
plt.scatter(range(len(y_test_np)), y_test_np, color='blue', label='True Value', alpha=0.7)
plt.scatter(range(len(predictions_np)), predictions_np, color='red', label='Predicted Value', alpha=0.7)

plt.plot(range(len(y_test_np)), y_test_np, color='blue', linestyle='--', linewidth=0.8, alpha=0.6)
plt.plot(range(len(predictions_np)), predictions_np, color='red', linestyle='--', linewidth=0.8, alpha=0.6)

plt.title("True Values vs Predicted Values", fontsize=16)
plt.xlabel("Sample Index", fontsize=14)
plt.ylabel("Calories", fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()


sorted_data = sorted(zip(y_test, predictions.flatten(), df_test['image_name'].values), key=lambda x: x[0])

print("\nSorted Predictions:")
for true, pred, filename in sorted_data:
    print(f"Filename: {filename}, Ground Truth: {true}, Prediction: {pred}")

def plot_trend_chart(y_test, predictions):
    indices = range(len(y_test))
    plt.figure(figsize=(10, 6))
    plt.plot(indices, y_test, label='True Value', color='blue', linestyle='--', marker='o') 
    plt.plot(indices, predictions.flatten(), label='Predicted Value', color='red', linestyle='--', marker='o')  # 預測值
    plt.title('True Values vs Predicted Values', fontsize=16)
    plt.xlabel('Sample Index', fontsize=14)  
    plt.ylabel('Calories', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)

# Compute metrics
mae_test = mean_absolute_error(y_test.numpy(), predictions.numpy())
rmse_test = math.sqrt(mean_squared_error(y_test.numpy(), predictions.numpy()))
mape_test = np.mean(np.abs((y_test.numpy().flatten() - predictions.numpy().flatten()) / y_test.numpy().flatten())) * 100
r2_test = 1 - (np.sum((y_test.numpy().flatten() - predictions.numpy().flatten())**2) / np.sum((y_test.numpy().flatten() - np.mean(y_test.numpy().flatten()))**2))
errors = np.abs(y_test.numpy().flatten() - predictions.numpy().flatten())
accuracy_50 = np.mean(errors < 50) * 100
accuracy_100 = np.mean(errors < 100) * 100
accuracy_150 = np.mean(errors < 150) * 100

print(f"MAE: {mae_test:.2f}, RMSE: {rmse_test:.2f}, MAPE: {mape_test:.2f}%, R² Score: {r2:.2f}")
print(f"Accuracy (<50): {accuracy_50:.2f}%")
print(f"Accuracy (<100): {accuracy_100:.2f}%")
print(f"Accuracy (<150): {accuracy_150:.2f}%")

output_text_path = f"Regression_{int(rmse_test)}.txt"
with open(output_text_path, "w") as f:
    f.write(f"MAE: {mae_test:.2f}, RMSE: {rmse_test:.2f}, MAPE: {mape_test:.2f}%, R^2: {r2_test:.2f}\n")
    f.write(f"Accuracy (<50): {accuracy_50:.2f}%\n")
    f.write(f"Accuracy (<100): {accuracy_100:.2f}%\n")
    f.write(f"Accuracy (<150): {accuracy_150:.2f}%\n")
    f.write("\nSorted Predictions:\n")
    for true, pred, filename in sorted_data:
        f.write(f"Filename: {filename}, Ground Truth: {true}, Prediction: {pred}\n")


plot_trend_chart(y_test, predictions)
output_path = f"Regression_{int(rmse_test)}.png" 
plt.savefig(output_path, dpi=300)
print(f"Regression plot saved as {output_path}")
plt.show()