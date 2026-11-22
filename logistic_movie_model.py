from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn



torch.manual_seed(42)

DATA_FOLDER = Path(__file__).resolve().parent

train_df = pd.read_csv(DATA_FOLDER / "movie_train_data.csv")
test_df = pd.read_csv(DATA_FOLDER / "movie_test_data.csv")

features = ["action_level", "romance_level"]
target = "like_movie"

X_train = torch.tensor(train_df[features].values, dtype=torch.float32)
y_train = torch.tensor(train_df[target].values, dtype=torch.float32).reshape(-1, 1)

X_test = torch.tensor(test_df[features].values, dtype=torch.float32)
y_test = torch.tensor(test_df[target].values, dtype=torch.float32).reshape(-1, 1)

print(f"Training examples: {len(X_train)}")
print(f"Testing examples:  {len(X_test)}")


feature_mean = X_train.mean(dim=0)
feature_std = X_train.std(dim=0, unbiased=False)


feature_std = torch.clamp(feature_std, min=1e-8)

X_train_scaled = (X_train - feature_mean) / feature_std
X_test_scaled = (X_test - feature_mean) / feature_std


model = nn.Linear(in_features=len(features), out_features=1)

loss_function = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)



epochs = 2000

print("\nTraining model:")

for epoch in range(1, epochs + 1):
    optimizer.zero_grad()

    logits = model(X_train_scaled)
    loss = loss_function(logits, y_train)

    loss.backward()
    optimizer.step()

    if epoch == 1 or epoch % 500 == 0:
        print(f"Epoch {epoch:4d} | Training loss: {loss.item():.4f}")


model.eval()

with torch.no_grad():
    test_logits = model(X_test_scaled)
    test_loss = loss_function(test_logits, y_test).item()

    probabilities = torch.sigmoid(test_logits)
    predictions = (probabilities >= 0.5).float()

correct = (predictions == y_test).sum().item()
accuracy = correct / len(y_test)

true_positive = int(((predictions == 1) & (y_test == 1)).sum().item())
true_negative = int(((predictions == 0) & (y_test == 0)).sum().item())
false_positive = int(((predictions == 1) & (y_test == 0)).sum().item())
false_negative = int(((predictions == 0) & (y_test == 1)).sum().item())

precision_denominator = true_positive + false_positive
recall_denominator = true_positive + false_negative

precision = (
    true_positive / precision_denominator
    if precision_denominator > 0
    else 0.0
)
recall = (
    true_positive / recall_denominator
    if recall_denominator > 0
    else 0.0
)


print("\nTest results:")
print(f"Test loss: {test_loss:.4f}")
print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")

print("\nConfusion matrix (rows = actual 0/1, columns = predicted 0/1):")
print(f"[[{true_negative:2d} {false_positive:2d}]")
print(f" [{false_negative:2d} {true_positive:2d}]]")


print("\nLearned parameters (for standardised features):")

for feature_name, weight in zip(features, model.weight.detach().flatten()):
    print(f"{feature_name}: {weight.item():.3f}")

print(f"Bias: {model.bias.item():.3f}")
