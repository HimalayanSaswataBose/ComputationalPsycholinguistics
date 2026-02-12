import pandas as pd
from scipy.stats import pearsonr
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('word_statistics.csv')

# Obtain the values in the first 3 columns as separate arrays
meanItemRT = df.iloc[:, 1].values
nItem = df.iloc[:, 2].values
word_length = df.iloc[:, 3].values
gpt3prob = df.iloc[:, 4].values

# Prepare features for Model 1: word freq + word length
X1 = np.column_stack([nItem, word_length])

# Prepare features for Model 2: -log(gpt3 probability) + word length
# Handle any zero or negative probabilities
gpt3prob_safe = np.where(gpt3prob > 0, gpt3prob, np.nan)
neg_log_gpt3 = -np.log(gpt3prob_safe)

# Remove rows with NaN values
valid_mask = ~np.isnan(neg_log_gpt3)
X2 = np.column_stack([neg_log_gpt3[valid_mask], word_length[valid_mask]])
y2 = meanItemRT[valid_mask]

# Fit Model 1
model1 = LinearRegression()
model1.fit(X1, meanItemRT)
y1_pred = model1.predict(X1)

# Fit Model 2
model2 = LinearRegression()
model2.fit(X2, y2)
y2_pred = model2.predict(X2)

# Compare models
r2_model1 = r2_score(meanItemRT, y1_pred)
r2_model2 = r2_score(y2, y2_pred)
mse_model1 = mean_squared_error(meanItemRT, y1_pred)
mse_model2 = mean_squared_error(y2, y2_pred)

print("Model 1 (word freq + word length):")
print(f"  R² = {r2_model1:.4f}")
print(f"  MSE = {mse_model1:.4f}")

print("\nModel 2 (-log(gpt3 prob) + word length):")
print(f"  R² = {r2_model2:.4f}")
print(f"  MSE = {mse_model2:.4f}")

print(f"\nBetter model: {'Model 2' if r2_model2 > r2_model1 else 'Model 1'}")

# Load content/function word labels (assuming there's a column in the CSV)
# If not in CSV, you'll need to load it from another source
word_type = df.iloc[:, 5].values  # Adjust column index as needed

# Separate content and function words
content_mask = word_type == 'content'
function_mask = word_type == 'function'

# Content words data
content_X1 = np.column_stack([nItem[content_mask], word_length[content_mask]])
content_y = meanItemRT[content_mask]

# Content words with GPT-3 (handle NaN)
content_valid_mask = content_mask & valid_mask
content_X2 = np.column_stack([neg_log_gpt3[content_valid_mask], word_length[content_valid_mask]])
content_y2 = meanItemRT[content_valid_mask]

# Function words data
function_X3 = np.column_stack([nItem[function_mask], word_length[function_mask]])
function_y = meanItemRT[function_mask]

# Function words with GPT-3 (handle NaN)
function_valid_mask = function_mask & valid_mask
function_X4 = np.column_stack([neg_log_gpt3[function_valid_mask], word_length[function_valid_mask]])
function_y2 = meanItemRT[function_valid_mask]

# Fit Model 1: Content words with word freq + length
model1_content = LinearRegression()
model1_content.fit(content_X1, content_y)
y1_content_pred = model1_content.predict(content_X1)

# Fit Model 2: Content words with -log(gpt3) + length
model2_content = LinearRegression()
model2_content.fit(content_X2, content_y2)
y2_content_pred = model2_content.predict(content_X2)

# Fit Model 3: Function words with word freq + length
model3_function = LinearRegression()
model3_function.fit(function_X3, function_y)
y3_function_pred = model3_function.predict(function_X3)

# Fit Model 4: Function words with -log(gpt3) + length
model4_function = LinearRegression()
model4_function.fit(function_X4, function_y2)
y4_function_pred = model4_function.predict(function_X4)

# Calculate metrics
r2_m1 = r2_score(content_y, y1_content_pred)
r2_m2 = r2_score(content_y2, y2_content_pred)
r2_m3 = r2_score(function_y, y3_function_pred)
r2_m4 = r2_score(function_y2, y4_function_pred)

mse_m1 = mean_squared_error(content_y, y1_content_pred)
mse_m2 = mean_squared_error(content_y2, y2_content_pred)
mse_m3 = mean_squared_error(function_y, y3_function_pred)
mse_m4 = mean_squared_error(function_y2, y4_function_pred)

print("\n--- Content vs Function Word Analysis ---")
print("\nModel 1 (Content: word freq + length):")
print(f"  R² = {r2_m1:.4f}, MSE = {mse_m1:.4f}")

print("\nModel 2 (Content: -log(gpt3) + length):")
print(f"  R² = {r2_m2:.4f}, MSE = {mse_m2:.4f}")

print("\nModel 3 (Function: word freq + length):")
print(f"  R² = {r2_m3:.4f}, MSE = {mse_m3:.4f}")

print("\nModel 4 (Function: -log(gpt3) + length):")
print(f"  R² = {r2_m4:.4f}, MSE = {mse_m4:.4f}")

print(f"\nBest content word model: {'Model 2' if r2_m2 > r2_m1 else 'Model 1'}")
print(f"Best function word model: {'Model 4' if r2_m4 > r2_m3 else 'Model 3'}")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Model 1 - Content words (word freq + length)
axes[0, 0].scatter(content_y, y1_content_pred, alpha=0.6)
axes[0, 0].plot([content_y.min(), content_y.max()], [content_y.min(), content_y.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Actual RT')
axes[0, 0].set_ylabel('Predicted RT')
axes[0, 0].set_title(f'Content Words: Freq+Length (R²={r2_m1:.4f})')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Model 2 - Content words (-log(gpt3) + length)
axes[0, 1].scatter(content_y2, y2_content_pred, alpha=0.6, color='green')
axes[0, 1].plot([content_y2.min(), content_y2.max()], [content_y2.min(), content_y2.max()], 'r--', lw=2)
axes[0, 1].set_xlabel('Actual RT')
axes[0, 1].set_ylabel('Predicted RT')
axes[0, 1].set_title(f'Content Words: -log(GPT3)+Length (R²={r2_m2:.4f})')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Model 3 - Function words (word freq + length)
axes[1, 0].scatter(function_y, y3_function_pred, alpha=0.6, color='orange')
axes[1, 0].plot([function_y.min(), function_y.max()], [function_y.min(), function_y.max()], 'r--', lw=2)
axes[1, 0].set_xlabel('Actual RT')
axes[1, 0].set_ylabel('Predicted RT')
axes[1, 0].set_title(f'Function Words: Freq+Length (R²={r2_m3:.4f})')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Model 4 - Function words (-log(gpt3) + length)
axes[1, 1].scatter(function_y2, y4_function_pred, alpha=0.6, color='red')
axes[1, 1].plot([function_y2.min(), function_y2.max()], [function_y2.min(), function_y2.max()], 'r--', lw=2)
axes[1, 1].set_xlabel('Actual RT')
axes[1, 1].set_ylabel('Predicted RT')
axes[1, 1].set_title(f'Function Words: -log(GPT3)+Length (R²={r2_m4:.4f})')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_predictions.png', dpi=300, bbox_inches='tight')
plt.show()

# Model comparison bar plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

models = ['Content\nFreq+Length', 'Content\n-log(GPT3)+Length', 'Function\nFreq+Length', 'Function\n-log(GPT3)+Length']
r2_scores = [r2_m1, r2_m2, r2_m3, r2_m4]
mse_scores = [mse_m1, mse_m2, mse_m3, mse_m4]
colors = ['blue', 'green', 'orange', 'red']

ax1.bar(models, r2_scores, color=colors, alpha=0.7)
ax1.set_ylabel('R² Score')
ax1.set_title('Model Comparison: R² Score')
ax1.set_ylim(0, max(r2_scores) * 1.1)
for i, v in enumerate(r2_scores):
    ax1.text(i, v + 0.01, f'{v:.4f}', ha='center')

ax2.bar(models, mse_scores, color=colors, alpha=0.7)
ax2.set_ylabel('Mean Squared Error')
ax2.set_title('Model Comparison: MSE')
for i, v in enumerate(mse_scores):
    ax2.text(i, v + 0.5, f'{v:.4f}', ha='center')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()