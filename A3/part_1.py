import pandas as pd
from scipy.stats import pearsonr
import numpy as np

import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('word_statistics.csv')

# Obtain the values in the first 3 columns as separate arrays
meanItemRT = df.iloc[:, 1].values
nItem = df.iloc[:, 2].values
word_length = df.iloc[:, 3].values

# Create correlation plots
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot 1: meanItemRT vs word_length
axes[0].scatter(word_length, meanItemRT, alpha=0.5)
axes[0].set_xlabel('word_length')
axes[0].set_ylabel('meanItemRT')
corr1, p1 = pearsonr(word_length, meanItemRT)
axes[0].set_title(f'meanItemRT vs word_length\nr={corr1:.3f}, p={p1:.3e}')

# Plot 2: meanItemRT vs nItem
axes[1].scatter(nItem, meanItemRT, alpha=0.5)
axes[1].set_xlabel('nItem')
axes[1].set_ylabel('meanItemRT')
corr2, p2 = pearsonr(nItem, meanItemRT)
axes[1].set_title(f'meanItemRT vs nItem\nr={corr2:.3f}, p={p2:.3e}')

# Plot 3: word_length vs nItem
axes[2].scatter(word_length, nItem, alpha=0.5)
axes[2].set_xlabel('word_length')
axes[2].set_ylabel('nItem')
corr3, p3 = pearsonr(word_length, nItem)
axes[2].set_title(f'word_length vs nItem\nr={corr3:.3f}, p={p3:.3e}')

plt.tight_layout()
plt.savefig('correlation_plots.png')