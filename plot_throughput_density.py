#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Read the CSV file
df = pd.read_csv('throughput_results.csv')

# Create figure and axis
plt.figure(figsize=(12, 6))

# Calculate the kernel density estimation
density = stats.gaussian_kde(df['throughput'])
xs = np.linspace(df['throughput'].min(), df['throughput'].max(), 200)
density.covariance_factor = lambda: .25
density._compute_covariance()

# Plot the density
plt.plot(xs, density(xs), 'b-', lw=2, label='Density')

# Add a histogram in the background
plt.hist(df['throughput'], bins=50, density=True, alpha=0.3, color='gray', label='Histogram')

# Add summary statistics as text
stats_text = f'Mean: {df.throughput.mean():.2f}\n'
stats_text += f'Median: {df.throughput.median():.2f}\n'
stats_text += f'Std Dev: {df.throughput.std():.2f}\n'
stats_text += f'Min: {df.throughput.min():.2f}\n'
stats_text += f'Max: {df.throughput.max():.2f}'

plt.text(0.95, 0.95, stats_text,
         transform=plt.gca().transAxes,
         verticalalignment='top',
         horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Customize the plot
plt.title('Distribution of Throughput Values', fontsize=14, pad=20)
plt.xlabel('Throughput', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()

# Save the plot
plt.savefig('throughput_density.png', dpi=300, bbox_inches='tight')
print("Plot saved as throughput_density.png")

# Display some basic statistics
print("\nSummary Statistics:")
print(df['throughput'].describe())

# Count number of grids in different throughput ranges
print("\nThroughput Range Distribution:")
ranges = [0, 1, 2, 3, 4, 5, 6, 7]
labels = [f"{ranges[i]}-{ranges[i+1]}" for i in range(len(ranges)-1)]
df['throughput_range'] = pd.cut(df['throughput'], bins=ranges, labels=labels)
print(df['throughput_range'].value_counts().sort_index())