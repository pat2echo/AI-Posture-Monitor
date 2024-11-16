import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Sample data
data = {
    'aspect_ratio_mean': [4.6608, 1.3538, 0.5159],  # Means for 'Lie', 'Sit', 'Stand'
    'relative_width_mean': [0.4850, 0.2650, 0.1875],
    'relative_height_mean': [0.1202, 0.2305, 0.3812]
}
df = pd.DataFrame(data)

# Calculate correlation matrix
correlation_matrix = df.corr()

# Display the correlation matrix
print("Correlation Matrix:\n", correlation_matrix)

# Visualize with a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Aspect Ratio, Relative Width, and Relative Height')
plt.show()
