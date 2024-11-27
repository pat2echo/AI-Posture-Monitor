import os.path

import pandas as pd
import sys
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from pose_estimation_dependencies import get_groundtruth_from_image_name
import seaborn as sns


def kmeans(file_path, n_clusters=3):
    df = pd.read_csv(file_path)

    # Handle invalid values (-1.0 is used as a placeholder for missing values)
    df.replace(-1.0, np.nan, inplace=True)
    df.replace(1.0, np.nan, inplace=True)
    df = df.dropna(axis=0, how='any')

    # Select features for clustering
    features = ['aspect_ratio', 'relative_height', 'relative_width']
    features = ['aspect_ratio', 'relative_height']
    reduced_data = df[features].values

    # code sourced from: https://scikit-learn.org/1.5/auto_examples/cluster/plot_kmeans_digits.html#sphx-glr-auto-examples-cluster-plot-kmeans-digits-py
    kmeans = KMeans(init="k-means++", n_clusters=n_clusters, n_init=4)
    kmeans.fit(reduced_data)

    # Obtain cluster labels for the actual data points (not mesh grid)
    labels = kmeans.predict(reduced_data)

    # Add cluster labels to the DataFrame
    df['cluster'] = labels

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = 0.02  # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(
        Z,
        interpolation="nearest",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        cmap=plt.cm.Paired,
        aspect="auto",
        origin="lower",
    )

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="x",
        s=169,
        linewidths=3,
        color="w",
        zorder=10,
    )
    plt.title(
        "K-means clustering on the digits dataset (PCA-reduced data)\n"
        "Centroids are marked with white cross"
    )
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()

    return df

def clustering_analysis(file_path, method='kmeans', field_name='aspect_ratio', n_clusters=3, eps=0.5, min_samples=5, bin_rule=None):
    # Load data into a DataFrame
    df = pd.read_csv(file_path)

    # Handle invalid values (-1.0 is used as a placeholder for missing values)
    df.replace(-1.0, np.nan, inplace=True)
    df.replace(1.0, np.nan, inplace=True)
    df = df.dropna(axis=0, how='any')

    # Initialize labels
    labels = None

    # Apply K-means Clustering
    if method == 'stats':
        df['label'] = df.apply(lambda row: get_groundtruth_from_image_name(image_name=row["image_name"]), axis=1)
        print(df['label'].value_counts())

        #grouped = df.groupby("label")[["aspect_ratio", "relative_width", "relative_height"]].agg(['min', 'max', 'mean', 'median'])
        grouped = df.groupby("label")[["aspect_ratio", "relative_width", "relative_height"]].agg(['mean'])
        print(grouped)

        correlation_matrix = grouped.corr()

        # Display the correlation matrix
        print("Correlation Matrix:\n", correlation_matrix)

        # Visualize with a heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix of Aspect Ratio, Relative Width, and Relative Height')
        plt.show()

        save_path = os.path.join( os.path.dirname(file_path), 'grouped_static_pose_boundingbox_data.csv')
        grouped.to_csv(save_path, index=False)
    else:
        k = field_name

        # Select features for clustering
        features = ['aspect_ratio', 'relative_height', 'relative_width']
        features = [k]
        X = df[features].values.reshape(-1, 1)

        # Standardize features for better performance of clustering algorithms
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if method == 'kmeans':
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(X_scaled)
            df['cluster'] = labels
            cluster_centers = kmeans.cluster_centers_
            print(f"K-means Cluster Centers:\n{cluster_centers}")

            print(df['cluster'].value_counts())

            grouped = df.groupby("cluster")[["aspect_ratio"]].agg(
                ['min', 'max', 'mean', 'count'])
            print(grouped)
            save_path = os.path.join(os.path.dirname(file_path), 'grouped_kmeans_static_pose_boundingbox_data.csv')
            grouped.to_csv(save_path, index=False)

        # Apply DBSCAN Clustering
        elif method == 'dbscan':
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X_scaled)
            df['cluster'] = labels
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            print(f"DBSCAN detected {n_clusters} clusters")
        elif method == 'histogram':
            # Convert the aspect ratios to a numpy array (optional, if not already)
            aspect_ratios = df[features].values

            bins = n_clusters

            if bin_rule == 'sturges':
                # Sturges Rule to get no of bins
                n = len(aspect_ratios)
                bins = int(np.ceil(np.log2(n) + 1))
            elif bin_rule == 'freedman-diaconis':
                # Freedman-Diaconis Rule
                iqr = np.percentile(aspect_ratios, 75) - np.percentile(aspect_ratios, 25)
                bin_width = 2 * iqr / len(aspect_ratios) ** (1 / 3)
                bins = int(np.ceil((max(aspect_ratios) - min(aspect_ratios)) / bin_width))
            elif bin_rule == 'scott':
                # Scott Rule
                bin_width = 3.5 * np.std(aspect_ratios) / len(aspect_ratios) ** (1 / 3)
                bins = int(np.ceil((max(aspect_ratios) - min(aspect_ratios)) / bin_width))


            # Plot histogram to visualize the distribution of aspect ratios
            plt.figure(figsize=(10, 6))
            if bins == 0:
                plt.hist(aspect_ratios, color='orange', edgecolor='black', alpha=0.7)
                plt.title(f'Histogram of {k.replace('_', ' ').capitalize()}')
            else:
                plt.hist(aspect_ratios, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
                plt.title(f'Histogram of {k.replace('_', ' ').capitalize()} \nNo of Bins: {bins} rule {bin_rule}')
            plt.xlabel(f'{k.replace('_', ' ').capitalize()}')
            plt.ylabel('Frequency')
            plt.show()

    if labels is not None:
        # Display the resulting clusters
        print(f"\nCluster Labels Assigned:", df[['image_name', 'cluster']] )

        # Visualization
        plt.figure(figsize=(10, 6))
        plt.scatter(X_scaled, np.zeros_like(X_scaled), c=labels, cmap=plt.cm.Paired, s=50)
        if method == 'kmeans':
            plt.scatter(cluster_centers, np.zeros_like(cluster_centers), s=200, c='red', marker='X')
            for i, center in enumerate(cluster_centers):
                plt.text(center[0], 0, f'{i}', fontsize=12, fontweight='bold', ha='center', va='bottom')

        plt.title(f"{method.upper()} Clustering \n No of Clusters: {n_clusters}")
        plt.xlabel(f'{k.replace('_', ' ').capitalize()} (scaled)')
        plt.colorbar(label='Cluster Label')
        plt.grid(True)
        plt.show()

    return df

if len(sys.argv) > 3:
    cluster = 3
    if len(sys.argv) > 4:
        cluster = int(sys.argv[4])

    if sys.argv[2] == 'kmeans':
        df_clustered = clustering_analysis(file_path=sys.argv[1], field_name=sys.argv[3], method='kmeans', n_clusters=cluster)
    elif sys.argv[2] == 'dbscan':
        df_clustered = clustering_analysis(file_path=sys.argv[1], field_name=sys.argv[3], method='dbscan', eps=0.5, min_samples=cluster)
    elif sys.argv[2] == 'stats':
        df_clustered = clustering_analysis(file_path=sys.argv[1], field_name=sys.argv[3], method='stats')
    else:
        df_clustered = clustering_analysis(file_path=sys.argv[1], field_name=sys.argv[3], method='histogram', n_clusters=cluster, bin_rule=None)
    #df_clustered = kmeans(file_path=sys.argv[1], n_clusters=6)
    #print(df_clustered)

