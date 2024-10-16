import numpy as np
import matplotlib.pyplot as plt

landmarks_3d = [
    0.0, 0.0, 0.0,  # nose
    0.1, -0.1, -0.1,  # left eye inner
    0.15, -0.15, -0.12,  # left eye
    0.2, -0.2, -0.1,  # left eye outer
    -0.1, -0.1, -0.1,  # right eye inner
    -0.15, -0.15, -0.12,  # right eye
    -0.2, -0.2, -0.1,  # right eye outer
    0.05, -0.25, -0.12,  # left ear
    -0.05, -0.25, -0.12,  # right ear
    0.05, -0.5, -0.05,  # mouth left
    -0.05, -0.5, -0.05,  # mouth right
    0.0, -0.7, 0.0,  # left shoulder
    0.0, -0.7, 0.0,  # right shoulder
    0.2, -1.0, 0.0,  # left elbow
    -0.2, -1.0, 0.0,  # right elbow
    0.3, -1.3, 0.1,  # left wrist
    -0.3, -1.3, 0.1,  # right wrist
    0.1, -0.9, -0.1,  # left pinky
    -0.1, -0.9, -0.1,  # right pinky
    0.1, -1.0, 0.0,  # left index
    -0.1, -1.0, 0.0,  # right index
    0.1, -1.1, 0.1,  # left thumb
    -0.1, -1.1, 0.1,  # right thumb
    0.0, -1.5, 0.0,  # left hip
    0.0, -1.5, 0.0,  # right hip
    0.1, -2.0, 0.0,  # left knee
    -0.1, -2.0, 0.0,  # right knee
    0.15, -2.5, 0.1,  # left ankle
    -0.15, -2.5, 0.1,  # right ankle
    0.2, -2.6, 0.0,  # left heel
    -0.2, -2.6, 0.0,  # right heel
    0.2, -2.7, 0.2,  # left foot index
    -0.2, -2.7, 0.2,  # right foot index
]

# Reshape the landmarks into a 33x3 array
landmarks_3d = np.array(landmarks_3d).reshape(-1, 3)
print(landmarks_3d)
# Define connections between landmarks
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),  # Left eye
    (0, 4), (4, 5), (5, 6), (6, 8),  # Right eye
    (0, 9), (9, 10),  # Mouth
    (11, 12),  # Shoulders
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),  # Left arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),  # Right arm
    (11, 23), (23, 25), (25, 27), (27, 29), (27, 31),  # Left leg
    (12, 24), (24, 26), (26, 28), (28, 30), (28, 32),  # Right leg
    (23, 24)  # Hips
]

# Create 3D plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot landmarks
ax.scatter(landmarks_3d[:, 0], landmarks_3d[:, 1], landmarks_3d[:, 2])

# Plot connections
for connection in CONNECTIONS:
    start, end = connection
    ax.plot([landmarks_3d[start, 0], landmarks_3d[end, 0]],
            [landmarks_3d[start, 1], landmarks_3d[end, 1]],
            [landmarks_3d[start, 2], landmarks_3d[end, 2]])

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('MediaPipe 3D Pose Landmarks')

# Adjust the view angle
ax.view_init(elev=90, azim=-90)

# Show the plot
plt.show()