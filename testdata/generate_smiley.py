import numpy as np
import matplotlib.pyplot as plt

def generate_smiley_data(show=False):
    rng = np.random.default_rng(42)

    # Two compact eyes
    left_eye = rng.normal(
        loc=(-0.5, 0.5), scale=0.06, size=(80, 2)
    )
    right_eye = rng.normal(
        loc=(0.5, 0.5), scale=0.06, size=(80, 2)
    )

    # Curved mouth
    theta = np.linspace(np.pi, 2 * np.pi, 240)
    mouth = np.column_stack((
        0.8 * np.cos(theta),
        0.6 * np.sin(theta),
    ))
    mouth += rng.normal(scale=0.02, size=mouth.shape)

    # Outer circle
    theta_circle = np.linspace(0, 2 * np.pi, 360, endpoint=False)

    circle = np.column_stack((
        1.0 * np.cos(theta_circle),
        1.0 * np.sin(theta_circle),
    ))
    circle += rng.normal(scale=0.01, size=circle.shape)

    # Random background points
    background = rng.uniform(-1.1, 1.1, size=(30, 2))

    X_smiley = np.vstack((
        left_eye, right_eye, mouth, circle, background
    ))

    # Shuffle so the algorithm cannot rely on the order of the shapes
    rng.shuffle(X_smiley)

    if show is True:
        plt.scatter(X_smiley[:, 0], X_smiley[:, 1], s=12)
        plt.axis("equal")
        plt.show()

    return X_smiley