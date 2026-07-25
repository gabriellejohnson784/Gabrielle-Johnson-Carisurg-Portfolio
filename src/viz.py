from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def save_confusion_matrix(model, X_test, y_test, path, title=None, cmap="RdPu"):
    """Predict on X_test and save a confusion matrix plot as a PNG.

    Rows are true ESI, columns are predicted ESI; the diagonal is correct.

    Returns
    -------
    Path to the saved PNG.
    """
    pred = np.asarray(model.predict(X_test)).ravel()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax, cmap=cmap)
    ax.set_title(title or "")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
