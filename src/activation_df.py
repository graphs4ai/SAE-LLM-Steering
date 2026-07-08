import pandas as pd
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple

METADATA_COLS = frozenset({"class", "pair_key", "row_id", "statement"})


class ActivationDataFrame:
    """
    Efficiently accumulates activation vectors and converts them to a pandas DataFrame.
    """

    def __init__(self, layers: List[int], d_features: int):
        """
        Args:
            layers: List of layer indices used for extraction (sorted).
            d_features: Dimension of activations per layer (d_model or d_sae).
        """
        # Store data in lists first to avoid expensive DataFrame resizing operations
        self._activations_list: List[np.ndarray] = []
        self._labels_list: List[str] = []
        self._metadata_lists: Dict[str, List[object]] = {}
        self._df: Optional[pd.DataFrame] = None
        self._layers = sorted(layers)
        self._d_features = d_features

    def feature_column_names(self) -> List[str]:
        """Return ordered SAE feature column names (layer_m-feature_n)."""
        cols: List[str] = []
        for layer in self._layers:
            for feature in range(self._d_features):
                cols.append(f"layer_{layer}-feature_{feature}")
        return cols

    def get_array(self) -> Tuple[np.ndarray, List[str]]:
        """
        Return activation matrix as numpy array.

        Returns:
            X: float32 array of shape (n_samples, n_features).
            feature_columns: ordered column names matching X's second axis.
        """
        feature_columns = self.feature_column_names()
        if self._df is not None:
            meta_in_df = [c for c in self._df.columns if c in METADATA_COLS]
            feature_cols = [c for c in self._df.columns if c not in meta_in_df]
            if not feature_cols:
                feature_cols = feature_columns
            return self._df[feature_cols].to_numpy(dtype=np.float32), feature_cols

        if not self._activations_list:
            return np.empty((0, len(feature_columns)), dtype=np.float32), feature_columns

        return np.concatenate(self._activations_list, axis=0), feature_columns

    def get_metadata(self) -> pd.DataFrame:
        """Return per-row metadata (class, pair_key, etc.) without feature columns."""
        df = self.get_df()
        if df.empty:
            return pd.DataFrame()
        meta_cols = [c for c in df.columns if c in METADATA_COLS or c in self._metadata_lists]
        return df[meta_cols].reset_index(drop=True)

    def add_batch(
        self,
        activations: torch.Tensor,
        labels: List[str],
        metadata: Optional[Dict[str, List[object]]] = None,
    ):
        """
        Adds a batch of activations and labels.

        Args:
            activations (torch.Tensor): Shape [batch_size, n_layers * d_features].
                IMPORTANT: The wrapper returns [batch, seq_len, n_layers * d_features].
                You must select a specific token (e.g., the last token) before passing it here.
            labels (List[str]): List of class labels for the batch.
            metadata: Optional dict of per-row metadata lists with the same
                length as the batch, e.g. {"pair_key": [...], "row_id": [...]}.
        """
        if activations.ndim != 2:
            raise ValueError(
                f"Expected activations shape [batch, n_layers * d_features], "
                f"got {activations.shape}. "
                "Did you forget to select a specific token (e.g., [:, -1, :])?"
            )

        if activations.shape[0] != len(labels):
            raise ValueError(
                f"Batch size mismatch: {activations.shape[0]} activations vs {len(labels)} labels")

        # Detach from graph, move to CPU, and convert to numpy float32
        # float32 is standard for pandas; float16 saves memory but can be tricky with some pandas backends
        acts_np = activations.detach().cpu().to(dtype=torch.float32).numpy()

        self._activations_list.append(acts_np)
        self._labels_list.extend(labels)
        if metadata:
            for key, values in metadata.items():
                if len(values) != activations.shape[0]:
                    raise ValueError(
                        f"Metadata column {key!r} has {len(values)} rows, "
                        f"expected {activations.shape[0]}."
                    )
                self._metadata_lists.setdefault(key, []).extend(values)
        self._df = None  # Invalidate cached DataFrame

    def get_df(self) -> pd.DataFrame:
        """
        Constructs and returns the pandas DataFrame.
        """
        if self._df is not None:
            return self._df

        if not self._activations_list:
            return pd.DataFrame()

        # 1. Concatenate all numpy arrays (Much faster than creating DF row by row)
        X = np.concatenate(self._activations_list, axis=0)

        # 2. Create column names (layer_m-feature_n)
        feature_cols = []
        for layer in self._layers:
            for feature in range(self._d_features):
                feature_cols.append(f"layer_{layer}-feature_{feature}")

        # 3. Create DataFrame
        self._df = pd.DataFrame(X, columns=feature_cols)

        # 4. Add class column
        self._df["class"] = self._labels_list
        for key, values in self._metadata_lists.items():
            self._df[key] = values

        return self._df

    def save(self, filepath: str):
        """
        Saves the DataFrame. Strongly recommend using .parquet extension.
        """
        df = self.get_df()
        if filepath.endswith(".parquet"):
            df.to_parquet(filepath, index=False)
        else:
            # CSV will be very slow for 4000+ columns
            df.to_csv(filepath, index=False)
