import numpy as np
import faiss
from typing import Tuple, Optional
import json

with open("config.json", "r") as f:
    config = json.load(f)

EMB_PER_USER = config["EMB_PER_USER"]
EMB_DIM = config["EMB_DIM"]

class FaissIndex:
    def __init__(self, mode="known"):
        self.mode = mode
        self.embeddings = None
        self.ids = None
        self.index = None
        self.index_metric = None

    def build_faiss(self, database, use_exact: bool = False) -> None:
        embeddings_list = []
        ids_list = []
        if isinstance(database, list):
            for user_id, emb in database:
                emb = np.array(emb, dtype=np.float32)
                if emb.shape[0] != EMB_DIM:
                    raise ValueError(
                        f"Expected embedding with dimension {EMB_DIM} for user {user_id}, got {emb.shape}"
                    )
                embeddings_list.append(emb)
                ids_list.append((user_id, 0))
        else:
            for user_id, embs in database.items():
                embs = np.array(embs, dtype=np.float32)
                if self.mode == "known":
                    # Allow variable number of embeddings (1 to EMB_PER_USER)
                    if embs.ndim != 2 or embs.shape[0] < 1 or embs.shape[0] > EMB_PER_USER or embs.shape[1] != EMB_DIM:
                        raise ValueError(
                            f"Expected embeddings with shape (1-{EMB_PER_USER}, {EMB_DIM}) for user {user_id}, got {embs.shape}"
                        )
                if self.mode == "unknown":
                    if embs.shape[1] != EMB_DIM:
                        raise ValueError(
                            f"Expected embeddings with dimension {EMB_DIM} for user {user_id}, got {embs.shape}"
                        )
                for sub_idx, emb in enumerate(embs):
                    embeddings_list.append(emb)
                    ids_list.append((user_id, sub_idx))

        if len(embeddings_list) == 0:
            print("[x] No embeddings, FAISS index not built.")
            self.index = None
            self.ids = []
            self.embeddings = None
            self.index_metric = None
            return

        embeddings = np.vstack(embeddings_list).astype(np.float32)
        faiss.normalize_L2(embeddings)

        if use_exact or embeddings.shape[0] <= 5000:
            self.index = faiss.IndexFlatIP(EMB_DIM)
            self.index_metric = "ip"
        else:
            try:
                self.index = faiss.IndexHNSWFlat(EMB_DIM, 32, faiss.METRIC_INNER_PRODUCT)
                self.index_metric = "ip"
            except TypeError:
                self.index = faiss.IndexHNSWFlat(EMB_DIM, 32)
                self.index_metric = "l2"

            self.index.hnsw.efConstruction = 200
            self.index.hnsw.efSearch = 200

        self.index.add(embeddings)
        self.embeddings = embeddings
        self.ids = ids_list

    def search_faiss_best(
        self,
        query_embedding: np.ndarray,
        unknown_threshold: Optional[float] = None
    ) -> Tuple[Optional[int], float]:
        if self.index is None or self.ids is None or len(self.ids) == 0:
            return None, 0.0

        query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        top_k_search = min(10, len(self.ids))
        distances, indices = self.index.search(query_embedding, top_k_search)
        best_system_id = None
        best_confidence = 0.0
        for rank, idx in enumerate(indices[0]):
            if idx == -1 or idx >= len(self.ids):
                continue
            system_id, _ = self.ids[idx] if isinstance(self.ids[idx], (tuple, list)) else (self.ids[idx], None)
            distance = float(distances[0][rank])
            confidence = distance if self.index_metric == "ip" else 1.0 - 0.5 * distance
            if confidence > best_confidence:
                best_confidence = confidence
                best_system_id = system_id

        if unknown_threshold is not None and best_confidence < unknown_threshold:
            return None, best_confidence

        return best_system_id, best_confidence
