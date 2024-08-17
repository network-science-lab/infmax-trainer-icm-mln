"""A naive seed set selector which just clusters embedding space and seeks for central points."""

import pathlib
import warnings

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class KMeansSeedSelector:

    def __init__(
        self,
        emb_path: pathlib.Path,
        num_segments: int,
        random_state: int = 42,
        experiment_name: str = "experiment"
    ) -> None:
        """
        Initialise the object.

        :param emb_path: path to the CSV file with embedding coords
        :param num_segments: number of cluster to divide space in
        :param random_state: RNG for k-means algorithm, defaults to 42
        :param experiment_name: name of the experiment for the optional visualisation
        """
        self.embeddings = pd.read_csv(emb_path, header=None)
        self.num_segments = num_segments
        self.random_state = random_state
        self.experiment_name = experiment_name
        self._emb_labels = self.embeddings.to_numpy()[:, 0]
        self._emb_vectors = self.embeddings.to_numpy()[:, 1:]

    @staticmethod
    def clusterise(x: np.ndarray, num_segments: int, random_state: int) -> KMeans:
        """
        Divide a given vector space into `num_segments`.

        :param x: vectors to cluster, shape: num_vectors, dim_size
        :param num_segments: number of clusters to obtain
        :param random_state: RNG seed
        :return: clusterised space
        """
        kmeans = KMeans(n_clusters=num_segments, random_state=random_state)
        kmeans.fit(X=x)
        return kmeans

    @staticmethod
    def extract_seeds(kmeans: KMeans, emb_vectors: np.ndarray, emb_labels: np.ndarray) -> list[int]:
        """
        Basing on k-means division extract the most central points.

        For each cluster select a point belonging to that cluster, which is closest to the centre of
        that cluster.

        :param kmeans: clusterised vector space
        :param emb_vectors: clusterised vectors - coordinates
        :param emb_labels: clusterised vectors - labels
        :return: list of the most central points (labels)
        """
        seeds_ids = []
        seeds_coords = []

        for segment in np.unique(kmeans.labels_):

            segment_vectors = emb_vectors[kmeans.labels_==segment]
            segment_labels = emb_labels[kmeans.labels_==segment]
            segment_centre = kmeans.cluster_centers_[segment, :][np.newaxis, :]

            seg_seed_id_, _ = pairwise_distances_argmin_min(segment_centre, segment_vectors)

            seg_seed_id = segment_labels[seg_seed_id_]
            seg_seed_coords = segment_vectors[seg_seed_id_]

            seeds_ids.append(seg_seed_id)
            seeds_coords.append(seg_seed_coords)

        seeds_ids = np.array(seeds_ids)
        seeds_coords = np.array(seeds_coords).squeeze(axis=1)

        return seeds_ids.squeeze().squeeze().astype(int)

    def _visualise(self, seeds_ids: list[int], kmeans: KMeans) -> None:
        """Plot a visualisaiton of the division."""
        if self._emb_vectors.shape[-1] > 2:
            warnings.warn("Visualisation is available only for 2D space!", stacklevel=10)
            return
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.scatter( # plot embedded actors
            x=self._emb_vectors[:, 0],
            y=self._emb_vectors[:, 1],
            color="green",
            s=20,
            label="actors",
        )
        for x, y, s in zip(self._emb_vectors[:, 0]+0.01, self._emb_vectors[:, 1]+0.01, self._emb_labels):
            ax.text(x=x, y=y, s=s)
        ax.scatter( # plot centroids
            x=kmeans.cluster_centers_[:, 0],
            y=kmeans.cluster_centers_[:, 1],
            color="red",
            s=7,
            label="centroids",
        )
        ax.scatter( # mark seeds
            x=self._emb_vectors[seeds_ids, :][:, 0],
            y=self._emb_vectors[seeds_ids, :][:, 1],
            color="yellow",
            s=7,
            label="seeds"
        )
        ax.legend()
        fig.set_size_inches(6, 6)
        fig.suptitle(
            f"multi-node2vec & k-means\n {self.experiment_name}, num seeds: {self.num_segments}\n"
            f"found actors: {seeds_ids.tolist()}"
        )
        plt.show()

    def __call__(self, visualise: bool = False) -> list[int]:
        """Select seeds from given embedded nodes."""
        kmeans = self.clusterise(x=self._emb_vectors, num_segments=self.num_segments, random_state=self.random_state)
        seeds = self.extract_seeds(kmeans=kmeans, emb_vectors=self._emb_vectors, emb_labels=self._emb_labels)
        if visualise:
            self._visualise(seeds_ids=seeds, kmeans=kmeans)
        return seeds
