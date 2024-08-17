"""A naive seed set selector which just clusters embedding space and seeks for central points."""

import pathlib
import warnings

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min, silhouette_score

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_closest_vec_to_X(
        coords: np.ndarray, labels: np.ndarray, X: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return ID of the closest point to gixen X in the euclidean space.

    :param coords: coordinates of the available points
    :param labels: unique labels of the available points (indicces are aligned with `coords`)
    :param X: a point of reference
    :return: label and coordinates of the closest point to x
    """
    seg_seed_id, _ = pairwise_distances_argmin_min(X, coords)
    return labels[seg_seed_id], coords[seg_seed_id]


class KMeansSeedSelector:

    def __init__(
        self,
        emb_path: pathlib.Path,
        nb_seeds: int,
        random_state: int = 42,
        experiment_name: str = "experiment"
    ) -> None:
        """
        Initialise the object.

        :param emb_path: path to the CSV file with embedding coords
        :param nb_seeds: number of cluster to divide space in, i.e. number of
            seeds to extract
        :param random_state: RNG for k-means algorithm, defaults to 42
        :param experiment_name: name of the experiment for the optional visualisation
        """
        if nb_seeds < 2:
            raise ValueError("This method cannot select less seeds than 2!")
        self.embeddings = pd.read_csv(emb_path, header=None)
        self.nb_seeds = nb_seeds
        self.random_state = random_state
        self.experiment_name = experiment_name
        self._emb_ids = self.embeddings.to_numpy()[:, 0]  # TODO: check if labels are unique!
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
    def extract_seeds(kmeans: KMeans, emb_vectors: np.ndarray, emb_labels: np.ndarray) -> np.ndarray:
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

            seg_seed_id, seg_seed_coords = get_closest_vec_to_X(
                segment_vectors, segment_labels, segment_centre
            )
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
        for x, y, s in zip(self._emb_vectors[:, 0]+0.01, self._emb_vectors[:, 1]+0.01, self._emb_ids):
            ax.text(x=x, y=y, s=s)
        ax.scatter( # plot centroids
            x=kmeans.cluster_centers_[:, 0],
            y=kmeans.cluster_centers_[:, 1],
            color="red",
            s=7,
            label="centroids",
        )
        # TODO: plot convex hull https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html
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
            f"multi-node2vec & k-means\n {self.experiment_name}, num seeds: {self.nb_seeds}\n"
            f"found actors: {seeds_ids.tolist()}"
        )
        plt.show()

    def __call__(self, visualise: bool = False) -> list[int]:
        """Select seeds from given embedded nodes."""
        kmeans = self.clusterise(x=self._emb_vectors, num_segments=self.nb_seeds, random_state=self.random_state)
        seeds = self.extract_seeds(kmeans=kmeans, emb_vectors=self._emb_vectors, emb_labels=self._emb_ids)
        if visualise:
            self._visualise(seeds_ids=seeds, kmeans=kmeans)
        return seeds.tolist()


class KMeansAutoSeedSelector(KMeansSeedSelector):

    def __init__(
        self,
        emb_path: pathlib.Path,
        nb_seeds: int,
        max_nb_segments: int,
        random_state: int = 42,
        experiment_name: str = "experiment"
    ) -> None:
        """
        Initialise the object.

        :param emb_path: path to the CSV file with embedding coords
        :param nb_seeds: number of seeds to extract
        :param max_segments: number of cluster to divide space in
        :param random_state: RNG for k-means algorithm, defaults to 42
        :param experiment_name: name of the experiment for the optional visualisation
        """
        self.embeddings = pd.read_csv(emb_path, header=None)
        self.nb_seeds = nb_seeds
        self.max_nb_segments = max_nb_segments
        self.random_state = random_state
        self.experiment_name = experiment_name
        self._emb_ids = self.embeddings.to_numpy()[:, 0]
        self._emb_vectors = self.embeddings.to_numpy()[:, 1:]

    def _visualise(self, silhouette_coefficients: list[float]) -> None:
        """Plot a visualisaiton of the division."""
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(
            np.arange(start=2, stop=len(silhouette_coefficients) + 2),
            silhouette_coefficients,
            color="green",
            marker=".",
            linestyle='-',
        )
        ax.set_xlabel("Number of clusters")
        ax.set_ylabel("Silhouette Coefficient")
        fig.set_size_inches(6, 6)
        plt.show()

    def extract_seeds(self, kmeans: KMeans) -> np.ndarray:
        seeds_ids = []
        avail_vectors = np.copy(self._emb_vectors)  # coords of vectors
        avail_ids = np.copy(self._emb_ids)  # ids of vectors, i.e. nodes' names
        avail_labels = np.copy(kmeans.labels_)  # labels of vectors, i.e. clusters they are assigned to

        while len(seeds_ids) < self.nb_seeds:  # TODO: sort clusters by size!!!
            for segment in np.unique(kmeans.labels_):

                # get vectors tht were assigned to this segment and the centre
                segment_vectors = avail_vectors[avail_labels==segment]
                segment_labels = avail_ids[avail_labels==segment]
                segment_centre = kmeans.cluster_centers_[segment, :][np.newaxis, :]

                # find a vector that is closest to the centre of the segment
                seed_id, _ = get_closest_vec_to_X(segment_vectors, segment_labels, segment_centre)
                seeds_ids.append(seed_id)
                print(seeds_ids)
                if len(seeds_ids) >= self.nb_seeds:
                    break

                # remove that vector from the list of available vectors
                seed_idx = np.where(avail_ids == seed_id)[0].item()
                avail_vectors = np.delete(avail_vectors, seed_idx, axis=0)
                avail_ids = np.delete(avail_ids, seed_idx, axis=0)
                avail_labels = np.delete(avail_labels, seed_idx, axis=0)

        seeds_ids = np.array(seeds_ids)
        return seeds_ids.squeeze().squeeze().astype(int)

    def __call__(self, visualise: bool = False) -> list[int]:
        """Select seeds from given embedded nodes."""
        split_silhouettes = []
        split_models = []
        for k in range(2, self.max_nb_segments + 1):
            kmeans = self.clusterise(self._emb_vectors, k, self.random_state)
            split_score = silhouette_score(self._emb_vectors, kmeans.labels_)
            split_silhouettes.append(split_score)
            split_models.append(kmeans)
        if visualise:
            self._visualise(silhouette_coefficients=split_silhouettes)
        optimal_model = split_models[np.argmax(split_silhouettes)]
        print(f"Optimal split into: {np.argmax(split_silhouettes) + 2} clusters.")
        seeds = self.extract_seeds(kmeans=optimal_model)
        if visualise:
            super()._visualise(seeds_ids=seeds, kmeans=optimal_model)
        return seeds.tolist()
