"""Method to select seeds with multi_node2vec and k-means."""

import tempfile

from pathlib import Path
from typing import Any

import docker
import network_diffusion as nd
import numpy as np
import pandas as pd

from infmax_models.utils import k_means


class MultiNode2VecKMeans:  # TODO: even if it's not necessary, consider modifying this class to be a child of nn.module
    """Method to select seeds with multi_node2vec and k-means."""

    docker_image = "multi-node2vec"
    docker_platform = "linux/amd64"
    docker_io_dir = "/data"
    
    def __init__(self, multi_node2vec: dict[str, Any], k_means: dict[str, Any]) -> None:
        """Initialise the object."""
        self.docker_client = docker.from_env()
        self.docker_client.images.get(self.docker_image)
        self.mn2v_pms = multi_node2vec
        self.km_pms = k_means

    @staticmethod
    def export_network(data_dir: str, network: nd.MultilayerNetworkTorch) -> None:
        """Export network into csv files for each layer."""
        for l_idx, l_name in enumerate(network.layers_order):
            pd.DataFrame(
                network.adjacency_tensor[l_idx, ...].to_dense()).to_csv(f"{data_dir}/{l_name}.csv"
            )

    @staticmethod
    def get_multi_node2vec_src_path() -> str:
        """Obtain a path to the sources of multi_node2vec."""
        candidate_path = Path(__file__).parent / "multi_node2vec"
        if not candidate_path.exists():
            raise FileNotFoundError(
                f"Couldn't find multi_node2vec sources - {candidate_path} doesn't exits!"
            )
        return str(candidate_path)
    
    @staticmethod
    def get_dim_size(network: nd.MultilayerNetworkTorch) -> float:
        """A simple heuristic that sets embedding dimension as a sqrt of number of nodes."""
        num_actors = network.nodes_mask.shape[1]
        return np.floor(np.sqrt(num_actors)).astype(int).item()

    def get_python_cmd(self, network: nd.MultilayerNetworkTorch):
        """Get command to execute in the docker."""
        return f"python multi_node2vec.py --dir {self.docker_io_dir} --output {self.docker_io_dir} \
            --d {self.get_dim_size(network) if self.mn2v_pms['d'] == 'auto' else self.mn2v_pms['d']} \
            --window_size {self.mn2v_pms['window_size']} --n_samples {self.mn2v_pms['n_samples']} \
            --rvals {self.mn2v_pms['rvals']} --pvals {self.mn2v_pms['pvals']} \
            --thresh {self.mn2v_pms['thresh']} --qvals {self.mn2v_pms['qvals']}"

    def multi_node2vec(self, data_dir: str, network: nd.MultilayerNetworkTorch) -> None:
        """Prepare embedding of the given input."""
        multi_node2vec_src_path = self.get_multi_node2vec_src_path()
        cmd_python = self.get_python_cmd(network=network)
        print(f"Running multi_node2vec with args: {cmd_python}")
        self.export_network(data_dir=data_dir, network=network)
        container = self.docker_client.containers.run(
            image=self.docker_image,
            remove=True,
            detach=True,
            volumes=[f"{multi_node2vec_src_path}:/app", f"{data_dir}:{self.docker_io_dir}"],
            platform=self.docker_platform,
            command=cmd_python,
        )
        for line in container.attach(stdout=True, stream=True, logs=True):
            print(line.decode("utf-8"))

    def __call__(self, network: nd.MultilayerNetworkTorch) -> np.ndarray:
        """Select seeds using multi_node2vec and kmeans."""
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Temporary directory: {temp_dir}")
            self.multi_node2vec(data_dir=temp_dir, network=network)
            seeds = k_means.KMeansSeedSelector(
                emb_path=f"{temp_dir}/mltn2v_results.csv",
                num_segments=self.km_pms["num_segments"],
                random_state=self.km_pms["random_state"],
                experiment_name=self.km_pms["experiment_name"],
            )(visualise=self.km_pms["visualise"])
            return seeds

# cmd_docker = "docker run --rm -v ./multi_node2vec:/app -v ./toy_network:/data --platform linux/amd64 multi-node2vec"
# cmd_python = "python multi_node2vec.py --dir /data --output /data --d 2 --window_size 10 --n_samples 1 --rvals 0.25 --pvals 1 --thresh 0.5 --qvals 0.5"
