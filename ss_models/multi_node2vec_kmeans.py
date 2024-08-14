"""Method to select seeds with multi_node2vec and k-means."""

import tempfile

from pathlib import Path
from typing import Any

import docker
import network_diffusion as nd
import pandas as pd

from ss_models.utils import k_means


class MultiNode2VecKMeans:

    docker_image = "multi-node2vec"
    docker_platform = "linux/amd64"
    
    def __init__(self, net: nd.MultilayerNetworkTorch) -> None:
        """Initialise the object."""
        self.docker_client = docker.from_env()
        self.docker_client.images.get(self.docker_image)
        self.net = net

    def export_network(self, data_dir: str) -> None:
        """Export network into csv files for each layer."""
        for l_idx, l_name in enumerate(self.net.layers_order):
            pd.DataFrame(
                self.net.adjacency_tensor[l_idx, ...].to_dense()).to_csv(f"{data_dir}/{l_name}.csv"
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
    def get_python_cmd():
        """Get command to execute in the docker."""
        return "python multi_node2vec.py --dir /data --output /data --d 2 --window_size 10 --n_samples 1 --rvals 0.25 --pvals 1 --thresh 0.5 --qvals 0.5"

    def multi_node2vec(self, data_dir: str) -> None:
        """Prepare embedding of the given input."""
        multi_node2vec_src_path = self.get_multi_node2vec_src_path()
        cmd_python = self.get_python_cmd()
        container = self.docker_client.containers.run(
            image=self.docker_image,
            remove=True,
            detach=True,
            volumes=[f"{multi_node2vec_src_path}:/app", f"{data_dir}:/data"],
            platform=self.docker_platform,
            command=cmd_python,
        )
        for line in container.attach(stdout=True, stream=True, logs=True):
            print(line.decode("utf-8"))

    def __call__(self) -> Any:
        """Select seeds using multi_node2vec and kmeans."""
        with tempfile.TemporaryDirectory() as temp_dir:
            print(temp_dir)
            self.export_network(data_dir=temp_dir)
            self.multi_node2vec(data_dir=temp_dir)
            k_means.KMeansSeedSelector(
                emb_path=f"{temp_dir}/mltn2v_results.csv",
                num_segments=3,
                random_state=42,
                experiment_name="aaa",
            )(visualise=True)

# cmd_docker = "docker run --rm -v ./multi_node2vec:/app -v ./toy_network:/data --platform linux/amd64 multi-node2vec"
# cmd_python = "python multi_node2vec.py --dir /data --output /data --d 2 --window_size 10 --n_samples 1 --rvals 0.25 --pvals 1 --thresh 0.5 --qvals 0.5"
