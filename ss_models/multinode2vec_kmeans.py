from typing import Any
from ss_models.utils import k_means
import docker
import subprocess
from pathlib import Path


# cmd_docker = "docker run --rm -v ./multi_node2vec:/app -v ./toy_network:/data --platform linux/amd64 multi-node2vec"
# cmd_python = "python multi_node2vec.py --dir /data --output /data --d 2 --window_size 10 --n_samples 1 --rvals 0.25 --pvals 1 --thresh 0.5 --qvals 0.5"


class Multi_Node2Vec_KMeans:

    docker_image = "multi-node2vec"
    docker_platform = "linux/amd64"
    
    def __init__(self) -> None:
        self.docker_client = docker.from_env()
        self.docker_client.images.get(self.docker_image)
        self.temp_dir = "/Users/michal/Development/multi-node2vec/toy_network"

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass

    @staticmethod
    def get_multi_node2vec_src_path() -> str:
        """Obtain a path to the sources of multi_node2vec."""
        candidate_path = Path(__file__).parent / "multi_node2vec"
        if not candidate_path.exists():
            raise FileNotFoundError(
                f"Couldn't find multi_node2vec sources - {candidate_path} doesn't exits!"
            )
        return str(candidate_path)
    

    def multi_node2vec(self):
        multi_node2vec_src_path = self.get_multi_node2vec_src_path()
        cmd_python = "python multi_node2vec.py --dir /data --output /data --d 2 --window_size 10 --n_samples 1 --rvals 0.25 --pvals 1 --thresh 0.5 --qvals 0.5"
        self.docker_client.containers.run(
            image=self.docker_image,
            remove=True,
            volumes=[f"{multi_node2vec_src_path}:/app", f"{self.temp_dir}:/data"],
            platform=self.docker_platform,
            command=cmd_python
        )
