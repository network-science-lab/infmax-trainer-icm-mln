from os import cpu_count
from typing import Any


def get_num_workers(config: dict[str, Any]) -> int:
    num_workers = config["data"]["num_workers"]
    return num_workers if num_workers != -1 else cpu_count() - 1
