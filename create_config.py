import argparse
from typing import Callable, Any

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate configuration file")
    parser.add_argument("--dev", action="store_true", help="Use development configuration")
    return parser.parse_args()


def generate_worker_config(dev: bool = False) -> dict[str, Any]:
    config: dict[str, Any] = {}
    config["PORT"] = input("Enter the port to run the orchestrator server on (default: 6920): ") or "6920"
    return config


def write_config_to_file(config: dict[str, Any], env: str) -> None:
    filename = f".{env}.env"
    with open(filename, "w") as f:
        for key, value in config.items():
            f.write(f"{key}={value}\n")

if __name__ == "__main__":
    args = parse_args()
    print("Welcome to the configuration generator!")

    config = generate_worker_config(dev=args.dev)
    name = "vali"

    write_config_to_file(config, name)
    print(f"Configuration has been written to .{name}.env")
