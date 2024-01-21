import argparse
from src.app import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--object", type=str)
    args = parser.parse_args()

    main(
        mesh_path=args.object
    )