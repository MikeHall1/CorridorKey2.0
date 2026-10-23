"""Owned subprocess backend availability probe; never loads models."""
import json


def main() -> None:
    import torch

    available = torch.cuda.is_available()
    print(json.dumps({'cuda_available': available, 'torch': torch.__version__, 'cuda_build': torch.version.cuda}))


if __name__ == '__main__':
    main()
