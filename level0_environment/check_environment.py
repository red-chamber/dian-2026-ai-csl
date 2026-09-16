import platform
import torch


def main():
    print("Python:", platform.python_version())
    print("PyTorch:", torch.__version__)
    print("PyTorch CUDA runtime:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("GPU count:", torch.cuda.device_count())

        x = torch.randn(1024, 1024, device="cuda")
        print("GPU calculation result:", x.mean().item())


if __name__ == "__main__":
    main()
