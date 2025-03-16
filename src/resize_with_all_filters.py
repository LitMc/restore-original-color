import argparse

from resize import resize_and_save_images

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="resize_with_all_filters.py", description="Resize images with all filters"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        help="Path to the image file or directory",
        required=True,
    )
    parser.add_argument(
        "--output_path", type=str, help="Output directory", required=True
    )
    parser.add_argument("--width", type=int, help="Target width", required=True)
    parser.add_argument("--height", type=int, help="Target height", required=True)
    args = parser.parse_args()
    resampling_filters = ["NEAREST", "BILINEAR", "BICUBIC", "LANCZOS", "HAMMING", "BOX"]
    for resample in resampling_filters:
        resize_and_save_images(
            args.image_path, args.output_path, args.width, args.height, resample
        )
