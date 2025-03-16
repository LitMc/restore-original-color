import argparse
from pathlib import Path

from PIL import Image


def resize_image(
    image: Image,
    target_width: int,
    target_height: int,
    resample: str = Image.Resampling.BICUBIC.name,
) -> Image:
    return image.resize(
        (target_width, target_height), resample=Image.Resampling[resample]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="resize.py", description="Resize images")
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
    parser.add_argument(
        "--resample",
        type=str,
        help="Resampling method",
        choices=["NEAREST", "BILINEAR", "BICUBIC", "LANCZOS", "HAMMING", "BOX"],
        default="BICUBIC",
    )
    args = parser.parse_args()
    image_path = Path(args.image_path)
    output_path = Path(args.output_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not output_path.exists():
        output_path.mkdir(parents=True)
    if image_path.is_dir():
        # Resize all images in the directory
        for image_file in image_path.iterdir():
            image = Image.open(image_file).convert("RGB")
            resize_image(image, args.width, args.height, args.resample).save(
                output_path / f"{image_file.stem}_{args.resample}.png"
            )
    else:
        # Resize single image
        image = Image.open(image_path).convert("RGB")
        resize_image(image, args.width, args.height, args.resample).save(
            output_path / f"{image_path.stem}_{args.resample}.png"
        )
