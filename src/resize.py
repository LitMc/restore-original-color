import argparse
from pathlib import Path

from PIL import Image


def __resize_image(
    image: Image,
    target_width: int,
    target_height: int,
    resample: str = Image.Resampling.BICUBIC.name,
) -> Image:
    print(f"{image.size} -> {target_width}x{target_height}", end=" ")
    return image.resize(
        (target_width, target_height), resample=Image.Resampling[resample]
    )


def __resize_and_save_image_file(
    image_file: Path,
    output_path: Path,
    width: int,
    height: int,
    resample: str = Image.Resampling.BICUBIC.name,
) -> Image:
    print(f"{image_file.name},", end=" ")
    image = Image.open(image_file).convert("RGB")
    resized_image = __resize_image(image, width, height, resample)
    resized_image.save(output_path / f"{image_file.stem}_{resample}.png")
    print(f", {output_path / f'{image_file.stem}_{resample}.png'}")


def resize_and_save_images(image_path, output_path, width, height, resample):
    print(f"Resizing images using {resample} resampling filter")
    image_path = Path(image_path)
    output_path = Path(output_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not output_path.exists():
        output_path.mkdir(parents=True)
    if image_path.is_dir():
        # Resize all images in the directory
        for image_file in image_path.iterdir():
            __resize_and_save_image_file(
                image_file, output_path, width, height, resample
            )
    else:
        # Resize single image
        __resize_and_save_image_file(image_path, output_path, width, height, resample)


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
    resampling_methods = [
        "NEAREST",
        "BILINEAR",
        "BICUBIC",
        "LANCZOS",
        "HAMMING",
        "BOX",
        "ALL",
    ]
    parser.add_argument(
        "--resample",
        type=str,
        help="Resampling method",
        choices=resampling_methods,
        default="BICUBIC",
    )
    args = parser.parse_args()
    if args.resample == "ALL":
        for resample in resampling_methods[:-1]:
            resize_and_save_images(
                args.image_path, args.output_path, args.width, args.height, resample
            )
    else:
        resize_and_save_images(
            args.image_path, args.output_path, args.width, args.height, args.resample
        )
