import argparse
from pathlib import Path

from PIL import Image, ImageFilter


def process_image_file(
    image_file: Path,
    output_dir: Path,
    width: int,
    height: int,
    resample_str: str,
    blur_radius: float,
):
    """
    1) 画像を読み込む
    2) 必要ならぼかし (PILの .filter(ImageFilter.GaussianBlur))
    3) リサイズ (PILの .resize を直接呼び出し)
    4) 出力ファイルに保存
    """
    print(f"Processing {image_file.name} ...", end=" ")
    img = Image.open(image_file).convert("RGB")

    # 液晶画面の赤青緑のサブピクセルを混ぜて白色に統一するためにぼかす
    if blur_radius > 0.0:
        print(f"Blurring radius={blur_radius}", end=" ")
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # 細かすぎる輝度マップで模様が生まれないようにリサイズ
    resample_map = {
        "NEAREST": Image.Resampling.NEAREST,
        "BILINEAR": Image.Resampling.BILINEAR,
        "BICUBIC": Image.Resampling.BICUBIC,
        "LANCZOS": Image.Resampling.LANCZOS,
        "HAMMING": Image.Resampling.HAMMING,
        "BOX": Image.Resampling.BOX,
    }
    if resample_str not in resample_map:
        raise ValueError(f"Unsupported resample method: {resample_str}")

    resample_method = resample_map[resample_str]
    print(
        f"Resizing from {img.size} -> ({width},{height}), method={resample_str}",
        end=" ",
    )
    img = img.resize((width, height), resample=resample_method)

    out_name = f"{image_file.stem}_{resample_str}"
    if blur_radius > 0.0:
        out_name += f"_blur{blur_radius}"
    out_path = output_dir / f"{out_name}.png"

    img.save(out_path)
    print(f"Saved -> {out_path}")


def process_images_in_path(
    image_path: Path,
    output_path: Path,
    width: int,
    height: int,
    resample: str,
    blur_radius: float,
):
    """
    - 指定パスがディレクトリなら内包する画像すべてを処理
    - ファイルなら単一処理
    """
    if image_path.is_dir():
        for img_file in image_path.iterdir():
            if img_file.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tif"]:
                process_image_file(
                    img_file, output_path, width, height, resample, blur_radius
                )
    else:
        process_image_file(
            image_path, output_path, width, height, resample, blur_radius
        )


def main():
    parser = argparse.ArgumentParser(
        description="Resize images and optionally blur them."
    )
    parser.add_argument(
        "--image_path", required=True, type=str, help="Path to image or folder"
    )
    parser.add_argument(
        "--output_path", required=True, type=str, help="Directory for output"
    )
    parser.add_argument("--width", type=int, required=True, help="Target width")
    parser.add_argument("--height", type=int, required=True, help="Target height")

    resample_methods = [
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
        default="BICUBIC",
        choices=resample_methods,
        help="Resampling method (e.g. BICUBIC)",
    )

    parser.add_argument(
        "--blur", type=float, default=0.0, help="Gaussian blur radius. 0 for no blur"
    )

    args = parser.parse_args()

    in_path = Path(args.image_path)
    out_path = Path(args.output_path)
    out_path.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    if args.resample == "ALL":
        for resample in resample_methods[:-1]:
            process_images_in_path(
                image_path=in_path,
                output_path=out_path,
                width=args.width,
                height=args.height,
                resample=resample,
                blur_radius=args.blur,
            )
    else:
        process_images_in_path(
            image_path=in_path,
            output_path=out_path,
            width=args.width,
            height=args.height,
            resample=args.resample,
            blur_radius=args.blur,
        )


if __name__ == "__main__":
    main()
