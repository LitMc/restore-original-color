import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def parse_color_code(code: str):
    """
    Parse a color code like "#RRGGBB" into an (R,G,B) tuple in [0..255].
    """
    code = code.strip()
    if code.startswith("#"):
        code = code[1:]
    if len(code) != 6:
        raise ValueError(f"Invalid color code: #{code}")
    r = int(code[0:2], 16)
    g = int(code[2:4], 16)
    b = int(code[4:6], 16)
    return (r, g, b)


def load_images_and_average(image_paths, width, height):
    """
    1) 画像群を読み込み:
       - もし (im.width, im.height) != (width, height) なら
         * radius=8.0 でガウスぼかし
         * 256x192 (指定の width×height) へ BICUBIC リサイズ
    2) 各画素(R,G,B)を足し合わせ
    3) 平均を返す: shape=(H,W,3), range=0..255 (float)
    """
    sum_array = None
    count = 0

    for path in image_paths:
        im = Image.open(path).convert("RGB")

        # 入力画像サイズが輝度マップの指定サイズと異なっても対応できるようリサイズ
        if (im.width != width) or (im.height != height):
            # 縮小後にモアレが生じないようにガウスぼかし
            im = im.filter(ImageFilter.GaussianBlur(radius=8.0))
            # モアレが生じないようにBICUBICでなだらかにリサイズ
            im = im.resize((width, height), resample=Image.Resampling.BICUBIC)

        arr = np.array(im, dtype=np.float32)  # shape=(H,W,3), 0..255
        if sum_array is None:
            sum_array = arr
        else:
            sum_array += arr
        count += 1

    if count == 0:
        raise RuntimeError("No images found to process.")

    avg_array = sum_array / count  # shape=(H,W,3), float
    return avg_array


def create_gain_map(avg_array: np.ndarray, target_color: tuple, max_gain: float):
    """
    avg_array: shape=(H,W,3), range=0..255 float
    target_color: (R_t, G_t, B_t) in 0..255
    1) For each pixel => gain_c = target_color[c] / measured_c
    2) clamp to [0..max_gain], then map to [0..255]
    """
    eps = 1e-6
    rt, gt, bt = target_color  # still 0..255
    gain_map = np.zeros_like(avg_array, dtype=np.float32)

    # ratio = (target_channel) / (measured_channel + eps)
    gain_map[:, :, 0] = rt / (avg_array[:, :, 0] + eps)
    gain_map[:, :, 1] = gt / (avg_array[:, :, 1] + eps)
    gain_map[:, :, 2] = bt / (avg_array[:, :, 2] + eps)

    # clamp to [0..max_gain]
    np.clip(gain_map, 0.0, max_gain, out=gain_map)

    # map [0..max_gain] => [0..255]
    gain_map_255 = (gain_map / max_gain) * 255.0
    gain_map_255 = np.clip(gain_map_255, 0, 255).astype(np.uint8)

    return gain_map_255


def main():
    parser = argparse.ArgumentParser(
        description="Create a color gain map from multiple images. "
        "If input images differ from the specified width/height, "
        "they are blurred (radius=8) then BICUBIC-resized."
    )
    parser.add_argument(
        "--image_path",
        required=True,
        type=str,
        help="File or directory. If directory, use all valid images within.",
    )
    parser.add_argument(
        "--output_path", required=True, type=str, help="Output PNG for the gain map."
    )
    parser.add_argument(
        "--target_color",
        required=True,
        type=str,
        help="Target color (hex code), e.g. #FFFFFF",
    )
    parser.add_argument(
        "--max_gain",
        type=float,
        default=4.0,
        help="Max gain factor. Gains above this are clamped.",
    )
    parser.add_argument("--width", type=int, default=256, help="Image width")
    parser.add_argument("--height", type=int, default=192, help="Image height")

    args = parser.parse_args()

    in_path = Path(args.image_path)
    out_path = Path(args.output_path)

    # collect image files
    image_files = []
    if in_path.is_dir():
        for p in in_path.iterdir():
            if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tif"]:
                image_files.append(p)
    else:
        image_files.append(in_path)

    if not image_files:
        raise FileNotFoundError(f"No valid images found in {in_path}")

    # load+average (with the new "blur+resize if needed" logic)
    avg_array = load_images_and_average(image_files, args.width, args.height)

    # parse target_color
    targ_col = parse_color_code(args.target_color)

    # create gain map
    gain_map_arr = create_gain_map(avg_array, targ_col, args.max_gain)

    # 出力先ディレクトリが存在しない場合は作成
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # save
    out_img = Image.fromarray(gain_map_arr, mode="RGB")
    out_img.save(out_path)
    print(f"Gain map saved to {out_path}")


if __name__ == "__main__":
    main()
