import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def parse_color_code(code: str):
    code = code.strip().lower()
    if code.endswith(".png"):
        code = code[:-4]
    if code.startswith("#"):
        code = code[1:]
    if len(code) != 6:
        raise ValueError(f"Invalid color code: {code}")
    r = int(code[0:2], 16)
    g = int(code[2:4], 16)
    b = int(code[4:6], 16)
    return (r, g, b)


# =========== MODEL: linear (with offset) ===============
def fit_linear_model(measured_array, target_array):
    """
    (r',g',b') = M*(r,g,b) + b  (オフセット有り)
    => param 12
    """
    K = measured_array.shape[0]
    ones = np.ones((K, 1), dtype=np.float64)
    X = np.hstack([measured_array, ones])  # shape(K,4)
    Y = target_array  # shape(K,3)
    T, residuals, rank, s = np.linalg.lstsq(X, Y, rcond=None)
    M = T[:3, :].T  # shape(3,3)
    b_ = T[3, :].T  # shape(3,)
    return (M, b_)


def apply_linear(M, b_, r_in, g_in, b_in):
    vec_in = np.array([r_in, g_in, b_in], dtype=float)
    vec_out = M @ vec_in + b_
    return np.clip(vec_out, 0, 255)


# =========== MODEL: linear_no_offset (3×3 only) =========
def fit_linear_no_offset_model(measured_array, target_array):
    """
    (r',g',b') = M*(r,g,b)   (オフセット無し)
    => param 9
    """
    # X: shape(K,3), Y: shape(K,3)
    # solve Y = X * M => M= (X^+)*Y, => np.linalg.lstsq(X, Y)
    X = measured_array  # (K,3)
    Y = target_array  # (K,3)
    T, residuals, rank, s = np.linalg.lstsq(X, Y, rcond=None)  # T: shape(3,3)
    M = T  # shape(3,3)
    return M


def apply_linear_no_offset(M, r_in, g_in, b_in):
    vec_in = np.array([r_in, g_in, b_in], dtype=float)
    vec_out = M @ vec_in
    return np.clip(vec_out, 0, 255)


# =========== MODEL: independent ===============
def fit_independent_model(measured_array, target_array):
    K = measured_array.shape[0]
    # R
    Xr = np.hstack([measured_array[:, 0:1], np.ones((K, 1))])
    Yr = target_array[:, 0]
    Tr, _, _, _ = np.linalg.lstsq(Xr, Yr, rcond=None)
    # G
    Xg = np.hstack([measured_array[:, 1:2], np.ones((K, 1))])
    Yg = target_array[:, 1]
    Tg, _, _, _ = np.linalg.lstsq(Xg, Yg, rcond=None)
    # B
    Xb = np.hstack([measured_array[:, 2:3], np.ones((K, 1))])
    Yb = target_array[:, 2]
    Tb, _, _, _ = np.linalg.lstsq(Xb, Yb, rcond=None)

    a_r, b_r = Tr
    a_g, b_g = Tg
    a_b, b_b = Tb

    M = np.array([[a_r, 0, 0], [0, a_g, 0], [0, 0, a_b]], dtype=float)
    b_ = np.array([b_r, b_g, b_b], dtype=float)
    return (M, b_)


def apply_independent(M, b_, r_in, g_in, b_in):
    vec_in = np.array([r_in, g_in, b_in], dtype=float)
    vec_out = M @ vec_in + b_
    return np.clip(vec_out, 0, 255)


# =========== MODEL: polynomial2 ==============
def poly2_feature(r, g, b_):
    r2 = r * r
    g2 = g * g
    b2 = b_ * b_
    rg = r * g
    rb = r * b_
    gb = g * b_
    return np.array([1, r, g, b_, r2, g2, b2, rg, rb, gb], dtype=float)


def fit_polynomial2_model(measured_array, target_array):
    K = measured_array.shape[0]
    Xlist = []
    for i in range(K):
        r_, g_, b_ = measured_array[i]
        Xlist.append(poly2_feature(r_, g_, b_))
    X = np.vstack(Xlist)  # (K,10)
    Y = target_array  # (K,3)
    T, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    return T  # shape(10,3)


def apply_polynomial2(T, r_in, g_in, b_in):
    feat = poly2_feature(r_in, g_in, b_in)
    vec_out = feat @ T
    return np.clip(vec_out, 0, 255)


# =========== identity ===========
def apply_identity(r_in, g_in, b_in):
    return np.array([r_in, g_in, b_in], dtype=float)


# =========== packing 64x64x64 => 512x512 ===========


def index_to_2d(z, x, y):
    tile_x = z % 8
    tile_y = z // 8
    px = tile_x * 64 + x
    py = tile_y * 64 + y
    return (px, py)


def main():
    parser = argparse.ArgumentParser(
        description="Generate 512x512 LUT from various models (including no-offset linear)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="identity",
        choices=[
            "identity",
            "linear",
            "linear_no_offset",
            "independent",
            "polynomial2",
        ],
        help="Which model to use/fitting. 'identity' => no training needed.",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        help="Directory of sample images named 'RRGGBB.png' for training",
    )
    parser.add_argument("--x", type=int, default=0, help="ROI x")
    parser.add_argument("--y", type=int, default=0, help="ROI y")
    parser.add_argument("--w", type=int, default=0, help="ROI w")
    parser.add_argument("--h", type=int, default=0, help="ROI h")

    parser.add_argument(
        "--output_png", required=True, type=str, help="Output 512x512 png LUT"
    )
    args = parser.parse_args()

    # If identity => direct pass
    if args.model == "identity":

        def apply_model(r_in, g_in, b_in):
            return apply_identity(r_in, g_in, b_in)
    else:
        if not args.input_path:
            print(f"Error: input_path is required for model={args.model}")
            return
        in_dir = Path(args.input_path)
        if not in_dir.is_dir():
            print(f"{in_dir} not dir.")
            return

        # Collect data
        meas_list = []
        targ_list = []

        pngs = sorted(in_dir.glob("*.png"))
        if not pngs:
            print("No .png in input_path")
            return

        for f in pngs:
            code = f.stem
            try:
                R, G, B = parse_color_code(code)
            except:
                print(f"skip {f.name}, parse error")
                continue
            target = (R, G, B)

            im = Image.open(f).convert("RGB")
            arr = np.array(im, dtype=float)
            H_img, W_img, _ = arr.shape
            x1 = args.x + args.w
            y1 = args.y + args.h
            if x1 > W_img or y1 > H_img or args.x < 0 or args.y < 0:
                print(f"skip {f.name}, ROI out of range")
                continue
            roi = arr[args.y : y1, args.x : x1, :]
            mean_c = roi.mean(axis=(0, 1))

            meas_list.append(mean_c)
            targ_list.append(target)
            print(f"{f.name} => measured={mean_c}, target={target}")

        if not meas_list:
            print("No valid sample data.")
            return
        measured_array = np.vstack(meas_list)
        target_array = np.vstack(targ_list)

        # Fit
        if args.model == "linear":
            M, b_ = fit_linear_model(measured_array, target_array)
            print("fitted linear => M,b")

            def apply_model(r_in, g_in, b_in):
                return apply_linear(M, b_, r_in, g_in, b_in)
        elif args.model == "linear_no_offset":
            M = fit_linear_no_offset_model(measured_array, target_array)
            print("fitted linear_no_offset => M(3x3)")

            def apply_model(r_in, g_in, b_in):
                return apply_linear_no_offset(M, r_in, g_in, b_in)
        elif args.model == "independent":
            M, b_ = fit_independent_model(measured_array, target_array)
            print("fitted independent => M,b")

            def apply_model(r_in, g_in, b_in):
                return apply_independent(M, b_, r_in, g_in, b_in)
        elif args.model == "polynomial2":
            T = fit_polynomial2_model(measured_array, target_array)
            print("fitted polynomial2 => T(10,3)")

            def apply_model(r_in, g_in, b_in):
                return apply_polynomial2(T, r_in, g_in, b_in)
        else:
            print("Unknown model")
            return

    # Build 512x512 from 64x64x64
    lut_size = 64
    w2d = 512
    h2d = 512
    lut_arr = np.zeros((h2d, w2d, 3), dtype=np.uint8)

    for z in range(lut_size):
        for y in range(lut_size):
            for x in range(lut_size):
                r_in = x * (255 / (lut_size - 1))
                g_in = y * (255 / (lut_size - 1))
                b_in = z * (255 / (lut_size - 1))
                vec_out = apply_model(r_in, g_in, b_in)
                R_out = int(round(vec_out[0]))
                G_out = int(round(vec_out[1]))
                B_out = int(round(vec_out[2]))
                px, py = index_to_2d(z, x, y)
                lut_arr[py, px] = (R_out, G_out, B_out)

    out_path = Path(args.output_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_img = Image.fromarray(lut_arr, "RGB")
    out_img.save(out_path)
    print(f"Saved LUT => {out_path}  (model={args.model})")


if __name__ == "__main__":
    main()
