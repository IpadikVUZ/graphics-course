import argparse
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
SIZE = 32


def make_example(directory):
    directory.mkdir(exist_ok=True, parents=True)
    b, g, r = np.indices((SIZE, SIZE, SIZE), dtype=np.float32)
    chart = np.stack(((r + 0.5) / SIZE, (g + 0.5) / SIZE, (b + 0.5) / SIZE), axis=-1)
    chart = chart.reshape(128, 256, 3)
    chart = np.repeat(np.repeat(chart, 2, axis=0), 2, axis=1)
    original = np.round(chart * 255).astype(np.uint8)
    Image.fromarray(original, 'RGB').save(directory / 'source.png')

    # Reference correction: mild filmic contrast, cooler shadows, warmer highlights.
    c = chart.astype(np.float32)
    c = np.clip((c - 0.48) * 1.06 + 0.48, 0, 1) ** 0.96
    luminance = c @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    shadow = np.clip((0.49 - luminance) / 0.49, 0, 1)[..., None]
    highlight = np.clip((luminance - 0.42) / 0.58, 0, 1)[..., None]
    c += shadow * np.array([-0.012, 0.002, 0.013], dtype=np.float32)
    c += highlight * np.array([0.012, 0.006, -0.008], dtype=np.float32)
    gray = (c @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32))[..., None]
    c = np.clip(gray + (c - gray) * 1.045, 0, 1)
    corrected = np.round(c * 255).astype(np.uint8)
    Image.fromarray(corrected, 'RGB').save(directory / 'graded.png')


def fill_empty_nearest(values, valid):
    # Full Euclidean nearest neighbour if SciPy is available.
    try:
        from scipy.ndimage import distance_transform_edt
        mask = valid.reshape((SIZE, SIZE, SIZE))
        volume = values.reshape((SIZE, SIZE, SIZE, 3))
        indices = distance_transform_edt(~mask, return_distances=False, return_indices=True)
        volume[~mask] = volume[tuple(indices[:, ~mask])]
        return volume.reshape((-1, 3))
    except ImportError:
        # Manhattan-distance nearest filled voxel without optional dependencies.
        grid = values.reshape((SIZE, SIZE, SIZE, 3))
        seen = valid.reshape((SIZE, SIZE, SIZE)).copy()
        q = deque(tuple(int(v) for v in pos) for pos in np.argwhere(seen))
        while q:
            z, y, x = q.popleft()
            for dz, dy, dx in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
                zz, yy, xx = z+dz, y+dy, x+dx
                if not (0 <= zz < SIZE and 0 <= yy < SIZE and 0 <= xx < SIZE) or seen[zz,yy,xx]:
                    continue
                grid[zz,yy,xx] = grid[z,y,x]
                seen[zz,yy,xx] = True
                q.append((zz,yy,xx))
        return grid.reshape((-1,3))


def build_lut(source, graded, output):
    original = np.asarray(Image.open(source).convert('RGB'), dtype=np.float32) / 255
    edited = np.asarray(Image.open(graded).convert('RGB'), dtype=np.float32) / 255
    if original.shape != edited.shape:
        raise ValueError('Original and edited image dimensions must match')
    src = original.reshape(-1, 3)
    dst = edited.reshape(-1, 3)
    cell = np.minimum((src * SIZE).astype(np.int32), SIZE - 1)
    indices = cell[:, 0] + SIZE * cell[:, 1] + SIZE * SIZE * cell[:, 2]
    counts = np.bincount(indices, minlength=SIZE**3)
    sums = np.zeros((SIZE**3, 3), dtype=np.float64)
    np.add.at(sums, indices, dst)
    populated = counts > 0
    if not np.any(populated):
        raise ValueError('No color samples in reference image')
    lut = np.zeros((SIZE**3, 3), dtype=np.float32)
    lut[populated] = (sums[populated] / counts[populated, None]).astype(np.float32)
    if not np.all(populated):
        lut = fill_empty_nearest(lut, populated)
    alpha = np.ones((SIZE**3, 1), dtype=np.float32)
    data = np.round(np.clip(np.concatenate((lut, alpha), axis=1), 0, 1) * 255).astype(np.uint8)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data.tobytes())
    print(f'Wrote {output}: {output.stat().st_size} bytes; {np.count_nonzero(populated)} / {SIZE**3} sampled cells')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build 32x32x32 RGBA8 3D LUT from original / graded RGB image pair')
    parser.add_argument('--source', type=Path)
    parser.add_argument('--graded', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'assets' / 'filmic_32_rgba8.bin')
    parser.add_argument('--example', action='store_true', help='Generate the included reproducible reference image pair')
    args = parser.parse_args()
    if args.example:
        make_example(ROOT / 'lut')
    source = args.source or (ROOT / 'lut' / 'source.png')
    graded = args.graded or (ROOT / 'lut' / 'graded.png')
    build_lut(source, graded, args.output)
