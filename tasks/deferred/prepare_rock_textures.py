from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent / 'assets'
ASSETS = {
    'boulder_01': {
        'albedo': 'textures/boulder_01_diff_1k.jpg',
        'normal': 'textures/boulder_01_nor_gl_1k.jpg',
        'arm': 'textures/boulder_01_arm_1k.jpg',
    },
    'rock_09': {
        'albedo': 'textures/rock_09_diff_1k.jpg',
        'normal': 'textures/rock_09_nor_gl_1k.jpg',
        'arm': 'textures/rock_09_arm_1k.jpg',
    },
}
MIPS = 11

for asset, maps in ASSETS.items():
    asset_dir = ROOT / asset
    for kind, rel in maps.items():
        img = Image.open(asset_dir / rel).convert('RGBA')
        if img.size != (1024, 1024):
            raise SystemExit(f'{asset} {kind}: expected 1024x1024, got {img.size}')
        out_path = ROOT / f'{asset}_{kind}_mips.bin'
        with out_path.open('wb') as out:
            cur = img
            for mip in range(MIPS):
                out.write(cur.tobytes())
                if mip + 1 < MIPS:
                    cur = cur.resize((max(1, cur.size[0] // 2), max(1, cur.size[1] // 2)), Image.Resampling.LANCZOS)
        print(out_path.name)
