import argparse
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine site-specific PNGs into one multi-panel figure.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    site_codes = ["AP", "CB", "CH", "CK", "LA", "LM"]
    paths = [input_dir / f"{args.prefix}_{site}.png" for site in site_codes]

    fig, axes = plt.subplots(2, 3, figsize=(24, 16), constrained_layout=True)
    if args.title:
        fig.suptitle(args.title, fontsize=18)

    for ax, site, path in zip(axes.flatten(), site_codes, paths):
        img = mpimg.imread(path)
        ax.imshow(img)
        ax.set_title(site, fontsize=14)
        ax.axis("off")

    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
