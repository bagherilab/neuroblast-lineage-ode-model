"""Generate all model figures as SVG.

Usage:
    python make_figures.py                     # everything into figures/
    python make_figures.py --only manuscript   # just the Figure 7 grid
    python make_figures.py --out some/dir
"""
import argparse
import os

# Fixed timestamp in the SVG metadata so re-running produces byte-identical files
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import matplotlib
matplotlib.use("Agg")

from neuroblast_ode.figures import make_all

GROUPS = ("manuscript", "supplemental")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="figures", help="output directory (default: figures)")
    parser.add_argument("--only", nargs="+", choices=GROUPS, default=GROUPS,
                        help="figure groups to generate (default: all)")
    args = parser.parse_args()
    make_all(args.out, groups=args.only)


if __name__ == "__main__":
    main()
