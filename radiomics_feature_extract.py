"""Compatibility CLI for radiomics extraction.

New projects should import :mod:`onem_radiomics` directly.
"""

import argparse


def main():
    from onem_radiomics import PRESET_CONFIGS, RadiomicsExtractor

    parser = argparse.ArgumentParser(description="Extract radiomics features")
    parser.add_argument("images_dir")
    parser.add_argument("masks_dir")
    parser.add_argument("output_csv")
    parser.add_argument("--preset", default="research", choices=sorted(PRESET_CONFIGS))
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    extractor = RadiomicsExtractor(PRESET_CONFIGS[args.preset])
    extractor.batch_extract_features(
        args.images_dir,
        args.masks_dir,
        args.output_csv,
        n_jobs=args.jobs,
    )


if __name__ == "__main__":
    main()
