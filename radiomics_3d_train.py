"""Deprecated compatibility entry point.

Use ``onem_modeling`` for feature-table models or ``onem_torch`` for image
models. Importing this file performs no training or data access.
"""


def main():
    raise SystemExit(
        "This prototype has been retired. Use onem_modeling or onem_torch."
    )


if __name__ == "__main__":
    main()
