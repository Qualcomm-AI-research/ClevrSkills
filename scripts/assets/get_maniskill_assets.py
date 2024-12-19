# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

"""Download the ManiSkill assets other than vima textures."""
from pathlib import Path

from mani_skill2.utils.download_asset import DATA_SOURCES, download, initialize_sources


def download_assets():
    ASSET_DIR = Path("clevr_skills/assets/")
    # Initialize the data sources. The DATA_SOURCES dict will then be populated with
    # the download links.
    initialize_sources()
    data_source_keys = ["ycb", "assembling_kits"]

    for key in data_source_keys:
        print(f"Downloading asset from {DATA_SOURCES[key]['url']}...")
        download(
            DATA_SOURCES[key]["url"],
            output_dir=ASSET_DIR,
            target_path=DATA_SOURCES[key]["target_path"],
            checksum=DATA_SOURCES[key]["checksum"],
            verbose=True,
            non_interactive=True,
        )

    print("Done.")


if __name__ == "__main__":
    download_assets()
