#!/usr/bin/env python3
"""
download_renpy_sdk.py

Downloads and extracts the Ren'Py SDK for the Visual Novel Development Toolkit.

This script:
- Auto-detects your platform (Windows/Mac/Linux)
- Downloads the appropriate Ren'Py 8.5.0 SDK
- Extracts it to the current directory
- Verifies the installation
"""

import os
import platform
import shutil
import sys
import zipfile
import tarfile
from pathlib import Path
from urllib.request import urlretrieve


# Ren'Py 8.5.0 SDK download URLs
SDK_URLS = {
    "Windows": "https://www.renpy.org/dl/8.5.0/renpy-8.5.0-sdk.zip",
    "Darwin": "https://www.renpy.org/dl/8.5.0/renpy-8.5.0-sdk.tar.bz2",  # macOS
    "Linux": "https://www.renpy.org/dl/8.5.0/renpy-8.5.0-sdk.tar.bz2",
}

SDK_VERSION = "8.5.0"
SDK_FOLDER_NAME = f"renpy-{SDK_VERSION}-sdk"


def show_progress(block_num, block_size, total_size):
    """Display download progress."""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(downloaded * 100 / total_size, 100)
        bar_length = 50
        filled = int(bar_length * percent / 100)
        bar = '=' * filled + '-' * (bar_length - filled)

        size_mb = total_size / (1024 * 1024)
        downloaded_mb = downloaded / (1024 * 1024)

        sys.stdout.write(f'\r[{bar}] {percent:.1f}% ({downloaded_mb:.1f}/{size_mb:.1f} MB)')
        sys.stdout.flush()


def get_platform():
    """Detect the current platform."""
    system = platform.system()
    if system in SDK_URLS:
        return system
    else:
        print(f"[ERROR] Unsupported platform: {system}")
        print("Supported platforms: Windows, macOS (Darwin), Linux")
        return None


def download_sdk(url, output_path):
    """Download the SDK from the given URL."""
    print(f"Downloading Ren'Py SDK {SDK_VERSION}...")
    print(f"URL: {url}")
    print(f"Destination: {output_path}\n")

    try:
        urlretrieve(url, output_path, reporthook=show_progress)
        print("\n[INFO] Download complete!")
        return True
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        return False


def extract_zip(zip_path, extract_to):
    """Extract a ZIP file."""
    print(f"\n[INFO] Extracting ZIP archive...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get total number of files for progress
            members = zip_ref.namelist()
            total = len(members)

            for i, member in enumerate(members):
                zip_ref.extract(member, extract_to)
                percent = (i + 1) * 100 / total
                sys.stdout.write(f'\rExtracting: {percent:.1f}% ({i + 1}/{total} files)')
                sys.stdout.flush()

        print("\n[INFO] Extraction complete!")
        return True
    except Exception as e:
        print(f"\n[ERROR] Extraction failed: {e}")
        return False


def extract_tar(tar_path, extract_to):
    """Extract a TAR.BZ2 file."""
    print(f"\n[INFO] Extracting TAR.BZ2 archive...")
    try:
        with tarfile.open(tar_path, 'r:bz2') as tar_ref:
            # Get total number of members
            members = tar_ref.getmembers()
            total = len(members)

            for i, member in enumerate(members):
                tar_ref.extract(member, extract_to)
                percent = (i + 1) * 100 / total
                sys.stdout.write(f'\rExtracting: {percent:.1f}% ({i + 1}/{total} files)')
                sys.stdout.flush()

        print("\n[INFO] Extraction complete!")
        return True
    except Exception as e:
        print(f"\n[ERROR] Extraction failed: {e}")
        return False


def verify_sdk(sdk_path):
    """Verify that the SDK was extracted correctly."""
    print(f"\n[INFO] Verifying SDK installation...")

    # Check for key files/folders
    required_items = [
        "renpy.py",
        "launcher",
    ]

    missing = []
    for item in required_items:
        if not (sdk_path / item).exists():
            missing.append(item)

    if missing:
        print(f"[WARN] Some expected files/folders are missing: {', '.join(missing)}")
        print("[WARN] The SDK may not be fully functional")
        return False

    print("[INFO] SDK verification passed!")
    return True


def main():
    """Main download and setup function."""
    print("=" * 70)
    print(" Ren'Py SDK Downloader")
    print("=" * 70)
    print(f"Version: {SDK_VERSION}")
    print()

    # Get platform
    platform_name = get_platform()
    if not platform_name:
        return 1

    print(f"Detected platform: {platform_name}")

    # Get download URL
    download_url = SDK_URLS[platform_name]
    file_extension = ".zip" if platform_name == "Windows" else ".tar.bz2"
    download_filename = f"renpy-{SDK_VERSION}-sdk{file_extension}"

    # Setup paths
    script_dir = Path(__file__).parent
    download_path = script_dir / download_filename
    sdk_path = script_dir / SDK_FOLDER_NAME

    # Check if SDK already exists
    if sdk_path.exists():
        print(f"\n[WARN] SDK folder already exists: {sdk_path}")
        response = input("Do you want to delete it and re-download? [y/N]: ").strip().lower()
        if response == 'y':
            print("[INFO] Removing existing SDK...")
            shutil.rmtree(sdk_path)
        else:
            print("[INFO] Keeping existing SDK. Exiting.")
            return 0

    # Download
    if not download_sdk(download_url, download_path):
        return 1

    # Extract
    print()
    if file_extension == ".zip":
        success = extract_zip(download_path, script_dir)
    else:
        success = extract_tar(download_path, script_dir)

    if not success:
        return 1

    # Verify
    if not verify_sdk(sdk_path):
        print("\n[WARN] SDK verification failed, but extraction completed")

    # Cleanup downloaded archive
    print(f"\n[INFO] Cleaning up downloaded archive...")
    try:
        download_path.unlink()
        print(f"[INFO] Deleted: {download_filename}")
    except Exception as e:
        print(f"[WARN] Could not delete archive: {e}")

    # Success!
    print("\n" + "=" * 70)
    print(" ✓ Ren'Py SDK Setup Complete!")
    print("=" * 70)
    print(f"SDK Location: {sdk_path}")
    print()
    print("Next steps:")
    print("1. Run the Visual Novel Development Toolkit:")
    print("   python pipeline_runner.py")
    print()
    print("2. Select option 1 to create a new Ren'Py project")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n[INFO] Download cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
