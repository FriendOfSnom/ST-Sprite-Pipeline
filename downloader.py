#!/usr/bin/env python3

"""
downloader.py

Bulk gallery crawler and downloader.

Features:
- User specifies destination folder name and starting page URL.
- Cross-platform support (Windows, Mac, Linux).
- Resume support: skips completed items on restart.
- Logs failures and performs a retry pass.
- Saves images in the user's OS-specific Downloads folder.
"""

import pathlib
import sys
import os
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# ----------------------------------------------------------------------
# Configuration Constants
# ----------------------------------------------------------------------
USER_AGENT = "Mozilla/5.0"
DOWNLOAD_DELAY = 1.0
RETRY_COUNT = 2
RETRY_DELAY = 5


def get_downloads_folder():
    """
    Returns the user's OS-specific Downloads folder path.
    """
    return pathlib.Path.home() / "Downloads"


def run_downloader(output_dir, start_url):
    """
    Runs the bulk downloader logic.
    Downloads all images in a gallery starting from start_url,
    saving them to output_dir, with resume and retry support.
    """

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Saving images to: {output_dir}")

    # Paths for logs and progress tracking
    completed_file = output_dir / "completed.txt"
    resume_file = output_dir / "resume.txt"
    failed_file = output_dir / "failed.txt"
    failed_final_file = output_dir / "failed_final.txt"
    log_file_path = output_dir / "log.txt"

    # Load any existing completed indices for resumption
    completed_indices = set()
    if completed_file.exists():
        with completed_file.open() as f:
            for line in f:
                try:
                    completed_indices.add(int(line.strip()))
                except ValueError:
                    pass

    # Set up logging to file and console simultaneously
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    logfile = log_file_path.open("a", encoding="utf-8")

    class Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()
        def flush(self):
            for s in self.streams:
                s.flush()

    sys.stdout = sys.stderr = Tee(sys.__stdout__, logfile)

    print("\n" + "=" * 60)
    print(f"[INFO] New download run started at {time.ctime()}")
    print("=" * 60)

    # Set up HTTP session with headers
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    def get_with_retries(url, description):
        """
        Fetches the given URL with retry logic.
        Sleeps between failures. Returns the response or None.
        """
        for attempt in range(1, RETRY_COUNT + 1):
            try:
                response = session.get(url)
                response.raise_for_status()
                return response
            except Exception as e:
                print(f"[WARN] Attempt {attempt}/{RETRY_COUNT} failed for {description}: {e}")
                time.sleep(RETRY_DELAY)
        print(f"[ERROR] All retries failed for {description}.")
        return None

    def save_completed(index):
        """
        Appends the completed index to the completed.txt file.
        """
        with completed_file.open("a") as f:
            f.write(f"{index}\n")

    def save_resume(index, url):
        """
        Writes current progress (index and URL) to resume.txt.
        """
        with resume_file.open("w") as f:
            f.write(f"{index}\n{url}\n")

    def log_failed_url(index, url, path):
        """
        Appends a failed download attempt to the given failure log file.
        """
        with path.open("a") as f:
            f.write(f"{index}\t{url}\n")
        print(f"[WARN] Logged failed download for image {index} to {path.name}")

    def get_next_link(soup):
        """
        Extracts the 'next' link from the gallery page soup.
        """
        next_link = soup.find("a", id="next")
        if next_link and "href" in next_link.attrs:
            return urljoin("https://e-hentai.org/", next_link["href"])
        return None

    def process_image(index, page_url, fail_log):
        """
        Downloads a single gallery image by:
        - Fetching the page URL.
        - Extracting the image link.
        - Downloading the image file.
        Logs failures as needed.
        Returns the next page URL, or None if at end.
        """
        print(f"\n[INFO] Fetching page for image {index}: {page_url}")
        resp = get_with_retries(page_url, "image page")
        if not resp:
            log_failed_url(index, page_url, fail_log)
            return None

        soup = BeautifulSoup(resp.text, "html.parser")
        img_tag = soup.find("img", id="img")
        if not img_tag:
            print("[WARN] Could not find image on this page.")
            log_failed_url(index, page_url, fail_log)
            return get_next_link(soup)

        img_url = img_tag["src"]
        print(f"[INFO] Downloading image: {img_url}")
        img_data = get_with_retries(img_url, "image file")
        if not img_data:
            log_failed_url(index, page_url, fail_log)
            return get_next_link(soup)

        ext = img_url.split(".")[-1].split("?")[0]
        filename = f"{index:03d}.{ext}"
        filepath = output_dir / filename

        with filepath.open("wb") as f:
            f.write(img_data.content)
        print(f"[INFO] Saved: {filename}")

        save_completed(index)
        return get_next_link(soup)

    # Main download loop
    image_count = 1
    current_url = start_url

    while current_url:
        if image_count in completed_indices:
            print(f"[INFO] Image {image_count} already downloaded. Skipping.")
            resp = get_with_retries(current_url, "image page")
            if not resp:
                break
            soup = BeautifulSoup(resp.text, "html.parser")
            current_url = get_next_link(soup)
            image_count += 1
            save_resume(image_count, current_url or "")
            continue

        next_url = process_image(image_count, current_url, failed_file)
        if next_url:
            if next_url == current_url:
                print("[WARN] Detected repeating next link. Stopping.")
                break
            image_count += 1
            current_url = next_url
            save_resume(image_count, current_url)
        else:
            print("[INFO] No more images or next link missing. Ending.")
            break

        time.sleep(DOWNLOAD_DELAY)

    print("\n[INFO] First pass complete.")

    # Retry pass for failed items
    if failed_file.exists():
        print("\n[INFO] Starting retry pass for failed items.")
        with failed_file.open() as f:
            failed_items = [line.strip().split("\t") for line in f if "\t" in line]
        failed_file.unlink()

        for index_str, url in failed_items:
            try:
                index = int(index_str)
            except ValueError:
                continue
            process_image(index, url, failed_final_file)
            time.sleep(DOWNLOAD_DELAY)

        print("\n[INFO] Retry pass complete. Check failed_final.txt for any remaining issues.")
    else:
        print("\n[INFO] No failed items to retry.")

    print(f"\n[INFO] All images saved in {output_dir}")
    print("=" * 60)

    # Restore output streams
    sys.stdout = orig_stdout
    sys.stderr = orig_stderr
    logfile.close()


def run_downloader_interactive():
    """
    Interactive CLI entry point.
    Prompts the user for a folder name and start URL.
    Returns the final output directory path as a string.
    """
    print("=" * 60)
    print(" E-HENTAI BULK SPRITE DOWNLOADER ")
    print("=" * 60)

    folder_name = input("\nEnter a folder name for saving images (in your Downloads folder):\n> ").strip()
    if not folder_name:
        print("\nERROR: Folder name cannot be empty.")
        sys.exit(1)

    start_url = input("\nEnter the starting image page URL.\nThis tool currently only works with E-Hentai links that point to the first image in the set you wish to downlad:\n> ").strip()
    if not start_url:
        print("\nERROR: Start URL cannot be empty.")
        sys.exit(1)

    downloads_folder = get_downloads_folder()
    output_dir = downloads_folder / folder_name

    run_downloader(output_dir, start_url)

    print("\n[INFO] Download step complete.")
    print("=" * 60)

    return str(output_dir)


if __name__ == "__main__":
    run_downloader_interactive()
