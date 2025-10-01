#!/usr/bin/env python3
"""
downloader.py

Bulk gallery crawler and downloader for E-Hentai / ExHentai.

Features:
- User specifies destination folder name and starting page URL.
- Cross-platform support (Windows, Mac, Linux).
- Resume support: skips completed items on restart.
- Logs failures and performs a retry pass.
- Saves images in the user's OS-specific Downloads folder.
- ExHentai cookie support with pasteable cookie header and local caching.
- Optional per-image delay override from the interactive prompt.
"""

import json
import pathlib
import sys
import time
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# ----------------------------------------------------------------------
# Configuration Constants
# ----------------------------------------------------------------------
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0 Safari/537.36"
)
DEFAULT_DOWNLOAD_DELAY = 0.25
RETRY_COUNT = 2
RETRY_DELAY = 0
HTTP_TIMEOUT = (5,8)
IMG_TIMEOUT = (5,10)

CONFIG_DIRNAME = ".st_sprite_tool"
CONFIG_FILENAME = "auth.json"

# ----------------------------------------------------------------------
# Utility: paths, cookies, filenames
# ----------------------------------------------------------------------
def get_downloads_folder() -> pathlib.Path:
    """
    Returns the user's OS-specific Downloads folder path.
    """
    return pathlib.Path.home() / "Downloads"


def _config_path() -> pathlib.Path:
    """
    Returns the path to the auth/config json file in the user's home directory.
    Creates the folder if it doesn't exist.
    """
    cfg_dir = pathlib.Path.home() / CONFIG_DIRNAME
    cfg_dir.mkdir(parents=True, exist_ok=True)
    return cfg_dir / CONFIG_FILENAME


def _load_saved_cookies(netloc: str) -> Dict[str, str]:
    """
    Loads cached cookies for a given host (netloc), if present.
    """
    path = _config_path()
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f) or {}
        return (data.get("cookies") or {}).get(netloc, {})
    except Exception:
        return {}


def _save_cookies(netloc: str, cookies: Dict[str, str]) -> None:
    """
    Saves cookies for a given host (netloc), merging with any existing config.
    """
    path = _config_path()
    cfg = {"cookies": {}}
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                cfg = json.load(f) or {"cookies": {}}
        except Exception:
            cfg = {"cookies": {}}
    cfg.setdefault("cookies", {})
    cfg["cookies"][netloc] = cookies
    with path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def _parse_cookie_header(cookie_str: str) -> Dict[str, str]:
    """
    Parses a typical 'k=v; k2=v2; ...' cookie header string into a dict.
    """
    jar: Dict[str, str] = {}
    for part in cookie_str.split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            k = k.strip()
            v = v.strip()
            if k and v:
                jar[k] = v
    return jar


def _needs_cookies(netloc: str) -> bool:
    """
    Returns True if the host is known to require cookies (ExHentai).
    """
    return "exhentai.org" in netloc.lower()


def _sanitize_filename(s: str) -> str:
    """
    Replaces unsafe filename characters with underscores.
    """
    s = re.sub(r"[^\w\-\.]+", "_", s)
    return s.strip("._") or "img"

def _sanitize_folder_name(name: str) -> str:
    """
    Make a safe folder name (Windows/macOS/Linux).
    """
    name = re.sub(r'[<>:"/\\|?*]+', "_", name).strip()
    return name.strip(" ._") or "untitled"

def _unique_child_dir(parent: pathlib.Path, desired: str) -> pathlib.Path:
    """
    Ensure a unique child directory under parent, appending _2, _3, ...
    """
    base = parent / desired
    if not base.exists():
        return base
    i = 2
    while True:
        cand = parent / f"{desired}_{i}"
        if not cand.exists():
            return cand
        i += 1



# ----------------------------------------------------------------------
# Network helpers
# ----------------------------------------------------------------------
@dataclass
class PageResult:
    page_url: str
    image_url: Optional[str]
    saved_to: Optional[str]
    error: Optional[str]


def _session_with_headers(referer: str) -> requests.Session:
    """
    Creates a requests session with a default UA and referer.
    """
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Referer": referer})
    return s


def _get_with_retries(session: requests.Session, url: str, description: str, timeout: int) -> Optional[requests.Response]:
    """
    Fetch the given URL with up to RETRY_COUNT attempts. We do NOT pause between
    attempts (RETRY_DELAY=0 by default) so a failure is retried immediately.
    """
    for attempt in range(1, RETRY_COUNT + 1):
        try:
            resp = session.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as e:
            print(f"[WARN] Attempt {attempt}/{RETRY_COUNT} failed for {description}: {e}")
            # Instant retry: only sleep if a nonzero RETRY_DELAY is configured
            if attempt < RETRY_COUNT and RETRY_DELAY > 0:
                time.sleep(RETRY_DELAY)
    print(f"[ERROR] All retries failed for {description}.")
    return None


def _fetch_image_and_next(session: requests.Session, page_url: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Downloads the HTML for a gallery page and extracts:
    - full-size image URL
    - next page URL

    Next page selector is robust across variants:
      - <a id="next">
      - or <a accesskey="n">
    """
    resp = _get_with_retries(session, page_url, "image page", HTTP_TIMEOUT)
    if not resp:
        return None, None

    soup = BeautifulSoup(resp.text, "html.parser")

    # Full-size image element
    img_tag = soup.find("img", id="img")
    img_url = img_tag["src"] if img_tag and img_tag.has_attr("src") else None

    # Next link: id="next" or accesskey="n"
    next_a = soup.find("a", id="next")
    if not next_a:
        next_a = soup.find("a", attrs={"accesskey": "n"})
    next_url = urljoin(page_url, next_a["href"]) if next_a and next_a.has_attr("href") else None

    # Normalize image URL (most pages are already absolute)
    if img_url:
        img_url = urljoin(page_url, img_url)

    return img_url, next_url


# ----------------------------------------------------------------------
# Core logic
# ----------------------------------------------------------------------
def run_downloader(output_dir: pathlib.Path, start_url: str, cookies: Optional[Dict[str, str]] = None, delay_override: Optional[float] = None):
    """
    Runs the bulk downloader logic.

    Parameters
    ----------
    output_dir : pathlib.Path
        Destination directory for downloaded images.
    start_url : str
        URL of the first gallery page (E-Hentai or ExHentai).
    cookies : Optional[Dict[str, str]]
        Cookies to send with the session (used for ExHentai).
    delay_override : Optional[float]
        If provided, overrides DEFAULT_DOWNLOAD_DELAY per image.

    Behavior
    --------
    - Creates/uses a requests session with UA + Referer.
    - For ExHentai, will use provided cookies and/or cached cookies.
    - Walks pages using the "next" link and downloads full-size images.
    - Logs progress and failures; performs a retry pass.
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Saving images to: {output_dir}")

    netloc = urlparse(start_url).netloc
    is_exh = _needs_cookies(netloc)

    # Determine base domain for navigation (mostly cosmetic here)
    base_domain = "https://exhentai.org/" if is_exh else "https://e-hentai.org/"
    print(f"[INFO] Using base domain: {base_domain}")

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

    # Construct HTTP session
    session = _session_with_headers(referer=start_url)

    # Merge cookies priority: provided > cached
    if is_exh:
        cached = _load_saved_cookies(netloc)
        merged = dict(cached)
        if cookies:
            merged.update(cookies)
        if merged:
            session.cookies.update(merged)
            print("[INFO] Using ExHentai cookies (provided and/or cached).")
            # Save back any new cookies
            _save_cookies(netloc, merged)
        else:
            print("[WARN] ExHentai detected but no cookies available. You will likely get image failures.")

    # Helper writers
    def save_completed(index: int) -> None:
        """
        Append the completed index to completed.txt.
        """
        with completed_file.open("a") as f:
            f.write(f"{index}\n")

    def save_resume(index: int, url: str) -> None:
        """
        Write current progress (index and URL) to resume.txt.
        """
        with resume_file.open("w") as f:
            f.write(f"{index}\n{url}\n")

    def log_failed_url(index: int, url: str, path: pathlib.Path) -> None:
        """
        Append a failed download attempt to the given failure log file.
        """
        with path.open("a") as f:
            f.write(f"{index}\t{url}\n")
        print(f"[WARN] Logged failed download for image {index} to {path.name}")

    def process_image(index: int, page_url: str, fail_log: pathlib.Path) -> Optional[str]:
        """
        Downloads a single gallery image by:
        - Fetching the page URL.
        - Extracting the image link.
        - Downloading the image file.
        Logs failures as needed.
        Returns the next page URL, or None if at end or unrecoverable.
        """
        print(f"\n[INFO] Fetching page for image {index}: {page_url}")
        img_url, next_url = _fetch_image_and_next(session, page_url)
        if not img_url:
            # Second chance: if we couldn't parse, log and move to next (if any)
            print("[WARN] Could not find image on this page.")
            log_failed_url(index, page_url, fail_log)
            return next_url

        print(f"[INFO] Downloading image: {img_url}")
        img_resp = _get_with_retries(session, img_url, "image file", IMG_TIMEOUT)
        if not img_resp:
            log_failed_url(index, page_url, fail_log)
            return next_url

        # Use 5-digit index and preserve remote extension (strip query).
        ext = img_url.split(".")[-1].split("?")[0]
        filename = f"{index:05d}.{ext}"
        filepath = output_dir / _sanitize_filename(filename)

        with filepath.open("wb") as f:
            f.write(img_resp.content)
        print(f"[INFO] Saved: {filepath.name}")

        save_completed(index)
        return next_url

    # Main download loop
    per_image_delay = DEFAULT_DOWNLOAD_DELAY if delay_override is None else max(0.0, float(delay_override))

    image_count = 1
    current_url = start_url

    while current_url:
        if image_count in completed_indices:
            print(f"[INFO] Image {image_count} already downloaded. Skipping.")
            resp = _get_with_retries(session, current_url, "image page", HTTP_TIMEOUT)
            if not resp:
                break
            soup = BeautifulSoup(resp.text, "html.parser")
            next_a = soup.find("a", id="next") or soup.find("a", attrs={"accesskey": "n"})
            current_url = urljoin(current_url, next_a["href"]) if next_a and next_a.has_attr("href") else None
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

        time.sleep(per_image_delay)

    print("\n[INFO] First pass complete.")

    # Retry pass for failed items
    if failed_file.exists():
        print("\n[INFO] Starting retry pass for failed items.")
        with failed_file.open() as f:
            failed_items = [line.strip().split("\t") for line in f if "\t" in line]
        failed_file.unlink()

        for index_str, url in failed_items:
            try:
                idx = int(index_str)
            except ValueError:
                continue
            process_image(idx, url, failed_final_file)

        print("\n[INFO] Retry pass complete. Check failed_final.txt for any remaining issues.")
    else:
        print("\n[INFO] No failed items to retry.")

    print(f"\n[INFO] All images saved in {output_dir}")
    print("=" * 60)

    # Restore output streams
    sys.stdout = orig_stdout
    sys.stderr = orig_stderr
    logfile.close()


# ----------------------------------------------------------------------
# Interactive entry point
# ----------------------------------------------------------------------
def run_downloader_interactive() -> str:
    """
    Interactive CLI entry point.
    Prompts for a game/forum title (used as the folder name),
    start URL, cookies (if needed), and optional per-image delay.
    Returns the output directory path as a string for pipeline integration.
    """
    print("=" * 60)
    print(" E-HENTAI / EXHENTAI BULK SPRITE DOWNLOADER ")
    print("=" * 60)

    title = input("\nEnter the game/forum post title (this becomes the folder name):\n> ").strip()
    if not title:
        print("\nERROR: Title cannot be empty.")
        sys.exit(1)

    start_url = input("\nEnter the starting image page URL (from e-hentai OR exhentai):\n> ").strip()
    if not start_url:
        print("\nERROR: Start URL cannot be empty.")
        sys.exit(1)

    # Optional per-image delay override
    delay_override = None
    delay_str = input(f"\nSeconds between images [default {DEFAULT_DOWNLOAD_DELAY}]:\n> ").strip()
    if delay_str:
        try:
            delay_override = float(delay_str)
        except ValueError:
            print("[WARN] Invalid delay; using default.")

    cookies: Dict[str, str] = {}
    netloc = urlparse(start_url).netloc
    is_exh = _needs_cookies(netloc)

    if is_exh:
        print("\n[NOTICE] ExHentai detected. ExHentai requires you to be logged in.")
        print("""
Option A: Paste your full cookie header (recommended)
  - Open DevTools on an exhentai page
  - Application/Storage -> Cookies -> https://exhentai.org
  - Copy all values as: k=v; k2=v2; k3=v3

Option B: Enter individual fields
  - Common cookies: ipb_member_id, ipb_pass_hash, igneous
""")
        choice = input("Paste full cookie header? [Y/n]: ").strip().lower()
        if choice in ("", "y", "yes"):
            raw = input("Cookie header: ").strip()
            cookies = _parse_cookie_header(raw)
        else:
            member_id = input(" - ipb_member_id: ").strip()
            pass_hash = input(" - ipb_pass_hash: ").strip()
            igneous = input(" - igneous (optional but often required): ").strip()
            if member_id:
                cookies["ipb_member_id"] = member_id
            if pass_hash:
                cookies["ipb_pass_hash"] = pass_hash
            if igneous:
                cookies["igneous"] = igneous

        if not cookies:
            print("\nERROR: No cookies provided for ExHentai; downloads will likely fail.")
            sys.exit(1)

    downloads_folder = get_downloads_folder()
    folder_name = _sanitize_folder_name(title)
    output_dir = _unique_child_dir(downloads_folder, folder_name)

    # Create the folder now and drop a tiny metadata file for later steps
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / "download_meta.json"
    try:
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "source_game": title,
                    "start_url": start_url,
                    "created_at_unix": int(time.time())
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
    except Exception as e:
        print(f"[WARN] Could not write download_meta.json: {e}")

    run_downloader(output_dir, start_url, cookies=cookies, delay_override=delay_override)

    print("\n[INFO] Download step complete.")
    print("=" * 60)
    return str(output_dir)



if __name__ == "__main__":
    run_downloader_interactive()
