import os
import json
import time
import traceback

import requests
from tqdm import tqdm

# API endpoints
API_VERSION_URL = "https://filamentcolors.xyz/api/version/"
SWATCH_API_URL = "https://filamentcolors.xyz/api/swatch/?m=manufacturer"

# Local file names
DATA_FILE = "../swatches.json"
VERSION_FILE = "../swatches_version.json"
LAST_CHECKED_FILE = "../last_checked.json"

def get_api_version():
    """
    Fetches the current version information from the API.

    Sends a GET request to the API version endpoint and returns the JSON response.

    Returns:
        dict: A dictionary containing the version information from the API.

    Raises:
        requests.RequestException: If the request to the API fails.
    """
    response = requests.get(API_VERSION_URL)
    response.raise_for_status()
    return response.json()


def load_local_version():
    """
    Loads the locally saved version info if it exists.

    Checks if the version file exists in the local file system. If it does,
    opens the file, reads the JSON content, and returns it as a dictionary.

    Returns:
        dict or None: A dictionary containing the version information if the file exists,
                      otherwise None.
    """
    if os.path.exists(VERSION_FILE):
        with open(VERSION_FILE, "r") as f:
            return json.load(f)
    return None


def save_local_version(version_info):
    """
    Saves the version info locally.

    Args:
        version_info (dict): A dictionary containing the version information to be saved.
    """
    with open(VERSION_FILE, "w") as f:
        json.dump(version_info, f)


def download_all_pages(url):
    """
    Downloads all pages of swatch data from the given URL.

    This function sends GET requests to the provided URL and continues to follow
    pagination links (if any) to download all pages of swatch data. The results
    from each page are combined into a single list.

    Args:
        url (str): The initial URL to start downloading the swatch data from.

    Returns:
        list: A list containing all swatch data from all pages.

    Raises:
        requests.RequestException: If any of the requests to the URL fail.
    """
    all_results = []
    tbar = tqdm(desc="Downloading", unit=" swatches page")
    while url:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        all_results.extend(data.get("results", []))
        url = data.get("next")
        tbar.update(1)
    return all_results


def download_filament_info():
    """
    Main function to manage the swatch data update process.

    This function performs the following steps:
    1. Retrieves the current API version details.
    2. Loads the local version info if available.
    3. Checks if the local data is up-to-date with the API version.
    4. Downloads new data if the local copy is outdated or missing.
    5. Saves the downloaded data and updates the local version info.

    Raises:
        requests.RequestException: If any of the requests to the API fail.
    """
    #load or create last checked file
    last_modified_checked = None
    try:
        if os.path.exists(LAST_CHECKED_FILE):
            with open(LAST_CHECKED_FILE, "r") as f:
                last_modified_checked = json.load(f).get("db_last_modified")
        else:
            with open(LAST_CHECKED_FILE, "w") as f:
                last_modified_checked = 0
                json.dump({"db_last_modified": time.time()}, f)
    except Exception as e:
        print("Failed to load last checked file:", e)
        traceback.print_exc()
    update_minimum_time = 60 * 60 * 24  # 24 hours

    if last_modified_checked and time.time() - last_modified_checked < update_minimum_time:
        print("Data was checked recently. Skipping download.")
        return

    try:
        api_version = get_api_version()
    except requests.RequestException as e:
        print("Failed to retrieve API version:", e)
        return

    local_version = load_local_version()
    if local_version and local_version.get("db_last_modified") == api_version.get(
        "db_last_modified"
    ):
        print(
            "Data is up-to-date. No download needed. File last modified:",
            api_version.get("db_last_modified"),
        )
        print("Swatch File location:", DATA_FILE)
        return

    print("Downloading new filament color data...")
    try:
        results = download_all_pages(SWATCH_API_URL)
    except requests.RequestException as e:
        print("Error downloading swatch data:", e)
        return

    with open(DATA_FILE, "w") as f:
        json.dump(results, f)

    save_local_version(api_version)
    print("Download complete, data saved to", DATA_FILE)


if __name__ == "__main__":
    download_filament_info()
