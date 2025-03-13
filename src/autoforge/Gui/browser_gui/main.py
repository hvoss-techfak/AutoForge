import subprocess
import threading

# set env variable
# STREAMLIT_SERVER_HEADLESS=true
import os

os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"


from autoforge.Helper.filamentcolors_library import download_filament_info


def run_streamlit():
    # Start the Streamlit app (by running app.py)
    subprocess.run(["streamlit", "run", "app.py"])


if __name__ == "__main__":
    # download filament info
    print("Downloading filament info...")
    print("If this is your first time running this script, it may take a few minutes.")
    download_filament_info()

    streamlit_thread = threading.Thread(target=run_streamlit, daemon=True)

    streamlit_thread.start()
    # Wait for threads to finish (they run indefinitely)
    streamlit_thread.join()
