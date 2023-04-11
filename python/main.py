from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

import gdown
from playwright.sync_api import Error, sync_playwright


def main() -> None:
    zip_file_path = Path("bin/rp_im.zip")
    if not zip_file_path.is_file():
        gdown.download("https://drive.google.com/uc?id=1ruTiKdmqhqdbE9xOEmjQGing76nrTK2m", zip_file_path.as_posix(), quiet=False)
        with ZipFile(zip_file_path, "r") as zip_file:
            zip_file.extractall("bin")
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(args=["--use-gl=egl"])
        context = browser.new_context(record_video_dir="bin/")
        page = context.new_page()
        timeout = 100000
        page.set_default_timeout(timeout)
        page.set_default_navigation_timeout(timeout)
        page.on("pageerror", page_error)
        page.goto("https://semi-automatic-annotation-tool.incisive.iti.gr/")
        page.locator("#model-download-div").wait_for(state="hidden")
        page.set_input_files("#load-files-input-file", "bin/rp_im/1.nii.gz")
        page.locator("#image-index-input-range").fill("2")
        page.locator("#label-color-div-1").click()
        page.screenshot(path="bin/before.png")
        page.locator("#reset-image-value-button").click()
        page.locator("#predict-image-current-button").click()
        with page.expect_download() as download_info:
            page.click("#save-predictions-to-disk-button")
        download = download_info.value
        download.save_as("bin/masks.nii")
        with Path("bin/masks.nii").open("rb") as file:
            assert sha256(file.read()).hexdigest() == "5c242f1e89e7f9ee6b066e687c9f382f4fce94339234a6a0f6cc643a6833c550"
        page.screenshot(path="bin/screenshot.png")
        with Path("bin/screenshot.png").open("rb") as file:
            assert sha256(file.read()).hexdigest() == "222201e6cdae9b12973dc65185ae913090e5c21d506c8d90e278eb321c2283ab"
        context.close()
        browser.close()


def page_error(exception: Error) -> None:
    raise exception


if __name__ == "__main__":
    main()
