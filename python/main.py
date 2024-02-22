import unittest
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

import gdown
from playwright.sync_api import Error, sync_playwright


class TestWebApplication(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:  # noqa: ANN102
        zip_file_path = Path("tmp/rp_im.zip")
        if not zip_file_path.is_file():
            gdown.download(
                "https://drive.google.com/uc?id=1ruTiKdmqhqdbE9xOEmjQGing76nrTK2m",
                zip_file_path.as_posix(),
                quiet=False,
            )
            with ZipFile(zip_file_path, "r") as zip_file:
                zip_file.extractall("tmp")

    def test_web_application(self: "TestWebApplication") -> None:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(args=["--use-gl=egl"])
            context = browser.new_context(record_video_dir="tmp/")
            page = context.new_page()
            timeout = 100000
            page.set_default_timeout(timeout)
            page.set_default_navigation_timeout(timeout)
            page.on("pageerror", self.page_error)
            page.goto("https://semi-automatic-annotation-tool.incisive.iti.gr/")
            page.locator("#model-download-div").wait_for(state="hidden")
            page.set_input_files("#load-files-input-file", "tmp/rp_im/1.nii.gz")
            page.locator("#image-index-input-range").fill("2")
            page.locator("#label-color-div-1").click()
            page.screenshot(path="tmp/before.png")
            page.locator("#reset-image-value-button").click()
            page.locator("#predict-image-current-button").click()
            with page.expect_download() as download_info:
                page.click("#save-predictions-to-disk-button")
            download = download_info.value
            download.save_as("tmp/masks.nii")
            with Path("tmp/masks.nii").open("rb") as file:
                if (
                    sha256(file.read()).hexdigest()
                    != "c3f3d3726d03498a6fdb9a0ba919f56252ef87e739f26210be580f9a8a2eaf89"  # noqa: E501
                ):
                    raise AssertionError
            page.screenshot(path="tmp/screenshot.png")
            with Path("tmp/screenshot.png").open("rb") as file:
                if (
                    sha256(file.read()).hexdigest()
                    != "ffb1a4b84d3e0a1eb1e60397d289459066e464149d29fff2a353851275f4e38b"  # noqa: E501
                ):
                    raise AssertionError
            context.close()
            browser.close()

    def page_error(self: "TestWebApplication", exception: Error) -> None:
        raise exception


if __name__ == "__main__":
    unittest.main()
