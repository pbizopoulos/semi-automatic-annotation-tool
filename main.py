from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

import gdown
from playwright.sync_api import sync_playwright


def main() -> None:
    zip_file_path = Path('bin/rp_im.zip')
    if not zip_file_path.is_file():
        gdown.download('https://drive.google.com/uc?id=1ruTiKdmqhqdbE9xOEmjQGing76nrTK2m', zip_file_path.as_posix(), quiet=False)
        with ZipFile(zip_file_path, 'r') as zip_file:
            zip_file.extractall('bin')
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(args=['--use-gl=egl'])
        context = browser.new_context(record_video_dir='bin/')
        page = context.new_page()
        timeout = 100000
        page.set_default_timeout(timeout)
        page.set_default_navigation_timeout(timeout)
        page.on('pageerror', lambda exception: (_ for _ in ()).throw(Exception(f'uncaught exception: {exception}')))
        page.goto('file:///usr/src/app/docs/index.html')
        page.locator('#model-download-div').wait_for(state='hidden')
        page.set_input_files('#load-files-input-file', 'bin/rp_im/1.nii.gz')
        page.locator('#image-index-input-range').fill('2')
        page.locator('#label-color-div-1').click()
        page.screenshot(path='bin/before.png')
        page.locator('#reset-image-value-button').click()
        page.locator('#predict-image-current-button').click()
        with page.expect_download() as download_info:
            page.click('#save-predictions-to-disk-button')
        download = download_info.value
        download.save_as('bin/masks.nii')
        with Path('bin/masks.nii').open('rb') as file:
            assert sha256(file.read()).hexdigest() == '6d1f1c28c38cab797d7500b01e5379223229b63c44bc857cbb38aab75fef75f2'
        page.screenshot(path='bin/screenshot.png')
        with Path('bin/screenshot.png').open('rb') as file:
            assert sha256(file.read()).hexdigest() == '664ba9b46e091ec97db7d60dbac519329e5aacbac9a3f9cc0d194efb02bdc105'
        context.close()
        browser.close()


if __name__ == '__main__':
    main()
