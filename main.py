from os.path import isfile, join
from playwright.sync_api import sync_playwright
from zipfile import ZipFile
import gdown
import hashlib


def main():
    zip_file_path = join('bin', 'rp_im.zip')
    if not isfile(zip_file_path):
        gdown.download('https://drive.google.com/uc?id=1ruTiKdmqhqdbE9xOEmjQGing76nrTK2m', zip_file_path, quiet=False)
        with ZipFile(zip_file_path, 'r') as zip_file:
            zip_file.extractall('bin')
    zip_file_path = join('bin', 'latest.zip')
    masks_multiclass_nifti_file_path = join('bin', 'masks-multiclass.nii')
    if not isfile(masks_multiclass_nifti_file_path):
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page()
            page.goto('https://github.com/pbizopoulos/semi-automatic-annotation-tool/releases/latest')
            page.click('text=Assets')
            with page.expect_download() as download_info:
                page.click('text=masks-multiclass.nii')
            download = download_info.value
            download.save_as(masks_multiclass_nifti_file_path)
            browser.close()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(args=['--use-gl=egl'])
        page = browser.new_page()
        page.on('pageerror', lambda exception: (_ for _ in ()).throw(Exception(f'uncaught exception: {exception}')))
        timeout = 100000
        page.set_default_navigation_timeout(timeout)
        page.set_default_timeout(timeout)
        page.goto('file:///work/docs/index.html')
        page.locator('#model-download-div').wait_for(state='hidden')
        page.set_input_files('#load-files-input-file', join('bin', 'rp_im', '1.nii.gz'))
        page.locator('#image-index-input-range').fill('2')
        page.locator('#label-color-div-1').click()
        page.screenshot(path=join('bin', 'before.png'))
        page.locator('#reset-image-value-button').click()
        page.locator('#predict-image-current-button').click()
        with page.expect_download() as download_info:
            page.click('#save-predictions-to-disk-button')
        download = download_info.value
        download.save_as(join('bin', 'masks.nii'))
        with open(join('bin', 'masks.nii'), 'rb') as file:
            assert hashlib.sha256(file.read()).hexdigest() == '6d1f1c28c38cab797d7500b01e5379223229b63c44bc857cbb38aab75fef75f2'
        page.screenshot(path=join('bin', 'screenshot.png'))
        with open(join('bin', 'screenshot.png'), 'rb') as file:
            assert hashlib.sha256(file.read()).hexdigest() == '664ba9b46e091ec97db7d60dbac519329e5aacbac9a3f9cc0d194efb02bdc105'
        page.set_input_files('#load-predictions-input-file', masks_multiclass_nifti_file_path)
        page.screenshot(path=join('bin', 'screenshot-2.png'))
        with open(join('bin', 'screenshot-2.png'), 'rb') as file:
            assert hashlib.sha256(file.read()).hexdigest() == '4874839618badb5fd70c078ddc9337141daf19ad433df18161272ac819dea3e6'
        browser.close()


if __name__ == '__main__':
    main()
