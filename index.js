'use strict';

const admzip = require('adm-zip');
const assert = require('assert');
const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
const puppeteer = require('puppeteer');


function waitFile(fileName) {
	while (!fs.existsSync(fileName)) {
		continue;
	}
}

(async () => {
	const browser = await puppeteer.launch({args: ['--use-gl=egl']});
	const page = await browser.newPage();
	page.on('pageerror', pageerr => {
		assert.fail(pageerr);
	});
	await page._client().send('Page.setDownloadBehavior', {behavior: 'allow', downloadPath: path.resolve('bin')});
	const inputNiftiFileName = 'rp_im.zip';
	if (!(fs.existsSync(`bin/${inputNiftiFileName}`))) {
		await page.goto('https://drive.google.com/uc?id=1ruTiKdmqhqdbE9xOEmjQGing76nrTK2m');
		await page.waitForSelector('#uc-download-link').then(selector => selector.click());
		waitFile(`bin/${inputNiftiFileName}`);
	}
	const zip = new admzip(`bin/${inputNiftiFileName}`);
	zip.extractAllTo('bin', true);
	await page.goto(`file:${path.join(__dirname, 'docs/index.html')}`);
	await page.waitForFunction('document.getElementById(\'model-load-fraction-div\').textContent == \'Model loaded.\'', {waitUntil: 'load', timeout: 0});
	await page.waitForSelector('#load-files-input-file:not([disabled])', {waitUntil: 'load', timeout: 0}).then(selector => selector.uploadFile('bin/rp_im/1.nii.gz'));
	await page.waitForSelector('#model-select:not([disabled])');
	await page.waitForSelector('#label-color-div-1').then(selector => selector.click());
	await page.evaluate(() => {
		document.querySelector('#image-index-input-range').value = 2;
		document.querySelector('#image-index-input-range').oninput();
	});
	await page.waitForSelector('#reset-image-value-button').then(selector => selector.click());
	await page.waitForSelector('#predict-image-current-button').then(selector => selector.click());
	const outputFileName = 'masks.nii';
	if (fs.existsSync(`bin/${outputFileName}`)) {
		await fs.unlinkSync(`bin/${outputFileName}`);
	}
	await page.waitForSelector('#save-predictions-to-disk-button:not([disabled])', {waitUntil: 'load', timeout: 0}).then(selector => selector.click());
	waitFile(`bin/${outputFileName}`);
	const outputBuffer = new fs.readFileSync(`bin/${outputFileName}`);
	const outputHash = crypto.createHash('sha256').update(outputBuffer).digest('hex');
	assert.strictEqual(outputHash, 'a986c5849a90fe1d8bf5e49c3fd751a72388445f5ff1b71c5f36684940d71508');
	await page.screenshot({path: 'bin/puppeteer-screenshot.png'});
	const screenshotBuffer = new fs.readFileSync('bin/puppeteer-screenshot.png');
	const screenshotHash = crypto.createHash('sha256').update(screenshotBuffer).digest('hex');
	assert.strictEqual(screenshotHash, '890152fc138f1d046ae9fce94ef44d7fef4cabe83306dce14793cf7e2b6f8d09');
	await page.close();
	await browser.close();
})();
