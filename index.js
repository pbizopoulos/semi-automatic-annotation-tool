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
	assert.strictEqual(outputHash, '3a8e856d715b1fc400b098098fb0deea70e7078f95434b9515e8d648667a5dde');
	await page.screenshot({path: 'bin/puppeteer-screenshot.png'});
	const screenshotBuffer = new fs.readFileSync('bin/puppeteer-screenshot.png');
	const screenshotHash = crypto.createHash('sha256').update(screenshotBuffer).digest('hex');
	assert.strictEqual(screenshotHash, 'f2d9936f89558d549c85e6c22b11e74d774070215158a94d3a3c7c01caf9a40f');
	await page.close();
	await browser.close();
})();
