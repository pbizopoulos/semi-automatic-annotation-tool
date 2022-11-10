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
	const browser = await puppeteer.launch({
		args: ['--use-gl=egl']
	});
	const page = await browser.newPage();
	page.on('pageerror', pageerr => {
		assert.fail(pageerr);
	});
	await page._client().send('Page.setDownloadBehavior', {
		behavior: 'allow',
		downloadPath: path.resolve('bin')
	});
	const inputNiftiFileName = 'rp_im.zip';
	if (!(fs.existsSync(`bin/${inputNiftiFileName}`))) {
		await page.goto('https://drive.google.com/uc?id=1ruTiKdmqhqdbE9xOEmjQGing76nrTK2m');
		await page.waitForSelector('#uc-download-link').then(selector => selector.click());
		waitFile(`bin/${inputNiftiFileName}`);
	}
	const zip = new admzip(`bin/${inputNiftiFileName}`);
	zip.extractAllTo('bin', true);
	const inputNiftiMasksFileName = 'masks-multiclass.nii';
	if (!(fs.existsSync(`bin/${inputNiftiMasksFileName}`))) {
		await page.goto('https://github.com/pbizopoulos/semi-automatic-annotation-tool/releases');
		await page.waitForSelector('#repo-content-pjax-container > div > div:nth-child(3) > section > div > div.col-md-9 > div > div.Box-footer > div.mb-3 > details > summary').then(selector => selector.click());
		await page.waitForSelector('#repo-content-pjax-container > div > div:nth-child(3) > section > div > div.col-md-9 > div > div.Box-footer > div.mb-3 > details > div > div > ul > li:nth-child(1) > div.d-flex.flex-justify-start.col-12.col-lg-9 > a').then(selector => selector.click());
		waitFile(`bin/${inputNiftiMasksFileName}`);
	}
	await page.goto(`file:${path.join(__dirname, 'docs/index.html')}`);
	await page.waitForFunction('document.getElementById(\'model-load-fraction-div\').textContent == \'Model loaded.\'', {
		waitUntil: 'load',
		timeout: 0
	});
	await page.waitForSelector('#load-files-input-file:not([disabled])', {
		waitUntil: 'load',
		timeout: 0
	}).then(selector => selector.uploadFile('bin/rp_im/1.nii.gz'));
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
	await page.waitForSelector('#save-predictions-to-disk-button:not([disabled])', {
		waitUntil: 'load',
		timeout: 0
	}).then(selector => selector.click());
	waitFile(`bin/${outputFileName}`);
	const outputBuffer = new fs.readFileSync(`bin/${outputFileName}`);
	const outputHash = crypto.createHash('sha256').update(outputBuffer).digest('hex');
	assert.strictEqual(outputHash, '6d1f1c28c38cab797d7500b01e5379223229b63c44bc857cbb38aab75fef75f2');
	await page.screenshot({
		path: 'bin/puppeteer-screenshot.png'
	});
	const screenshotBuffer = new fs.readFileSync('bin/puppeteer-screenshot.png');
	const screenshotHash = crypto.createHash('sha256').update(screenshotBuffer).digest('hex');
	assert.strictEqual(screenshotHash, '532c931932481097cad66314adff8badd8e5fce1f3fb992f50159dd031af33ad');
	await page.waitForSelector('#load-predictions-input-file').then(selector => selector.uploadFile(`bin/${inputNiftiMasksFileName}`));
	await page.evaluate(() => {
		document.querySelector('#image-index-input-range').value = 10;
		document.querySelector('#image-index-input-range').oninput();
	});
	await page.waitForTimeout(1000);
	await page.screenshot({
		path: 'bin/puppeteer-screenshot-2.png'
	});
	const screenshotBuffer2 = new fs.readFileSync('bin/puppeteer-screenshot-2.png');
	const screenshotHash2 = crypto.createHash('sha256').update(screenshotBuffer2).digest('hex');
	assert.strictEqual(screenshotHash2, 'e7ed9009047a9e3bf541b600d1aafd30d345b7147c1ff6f41dc6cfb8218fec10');
	await page.close();
	await browser.close();
})();
