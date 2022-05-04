'use strict';

const fs = require('fs');
const path = require('path');
const puppeteer = require('puppeteer');

(async () => {
	const browser = await puppeteer.launch({ headless: true, slowMo: 300 });
	const page = await browser.newPage();
	await page.goto(`file:${path.join(__dirname, '../index.html')}`);
	await page.waitForFunction("document.getElementById('modelLoadFractionDiv').textContent == 'Model loaded.'");
	await page.evaluate(() => {
		document.querySelector('#modelSelect').selectedIndex = 1;
		document.querySelector('#modelSelect').onchange();
	});
	await page.waitForSelector('#loadImagesInputFile:not([disabled])');
	const inputUploadHandle = await page.$('#loadImagesInputFile');
	const niftiFileName = 'artifacts/val_im.nii.gz';
	const nifti = fs.readFileSync(niftiFileName);
	inputUploadHandle.uploadFile(niftiFileName);
	await page.waitForSelector('#modelSelect:not([disabled])');
	await page.waitForSelector('#labelColorDiv1').then(selector => selector.click());
	await page.waitForSelector('#predictCurrentImageButton').then(selector => selector.click());
	await page.waitForTimeout(5000);
	await page.screenshot({
		path: 'artifacts/test.png',
		fullpage: true
	});
	await page.close();
	await browser.close();
})();
