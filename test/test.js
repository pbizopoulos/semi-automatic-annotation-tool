'use strict';

const fs = require('fs');
const path = require('path');
const puppeteer = require('puppeteer');

(async () => {
	const browser = await puppeteer.launch({ headless: true });
	const page = await browser.newPage();
	await page.goto(`file:${path.join(__dirname, '../index.html')}`);
	await page.waitForTimeout(1000);
	await page.waitForSelector('#modelSelect:not([disabled])');
	await page.evaluate(() => {
		document.querySelector('#modelSelect').selectedIndex = 1;
		document.querySelector('#modelSelect').onchange();
	});
	await page.waitForTimeout(1000);
	await page.waitForSelector('#loadImagesInputFile:not([disabled])');
	const inputUploadHandle = await page.$('#loadImagesInputFile');
	const niftiFileName = 'artifacts/val_im.nii.gz';
	const nifti = fs.readFileSync(niftiFileName);
	inputUploadHandle.uploadFile(niftiFileName);
	await page.waitForTimeout(1000);
	await page.waitForSelector('#modelSelect:not([disabled])');
	await page.evaluate(() => {
		document.querySelector('#labelColorDiv1').click();
		document.querySelector('#predictCurrentImageButton').click();
	});
	await page.waitForTimeout(10000);
	await page.screenshot({
		path: 'artifacts/test.png',
		fullpage: true
	});
	await page.close();
	await browser.close();
})();
