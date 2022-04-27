'use strict';

const fs = require('fs');
const path = require('path');
const puppeteer = require('puppeteer');

(async () => {
	const browser = await puppeteer.launch({ headless: true });
	const page = await browser.newPage();
	await page.goto(`file:${path.join(__dirname, '../index.html')}`);
	await page.waitForTimeout(1000);
	await page.waitForSelector('#selectModel:not([disabled])');
	await page.evaluate(() => {
		document.querySelector('#selectModel').selectedIndex = 1;
		document.querySelector('#selectModel').onchange();
	});
	await page.waitForTimeout(1000);
	await page.waitForSelector('#inputLoadImages:not([disabled])');
	const inputUploadHandle = await page.$('#inputLoadImages');
	const niftiFileName = 'artifacts/val_im.nii.gz';
	const nifti = fs.readFileSync(niftiFileName);
	inputUploadHandle.uploadFile(niftiFileName);
	await page.waitForTimeout(1000);
	await page.waitForSelector('#selectModel:not([disabled])');
	await page.evaluate(() => {
		document.querySelector('#divLabelColor1').click();
		document.querySelector('#buttonPredictCurrentImage').click();
	});
	await page.waitForTimeout(10000);
	await page.screenshot({
		path: 'artifacts/test.png',
		fullpage: true
	});
	await page.close();
	await browser.close();
})();
