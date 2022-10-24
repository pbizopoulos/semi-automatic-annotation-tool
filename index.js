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
	const browser = await puppeteer.launch({headless: true, args: ['--use-gl=egl']});
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
	await page.waitForFunction('document.getElementById(\'modelLoadFractionDiv\').textContent == \'Model loaded.\'', {waitUntil: 'load', timeout: 0});
	await page.waitForTimeout(1000);
	await page.waitForSelector('#loadFilesInputFile:not([disabled])', {waitUntil: 'load', timeout: 0}).then(selector => selector.uploadFile('bin/rp_im/1.nii.gz'));
	await page.waitForSelector('#modelSelect:not([disabled])');
	await page.waitForSelector('#labelColorDiv1').then(selector => selector.click());
	await page.evaluate(() => {
		document.querySelector('#imageIndexInputRange').value = 2;
		document.querySelector('#imageIndexInputRange').oninput();
	});
	await page.waitForSelector('#resetImageValueButton').then(selector => selector.click());
	await page.waitForSelector('#predictImagesAllButton').then(selector => selector.click());
	const outputFileName = 'masks.nii';
	if (fs.existsSync(`bin/${outputFileName}`)) {
		await fs.unlinkSync(`bin/${outputFileName}`);
	}
	await page.waitForSelector('#savePredictionsToDiskButton:not([disabled])', {waitUntil: 'load', timeout: 0}).then(selector => selector.click());
	waitFile(`bin/${outputFileName}`);
	const outputBuffer = new fs.readFileSync(`bin/${outputFileName}`);
	const outputHash = crypto.createHash('sha256').update(outputBuffer).digest('hex');
	assert.strictEqual(outputHash, '819104fac59f81f4aa40ed155db475a0598384c6071e3d84d1974ff67a116734');
	await page.screenshot({path: 'bin/puppeteer-screenshot.png'});
	const screenshotBuffer = new fs.readFileSync('bin/puppeteer-screenshot.png');
	const screenshotHash = crypto.createHash('sha256').update(screenshotBuffer).digest('hex');
	assert.strictEqual(screenshotHash, '187d4157a8f27246784c5a46b74fa08f393a114da0df28128ac091596d6ae4d1');
	await page.close();
	await browser.close();
})();
