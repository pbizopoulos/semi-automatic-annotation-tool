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
	const browser = await puppeteer.launch({ headless: true, args: ['--use-gl=egl'] });
	const page = await browser.newPage();
	const artifactsDir = process.env.ARTIFACTS_DIR;
	await page._client().send('Page.setDownloadBehavior', {behavior: 'allow', downloadPath: path.resolve(artifactsDir)});
	const inputNiftiFileName = 'rp_im.zip';
	if (!(fs.existsSync(`${artifactsDir}/${inputNiftiFileName}`))) {
		await page.goto('https://drive.google.com/uc?id=1ruTiKdmqhqdbE9xOEmjQGing76nrTK2m');
		await page.waitForSelector('#uc-download-link').then(selector => selector.click());
		waitFile(`${artifactsDir}/${inputNiftiFileName}`);
	}
	const zip = new admzip(`${artifactsDir}/${inputNiftiFileName}`);
	zip.extractAllTo(artifactsDir, true);
	const inputDicomFileName1 = 'N2D_0001.dcm';
	if (!(fs.existsSync(`${artifactsDir}/${inputDicomFileName1}`))) {
		await page.goto('https://github.com/datalad/example-dicom-structural/blob/master/dicoms/N2D_0001.dcm');
		await page.waitForSelector('#raw-url').then(selector => selector.click());
		waitFile(`${artifactsDir}/${inputDicomFileName1}`);
	}
	const inputDicomFileName2 = 'N2D_0002.dcm';
	if (!(fs.existsSync(`${artifactsDir}/${inputDicomFileName2}`))) {
		await page.goto('https://github.com/datalad/example-dicom-structural/blob/master/dicoms/N2D_0002.dcm');
		await page.waitForSelector('#raw-url').then(selector => selector.click());
		waitFile(`${artifactsDir}/${inputDicomFileName2}`);
	}
	await page.goto(`file:${path.join(__dirname, 'docs/index.html')}`);
	await page.waitForFunction("document.getElementById('modelLoadFractionDiv').textContent == 'Model loaded.'", {waitUntil: 'load', timeout: 0});
	await page.evaluate(() => {
		document.querySelector('#modelSelect').selectedIndex = 1;
		document.querySelector('#modelSelect').onchange();
	});
	await page.waitForTimeout(1000);
	await page.waitForSelector('#loadFilesButton:not([disabled])', {waitUntil: 'load', timeout: 0}).then(selector => selector.uploadFile(`${artifactsDir}/rp_im/1.nii.gz`));
	await page.waitForSelector('#modelSelect:not([disabled])');
	await page.waitForSelector('#labelColorDiv1').then(selector => selector.click());
	await page.evaluate(() => {
		document.querySelector('#imageIndexInputRange').value = 2;
		document.querySelector('#imageIndexInputRange').oninput();
	});
	await page.waitForSelector('#resetImageValueButton').then(selector => selector.click());
	await page.waitForSelector('#predictImagesAllButton').then(selector => selector.click());
	const outputFileName = 'masks.nii';
	if (fs.existsSync(`${artifactsDir}/${outputFileName}`)) {
		await fs.unlinkSync(`${artifactsDir}/${outputFileName}`);
	}
	await page.waitForSelector('#savePredictionsToDiskButton:not([disabled])', {waitUntil: 'load', timeout: 0}).then(selector => selector.click());
	waitFile(`${artifactsDir}/${outputFileName}`);
	const outputBuffer = new fs.readFileSync(`${artifactsDir}/${outputFileName}`);
	const outputHash = crypto.createHash('sha256').update(outputBuffer).digest('hex');
	assert.strictEqual(outputHash, '819104fac59f81f4aa40ed155db475a0598384c6071e3d84d1974ff67a116734');
	await page.screenshot({
		path: `${artifactsDir}/puppeteer-screenshot.png`
	});
	const screenshotBuffer = new fs.readFileSync(`${artifactsDir}/puppeteer-screenshot.png`);
	const screenshotHash = crypto.createHash('sha256').update(screenshotBuffer).digest('hex');
	if (process.env.GITHUB_ACTIONS === undefined) {
		assert.strictEqual(screenshotHash, 'eaadd57bf9055af19d2852c528f1a498f21502e4feee4ac2636bc6823ae427f3');
	}
	await page.close();
	await browser.close();
})();
