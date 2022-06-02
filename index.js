'use strict';

const assert = require('assert');
const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
const puppeteer = require('puppeteer');


async function waitFile(filename) {
	return new Promise(async (resolve, reject) => {
		if (!fs.existsSync(filename)) {
			await delay(1000);
			await waitFile(filename);
		}
		resolve();
	})
}

function delay(time) {
	return new Promise(function(resolve) {
		setTimeout(resolve, time)
	});
}

(async () => {
	const browser = await puppeteer.launch({ headless: true, slowMo: 300 });
	const page = await browser.newPage();
	const artifactsDir = process.env.ARTIFACTS_DIR;
	await page._client.send('Page.setDownloadBehavior', {behavior: 'allow', downloadPath: path.resolve(artifactsDir)});
	const inputNiftiFileName = 'tr_im.nii.gz';
	if (!(fs.existsSync(`${artifactsDir}/${inputNiftiFileName}`))) {
		await page.goto('https://drive.google.com/uc?id=1SJoMelgRqb0EuqlTuq6dxBWf2j9Kno8S');
		await page.waitForSelector('#uc-download-link').then(selector => selector.click());
		await waitFile(`${artifactsDir}/${inputNiftiFileName}`);
	}
	const inputDicomFileName1 = 'N2D_0001.dcm';
	if (!(fs.existsSync(`${artifactsDir}/${inputDicomFileName1}`))) {
		await page.goto('https://github.com/datalad/example-dicom-structural/blob/master/dicoms/N2D_0001.dcm');
		await page.waitForSelector('#raw-url').then(selector => selector.click());
		await waitFile(`${artifactsDir}/${inputDicomFileName1}`);
	}
	const inputDicomFileName2 = 'N2D_0002.dcm';
	if (!(fs.existsSync(`${artifactsDir}/${inputDicomFileName2}`))) {
		await page.goto('https://github.com/datalad/example-dicom-structural/blob/master/dicoms/N2D_0002.dcm');
		await page.waitForSelector('#raw-url').then(selector => selector.click());
		await waitFile(`${artifactsDir}/${inputDicomFileName2}`);
	}
	await page.goto(`file:${path.join(__dirname, 'docs/index.html')}`);
	await page.waitForFunction("document.getElementById('modelLoadFractionDiv').textContent == 'Model loaded.'");
	await page.evaluate(() => {
		document.querySelector('#modelSelect').selectedIndex = 1;
		document.querySelector('#modelSelect').onchange();
	});
	await page.waitForSelector('#loadFilesInputFile:not([disabled])').then(selector => selector.uploadFile(`${artifactsDir}/${inputNiftiFileName}`));
	await page.waitForSelector('#modelSelect:not([disabled])');
	await page.waitForSelector('#labelColorDiv1').then(selector => selector.click());
	await page.evaluate(() => {
		document.querySelector('#imageIndexInputRange').value = 2;
		document.querySelector('#imageIndexInputRange').oninput();
	});
	await page.waitForSelector('#resetImageValueButton').then(selector => selector.click());
	await page.waitForSelector('#predictImageCurrentButton').then(selector => selector.click());
	const outputFileName = 'masks.nii';
	if (fs.existsSync(`${artifactsDir}/${outputFileName}`)) {
		await fs.unlinkSync(`${artifactsDir}/${outputFileName}`);
	}
	await page.waitForSelector('#savePredictionsToDiskButton:not([disabled])').then(selector => selector.click());
	await waitFile(`${artifactsDir}/${outputFileName}`);
	const outputBuffer = new fs.readFileSync(`${artifactsDir}/${outputFileName}`);
	const outputHash = crypto.createHash('sha256').update(outputBuffer).digest('hex');
	assert(outputHash === 'c4abfd6bb2acc4cf729076031fbfa4521b5c35302c3f3e6accae69f601c247f9');
	await page.screenshot({
		path: `${artifactsDir}/puppeteer-screenshot.png`
	});
	const screenshotBuffer = new fs.readFileSync(`${artifactsDir}/puppeteer-screenshot.png`);
	const screenshotHash = crypto.createHash('sha256').update(screenshotBuffer).digest('hex');
	if (process.env.GITHUB_ACTIONS === undefined) {
		assert(screenshotHash === '86bf8f85043b541a56db08fbbec2cd6e9b283934e89cdbd4e418446c70c85a6c');
	}
	await page.close();
	await browser.close();
})();
