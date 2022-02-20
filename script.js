'use strict';

const configURLarray = [
	'https://raw.githubusercontent.com/pbizopoulos/comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-in-ct/master/wbml/lesion-segmentation.json',
	'https://raw.githubusercontent.com/pbizopoulos/comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-in-ct/master/wbml/lung-segmentation.json',
	'https://raw.githubusercontent.com/pbizopoulos/signal2image-modules-in-deep-neural-networks-for-eeg-classification/master/wbml/eeg-classification.json',
	'https://raw.githubusercontent.com/pbizopoulos/tmp/master/wbml/lung-classification.json',
]

const canvasWidth = 256;
const canvasHeight = 256;
const divInput = document.getElementById('divInput');

const signalFileReader = new FileReader();
function signalLoadView() {
	const files = event.currentTarget.files;
	if (files[0]) {
		signalFileReader.readAsText(files[0]);
	}
}

const imageFileReader = new FileReader();
function imageLoadView() {
	const files = event.currentTarget.files;
	if (files[0]) {
		imageFileReader.readAsDataURL(files[0]);
	}
}

function disableUI(argument) {
	const nodes = document.getElementById('divInputControl').getElementsByTagName('*');
	for(let i = 0; i < nodes.length; i++){
		nodes[i].disabled = argument;
	}
}

function imageClassificationUI() {
	inputLoadData.onchange = () => imageLoadView();
	inputLoadData.accept = 'image/*';

	const canvasImage = document.createElement('canvas');
	canvasImage.id = 'canvasImage';
	canvasImage.width = canvasWidth;
	canvasImage.height = canvasHeight;
	divInput.appendChild(canvasImage);
	const contextImage = canvasImage.getContext('2d');

	const pixelScaling = 3/255;
	const pixelBaseline = 1.5;

	buttonPredict.onclick = () => {
		if (model === undefined) {
			return;
		}
		tf.tidy(() => {
			let fromPixels = tf.browser.fromPixels(canvasImage);
			fromPixels = tf.image.resizeBilinear(fromPixels, [model.inputs[0].shape[1], model.inputs[0].shape[2]]);
			let pixels = fromPixels.slice([0, 0, 2]).expandDims(0);
			pixels = pixels.mul(pixelScaling);
			pixels = pixels.sub(pixelBaseline);
			const modelOutput = model.predict(pixels);
			const classProbabilities = modelOutput.softmax().mul(100).arraySync();
			document.getElementById('divResults').innerHTML = '';
			for (let i = 0; i < classProbabilities[0].length; i++) {
				document.getElementById('divResults').innerHTML += `<div>${window.configSelected.classNames[i]}: ${(classProbabilities[0][i]).toFixed(2)}%</div>`;
			}
		});
	}

	let image = new Image();
	imageFileReader.onload = () => image.src = imageFileReader.result;

	buttonLoadModelExampleData.onclick = () => {
		disableUI(true);
		image.crossOrigin = 'anonymous';
		image.src = window.configSelected.exampleData;
	}

	image.onload = () => {
		contextImage.clearRect(0, 0, canvasWidth, canvasHeight);
		contextImage.drawImage(image, 0, 0, image.width, image.height, 0, 0, canvasWidth, canvasHeight);
		disableUI(false);
	};
}

function imageSegmentationUI() {
	inputLoadData.onchange = () => imageLoadView();
	inputLoadData.accept = 'image/*';

	const canvasImage = document.createElement('canvas');
	canvasImage.id = 'canvasImage';
	canvasImage.width = canvasWidth;
	canvasImage.height = canvasHeight;
	divInput.appendChild(canvasImage);
	const contextImage = canvasImage.getContext('2d');

	const canvasMask = document.createElement('canvas');
	canvasMask.id = 'canvasMask';
	canvasMask.width = canvasWidth;
	canvasMask.height = canvasHeight;
	canvasMask.style.position = 'absolute';
	canvasMask.style.top = '0px';
	canvasMask.style.left = '0px';
	divInput.appendChild(canvasMask);
	const contextMask = canvasMask.getContext('2d');

	const pixelScaling = 3/255;
	const pixelBaseline = 1.5;

	buttonPredict.onclick = () => {
		if (model === undefined) {
			return;
		}
		tf.tidy(() => {
			let fromPixels = tf.browser.fromPixels(canvasImage);
			const originalShape = fromPixels.shape.slice(0, 2);
			fromPixels = tf.image.resizeBilinear(fromPixels, [model.inputs[0].shape[2], model.inputs[0].shape[3]]);
			let pixels = fromPixels.slice([0, 0, 2]).squeeze(-1).expandDims(0).expandDims(0);
			pixels = pixels.mul(pixelScaling);
			pixels = pixels.sub(pixelBaseline);
			const mask = model.predict(pixels);
			let maskToPixels = mask.squeeze(0).squeeze(0);
			const alphaTensor = tf.tensor([0.3]);
			const alphaChannel = alphaTensor.where(maskToPixels.greaterEqual(0.5), 0);
			maskToPixels = tf.stack([maskToPixels, tf.zerosLike(maskToPixels), tf.zerosLike(maskToPixels), alphaChannel], -1);
			maskToPixels = tf.image.resizeBilinear(maskToPixels, originalShape);
			contextMask.clearRect(0, 0, canvasWidth, canvasHeight);
			tf.browser.toPixels(maskToPixels, canvasMask);
		});
	}

	let image = new Image();
	imageFileReader.onload = () => image.src = imageFileReader.result;

	buttonLoadModelExampleData.onclick = () => {
		disableUI(true);
		image.crossOrigin = 'anonymous';
		image.src = window.configSelected.exampleData;
	}

	image.onload = () => {
		contextImage.clearRect(0, 0, canvasWidth, canvasHeight);
		contextMask.clearRect(0, 0, canvasWidth, canvasHeight);
		contextImage.drawImage(image, 0, 0, image.width, image.height, 0, 0, canvasWidth, canvasHeight);
		disableUI(false);
	};
}

function signalClassificationUI() {
	inputLoadData.onchange = () => signalLoadView();
	inputLoadData.accept = '.txt,.csv';

	let csvDataset;
	function drawSignal(text) {
		const array = text.match(/\d+(?:\.\d+)?/g).map(Number);
		csvDataset = tf.tensor(array);
		const x = d3.scaleLinear()
			.domain([0, csvDataset.size])
			.range([0, canvasWidth]);
		const y = d3.scaleLinear()
			.domain([csvDataset.min().arraySync(), csvDataset.max().arraySync()])
			.range([canvasHeight, 0]);
		const line = d3.line()
			.x((d,i) => x(i))
			.y(d => y(d));
		d3.select('#pathInput')
			.attr('d', line(csvDataset.arraySync()));
	}

	buttonLoadModelExampleData.onclick = () => {
		disableUI(true);
		fetch(window.configSelected.exampleData)
			.then(response => response.text())
			.then((text) => {
				drawSignal(text);
				disableUI(false);
			})
	}

	const svgInput = d3.select('#divInput')
		.append('svg')
		.attr('viewBox', [0, 0, canvasWidth, canvasHeight]);
	svgInput.append('path')
		.attr('id', 'pathInput')
		.style('fill', 'none')
		.style('stroke', 'blue');

	signalFileReader.onload = () => drawSignal(signalFileReader.result);

	buttonPredict.onclick = async function() {
		if (csvDataset === undefined) {
			return;
		}
		if (model === undefined) {
			return;
		}
		let csvDatasetTmp = csvDataset.expandDims(0).expandDims(2);
		csvDatasetTmp = tf.image.resizeBilinear(csvDatasetTmp, [1, model.inputs[0].shape[2]]);
		csvDatasetTmp = csvDatasetTmp.reshape([1, 1, model.inputs[0].shape[2]]);
		const modelOutput = await model.executeAsync(csvDatasetTmp);
		const classProbabilities = modelOutput.softmax().mul(100).arraySync();
		document.getElementById('divResults').innerHTML = '';
		for (let i = 0; i < classProbabilities[0].length; i++) {
			document.getElementById('divResults').innerHTML += `<div>${window.configSelected.classNames[i]}: ${(classProbabilities[0][i]).toFixed(2)}%</div>`;
		}
	}
}

let configArray = [];
let model;

async function initialize() {
	for (const [i, configURL] of configURLarray.entries()) {
		await fetch(configURL)
			.then(response => response.text())
			.then((text) => {
				configArray[i] = JSON.parse(text);
				let option = document.createElement('option');
				option.value = configArray[i].URL;
				option.innerHTML = configArray[i].name;
				selectModel.appendChild(option);
			})
	}
	selectModel.onchange = async function () {
		divInput.innerHTML = '';
		window.configSelected = configArray.find(config => config.URL == selectModel.value);
		if (window.configSelected.type == 'image-classification') {
			imageClassificationUI();
		} else if (window.configSelected.type == 'image-segmentation') {
			imageSegmentationUI();
		} else if (window.configSelected.type == 'signal-classification') {
			signalClassificationUI();
		}
		let loadModelFunction;
		await fetch(window.configSelected.URL)
			.then(response => response.text())
			.then((text) => {
				if (JSON.parse(text).format == 'graph-model') {
					loadModelFunction = tf.loadGraphModel;
				} else if (JSON.parse(text).format == 'layers-model') {
					loadModelFunction = tf.loadLayersModel;
				}
			})
		model = await loadModelFunction(window.configSelected.URL, {
			onProgress: function (fraction) {
				document.getElementById('divModelDownloadFraction').innerHTML = `Downloading model, please wait ${Math.round(100*fraction)}%.`;
				document.getElementById('divModelInputShape').innerHTML = '<b>Model input shape</b>: NaN';
				document.getElementById('divModelOutputShape').innerHTML = '<b>Model output shape</b>: NaN';
				document.getElementById('divResults').innerHTML = '';
				disableUI(true);
			}
		});
		document.getElementById('divModelDownloadFraction').innerHTML = 'Model downloaded.';
		document.getElementById('divModelInputShape').innerHTML = `<b>Model input shape</b>:${model.inputs[0].shape}`;
		document.getElementById('divModelOutputShape').innerHTML = `<b>Model output shape</b>:${model.outputs[0].shape}`;
		document.getElementById('divResults').innerHTML = '';
		disableUI(false);
	}
	selectModel.onchange();
}

initialize();
