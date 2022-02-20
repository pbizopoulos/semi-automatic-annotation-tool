'use strict';

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

let model;
async function loadModel() {
	window.selectedModel = modelDetailsArray.find(modelDetails => modelDetails.URL == selectModel.value);
	let loadModelFunction;
	await fetch(window.selectedModel.URL)
		.then(response => response.text())
		.then((text) => {
			if (JSON.parse(text).format == 'graph-model') {
				loadModelFunction = tf.loadGraphModel;
			} else if (JSON.parse(text).format == 'layers-model') {
				loadModelFunction = tf.loadLayersModel;
			}
		})
	model = undefined;
	model = await loadModelFunction(window.selectedModel.URL, {
		onProgress: function (fraction) {
			document.getElementById('divModelDownloadFraction').innerHTML = `Downloading model, please wait ${Math.round(100*fraction)}%.`;
			document.getElementById('divModelInputShape').innerHTML = '<b>Model input shape</b>: NaN';
			document.getElementById('divModelOutputShape').innerHTML = '<b>Model output shape</b>: NaN';
			document.getElementById('divResults').innerHTML = '';
			if (fraction == 1) {
				document.getElementById('divModelDownloadFraction').innerHTML = 'Model downloaded.';
			}
			disableUI(true);
		}
	});
	document.getElementById('divModelInputShape').innerHTML = `<b>Model input shape</b>:${model.inputs[0].shape}`;
	document.getElementById('divModelOutputShape').innerHTML = `<b>Model output shape</b>:${model.outputs[0].shape}`;
	document.getElementById('divResults').innerHTML = '';
	disableUI(false);
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

	function predictView() {
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
				document.getElementById('divResults').innerHTML += `<div>${window.selectedModel.classNames[i]}: ${(classProbabilities[0][i]).toFixed(2)}%</div>`;
			}
		});
	}
	buttonPredict.onclick = () => predictView();

	let image = new Image();
	imageFileReader.onload = () => image.src = imageFileReader.result;

	buttonLoadExampleData.onclick = () => {
		image.crossOrigin = 'anonymous';
		image.src = window.selectedModel.exampleData;
	}

	image.onload = () => {
		contextImage.clearRect(0, 0, canvasWidth, canvasHeight);
		contextImage.drawImage(image, 0, 0, image.width, image.height, 0, 0, canvasWidth, canvasHeight);
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

	function predictView() {
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
	buttonPredict.onclick = () => predictView();

	let image = new Image();
	imageFileReader.onload = () => image.src = imageFileReader.result;

	buttonLoadExampleData.onclick = () => {
		image.crossOrigin = 'anonymous';
		image.src = window.selectedModel.exampleData;
	}

	image.onload = () => {
		contextImage.clearRect(0, 0, canvasWidth, canvasHeight);
		contextMask.clearRect(0, 0, canvasWidth, canvasHeight);
		contextImage.drawImage(image, 0, 0, image.width, image.height, 0, 0, canvasWidth, canvasHeight);
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

	buttonLoadExampleData.onclick = () => {
		fetch(window.selectedModel.exampleData)
			.then(response => response.text())
			.then((text) => {
				drawSignal(text);
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

	async function predictView() {
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
			document.getElementById('divResults').innerHTML += `<div>${window.selectedModel.classNames[i]}: ${(classProbabilities[0][i]).toFixed(2)}%</div>`;
		}
	}
	buttonPredict.onclick = () => predictView();
}

selectMLtype.onchange = function () {
	divInput.innerHTML = '';
	selectModel.options.length = 0;
	for (const modelDetails of modelDetailsArray) {
		if (modelDetails.type == selectMLtype.value) {
			let option = document.createElement('option');
			option.value = modelDetails.URL;
			option.innerHTML = modelDetails.name;
			selectModel.appendChild(option);
		}
	}
	window.selectedModel = modelDetailsArray.find(modelDetails => modelDetails.URL == selectModel.value);
	if (window.selectedModel.type == 'image-classification') {
		imageClassificationUI();
	} else if (window.selectedModel.type == 'image-segmentation') {
		imageSegmentationUI();
	} else if (window.selectedModel.type == 'signal-classification') {
		signalClassificationUI();
	}
	selectModel.onchange = () => loadModel();
	selectModel.onchange();
}

selectMLtype.onchange();
