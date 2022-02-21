'use strict';

const configURLarray = [
	'https://raw.githubusercontent.com/pbizopoulos/comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-in-ct/master/wbml/lesion-segmentation.json',
	'https://raw.githubusercontent.com/pbizopoulos/comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-in-ct/master/wbml/lung-segmentation.json',
	'https://raw.githubusercontent.com/pbizopoulos/signal2image-modules-in-deep-neural-networks-for-eeg-classification/master/wbml/eeg-classification.json',
	'https://raw.githubusercontent.com/pbizopoulos/tmp/master/wbml/lung-classification.json',
]
const divInput = document.getElementById('divInput');
divInput.style.width = '512px';
divInput.style.height = '512px';
const canvasWidth = parseInt(divInput.style.width);
const canvasHeight = parseInt(divInput.style.height);
let configArray = [];
let configSelected;
let model;
let columns;
let imageIndex = 0;
let imageOffset;
let imageSize;
let imageValueMin = 0;
let imageValueRange = 1;
let images;
let masks = [];
let numImages = 0;
let rows;
const pixelScaling = 3/255;
const pixelBaseline = 1.5;
let csvDataset;
let image = new Image();
const imageFileReader = new FileReader();
imageFileReader.onload = () => image.src = imageFileReader.result;
const signalFileReader = new FileReader();
signalFileReader.onload = () => drawSignal(signalFileReader.result);

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

document.getElementById('buttonLoadModelExampleData').onclick = () => {
	disableUI(true);
	images = undefined;
	masks = [];
	if ((configSelected.machineLearningType == 'image classification') || (configSelected.machineLearningType == 'image segmentation')) {
		image.crossOrigin = 'anonymous';
		image.src = configSelected.exampleDataURL;
	} else if (configSelected.machineLearningType == 'signal classification') {
		fetch(configSelected.exampleDataURL)
			.then(response => response.text())
			.then((text) => {
				drawSignal(text);
				disableUI(false);
			})
	}
}

document.getElementById('checkboxShowMask').onchange = () => {
	if (event.currentTarget.checked) {
		canvasMask.style.display = '';
	} else {
		canvasMask.style.display = 'none';
	}
}

document.getElementById('inputLoadData').onchange = () => {
	const files = event.currentTarget.files;
	document.getElementById('spanNumberOfFiles').textContent = files.length;
	if (files[0] == undefined) {
		return;
	}
	if ((configSelected.machineLearningType == 'image classification') || (configSelected.machineLearningType == 'image segmentation')) {
		if (files[0].name.includes('.png') || files[0].name.includes('.jpg')) {
			imageFileReader.readAsDataURL(files[0]);
		} else if (files[0].name.includes('.nii')) {
			readNiiFile(files[0]);
		} else if (files[0].name.includes('.dcm')) {
			for (let i = 0; i < files.length; i++) {
				const reader = new FileReader();
				reader.readAsArrayBuffer(files[i]);
			}
			itk.readImageDICOMFileSeries(files)
				.then(function ({ image }) {
					itk.writeImageArrayBuffer(null, false, image, 'unnamed.nii')
						.then((data) => {
							const blob = new Blob([data.arrayBuffer]);
							readNiiFile(blob);
						});
				});
		}
	} else if (configSelected.machineLearningType == 'signal classification') {
		if (files[0].name.includes('.txt') || files[0].name.includes('.png')) {
			signalFileReader.readAsText(files[0]);
		}
	}
}

window.addEventListener('keydown', function (event) {
	if (event.key === 'ArrowDown' && (imageIndex > 0)) {
		imageIndex--;
	} else if (event.key === 'ArrowUp' && (imageIndex < numImages)) {
		imageIndex++;
	} else {
		return;
	}
	imageOffset = imageSize * imageIndex;
	document.getElementById('spanFileIndex').textContent = imageIndex;
	visualizeImageDataImage();
	visualizeImageDataMask();
});

function visualizeImageDataImage() {
	imageOffset = imageSize * imageIndex;
	const imageDataImage = new ImageData(columns, rows);
	for (let i = 0; i < rows; i++) {
		const rowOffset = i * columns;
		for (let j = 0; j < columns; j++) {
			const imageValueOffset = imageOffset + rowOffset + j;
			const imageValue = (images[imageValueOffset] - imageValueMin) / imageValueRange;
			const offsetMult4 = (rowOffset + j) * 4;
			imageDataImage.data[offsetMult4] = imageValue;
			imageDataImage.data[offsetMult4 + 1] = imageValue;
			imageDataImage.data[offsetMult4 + 2] = imageValue;
			imageDataImage.data[offsetMult4 + 3] = 255;
		}
	}
	let contextImage = canvasImage.getContext('2d');
	contextImage.putImageData(imageDataImage, 0, 0);
}

function visualizeImageDataMask() {
	let contextMask = canvasMask.getContext('2d');
	contextMask.clearRect(0, 0, canvasWidth, canvasHeight);
	if (masks[imageIndex]) {
		tf.browser.toPixels(masks[imageIndex], canvasMask);
	}
}

function readNiiFile(file) {
	const reader = new FileReader();
	reader.onloadend = function (event) {
		if (event.target.readyState === FileReader.DONE) {
			let niftiHeader;
			let niftiImage;
			let decompressedFile;
			if (nifti.isCompressed(event.target.result)) {
				decompressedFile = nifti.decompress(event.target.result);
			} else {
				decompressedFile = event.target.result;
			}
			if (nifti.isNIFTI(decompressedFile)) {
				niftiHeader = nifti.readHeader(decompressedFile);
				niftiImage = nifti.readImage(niftiHeader, decompressedFile);
			}
			switch (niftiHeader.datatypeCode) {
				case nifti.NIFTI1.TYPE_UINT8:
					images = new Uint8Array(niftiImage);
					break;
				case nifti.NIFTI1.TYPE_INT16:
					images = new Int16Array(niftiImage);
					break;
				case nifti.NIFTI1.TYPE_INT32:
					images = new Int32Array(niftiImage);
					break;
				case nifti.NIFTI1.TYPE_FLOAT32:
					images = new Float32Array(niftiImage);
					break;
				case nifti.NIFTI1.TYPE_FLOAT64:
					images = new Float64Array(niftiImage);
					break;
				case nifti.NIFTI1.TYPE_INT8:
					images = new Int8Array(niftiImage);
					break;
				case nifti.NIFTI1.TYPE_UINT16:
					images = new Uint16Array(niftiImage);
					break;
				case nifti.NIFTI1.TYPE_UINT32:
					images = new Uint32Array(niftiImage);
					break;
				default:
					return;
			}
			numImages = niftiHeader.dims[3] - 1;
			document.getElementById('spanNumberOfFiles').textContent = numImages;
			imageIndex = 0;
			document.getElementById('spanFileIndex').textContent = imageIndex;
			rows = niftiHeader.dims[2];
			columns = niftiHeader.dims[1];
			document.getElementById('spanFileShape').textContent = `${rows}x${columns}`;
			imageSize = rows * columns;
			canvasImage.height = rows;
			canvasImage.width = columns;
			canvasMask.height = rows;
			canvasMask.width = columns;
			visualizeImageDataImage();
		}
		disableUI(false);
	};
	reader.readAsArrayBuffer(file);
}

function disableUI(argument) {
	const nodes = document.getElementById('divInputControl').getElementsByTagName('*');
	for(let i = 0; i < nodes.length; i++){
		nodes[i].disabled = argument;
	}
}

async function initialize() {
	for (const [i, configURL] of configURLarray.entries()) {
		await fetch(configURL)
			.then(response => response.text())
			.then((text) => {
				configArray[i] = JSON.parse(text);
				let option = document.createElement('option');
				option.value = configArray[i].modelURL;
				option.textContent = configArray[i].name;
				selectModel.appendChild(option);
			})
	}
	selectModel.onchange = async function () {
		divInput.textContent = '';
		configSelected = configArray.find(config => config.modelURL == selectModel.value);
		document.getElementById('aProjectURL').href = configSelected.projectURL;
		if (configSelected.machineLearningType == 'image classification') {
			document.getElementById('inputLoadData').accept = 'image/*,.dcm,.nii,.nii.gz';
			divCheckboxShowMask.style.display = 'none';

			const canvasImage = document.createElement('canvas');
			canvasImage.id = 'canvasImage';
			canvasImage.width = canvasWidth;
			canvasImage.height = canvasHeight;
			divInput.appendChild(canvasImage);
			const contextImage = canvasImage.getContext('2d');

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
					document.getElementById('divResults').textContent = '';
					for (let i = 0; i < classProbabilities[0].length; i++) {
						let divElement = document.createElement('div');
						divElement.textContent = `${configSelected.classNames[i]}: ${(classProbabilities[0][i]).toFixed(2)}%`
						document.getElementById('divResults').append(divElement);
					}
				});
			}

			image.onload = () => {
				contextImage.clearRect(0, 0, canvasWidth, canvasHeight);
				contextImage.drawImage(image, 0, 0, image.width, image.height, 0, 0, canvasWidth, canvasHeight);
				disableUI(false);
			};
		} else if (configSelected.machineLearningType == 'image segmentation') {
			document.getElementById('inputLoadData').accept = 'image/*,.dcm,.nii,.nii.gz';
			divCheckboxShowMask.style.display = '';

			const canvasImage = document.createElement('canvas');
			canvasImage.id = 'canvasImage';
			canvasImage.width = canvasWidth;
			canvasImage.height = canvasHeight;
			canvasImage.style.position = 'absolute';
			divInput.appendChild(canvasImage);
			const contextImage = canvasImage.getContext('2d');

			const canvasMask = document.createElement('canvas');
			canvasMask.id = 'canvasMask';
			canvasMask.width = canvasWidth;
			canvasMask.height = canvasHeight;
			canvasMask.style.position = 'absolute';
			divInput.appendChild(canvasMask);
			const contextMask = canvasMask.getContext('2d');

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
					masks[imageIndex] = tf.keep(maskToPixels);
					tf.browser.toPixels(maskToPixels, canvasMask);
				});
			}

			image.onload = () => {
				contextImage.clearRect(0, 0, canvasWidth, canvasHeight);
				contextMask.clearRect(0, 0, canvasWidth, canvasHeight);
				contextImage.drawImage(image, 0, 0, image.width, image.height, 0, 0, canvasWidth, canvasHeight);
				disableUI(false);
			};
		} else if (configSelected.machineLearningType == 'signal classification') {
			document.getElementById('inputLoadData').accept = '.txt,.csv';
			divCheckboxShowMask.style.display = 'none';

			const svgInput = d3.select('#divInput')
				.append('svg')
				.attr('viewBox', [0, 0, canvasWidth, canvasHeight]);
			svgInput.append('path')
				.attr('id', 'pathInput')
				.style('fill', 'none')
				.style('stroke', 'blue');

			buttonPredict.onclick = async function() {
				if (model === undefined) {
					return;
				}
				if (csvDataset === undefined) {
					return;
				}
				let csvDatasetTmp = csvDataset.expandDims(0).expandDims(2);
				csvDatasetTmp = tf.image.resizeBilinear(csvDatasetTmp, [1, model.inputs[0].shape[2]]);
				csvDatasetTmp = csvDatasetTmp.reshape([1, 1, model.inputs[0].shape[2]]);
				const modelOutput = await model.executeAsync(csvDatasetTmp);
				const classProbabilities = modelOutput.softmax().mul(100).arraySync();
				document.getElementById('divResults').textContent = '';
				for (let i = 0; i < classProbabilities[0].length; i++) {
					let divElement = document.createElement('div');
					divElement.textContent = `${configSelected.classNames[i]}: ${(classProbabilities[0][i]).toFixed(2)}%`
					document.getElementById('divResults').append(divElement);
				}
			}
		}
		let loadModelFunction;
		await fetch(configSelected.modelURL)
			.then(response => response.text())
			.then((text) => {
				if (JSON.parse(text).format == 'graph-model') {
					loadModelFunction = tf.loadGraphModel;
				} else if (JSON.parse(text).format == 'layers-model') {
					loadModelFunction = tf.loadLayersModel;
				}
			})
		model = await loadModelFunction(configSelected.modelURL, {
			onProgress: function (fraction) {
				document.getElementById('divModelLoadFraction').textContent = `${Math.round(100*fraction)}%.`;
				document.getElementById('spanModelInputShape').textContent = 'NaN';
				document.getElementById('spanModelOutputShape').textContent = 'NaN';
				document.getElementById('spanModelTrainable').textContent = 'NaN';
				document.getElementById('divResults').textContent = '';
				disableUI(true);
			}
		});
		document.getElementById('divModelLoadFraction').textContent = 'Model loaded.';
		document.getElementById('spanModelInputShape').textContent = model.inputs[0].shape;
		document.getElementById('spanModelOutputShape').textContent = model.outputs[0].shape;
		if (model.trainable_) {
			document.getElementById('spanModelTrainable').textContent = 'True';
		} else {
			document.getElementById('spanModelTrainable').textContent = 'False';
		}
		document.getElementById('divResults').textContent = '';
		disableUI(false);
	}
	selectModel.onchange();
}

initialize();
