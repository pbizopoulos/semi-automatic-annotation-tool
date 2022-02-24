'use strict';

async function saveModelToDisk() {
	await model.save('downloads://saved-model');
}

async function uploadModelToServer() {
	await model.save('http://172.17.0.2:5000/upload');
}

const labelsColormap = [
	[ 255, 255, 255 ],
	[ 31, 119, 180 ],
	[ 174, 199, 232 ],
	[ 255, 127, 14 ],
	[ 255, 187, 120 ],
	[ 44, 160, 44 ],
	[ 152, 223, 138 ],
	[ 214, 39, 40 ],
	[ 255, 152, 150 ],
	[ 148, 103, 189 ],
	[ 197, 176, 213 ],
	[ 140, 86, 75 ],
	[ 196, 156, 148 ],
	[ 227, 119, 194 ],
	[ 247, 182, 210 ],
	[ 127, 127, 127 ],
	[ 199, 199, 199 ],
	[ 188, 189, 34 ],
	[ 219, 219, 141 ],
	[ 23, 190, 207 ],
	[ 158, 218, 229 ]
];

const configURLarray = [
	'https://raw.githubusercontent.com/pbizopoulos/comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-in-ct/master/wbml/lesion-segmentation.json',
	'https://raw.githubusercontent.com/pbizopoulos/comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-in-ct/master/wbml/lung-segmentation.json',
	'https://raw.githubusercontent.com/pbizopoulos/tmp/main/wbml/lung-classification.json',
]

let configArray = [];
let configSelected;
let currentLabel = 0;
let decompressedFile;
let drawActivated = false;
let files;
let imageCurrentIndex;
let imageOffset;
let imageSize = 1;
let imageValueMin;
let imageValueRange;
let model;
let numImages;
let offsetX;
let offsetY;
let valueRangeActivated = false;
let modelInputShape;

let images = new Uint8Array(imageSize);
let masks = new Uint8Array(imageSize);
let classes = new Uint8Array(1000).fill(1); // tmp hardcoded max value, remove later

const buttonSaveModelToDisk = document.getElementById('buttonSaveModelToDisk');
const buttonTrain = document.getElementById('buttonTrain');
const buttonUploadModelToServer = document.getElementById('buttonUploadModelToServer');
const canvasBrush = document.getElementById('canvasBrush');
const canvasImage = document.getElementById('canvasImage');
const canvasMask = document.getElementById('canvasMask');
const divAccuracy = document.getElementById('divAccuracy');
const divBrushSize = document.getElementById('divBrushSize');
const divControl = document.getElementById('divControl');
const divCurrentEpoch = document.getElementById('divCurrentEpoch');
const divLabelList = document.getElementById('divLabelList');
const divLoss = document.getElementById('divLoss');
const divModelLoadFraction = document.getElementById('divModelLoadFraction');
const divNumEpochs = document.getElementById('divNumEpochs');
const inputFile = document.getElementById('inputFile');
const inputLoadPredictions = document.getElementById('inputLoadPredictions');
const inputNumEpochs = document.getElementById('inputNumEpochs');
const inputRangeBrushSize = document.getElementById('inputRangeBrushSize');
const selectModel = document.getElementById('selectModel');
const spanAccuracy = document.getElementById('spanAccuracy');
const spanCurrentEpoch = document.getElementById('spanCurrentEpoch');
const spanHeightXwidth = document.getElementById('spanHeightXwidth');
const spanImageIndex = document.getElementById('spanImageIndex');
const spanImageValueMax = document.getElementById('spanImageValueMax');
const spanImageValueMin = document.getElementById('spanImageValueMin');
const spanLoss = document.getElementById('spanLoss');
const spanModelInputShape = document.getElementById('spanModelInputShape');
const spanModelPredictionShape = document.getElementById('spanModelPredictionShape');
const spanModelTrainable = document.getElementById('spanModelTrainable');

const contextBrush = canvasBrush.getContext('2d');
const contextImage = canvasImage.getContext('2d');
const contextMask = canvasMask.getContext('2d');

canvasBrush.addEventListener('mousedown', (event) => {
	if (event.button === 0) {
		drawActivated = true;
	} else if (event.button === 2) {
		valueRangeActivated = true;
	}
});

window.addEventListener('contextmenu', function (event) {
	event.preventDefault();
}, false);

window.addEventListener('mouseleave', () => {
	drawActivated = false;
	valueRangeActivated = false;
});

window.addEventListener('mouseup', () => {
	drawActivated = false;
	valueRangeActivated = false;
});

function disableUI(argument) {
	const nodes = divControl.getElementsByTagName('*');
	for(let i = 0; i < nodes.length; i++){
		nodes[i].disabled = argument;
	}
}

function drawCanvas() {
	const imageDataImage = new ImageData(canvasImage.width, canvasImage.height);
	for (let i = 0; i < canvasImage.height; i++) {
		const rowOffset = i * canvasImage.width;
		for (let j = 0; j < canvasImage.width; j++) {
			const imageValueOffset = imageOffset + rowOffset + j;
			const imageValue = (images[imageValueOffset] - imageValueMin) / imageValueRange;
			const offsetMult4 = (rowOffset + j) * 4;
			imageDataImage.data[offsetMult4] = imageValue;
			imageDataImage.data[offsetMult4 + 1] = imageValue;
			imageDataImage.data[offsetMult4 + 2] = imageValue;
			imageDataImage.data[offsetMult4 + 3] = 255;
		}
	}
	contextImage.putImageData(imageDataImage, 0, 0);
	if (configSelected.machineLearningType === 'image segmentation') {
		contextBrush.clearRect(0, 0, canvasBrush.width, canvasBrush.height);
		contextBrush.fillStyle = `rgb(${labelsColormap[currentLabel]})`;
		contextBrush.beginPath();
		contextBrush.arc(Math.floor(offsetX) + 0.5, Math.floor(offsetY) + 0.5, parseFloat(inputRangeBrushSize.value), 0, 2 * Math.PI);
		contextBrush.fill();
		const imageDataBrush = contextBrush.getImageData(0, 0, canvasImage.width, canvasImage.height);
		contextBrush.putImageData(imageDataBrush, 0, 0);
		if (drawActivated) {
			for (let i = 0; i < imageDataBrush.data.length; i += 4) {
				if (imageDataBrush.data[i] > 0) {
					masks[imageOffset + i / 4] = currentLabel;
				}
			}
		}
		contextBrush.drawImage(canvasBrush, 0, 0);
		contextMask.drawImage(canvasMask, 0, 0);
		const imageDataMask = new ImageData(canvasImage.width, canvasImage.height);
		for (let i = 0; i < imageSize; i++) {
			const maskValue = masks[imageOffset + i];
			imageDataMask.data[4*i] = labelsColormap[maskValue][0];
			imageDataMask.data[4*i + 1] = labelsColormap[maskValue][1];
			imageDataMask.data[4*i + 2] = labelsColormap[maskValue][2];
			if (maskValue > 0) {
				imageDataMask.data[4*i + 3] = 255;
			} else {
				imageDataMask.data[4*i + 3] = 0;
			}
		}
		contextMask.putImageData(imageDataMask, 0, 0);
	}
}

function resetView() {
	const imageSlice = images.slice(imageOffset, imageOffset + imageSize);
	let max = -Infinity;
	imageValueMin = Infinity;
	for (let i = 0; i < imageSlice.length; i++) {
		if (imageSlice[i] > max) {
			max = imageSlice[i];
		}
		if (imageSlice[i] < imageValueMin) {
			imageValueMin = imageSlice[i];
		}
	}
	imageValueRange = (max - imageValueMin) / 255;
	spanImageValueMax.textContent = Math.round(max);
	spanImageValueMin.textContent = Math.round(imageValueMin);
	drawCanvas();
}

function readNiftiFile(file) {
	const reader = new FileReader();
	reader.onloadend = function (event) {
		if (event.target.readyState === FileReader.DONE) {
			let niftiHeader;
			let niftiImage;
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
				case nifti.NIFTI1.TYPE_RGB24:
					images = new Uint8Array(niftiImage);
					images = images.filter(function(value, index, Arr) {
						return index % 3 === 0;
					});
					break;
				case 2304:
					images = new Uint32Array(niftiImage);
					break;
				default:
					return;
			}
			numImages = niftiHeader.dims[3] - 1;
			imageCurrentIndex = 0;
			canvasImage.height = niftiHeader.dims[2];
			canvasImage.width = niftiHeader.dims[1];
			imageSize = canvasImage.height * canvasImage.width;
			if (configSelected.machineLearningType === 'image segmentation') {
				canvasMask.height = canvasImage.height;
				canvasMask.width = canvasImage.width;
				canvasBrush.height = canvasImage.height;
				canvasBrush.width = canvasImage.width;
				masks = new Uint8Array(images.length);
			} else if (configSelected.machineLearningType === 'image classification') {
				classes = classes.slice(0, numImages+1);
			}
			spanHeightXwidth.textContent = `${canvasImage.height}x${canvasImage.width}`;
			spanImageIndex.textContent = `${imageCurrentIndex}/${numImages}`;
			resetView();
		}
		disableUI(false);
	};
	reader.readAsArrayBuffer(file);
}

async function predictCurrentImage() {
	let imageSlice = images.slice(imageOffset, imageOffset + imageSize);
	imageSlice = new Float32Array(imageSlice);
	tf.tidy(() => {
		let tensor = tf.tensor(imageSlice);
		tensor = tf.reshape(tensor, [canvasImage.height, canvasImage.width]);
		tensor = tensor.expandDims(-1);
		tensor = tf.image.resizeBilinear(tensor, modelInputShape);
		const tensorMax = tensor.max();
		const tensorMin = tensor.min();
		tensor = tensor.sub(tensorMin).div(tensorMax.sub(tensorMin));
		let preProcessedImage;
		if (configSelected.machineLearningType === 'image segmentation') {
			preProcessedImage = tensor.squeeze(-1).expandDims(0).expandDims(0);
		} else if (configSelected.machineLearningType === 'image classification') {
			preProcessedImage = tensor.expandDims(0);
		}
		let modelPrediction = model.predict(preProcessedImage);
		if (configSelected.machineLearningType === 'image segmentation') {
			if (modelPrediction.size !== imageSize) {
				modelPrediction = modelPrediction.reshape([512, 512, 1]);
				modelPrediction = tf.image.resizeNearestNeighbor(modelPrediction, [canvasImage.height, canvasImage.width]);
			}
			modelPrediction = modelPrediction.dataSync();
			for (let i = 0; i < modelPrediction.length; i++) {
				if (modelPrediction[i] > 0.5) {
					masks[imageOffset + i] = currentLabel;
				}
			}
		} else if (configSelected.machineLearningType === 'image classification') {
			const classProbabilities = modelPrediction.softmax().mul(100).arraySync();
			console.log(classProbabilities[0]);
		}
	});
	drawCanvas();
}

function predictAllImages() {
	for (let i = 0; i <= numImages; i++) {
		imageCurrentIndex = i;
		imageOffset = imageSize * imageCurrentIndex;
		predictCurrentImage();
	}
}

function saveData(data, fileName) {
	const a = document.createElement('a');
	document.body.appendChild(a);
	a.style = 'display: none';
	const blob = new Blob(data);
	const url = window.URL.createObjectURL(blob);
	a.href = url;
	a.download = fileName;
	a.click();
	window.URL.revokeObjectURL(url);
}

function savePredictionsToDisk() {
	if (files === undefined) {
		return;
	}
	const niftiHeaderTmp = decompressedFile.slice(0, 352);
	const data = [new Uint8Array(niftiHeaderTmp, 0, niftiHeaderTmp.length), new Uint8Array(masks.buffer, 0, masks.buffer.length)];
	saveData(data, 'masks.nii');
}

function loadImages() {
	disableUI(true);
	files = event.currentTarget.files;
	if (files[0] === undefined) {
		disableUI(false);
		return;
	}
	if (files[0].name.includes('.nii')) {
		readNiftiFile(files[0]);
	} else if (files[0].name.includes('.dcm')) {
		itk.readImageDICOMFileSeries(files)
			.then(function ({ image }) {
				itk.writeImageArrayBuffer(null, false, image, 'unnamed.nii')
					.then((data) => {
						const blob = new Blob([data.arrayBuffer]);
						readNiftiFile(blob);
					});
			});
	} else if ((files[0].name.includes('.png')) || (files[0].name.includes('.jpg'))) {
		itk.readImageFileSeries(files)
			.then(function ({ image }) {
				itk.writeImageArrayBuffer(null, false, image, 'unnamed.nii')
					.then((data) => {
						const blob = new Blob([data.arrayBuffer]);
						readNiftiFile(blob);
					});
			});
	}
}

function loadPredictions() {
	disableUI(true);
	const file = event.currentTarget.files[0];
	if (file === undefined) {
		return;
	}
	const reader = new FileReader();
	reader.onloadend = function (event) {
		if (event.target.readyState === FileReader.DONE) {
			if (configSelected.machineLearningType === 'image segmentation') {
				const niftiHeader = nifti.readHeader(event.target.result);
				const niftiImage = nifti.readImage(niftiHeader, event.target.result);
				masks = new Uint8Array(niftiImage);
				drawCanvas();
			}
		}
		disableUI(false);
	};
	reader.readAsArrayBuffer(file);
}

async function train() {
	disableUI(true);
	let tensor = tf.tensor(images).reshape([numImages + 1, canvasImage.height, canvasImage.width]);
	tensor = tensor.expandDims(-1);
	tensor = tf.image.resizeBilinear(tensor, modelInputShape);
	const tensorMax = tensor.max();
	const tensorMin = tensor.min();
	tensor = tensor.sub(tensorMin).div(tensorMax.sub(tensorMin));
	let preProcessedImage;
	let predictions;
	if (configSelected.machineLearningType === 'image segmentation') {
		preProcessedImage = tensor.squeeze(-1).expandDims(0).expandDims(0);
		predictions = tf.tensor(masks);
	} else if (configSelected.machineLearningType === 'image classification') {
		preProcessedImage = tensor;
		predictions = tf.oneHot(classes, configSelected.classNames.length);
	}
	model.compile({
		optimizer: 'adam',
		loss: 'categoricalCrossentropy',
		metrics: ['accuracy'],
	});
	await model.fit(preProcessedImage, predictions, {
		epochs: inputNumEpochs.value,
		callbacks: [
			new tf.CustomCallback({
				onEpochEnd: (epoch, logs) => {
					spanCurrentEpoch.textContent = epoch;
					spanLoss.textContent = logs.loss;
					spanAccuracy.textContent = logs.acc;
				}
			})
		]
	});
	tf.dispose(predictions);
	tf.dispose(preProcessedImage);
	tf.dispose(tensor);
	tf.dispose(tensorMax);
	tf.dispose(tensorMin);
	console.log(tf.memory());
	disableUI(false);
}

async function selectModelName() {
	contextBrush.clearRect(0, 0, canvasBrush.width, canvasBrush.height);
	contextImage.clearRect(0, 0, canvasImage.width, canvasImage.height);
	contextMask.clearRect(0, 0, canvasMask.width, canvasMask.height);
	currentLabel = 0;
	decompressedFile = null;
	divLabelList.textContent = '';
	imageCurrentIndex = 0;
	imageOffset = 0;
	imageSize = 1;
	imageValueMin = 0;
	imageValueRange = 1;
	images = new Uint8Array(imageSize);
	inputFile.value = '';
	inputLoadPredictions.value = '';
	masks = new Uint8Array(imageSize);
	numImages = 0;
	spanHeightXwidth.textContent = '';
	spanImageIndex.textContent = '';
	spanImageValueMax.textContent = '';
	spanImageValueMin.textContent = '';
	let loadModelFunction;
	configSelected = configArray.find(config => config.modelURL === selectModel.value);
	if (configSelected.machineLearningType === 'image classification') {
		canvasMask.style.display = 'none';
		canvasBrush.style.display = 'none';
		divBrushSize.style.display = 'none';
	} else if (configSelected.machineLearningType === 'image segmentation') {
		canvasMask.style.display = '';
		canvasBrush.style.display = '';
		divBrushSize.style.display = '';
	}
	await fetch(selectModel.value)
		.then(response => response.text())
		.then((text) => {
			if (JSON.parse(text).format === 'graph-model') {
				loadModelFunction = tf.loadGraphModel;
			} else if (JSON.parse(text).format === 'layers-model') {
				loadModelFunction = tf.loadLayersModel;
			}
		})
	model = await loadModelFunction(selectModel.value, {
		onProgress: function (fraction) {
			divModelLoadFraction.textContent = `${Math.round(100*fraction)}%.`;
			spanModelInputShape.textContent = 'NaN';
			spanModelPredictionShape.textContent = 'NaN';
			spanModelTrainable.textContent = 'NaN';
			disableUI(true);
		}
	});
	modelInputShape = model.inputs[0].shape.filter(x => x>3);
	canvasImage.width = modelInputShape[0];
	canvasImage.height = modelInputShape[1];
	canvasMask.width = modelInputShape[0];
	canvasMask.height = modelInputShape[1];
	canvasBrush.width = modelInputShape[0];
	canvasBrush.height = modelInputShape[1];
	divModelLoadFraction.textContent = 'Model loaded.';
	spanModelInputShape.textContent = modelInputShape;
	spanModelPredictionShape.textContent = model.outputs[0].shape;
	if (model.trainable) {
		buttonSaveModelToDisk.style.display = '';
		buttonTrain.style.display = '';
		buttonUploadModelToServer.style.display = '';
		divAccuracy.style.display = '';
		divCurrentEpoch.style.display = '';
		divLoss.style.display = '';
		divNumEpochs.style.display = '';
		spanModelTrainable.textContent = 'True';
	} else {
		buttonSaveModelToDisk.style.display = 'none';
		buttonTrain.style.display = 'none';
		buttonUploadModelToServer.style.display = 'none';
		divAccuracy.style.display = 'none';
		divCurrentEpoch.style.display = 'none';
		divLoss.style.display = 'none';
		divNumEpochs.style.display = 'none';
		spanModelTrainable.textContent = 'False';
	}
	for (const [i, labelText] of configSelected.classNames.entries()) {
		const divLabel = document.createElement('div');
		divLabel.id = `divLabel${i}`;
		divLabelList.appendChild(divLabel);
		const divLabelColor = document.createElement('div');
		divLabelColor.id = `divLabelColor${i}`;
		divLabelColor.style.backgroundColor = `rgb(${labelsColormap[i]})`;
		divLabelColor.style.float = 'left';
		divLabelColor.style.height = '15px';
		divLabelColor.style.opacity = 0.3;
		divLabelColor.style.width = '15px';
		divLabelColor.onclick = function (event) {
			const nodeList = document.querySelectorAll('*[id^="divLabelColor"]')
			for (let i = 0; i < nodeList.length; i++) {
				nodeList[i].style.opacity = 0.3;
			}
			event.currentTarget.style.opacity = 1;
			currentLabel = parseInt(event.currentTarget.id.match(/\d+/)[0]);
			if (configSelected.machineLearningType === 'image classification') {
				classes[imageCurrentIndex] = currentLabel;
			}
		};
		divLabel.appendChild(divLabelColor);
		const divLabelText = document.createElement('text');
		divLabelText.id = `divLabelText${i}`;
		divLabelText.style.right = '50%';
		divLabelText.style.width = '30%';
		divLabelText.textContent = labelText;
		divLabel.appendChild(divLabelText);
	}
	disableUI(false);
}

canvasBrush.addEventListener('mousemove', (event) => {
	if (valueRangeActivated) {
		const rangeTmp = imageValueRange * (1 + event.movementX * 0.01);
		let midTmp = imageValueMin + imageValueRange / 2;
		midTmp -= Math.abs(midTmp) * event.movementY / 1000;
		imageValueMin = midTmp - rangeTmp / 2;
		imageValueRange = rangeTmp;
		spanImageValueMax.textContent = Math.round(2*imageValueRange);
		spanImageValueMin.textContent = Math.round(imageValueMin);
	} else {
		offsetX = event.offsetX;
		offsetY = event.offsetY;
	}
	requestAnimationFrame(drawCanvas);
});

window.addEventListener('keydown', function (event) {
	if (event.code === 'ArrowUp' && (imageCurrentIndex > 0)) {
		imageCurrentIndex--;
	} else if (event.code === 'ArrowDown' && (imageCurrentIndex < numImages)) {
		imageCurrentIndex++;
	} else {
		return;
	}
	imageOffset = imageSize * imageCurrentIndex;
	currentLabel = classes[imageCurrentIndex];
	document.getElementById(`divLabelColor${currentLabel}`).click();
	spanHeightXwidth.textContent = `${canvasImage.height}x${canvasImage.width}`;
	spanImageIndex.textContent = `${imageCurrentIndex}/${numImages}`;
	drawCanvas();
});

(async () => {
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
	selectModelName();
})();
