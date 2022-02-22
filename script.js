'use strict';

function loadImages() {
	disableUI(true);
	files = event.currentTarget.files;
	if (files[0] == undefined) {
		disableUI(false);
		return;
	}
	if (files[0].name.includes('.nii')) {
		fileName = files[0].name.split('.nii')[0];
		readNiiFile(files[0]);
	} else if (files[0].name.includes('.dcm')) {
		fileName = files[0].name;
		itk.readImageDICOMFileSeries(files)
			.then(function ({ image }) {
				itk.writeImageArrayBuffer(null, false, image, 'unnamed.nii')
					.then((data) => {
						const blob = new Blob([data.arrayBuffer]);
						readNiiFile(blob);
					});
			});
	} else if ((files[0].name.includes('.png')) || (files[0].name.includes('.jpg'))) {
		fileName = files[0].name;
		itk.readImageFileSeries(files)
			.then(function ({ image }) {
				itk.writeImageArrayBuffer(null, false, image, 'unnamed.nii')
					.then((data) => {
						const blob = new Blob([data.arrayBuffer]);
						readNiiFile(blob);
					});
			});
	}
	resetData();
}

function loadOutput() {
	const file = event.currentTarget.files[0];
	if (file == undefined) {
		return;
	}
	const reader = new FileReader();
	reader.onloadend = function (event) {
		if (event.target.readyState === FileReader.DONE) {
			const niftiHeader = nifti.readHeader(event.target.result);
			const niftiImage = nifti.readImage(niftiHeader, event.target.result);
			masks = new Uint8Array(niftiImage);
			drawCanvas();
		}
		disableUI(false);
	};
	reader.readAsArrayBuffer(file);
}

function addLabel(labelText) {
	if (numLabels >= maxNumLabels) {
		return;
	}
	const divLabel = document.createElement('div');
	divLabel.id = `divLabel${numLabels}`;
	document.getElementById('divLabelList').appendChild(divLabel);
	const divLabelColor = document.createElement('div');
	divLabelColor.style.height = '15px';
	divLabelColor.style.width = '15px';
	divLabelColor.style.opacity = 0.3;
	divLabelColor.style.backgroundColor = hexValueList[numLabels];
	divLabelColor.style.float = 'left';
	divLabelColor.id = `divLabelColor${numLabels}`;
	divLabelColor.onclick = function (event) {
		document.getElementById(`divLabelColor${currentLabel}`).style.opacity = 0.3;
		event.currentTarget.style.opacity = 1;
		currentLabel = parseInt(event.currentTarget.id.match(/\d+/)[0]);
		visualizeImageDataMask();
	};
	divLabel.appendChild(divLabelColor);
	const divLabelText = document.createElement('text');
	divLabelText.textContent = labelText;
	divLabelText.style.right = '50%';
	divLabelText.style.width = '30%';
	divLabelText.id = `divLabelText${numLabels}`;
	divLabel.appendChild(divLabelText);
	divLabelColor.click();
	numLabels++;
}

function resetView() {
	const imageOffset = imageSize * imageIndex;
	const imageSlice = images.slice(imageOffset, imageOffset + imageSize);
	let max = -Infinity;
	let min = Infinity;
	for (let i = 0; i < imageSlice.length; i++) {
		if (imageSlice[i] > max) {
			max = imageSlice[i];
		}
		if (imageSlice[i] < min) {
			min = imageSlice[i];
		}
	}
	imageValueMin = min;
	imageValueRange = (max - min) / 255;
	document.getElementById('spanImageValueMin').textContent = Math.round(min);
	document.getElementById('spanImageValueMax').textContent = Math.round(max);
	requestAnimationFrame(drawCanvas);
}

function predictAndViewImage() {
	predictImage();
	drawCanvas();
}

function predictAndViewAllImages() {
	for (let i = 0; i <= numImages; i++) {
		imageIndex = i;
		predictImage();
		drawCanvas();
	}
}

async function train() {
	disableUI(true);
	const imageOffset = imageSize * imageIndex;
	let imageSlice = images.slice(imageOffset, imageOffset + imageSize);
	imageSlice = new Float32Array(imageSlice);
	let tensor = tf.tensor(imageSlice);
	tensor = tf.reshape(tensor, [rows, columns]);
	const modelInputShape = model.inputs[0].shape.filter(x => x>3);
	tensor = tensor.expandDims(-1);
	tensor = tf.image.resizeBilinear(tensor, modelInputShape);
	const tensorMax = tensor.max();
	const tensorMin = tensor.min();
	tensor = tensor.sub(tensorMin).div(tensorMax.sub(tensorMin));
	let preProcessedImage;
	if (configSelected.machineLearningType == 'image segmentation') {
		preProcessedImage = tensor.squeeze(-1).expandDims(0).expandDims(0);
	} else if (configSelected.machineLearningType == 'image classification') {
		preProcessedImage = tensor.expandDims(0);
	}
	model.compile({
		optimizer: 'adam',
		loss: 'categoricalCrossentropy',
		metrics: ['accuracy'],
	});
	let output = tf.tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
	await model.fit(preProcessedImage, output.expandDims(0), {
		epochs: document.getElementById('inputNumEpochs').value,
		callbacks: [
			new tf.CustomCallback({
				onEpochEnd: (epoch, logs) => {
					document.getElementById('spanCurrentEpoch').textContent = epoch;
					document.getElementById('spanLoss').textContent = logs.loss;
					document.getElementById('spanAccuracy').textContent = logs.acc;
				}
			})
		]
	});
	disableUI(false);
}

function saveOutput() {
	if (files == undefined) {
		return;
	}
	const niftiHeaderTmp = decompressedFile.slice(0, 352);
	const data = [new Uint8Array(niftiHeaderTmp, 0, niftiHeaderTmp.length), new Uint8Array(masks.buffer, 0, masks.buffer.length)];
	saveData(data, `${fileName}-masks.nii`);
}

function readNiiFile(file) {
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
						return index % 3 == 0;
					});
					break;
				case 2304:
					images = new Uint32Array(niftiImage);
					break;
				default:
					return;
			}
			masks = new Uint8Array(images.length);
			numImages = niftiHeader.dims[3] - 1;
			imageIndex = 0;
			rows = niftiHeader.dims[2];
			columns = niftiHeader.dims[1];
			imageSize = rows * columns;
			canvasImage.height = rows;
			canvasImage.width = columns;
			canvasMask.height = rows;
			canvasMask.width = columns;
			canvasBrush.height = rows;
			canvasBrush.width = columns;
			updateUI();
			resetView();
		}
		disableUI(false);
	};
	reader.readAsArrayBuffer(file);
}

function drawCanvas() {
	visualizeImageDataImage();
	visualizeImageDataMask();
	draw(offsetX, offsetY);
}

function visualizeImageDataImage() {
	const imageOffset = imageSize * imageIndex;
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
	contextImage.putImageData(imageDataImage, 0, 0);
}

function visualizeImageDataMask() {
	const imageOffset = imageSize * imageIndex;
	const imageDataMask = new ImageData(columns, rows);
	for (let i = 0; i < imageSize; i++) {
		const maskValue = masks[imageOffset + i];
		if (maskValue > 0) {
			imageDataMask.data[4*i] = labelsColormap[maskValue][0];
			imageDataMask.data[4*i + 1] = labelsColormap[maskValue][1];
			imageDataMask.data[4*i + 2] = labelsColormap[maskValue][2];
			imageDataMask.data[4*i + 3] = 255;
		}
	}
	contextMask.putImageData(imageDataMask, 0, 0);
}

function disableUI(argument) {
	const nodes = document.getElementById('divControl').getElementsByTagName('*');
	for(let i = 0; i < nodes.length; i++){
		nodes[i].disabled = argument;
	}
}

function updateUI() {
	document.getElementById('spanImageIndex').textContent = `${imageIndex}/${numImages}`;
	document.getElementById('spanRowsXColumns').textContent = `${rows}x${columns}`;
}

async function selectModelName() {
	contextImage.clearRect(0, 0, canvasImage.width, canvasImage.height);
	contextMask.clearRect(0, 0, canvasMask.width, canvasMask.height);
	contextBrush.clearRect(0, 0, canvasBrush.width, canvasBrush.height);
	document.getElementById('inputFile').value = '';
	let loadModelFunction;
	configSelected = configArray.find(config => config.modelURL == selectModel.value);
	if (configSelected.machineLearningType == 'image classification') {
		canvasMask.style.display = 'none';
		canvasBrush.style.display = 'none';
		divBrushSize.style.display = 'none';
	} else if (configSelected.machineLearningType == 'image segmentation') {
		canvasMask.style.display = '';
		canvasBrush.style.display = '';
		divBrushSize.style.display = '';
	}

	await fetch(selectModel.value)
		.then(response => response.text())
		.then((text) => {
			if (JSON.parse(text).format == 'graph-model') {
				loadModelFunction = tf.loadGraphModel;
			} else if (JSON.parse(text).format == 'layers-model') {
				loadModelFunction = tf.loadLayersModel;
			}
		})
	model = await loadModelFunction(selectModel.value, {
		onProgress: function (fraction) {
			document.getElementById('divModelLoadFraction').textContent = `${Math.round(100*fraction)}%.`;
			document.getElementById('spanModelInputShape').textContent = 'NaN';
			document.getElementById('spanModelOutputShape').textContent = 'NaN';
			document.getElementById('spanModelTrainable').textContent = 'NaN';
			disableUI(true);
		}
	});
	document.getElementById('divModelLoadFraction').textContent = 'Model loaded.';
	document.getElementById('spanModelInputShape').textContent = model.inputs[0].shape;
	document.getElementById('spanModelOutputShape').textContent = model.outputs[0].shape;
	if (model.trainable) {
		document.getElementById('spanModelTrainable').textContent = 'True';
		document.getElementById('buttonTrain').style.display = '';
		document.getElementById('divNumEpochs').style.display = '';
		document.getElementById('divCurrentEpoch').style.display = '';
		document.getElementById('divLoss').style.display = '';
		document.getElementById('divAccuracy').style.display = '';
	} else {
		document.getElementById('spanModelTrainable').textContent = 'False';
		document.getElementById('buttonTrain').style.display = 'none';
		document.getElementById('divNumEpochs').style.display = 'none';
		document.getElementById('divCurrentEpoch').style.display = 'none';
		document.getElementById('divLoss').style.display = 'none';
		document.getElementById('divAccuracy').style.display = 'none';
	}
	document.getElementById('divLabelList').textContent = '';
	resetView();
	resetData();
	updateUI();
	for (const labelText of configSelected.classNames) {
		if (labelText != "None") { // remove after updating the protocol
			addLabel(labelText);
		}
	}
	disableUI(false);
}

async function predictImage() {
	const imageOffset = imageSize * imageIndex;
	let imageSlice = images.slice(imageOffset, imageOffset + imageSize);
	imageSlice = new Float32Array(imageSlice);
	tf.tidy(() => {
		let tensor = tf.tensor(imageSlice);
		tensor = tf.reshape(tensor, [rows, columns]);
		const modelInputShape = model.inputs[0].shape.filter(x => x>3);
		tensor = tensor.expandDims(-1);
		tensor = tf.image.resizeBilinear(tensor, modelInputShape);
		const tensorMax = tensor.max();
		const tensorMin = tensor.min();
		tensor = tensor.sub(tensorMin).div(tensorMax.sub(tensorMin));

		let preProcessedImage;
		if (configSelected.machineLearningType == 'image segmentation') {
			preProcessedImage = tensor.squeeze(-1).expandDims(0).expandDims(0);
		} else if (configSelected.machineLearningType == 'image classification') {
			preProcessedImage = tensor.expandDims(0);
		}
		let modelPrediction = model.predict(preProcessedImage);
		if (configSelected.machineLearningType == 'image segmentation') {
			if (modelPrediction.size !== imageSize) {
				modelPrediction = modelPrediction.reshape([512, 512, 1]);
				modelPrediction = tf.image.resizeNearestNeighbor(modelPrediction, [rows, columns]);
			}
			modelPrediction = modelPrediction.dataSync();
			for (let i = 0; i < modelPrediction.length; i++) {
				if (modelPrediction[i] > 0.5) {
					masks[imageOffset + i] = currentLabel;
				}
			}
		} else if (configSelected.machineLearningType == 'image classification') {
			const classProbabilities = modelPrediction.softmax().mul(100).arraySync();
			console.log(classProbabilities[0]);
		}
	});
	visualizeImageDataMask();
}

function simpleDraw() {
	draw(offsetX, offsetY);
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

function resetData() {
	document.getElementById('inputFile').value = '';
	document.getElementById('inputLoadOutput').value = '';
	columns = 512;
	currentLabel = 1;
	decompressedFile = null;
	imageIndex = 0;
	imageOffset = 0;
	imageSize = columns * rows;
	imageValueMin = 0;
	imageValueRange = 1;
	images = new Uint8Array(imageSize);
	masks = new Uint8Array(imageSize);
	numImages = 0;
	numLabels = 1;
	rows = 512;
}

function draw(offsetX, offsetY) {
	contextBrush.clearRect(0, 0, canvasBrush.width, canvasBrush.height);
	contextBrush.fillStyle = hexValueList[currentLabel];
	contextBrush.beginPath();
	contextBrush.arc(Math.floor(offsetX) + 0.5, Math.floor(offsetY) + 0.5, parseFloat(inputRangeBrushSize.value), 0, 2 * Math.PI);
	contextBrush.fill();
	const imageDataBrush = contextBrush.getImageData(0, 0, columns, rows);
	contextBrush.putImageData(imageDataBrush, 0, 0);
	if (drawActivated) {
		let currentLabelTmp = currentLabel;
		if (eraseActivated) {
			currentLabelTmp = 0;
		}
		const imageOffset = imageSize * imageIndex;
		for (let i = 0; i < imageDataBrush.data.length; i += 4) {
			if (imageDataBrush.data[i] > 0) {
				masks[imageOffset + i / 4] = currentLabelTmp;
			}
		}
		visualizeImageDataMask();
	}
	contextBrush.drawImage(canvasBrush, 0, 0);
	contextMask.drawImage(canvasMask, 0, 0);
}

function hexToRgb(hex) {
	const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
	return [parseInt(result[1], 16), parseInt(result[2], 16), parseInt(result[3], 16)];
}

const hexValueList = [
	null,
	'#1f77b4',
	'#aec7e8',
	'#ff7f0e',
	'#ffbb78',
	'#2ca02c',
	'#98df8a',
	'#d62728',
	'#ff9896',
	'#9467bd',
	'#c5b0d5',
	'#8c564b',
	'#c49c94',
	'#e377c2',
	'#f7b6d2',
	'#7f7f7f',
	'#c7c7c7',
	'#bcbd22',
	'#dbdb8d',
	'#17becf',
	'#9edae5',
];
const maxNumLabels = hexValueList.length - 1;

const canvasBrush = document.getElementById('canvasBrush');
const canvasImage = document.getElementById('canvasImage');
const canvasMask = document.getElementById('canvasMask');
const contextBrush = canvasBrush.getContext('2d');
const contextImage = canvasImage.getContext('2d');
const contextMask = canvasMask.getContext('2d');
const divBrushSize = document.getElementById('divBrushSize');
const inputRangeBrushSize = document.getElementById('inputRangeBrushSize');
const selectModel = document.getElementById('selectModel');

canvasBrush.addEventListener('mousedown', (event) => {
	if (event.button === 0 && !event.ctrlKey) {
		drawActivated = true;
	} else if (event.button === 1) {
		valueRangeActivated = true;
	} else if (event.button === 2 && !event.ctrlKey) {
		eraseActivated = true;
		drawActivated = true;
	}
});
canvasBrush.addEventListener('mousemove', (event) => {
	if (valueRangeActivated) {
		let rangeTmp = imageValueRange * (1 + event.movementX * 0.01);
		let midTmp = imageValueMin + imageValueRange / 2;
		midTmp -= Math.abs(midTmp) * event.movementY / 1000;
		imageValueMin = midTmp - rangeTmp / 2;
		imageValueRange = rangeTmp;
		document.getElementById('spanImageValueMin').textContent = Math.round(imageValueMin);
		document.getElementById('spanImageValueMax').textContent = Math.round(imageValueRange + rangeTmp);
		requestAnimationFrame(drawCanvas);
	} else {
		offsetX = event.offsetX;
		offsetY = event.offsetY;
		requestAnimationFrame(simpleDraw);
	}
});

window.addEventListener('contextmenu', function (event) {
	event.preventDefault();
}, false);
document.getElementById('divCanvases').addEventListener('wheel', function (event) {
	if (event.deltaY > 0 && (imageIndex > 0)) {
		imageIndex--;
	} else if (event.deltaY < 0 && (imageIndex < numImages)) {
		imageIndex++;
	} else {
		return;
	}
	imageOffset = imageSize * imageIndex;
	updateUI();
	drawCanvas();
});
window.addEventListener('mouseleave', () => {
	drawActivated = false;
	eraseActivated = false;
	valueRangeActivated = false;
});
window.addEventListener('mouseup', () => {
	drawActivated = false;
	eraseActivated = false;
	valueRangeActivated = false;
});

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
	resetView();
	resetData();
	updateUI();
	selectModelName();
}

const configURLarray = [
	'https://raw.githubusercontent.com/pbizopoulos/comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-in-ct/master/wbml/lesion-segmentation.json',
	'https://raw.githubusercontent.com/pbizopoulos/comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-in-ct/master/wbml/lung-segmentation.json',
	'https://raw.githubusercontent.com/pbizopoulos/tmp/main/wbml/lung-classification.json',
]

let columns;
let configArray = [];
let configSelected;
let currentLabel = 1;
let decompressedFile;
let drawActivated = false;
let eraseActivated = false;
let fileName;
let files;
let imageIndex;
let imageOffset;
let imageValueMin;
let imageValueRange;
let labelsColormap = [ null ];
let model;
let numImages;
let numLabels = 1;
let offsetX = undefined;
let offsetY = undefined;
let rows;
let valueRangeActivated = false;

let imageSize = rows * columns;
let images = new Uint8Array(imageSize);
let masks = new Uint8Array(imageSize);
for (let i = 1; i < hexValueList.length; i++) {
	labelsColormap[i] = hexToRgb(hexValueList[i]);
}

initialize();
