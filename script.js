'use strict';

const accuracyDiv = document.getElementById('accuracyDiv');
const accuracySpan = document.getElementById('accuracySpan');
const brushCanvas = document.getElementById('brushCanvas');
const brushContext = brushCanvas.getContext('2d');
const brushSizeDiv = document.getElementById('brushSizeDiv');
const brushSizeInputRange = document.getElementById('brushSizeInputRange');
const epochCurrentDiv = document.getElementById('epochCurrentDiv');
const epochCurrentSpan = document.getElementById('epochCurrentSpan');
const epochsNumDiv = document.getElementById('epochsNumDiv');
const epochsNumInputNumber = document.getElementById('epochsNumInputNumber');
const imageCanvas = document.getElementById('imageCanvas');
const imageContext = imageCanvas.getContext('2d');
const imageHeightSpan = document.getElementById('imageHeightSpan');
const imageIndexSpan = document.getElementById('imageIndexSpan');
const imageValueMaxSpan = document.getElementById('imageValueMaxSpan');
const imageValueMinSpan = document.getElementById('imageValueMinSpan');
const imageWidthSpan = document.getElementById('imageWidthSpan');
const labelListDiv = document.getElementById('labelListDiv');
const loadImagesInputFile = document.getElementById('loadImagesInputFile');
const loadPredictionsInputFile = document.getElementById('loadPredictionsInputFile');
const lossDiv = document.getElementById('lossDiv');
const lossSpan = document.getElementById('lossSpan');
const maskCanvas = document.getElementById('maskCanvas');
const maskContext = maskCanvas.getContext('2d');
const modelInputShapeSpan = document.getElementById('modelInputShapeSpan');
const modelLoadFractionDiv = document.getElementById('modelLoadFractionDiv');
const modelPredictionShapeSpan = document.getElementById('modelPredictionShapeSpan');
const modelSelect = document.getElementById('modelSelect');
const modelTrainableSpan = document.getElementById('modelTrainableSpan');
const predictAllImagesButton = document.getElementById('predictAllImagesButton');
const predictCurrentImageButton = document.getElementById('predictCurrentImageButton');
const resetImageValueButton = document.getElementById('resetImageValueButton');
const saveModelToDiskButton = document.getElementById('saveModelToDiskButton');
const saveModelToServerButton = document.getElementById('saveModelToServerButton');
const savePredictionsToDiskButton = document.getElementById('savePredictionsToDiskButton');
const trainModelLocallyButton = document.getElementById('trainModelLocallyButton');

const configURLarray = [
	'https://raw.githubusercontent.com/pbizopoulos/comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-in-ct/main/docs/lesion-segmentation.json',
	'https://raw.githubusercontent.com/pbizopoulos/comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-in-ct/main/docs/lung-segmentation.json',
	'https://raw.githubusercontent.com/pbizopoulos/tmp/main/docs/lung-classification.json',
]
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

let classAnnotations = new Uint8Array(1000); // tmp hardcoded max value, remove later
let configArray = [];
let configSelected;
let drawActivated = false;
let fileDecompressed;
let files;
let imageCurrentIndex;
let imageOffset;
let imageSize;
let imageValueMin;
let imageValueRange;
let imageValueRangeActivated = false;
let images = new Uint8Array(imageSize);
let imagesNum;
let labelCurrent = 0;
let masks = new Uint8Array(imageSize);
let model;
let modelInputShape;
let offsetX;
let offsetY;

function disableUI(argument) {
	brushSizeInputRange.disabled = argument;
	epochsNumInputNumber.disabled = argument;
	loadImagesInputFile.disabled = argument;
	loadPredictionsInputFile.disabled = argument;
	modelSelect.disabled = argument;
	predictAllImagesButton.disabled = argument;
	predictCurrentImageButton.disabled = argument;
	resetImageValueButton.disabled = argument;
	saveModelToDiskButton.disabled = argument;
	saveModelToServerButton.disabled = argument;
	savePredictionsToDiskButton.disabled = argument;
	trainModelLocallyButton.disabled = argument;
}

function drawCanvas() {
	const imageDataImage = new ImageData(imageCanvas.width, imageCanvas.height);
	for (let i = 0; i < imageCanvas.height; i++) {
		const rowOffset = i * imageCanvas.width;
		for (let j = 0; j < imageCanvas.width; j++) {
			const imageValueOffset = imageOffset + rowOffset + j;
			const imageValue = (images[imageValueOffset] - imageValueMin) / imageValueRange;
			const offsetMult4 = (rowOffset + j) * 4;
			imageDataImage.data[offsetMult4] = imageValue;
			imageDataImage.data[offsetMult4 + 1] = imageValue;
			imageDataImage.data[offsetMult4 + 2] = imageValue;
			imageDataImage.data[offsetMult4 + 3] = 255;
		}
	}
	imageContext.putImageData(imageDataImage, 0, 0);
	if (configSelected.machineLearningType === 'image segmentation') {
		brushContext.clearRect(0, 0, brushCanvas.width, brushCanvas.height);
		brushContext.fillStyle = `rgb(${labelsColormap[labelCurrent]})`;
		brushContext.beginPath();
		brushContext.arc(Math.floor(offsetX) + 0.5, Math.floor(offsetY) + 0.5, parseFloat(brushSizeInputRange.value), 0, 2 * Math.PI);
		brushContext.fill();
		const brushImageData = brushContext.getImageData(0, 0, imageCanvas.width, imageCanvas.height);
		brushContext.putImageData(brushImageData, 0, 0);
		if (drawActivated) {
			for (let i = 0; i < brushImageData.data.length; i += 4) {
				if (brushImageData.data[i] > 0) {
					masks[imageOffset + i / 4] = labelCurrent;
				}
			}
		}
		brushContext.drawImage(brushCanvas, 0, 0);
		maskContext.drawImage(maskCanvas, 0, 0);
		const maskImageData = new ImageData(imageCanvas.width, imageCanvas.height);
		for (let i = 0; i < imageSize; i++) {
			const maskValue = masks[imageOffset + i];
			maskImageData.data[4*i] = labelsColormap[maskValue][0];
			maskImageData.data[4*i + 1] = labelsColormap[maskValue][1];
			maskImageData.data[4*i + 2] = labelsColormap[maskValue][2];
			if (maskValue > 0) {
				maskImageData.data[4*i + 3] = 255;
			} else {
				maskImageData.data[4*i + 3] = 0;
			}
		}
		maskContext.putImageData(maskImageData, 0, 0);
	}
}

async function predictCurrentImage() {
	let imageSlice = images.slice(imageOffset, imageOffset + imageSize);
	imageSlice = new Float32Array(imageSlice);
	tf.tidy(() => {
		let tensor = tf.tensor(imageSlice);
		tensor = tf.reshape(tensor, [imageCanvas.height, imageCanvas.width]);
		tensor = tensor.expandDims(-1);
		tensor = tf.image.resizeNearestNeighbor(tensor, modelInputShape);
		const tensorMomentsBefore = tf.moments(tensor);
		tensor = tensor.sub(tensorMomentsBefore.mean);
		const tensorMomentsAfter = tf.moments(tensor);
		tensor = tensor.div(tf.sqrt(tensorMomentsAfter.variance));
		let preProcessedImage;
		if (configSelected.machineLearningType === 'image classification') {
			preProcessedImage = tensor.expandDims(0);
		} else if (configSelected.machineLearningType === 'image segmentation') {
			preProcessedImage = tensor.squeeze(-1).expandDims(0).expandDims(0);
		}
		let modelPrediction = model.predict(preProcessedImage);
		if (configSelected.machineLearningType === 'image classification') {
			const classProbabilities = modelPrediction.softmax().mul(100).arraySync();
			const nodeList = document.querySelectorAll('*[id^="labelPredictionDiv"]')
			for (let i = 0; i < nodeList.length; i++) {
				nodeList[i].textContent = `${classProbabilities[0][i].toFixed(2)}%`;
			}
		} else if (configSelected.machineLearningType === 'image segmentation') {
			if (modelPrediction.size !== imageSize) {
				modelPrediction = modelPrediction.reshape([512, 512, 1]);
				modelPrediction = tf.image.resizeNearestNeighbor(modelPrediction, [imageCanvas.height, imageCanvas.width]);
			}
			modelPrediction = modelPrediction.dataSync();
			for (let i = 0; i < modelPrediction.length; i++) {
				if (modelPrediction[i] > 0.5) {
					masks[imageOffset + i] = labelCurrent;
				}
			}
		}
	});
	drawCanvas();
}

function readNiftiFile(file) {
	const reader = new FileReader();
	reader.onloadend = (event) => {
		if (event.target.readyState === FileReader.DONE) {
			let niftiHeader;
			let niftiImage;
			if (nifti.isCompressed(event.target.result)) {
				fileDecompressed = nifti.decompress(event.target.result);
			} else {
				fileDecompressed = event.target.result;
			}
			if (nifti.isNIFTI(fileDecompressed)) {
				niftiHeader = nifti.readHeader(fileDecompressed);
				niftiImage = nifti.readImage(niftiHeader, fileDecompressed);
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
			imagesNum = niftiHeader.dims[3] - 1;
			imageCurrentIndex = 0;
			imageCanvas.height = niftiHeader.dims[2];
			imageCanvas.width = niftiHeader.dims[1];
			imageSize = imageCanvas.height * imageCanvas.width;
			if (configSelected.machineLearningType === 'image classification') {
				classAnnotations = classAnnotations.slice(0, imagesNum+1);
			} else if (configSelected.machineLearningType === 'image segmentation') {
				maskCanvas.height = imageCanvas.height;
				maskCanvas.width = imageCanvas.width;
				brushCanvas.height = imageCanvas.height;
				brushCanvas.width = imageCanvas.width;
				masks = new Uint8Array(images.length);
			}
			imageHeightSpan.textContent = imageCanvas.height;
			imageIndexSpan.textContent = `${imageCurrentIndex}/${imagesNum}`;
			imageWidthSpan.textContent = imageCanvas.width;
			resetImageValue();
		}
		disableUI(false);
	};
	reader.readAsArrayBuffer(file);
}

function resetData() {
	brushContext.clearRect(0, 0, brushCanvas.width, brushCanvas.height);
	fileDecompressed = null;
	imageContext.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
	imageCurrentIndex = 0;
	imageHeightSpan.textContent = '';
	imageIndexSpan.textContent = '';
	imageOffset = 0;
	imageSize = 1;
	imageValueMaxSpan.textContent = '';
	imageValueMin = 0;
	imageValueMinSpan.textContent = '';
	imageValueRange = 1;
	imageWidthSpan.textContent = '';
	images = new Uint8Array(imageSize);
	imagesNum = 0;
	labelCurrent = 0;
	maskContext.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
	masks = new Uint8Array(imageSize);
}

function resetImageValue() {
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
	imageValueMaxSpan.textContent = Math.round(max);
	imageValueMinSpan.textContent = Math.round(imageValueMin);
	drawCanvas();
}

function saveData(data, filename) {
	const a = document.createElement('a');
	document.body.appendChild(a);
	a.style = 'display: none';
	const blob = new Blob(data);
	const url = window.URL.createObjectURL(blob);
	a.href = url;
	a.download = filename;
	a.click();
	window.URL.revokeObjectURL(url);
}

async function selectModelName() {
	labelListDiv.textContent = '';
	loadImagesInputFile.value = '';
	loadPredictionsInputFile.value = '';
	resetData();
	configSelected = configArray.find(config => config.modelURL === modelSelect.value);
	let loadModelFunction;
	if (configSelected.format === 'graph-model') {
		loadModelFunction = tf.loadGraphModel;
	} else if (configSelected.format === 'layers-model') {
		loadModelFunction = tf.loadLayersModel;
	}
	model = await loadModelFunction(configSelected.modelURL, {
		onProgress: function (fraction) {
			modelLoadFractionDiv.textContent = `${Math.round(100*fraction)}%.`;
			modelInputShapeSpan.textContent = 'NaN';
			modelPredictionShapeSpan.textContent = 'NaN';
			modelTrainableSpan.textContent = 'NaN';
			disableUI(true);
		}
	});
	modelInputShape = model.inputs[0].shape.filter(x => x>3);
	imageCanvas.width = modelInputShape[0];
	imageCanvas.height = modelInputShape[1];
	maskCanvas.width = modelInputShape[0];
	maskCanvas.height = modelInputShape[1];
	brushCanvas.width = modelInputShape[0];
	brushCanvas.height = modelInputShape[1];
	modelLoadFractionDiv.textContent = 'Model loaded.';
	modelInputShapeSpan.textContent = modelInputShape;
	modelPredictionShapeSpan.textContent = model.outputs[0].shape;
	if (model.trainable) {
		accuracyDiv.style.display = '';
		epochCurrentDiv.style.display = '';
		epochsNumDiv.style.display = '';
		lossDiv.style.display = '';
		modelTrainableSpan.textContent = 'True';
		saveModelToDiskButton.style.display = '';
		saveModelToServerButton.style.display = '';
		trainModelLocallyButton.style.display = '';
	} else {
		accuracyDiv.style.display = 'none';
		epochCurrentDiv.style.display = 'none';
		epochsNumDiv.style.display = 'none';
		lossDiv.style.display = 'none';
		modelTrainableSpan.textContent = 'False';
		saveModelToDiskButton.style.display = 'none';
		saveModelToServerButton.style.display = 'none';
		trainModelLocallyButton.style.display = 'none';
	}
	for (const [i, labelText] of configSelected.classNames.entries()) {
		const labelDiv = document.createElement('div');
		labelDiv.id = `labelDiv${i}`;
		labelListDiv.appendChild(labelDiv);
		const labelPredictionDiv = document.createElement('text');
		labelPredictionDiv.id = `labelPredictionDiv${i}`;
		labelPredictionDiv.style.float = 'left';
		labelPredictionDiv.style.width = '50px';
		labelPredictionDiv.style.height = '15px';
		labelPredictionDiv.textContent = 'NaN';
		labelDiv.appendChild(labelPredictionDiv);
		const labelColorDiv = document.createElement('div');
		labelColorDiv.id = `labelColorDiv${i}`;
		labelColorDiv.style.backgroundColor = `rgb(${labelsColormap[i]})`;
		labelColorDiv.style.float = 'left';
		labelColorDiv.style.height = '15px';
		labelColorDiv.style.opacity = 0.3;
		labelColorDiv.style.width = '15px';
		labelColorDiv.onclick = (event) => {
			const nodeList = document.querySelectorAll('*[id^="labelColorDiv"]')
			for (let i = 0; i < nodeList.length; i++) {
				nodeList[i].style.opacity = 0.3;
			}
			event.currentTarget.style.opacity = 1;
			labelCurrent = parseInt(event.currentTarget.id.match(/\d+/)[0]);
			if (configSelected.machineLearningType === 'image classification') {
				classAnnotations[imageCurrentIndex] = labelCurrent;
			}
		};
		labelDiv.appendChild(labelColorDiv);
		const labelTextDiv = document.createElement('text');
		labelTextDiv.id = `labelTextDiv${i}`;
		labelTextDiv.style.float = 'left';
		labelTextDiv.style.height = '15px';
		labelTextDiv.textContent = labelText;
		labelDiv.appendChild(labelTextDiv);
		const br = document.createElement('br');
		labelDiv.appendChild(br);
	}
	if (configSelected.machineLearningType === 'image classification') {
		const nodeList = document.querySelectorAll('*[id^="labelPredictionDiv"]')
		for (let i = 0; i < nodeList.length; i++) {
			nodeList[i].style.display = '';
		}
		maskCanvas.style.display = 'none';
		brushCanvas.style.display = 'none';
		brushSizeDiv.style.display = 'none';
	} else if (configSelected.machineLearningType === 'image segmentation') {
		const nodeList = document.querySelectorAll('*[id^="labelPredictionDiv"]')
		for (let i = 0; i < nodeList.length; i++) {
			nodeList[i].style.display = 'none';
		}
		maskCanvas.style.display = '';
		brushCanvas.style.display = '';
		brushSizeDiv.style.display = '';
	}
	loadImagesInputFile.disabled = false;
	modelSelect.disabled = false;
}

brushCanvas.oncontextmenu = (event) => {
	event.preventDefault();
}

brushCanvas.onmousedown = (event) => {
	if (event.button === 0) {
		drawActivated = true;
	} else if (event.button === 2) {
		imageValueRangeActivated = true;
	}
}

brushCanvas.onmouseleave = () => {
	drawActivated = false;
	imageValueRangeActivated = false;
}

brushCanvas.onmousemove = (event) => {
	if (imageValueRangeActivated) {
		const rangeTmp = imageValueRange * (1 + event.movementX * 0.01);
		let midTmp = imageValueMin + imageValueRange / 2;
		midTmp -= Math.abs(midTmp) * event.movementY / 1000;
		imageValueMin = midTmp - rangeTmp / 2;
		imageValueRange = rangeTmp;
		imageValueMaxSpan.textContent = Math.round(2*imageValueRange);
		imageValueMinSpan.textContent = Math.round(imageValueMin);
	} else {
		offsetX = event.offsetX;
		offsetY = event.offsetY;
	}
	requestAnimationFrame(drawCanvas);
}

brushCanvas.onmouseup = () => {
	drawActivated = false;
	imageValueRangeActivated = false;
}

loadImagesInputFile.onchange = (event) => {
	resetData();
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
				itk.writeImageArrayBuffer(null, false, image, 'tmp.nii')
					.then((data) => {
						const blob = new Blob([data.arrayBuffer]);
						readNiftiFile(blob);
					});
			});
	} else if ((files[0].name.includes('.png')) || (files[0].name.includes('.jpg'))) {
		itk.readImageFileSeries(files)
			.then(function ({ image }) {
				itk.writeImageArrayBuffer(null, false, image, 'tmp.nii')
					.then((data) => {
						const blob = new Blob([data.arrayBuffer]);
						readNiftiFile(blob);
					});
			});
	}
}

loadPredictionsInputFile.onchange = (event) => {
	const file = event.currentTarget.files[0];
	if (file === undefined) {
		return;
	}
	const reader = new FileReader();
	reader.onloadend = (event) => {
		if (event.target.readyState === FileReader.DONE) {
			if (configSelected.machineLearningType === 'image classification') {
				const rows = event.target.result.split('\n');
				for (const [i, row] of rows.entries()) {
					const row_elements = row.split(',');
					classAnnotations[i] = configSelected.classNames.findIndex((element) => element == row_elements[1]);
				}
			} else if (configSelected.machineLearningType === 'image segmentation') {
				const niftiHeader = nifti.readHeader(event.target.result);
				const niftiImage = nifti.readImage(niftiHeader, event.target.result);
				masks = new Uint8Array(niftiImage);
				drawCanvas();
			}
		}
	};
	if (configSelected.machineLearningType === 'image classification') {
		reader.readAsText(file);
	} else if (configSelected.machineLearningType === 'image segmentation') {
		reader.readAsArrayBuffer(file);
	}
}

predictAllImagesButton.onclick = () => {
	for (let i = 0; i <= imagesNum; i++) {
		imageCurrentIndex = i;
		imageOffset = imageSize * imageCurrentIndex;
		predictCurrentImage();
	}
}

saveModelToDiskButton.onclick = async () => {
	await model.save('downloads://saved-model');
}

saveModelToServerButton.onclick = async () => {
	await model.save('http://172.17.0.2:5000/upload');
}

savePredictionsToDiskButton.onclick = async () => {
	if (files === undefined) {
		return;
	}
	let filename;
	let data;
	if (configSelected.machineLearningType === 'image classification') {
		data = '';
		for (const [i, classAnnotation] of classAnnotations.entries()) {
			data += [files[i].name, configSelected.classNames[classAnnotation]].join(',');
			data += '\n';
		}
		data = [data.slice(0, -1)];
		filename = 'classAnnotations.txt';
	} else if (configSelected.machineLearningType === 'image segmentation') {
		const niftiHeaderTmp = fileDecompressed.slice(0, 352);
		data = [new Uint8Array(niftiHeaderTmp, 0, niftiHeaderTmp.length), new Uint8Array(masks.buffer, 0, masks.buffer.length)];
		filename = 'masks.nii';
	}
	saveData(data, filename);
}

trainModelLocallyButton.onclick = async () => {
	disableUI(true);
	const images_ = new Uint8Array(images);
	let tensor = tf.tensor(images_).reshape([imagesNum + 1, imageCanvas.height, imageCanvas.width]);
	tensor = tensor.expandDims(-1);
	tensor = tf.image.resizeNearestNeighbor(tensor, modelInputShape);
	const tensorMomentsBefore = tf.moments(tensor);
	tensor = tensor.sub(tensorMomentsBefore.mean);
	const tensorMomentsAfter = tf.moments(tensor);
	tensor = tensor.div(tf.sqrt(tensorMomentsAfter.variance));
	let preProcessedImage;
	let predictions;
	if (configSelected.machineLearningType === 'image classification') {
		preProcessedImage = tensor;
		predictions = tf.oneHot(classAnnotations, configSelected.classNames.length);
	} else if (configSelected.machineLearningType === 'image segmentation') {
		preProcessedImage = tensor.squeeze(-1).expandDims(0).expandDims(0);
		predictions = tf.tensor(masks);
	}
	model.compile({
		optimizer: 'adam',
		loss: 'categoricalCrossentropy',
		metrics: ['accuracy'],
	});
	await model.fit(preProcessedImage, predictions, {
		epochs: epochsNumInputNumber.value,
		callbacks: [
			new tf.CustomCallback({
				onEpochEnd: (epoch, logs) => {
					epochCurrentSpan.textContent = epoch;
					lossSpan.textContent = logs.loss;
					accuracySpan.textContent = logs.acc;
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

window.addEventListener('keydown', (event) => {
	if (event.code === 'ArrowUp' && (imageCurrentIndex > 0)) {
		imageCurrentIndex--;
	} else if (event.code === 'ArrowDown' && (imageCurrentIndex < imagesNum)) {
		imageCurrentIndex++;
	} else {
		return;
	}
	imageOffset = imageSize * imageCurrentIndex;
	if (configSelected.machineLearningType === 'image classification') {
		labelCurrent = classAnnotations[imageCurrentIndex];
		document.getElementById(`labelColorDiv${labelCurrent}`).click();
	}
	imageHeightSpan.textContent = imageCanvas.height;
	imageIndexSpan.textContent = `${imageCurrentIndex}/${imagesNum}`;
	imageWidthSpan.textContent = imageCanvas.width;
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
				fetch(option.value)
					.then(response => response.text())
					.then((text) => {
						configArray[i].format = JSON.parse(text).format;
						configArray[i].weightPaths = JSON.parse(text).weightsManifest[0].paths;
					})
				option.textContent = configArray[i].name;
				modelSelect.appendChild(option);
			})
	}
	selectModelName();
})();
