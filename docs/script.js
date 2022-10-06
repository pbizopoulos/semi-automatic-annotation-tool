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
const imageHeightWidthSpan = document.getElementById('imageHeightWidthSpan');
const imageIndexInputRange = document.getElementById('imageIndexInputRange');
const imageIndexSpan = document.getElementById('imageIndexSpan');
const labelListDiv = document.getElementById('labelListDiv');
const loadFilesInputFile = document.getElementById('loadFilesInputFile');
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
const predictImageCurrentButton = document.getElementById('predictImageCurrentButton');
const predictImagesAllButton = document.getElementById('predictImagesAllButton');
const resetImageValueButton = document.getElementById('resetImageValueButton');
const saveModelToDiskButton = document.getElementById('saveModelToDiskButton');
const saveModelToServerButton = document.getElementById('saveModelToServerButton');
const savePredictionsToDiskButton = document.getElementById('savePredictionsToDiskButton');
const trainModelLocallyButton = document.getElementById('trainModelLocallyButton');

const configUrlArray = [
	'https://raw.githubusercontent.com/pbizopoulos/comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-in-ct/main/python/dist/lesion-segmentation.json',
	'https://raw.githubusercontent.com/pbizopoulos/comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-in-ct/main/python/dist/lung-segmentation.json',
	// 'https://raw.githubusercontent.com/pbizopoulos/tmp/main/lung-classification.json',
]
const labelColorArray = [
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
let imageIndexCurrent;
let imageOffset;
let imageSize;
let imageValueMax;
let imageValueMin;
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
	imageIndexInputRange.disabled = argument;
	loadFilesInputFile.disabled = argument;
	loadPredictionsInputFile.disabled = argument;
	modelSelect.disabled = argument;
	predictImageCurrentButton.disabled = argument;
	predictImagesAllButton.disabled = argument;
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
			const imageValue = 255 * (images[imageOffset + rowOffset + j] - imageValueMin) / (imageValueMax - imageValueMin);
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
		brushContext.fillStyle = `rgb(${labelColorArray[labelCurrent]})`;
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
			maskImageData.data[4*i] = labelColorArray[maskValue][0];
			maskImageData.data[4*i + 1] = labelColorArray[maskValue][1];
			maskImageData.data[4*i + 2] = labelColorArray[maskValue][2];
			if (maskValue > 0) {
				maskImageData.data[4*i + 3] = 255;
			} else {
				maskImageData.data[4*i + 3] = 0;
			}
		}
		maskContext.putImageData(maskImageData, 0, 0);
	}
}

function predictImageCurrent() {
	let imageSlice = images.slice(imageOffset, imageOffset + imageSize);
	imageSlice = new Float32Array(imageSlice);
	tf.tidy(() => {
		let imageTensor = tf.tensor(imageSlice);
		imageTensor = tf.reshape(imageTensor, [imageCanvas.height, imageCanvas.width, 1]);
		imageTensor = tf.image.resizeNearestNeighbor(imageTensor, modelInputShape);
		const tensorMomentsBefore = tf.moments(imageTensor);
		imageTensor = imageTensor.sub(tensorMomentsBefore.mean);
		const tensorMomentsAfter = tf.moments(imageTensor);
		imageTensor = imageTensor.div(tf.sqrt(tensorMomentsAfter.variance));
		let preProcessedImage;
		if (configSelected.machineLearningType === 'image classification') {
			preProcessedImage = imageTensor.expandDims(0);
		} else if (configSelected.machineLearningType === 'image segmentation') {
			preProcessedImage = imageTensor.squeeze(-1).expandDims(0).expandDims(0);
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

function readFileNifti(file) {
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
			imageIndexInputRange.max = imagesNum;
			imageIndexCurrent = 0;
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
			imageHeightWidthSpan.textContent = `${imageCanvas.height}\u00D7${imageCanvas.width}`;
			imageIndexSpan.textContent = `${imageIndexCurrent}/${imagesNum}`;
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
	imageHeightWidthSpan.textContent = '0\u00D70';
	imageIndexCurrent = 0;
	imageIndexInputRange.value = 0;
	imageIndexSpan.textContent = '0/0';
	imageOffset = 0;
	imageSize = 1;
	imageValueMax = 0;
	imageValueMin = 0;
	images = new Uint8Array(imageSize);
	imagesNum = 0;
	labelCurrent = 0;
	maskContext.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
	masks = new Uint8Array(imageSize);
}

function resetImageValue() {
	const imageSlice = images.slice(imageOffset, imageOffset + imageSize);
	imageValueMax = -Infinity;
	imageValueMin = Infinity;
	for (let i = 0; i < imageSlice.length; i++) {
		if (imageSlice[i] > imageValueMax) {
			imageValueMax = imageSlice[i];
		}
		if (imageSlice[i] < imageValueMin) {
			imageValueMin = imageSlice[i];
		}
	}
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
	resetData();
	configSelected = configArray.find(config => config.modelDownloadUrl === modelSelect.value);
	let loadModelFunction = tf.loadGraphModel;
	if (configSelected.format === 'graph-model') {
		loadModelFunction = tf.loadGraphModel;
	} else if (configSelected.format === 'layers-model') {
		loadModelFunction = tf.loadLayersModel;
	}
	model = await loadModelFunction(configSelected.modelDownloadUrl, {
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
		labelColorDiv.style.backgroundColor = `rgb(${labelColorArray[i]})`;
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
				classAnnotations[imageIndexCurrent] = labelCurrent;
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
	document.getElementById('labelTextDiv1').textContent = 'Lesion'; // for review
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
	loadFilesInputFile.disabled = false;
	modelSelect.disabled = false;
}

brushCanvas.oncontextmenu = (event) => {
	event.preventDefault();
}

brushCanvas.onmousedown = (event) => {
	if (event.button === 0) {
		drawActivated = true;
	}
}

brushCanvas.onmouseleave = () => {
	drawActivated = false;
}

brushCanvas.onmousemove = (event) => {
	offsetX = event.offsetX;
	offsetY = event.offsetY;
	requestAnimationFrame(drawCanvas);
}

brushCanvas.onmouseup = () => {
	drawActivated = false;
}

imageIndexInputRange.oninput = () => {
	imageIndexCurrent = imageIndexInputRange.value;
	imageOffset = imageSize * imageIndexCurrent;
	if (configSelected.machineLearningType === 'image classification') {
		labelCurrent = classAnnotations[imageIndexCurrent];
		document.getElementById(`labelColorDiv${labelCurrent}`).click();
	}
	imageHeightWidthSpan.textContent = `${imageCanvas.height}\u00D7${imageCanvas.width}`;
	imageIndexSpan.textContent = `${imageIndexCurrent}/${imagesNum}`;
	drawCanvas();
}

loadFilesInputFile.onchange = (event) => {
	resetData();
	disableUI(true);
	files = event.currentTarget.files;
	if (files[0] === undefined) {
		disableUI(false);
		return;
	}
	if (files[0].name.includes('.nii')) {
		readFileNifti(files[0]);
	} else if (files[0].name.includes('.dcm')) {
		itk.readImageDICOMFileSeries(files)
			.then(function ({ image }) {
				itk.writeImageArrayBuffer(null, false, image, 'tmp.nii')
					.then((data) => {
						const blob = new Blob([data.arrayBuffer]);
						readFileNifti(blob);
					});
			});
	} else if ((files[0].name.includes('.png')) || (files[0].name.includes('.jpg'))) {
		itk.readImageFileSeries(files)
			.then(function ({ image }) {
				itk.writeImageArrayBuffer(null, false, image, 'tmp.nii')
					.then((data) => {
						const blob = new Blob([data.arrayBuffer]);
						readFileNifti(blob);
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

predictImagesAllButton.onclick = () => {
	let interval;
	imageIndexCurrent = 0;
	disableUI(true);
	interval = setInterval(() => {
		imageIndexInputRange.value = imageIndexCurrent;
		imageIndexInputRange.oninput();
		predictImageCurrent();
		imageIndexCurrent++;
		if (imageIndexCurrent > imagesNum) {
			clearInterval(interval);
			disableUI(false);
		}
	}, 100);
}

saveModelToDiskButton.onclick = async () => {
	await model.save('downloads://saved-model');
}

saveModelToServerButton.onclick = async () => {
	await model.save(configSelected.modelUploadUrl);
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
		const tmp = new Uint16Array(niftiHeaderTmp, 0, niftiHeaderTmp.length);
		tmp[35] = 2;
		data = [tmp, new Uint16Array(masks.buffer, 0, masks.buffer.length)];
		filename = 'masks.nii';
	}
	saveData(data, filename);
}

trainModelLocallyButton.onclick = async () => {
	disableUI(true);
	const imagesArray = new Uint8Array(images);
	let [preProcessedImage, predictions] = tf.tidy(() => {
		let imagesTensor = tf.tensor(imagesArray).reshape([imagesNum + 1, imageCanvas.height, imageCanvas.width, 1]);
		imagesTensor = tf.image.resizeNearestNeighbor(imagesTensor, modelInputShape);
		const tensorMomentsBefore = tf.moments(imagesTensor);
		imagesTensor = imagesTensor.sub(tensorMomentsBefore.mean);
		const tensorMomentsAfter = tf.moments(imagesTensor);
		imagesTensor = imagesTensor.div(tf.sqrt(tensorMomentsAfter.variance));
		let preProcessedImage;
		let predictions;
		if (configSelected.machineLearningType === 'image classification') {
			preProcessedImage = imagesTensor;
			predictions = tf.oneHot(classAnnotations, configSelected.classNames.length);
		} else if (configSelected.machineLearningType === 'image segmentation') {
			preProcessedImage = imagesTensor.squeeze(-1).expandDims(0).expandDims(0);
			predictions = tf.tensor(masks);
		}
		return [preProcessedImage, predictions];
	})
	model.compile({
		optimizer: configSelected.optimizer,
		loss: configSelected.loss,
		metrics: configSelected.metrics,
	});
	await model.fit(preProcessedImage, predictions, {
		epochs: epochsNumInputNumber.value,
		callbacks: [
			new tf.CustomCallback({
				onEpochEnd: (epoch, logs) => {
					epochCurrentSpan.textContent = epoch;
					lossSpan.textContent = logs.loss.toExponential(3);
					accuracySpan.textContent = logs.acc;
				}
			})
		]
	});
	tf.dispose(preProcessedImage);
	tf.dispose(predictions);
	disableUI(false);
}

(async () => {
	for (const [i, configUrl] of configUrlArray.entries()) {
		await fetch(configUrl)
			.then(response => response.text())
			.then((text) => {
				configArray[i] = JSON.parse(text);
				let option = document.createElement('option');
				option.value = configArray[i].modelDownloadUrl;
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
