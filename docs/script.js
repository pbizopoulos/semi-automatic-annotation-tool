const brushCanvas = document.getElementById("brush-canvas");
const brushContext = brushCanvas.getContext("2d");
const brushSizeDiv = document.getElementById("brush-size-div");
const brushSizeInputRange = document.getElementById("brush-size-input-range");
const imageCanvas = document.getElementById("image-canvas");
const imageContext = imageCanvas.getContext("2d");
const imageHeightWidthSpan = document.getElementById("image-height-width-span");
const imageIndexInputRange = document.getElementById("image-index-input-range");
const imageIndexSpan = document.getElementById("image-index-span");
const labelListDiv = document.getElementById("label-list-div");
const loadFilesInputFile = document.getElementById("load-files-input-file");
const loadPredictionsInputFile = document.getElementById(
	"load-predictions-input-file",
);
const maskCanvas = document.getElementById("mask-canvas");
const maskContext = maskCanvas.getContext("2d");
const modelDownloadDiv = document.getElementById("model-download-div");
const modelDownloadProgress = document.getElementById(
	"model-download-progress",
);
const modelInputShapeSpan = document.getElementById("model-input-shape-span");
const modelPredictionShapeSpan = document.getElementById(
	"model-prediction-shape-span",
);
const modelSelect = document.getElementById("model-select");
const predictImageCurrentButton = document.getElementById(
	"predict-image-current-button",
);
const predictImagesAllButton = document.getElementById(
	"predict-images-all-button",
);
const resetImageValueButton = document.getElementById(
	"reset-image-value-button",
);
const savePredictionsToDiskButton = document.getElementById(
	"save-predictions-to-disk-button",
);
let modelConfigurationArray = [
	{
		classNames: ["None", "Lung", "Covid-19"],
		exampleDataUrl:
			"https://raw.githubusercontent.com/pbizopoulos/comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-in-ct/main/latex/python/prm/lesion-segmentation-example-data.jpg",
		machineLearningType: "image segmentation",
		modelDownloadUrl:
			"https://raw.githubusercontent.com/pbizopoulos/multiclass-covid-19-segmentation/main/python/prm/model/model.json",
		modelUploadUrl: "http://172.17.0.2:5000/upload",
		name: "CT lung and covid-19 segmentation (custom)",
		projectUrl:
			"https://github.com/pbizopoulos/multiclass-covid-19-segmentation",
	},
];
modelConfigurationArray = modelConfigurationArray.slice(0, 1);
const labelColorArray = [
	[255, 255, 255],
	[31, 119, 180],
	[174, 199, 232],
	[255, 127, 14],
	[255, 187, 120],
	[44, 160, 44],
	[152, 223, 138],
	[214, 39, 40],
	[255, 152, 150],
	[148, 103, 189],
	[197, 176, 213],
	[140, 86, 75],
	[196, 156, 148],
	[227, 119, 194],
	[247, 182, 210],
	[127, 127, 127],
	[199, 199, 199],
	[188, 189, 34],
	[219, 219, 141],
	[23, 190, 207],
	[158, 218, 229],
];
const classAnnotations = new Uint8Array(1000); // tmp hardcoded max value, remove later
let modelConfigurationSelected;
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
brushCanvas.oncontextmenu = brushCanvasOnContextMenu;
brushCanvas.onmousedown = brushCanvasOnMouseDown;
brushCanvas.onmouseleave = brushCanvasOnMouseLeave;
brushCanvas.onmousemove = brushCanvasOnMouseMove;
brushCanvas.onmouseup = brushCanvasOnMouseUp;
imageIndexInputRange.oninput = imageIndexInputRangeOnInput;
loadFilesInputFile.onchange = loadFilesInputFileOnChange;
loadPredictionsInputFile.onchange = loadPredictionsInputFileOnChange;
predictImagesAllButton.onclick = predictImagesAllButtonOnClick;
savePredictionsToDiskButton.onclick = savePredictionsToDiskButtonOnClick;

function brushCanvasOnContextMenu(event) {
	event.preventDefault();
}

function brushCanvasOnMouseDown(event) {
	if (event.button === 0) {
		drawActivated = true;
	}
}

function brushCanvasOnMouseLeave() {
	drawActivated = false;
}

function brushCanvasOnMouseMove(event) {
	offsetX = event.offsetX;
	offsetY = event.offsetY;
	requestAnimationFrame(drawCanvas);
}

function brushCanvasOnMouseUp() {
	drawActivated = false;
}

function disableUI(argument) {
	brushSizeInputRange.disabled = argument;
	imageIndexInputRange.disabled = argument;
	loadFilesInputFile.disabled = argument;
	loadPredictionsInputFile.disabled = argument;
	modelSelect.disabled = argument;
	predictImageCurrentButton.disabled = argument;
	predictImagesAllButton.disabled = argument;
	resetImageValueButton.disabled = argument;
	savePredictionsToDiskButton.disabled = argument;
}

function drawCanvas() {
	const imageDataImage = new ImageData(imageCanvas.width, imageCanvas.height);
	for (let i = 0; i < imageCanvas.height; i++) {
		const rowOffset = i * imageCanvas.width;
		for (let j = 0; j < imageCanvas.width; j++) {
			const imageValue =
				(255 * (images[imageOffset + rowOffset + j] - imageValueMin)) /
				(imageValueMax - imageValueMin);
			const offsetMult4 = (rowOffset + j) * 4;
			imageDataImage.data[offsetMult4] = imageValue;
			imageDataImage.data[offsetMult4 + 1] = imageValue;
			imageDataImage.data[offsetMult4 + 2] = imageValue;
			imageDataImage.data[offsetMult4 + 3] = 255;
		}
	}
	imageContext.putImageData(imageDataImage, 0, 0);
	if (modelConfigurationSelected.machineLearningType === "image segmentation") {
		brushContext.clearRect(0, 0, brushCanvas.width, brushCanvas.height);
		brushContext.fillStyle = `rgb(${labelColorArray[labelCurrent]})`;
		brushContext.beginPath();
		brushContext.arc(
			Math.floor(offsetX) + 0.5,
			Math.floor(offsetY) + 0.5,
			parseFloat(brushSizeInputRange.value),
			0,
			2 * Math.PI,
		);
		brushContext.fill();
		const brushImageData = brushContext.getImageData(
			0,
			0,
			imageCanvas.width,
			imageCanvas.height,
		);
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
			maskImageData.data[4 * i] = labelColorArray[maskValue][0];
			maskImageData.data[4 * i + 1] = labelColorArray[maskValue][1];
			maskImageData.data[4 * i + 2] = labelColorArray[maskValue][2];
			if (maskValue === 0) {
				maskImageData.data[4 * i + 3] = 0;
			} else {
				maskImageData.data[4 * i + 3] = 255;
			}
		}
		maskContext.putImageData(maskImageData, 0, 0);
	}
}

function imageIndexInputRangeOnInput() {
	imageIndexCurrent = imageIndexInputRange.value;
	imageOffset = imageSize * imageIndexCurrent;
	imageHeightWidthSpan.textContent = `${imageCanvas.height}\u00D7${imageCanvas.width}`;
	imageIndexSpan.textContent = `${imageIndexCurrent}/${imagesNum}`;
	drawCanvas();
}

function loadFilesInputFileOnChange(event) {
	resetData();
	disableUI(true);
	files = event.currentTarget.files;
	if (files[0] === undefined) {
		disableUI(false);
		return;
	}
	if (files[0].name.includes(".nii")) {
		readFileNifti(files[0]);
	} else if (files[0].name.includes(".dcm")) {
		itk.readImageDICOMFileSeries(files).then(({ image }) => {
			itk.writeImageArrayBuffer(null, false, image, "tmp.nii").then((data) => {
				const blob = new Blob([data.arrayBuffer]);
				readFileNifti(blob);
			});
		});
	} else if (files[0].name.includes(".png") || files[0].name.includes(".jpg")) {
		itk.readImageFileSeries(files).then(({ image }) => {
			itk.writeImageArrayBuffer(null, false, image, "tmp.nii").then((data) => {
				const blob = new Blob([data.arrayBuffer]);
				readFileNifti(blob);
			});
		});
	}
	loadFilesInputFile.value = "";
	loadPredictionsInputFile.value = "";
}

function loadPredictionsInputFileOnChange(event) {
	const file = event.currentTarget.files[0];
	const fileReader = new FileReader();
	fileReader.readAsArrayBuffer(file);
	fileReader.onloadend = (event) => {
		if (event.target.readyState === FileReader.DONE) {
			const niftiHeader = nifti.readHeader(event.target.result);
			const niftiImage = nifti.readImage(niftiHeader, event.target.result);
			masks = new Uint8Array(niftiImage);
			drawCanvas();
		}
	};
}

function predictImageCurrent() {
	let imageSlice = images.slice(imageOffset, imageOffset + imageSize);
	imageSlice = new Float32Array(imageSlice);
	tf.tidy(() => {
		let imageTensor = tf.tensor(imageSlice);
		imageTensor = tf.reshape(imageTensor, [
			imageCanvas.height,
			imageCanvas.width,
			1,
		]);
		imageTensor = tf.image.resizeNearestNeighbor(imageTensor, modelInputShape);
		if (
			modelConfigurationSelected.modelDownloadUrl ===
			"https://raw.githubusercontent.com/pbizopoulos/multiclass-covid-19-segmentation/main/python/prm/model/model.json"
		) {
			imageTensor = imageTensor.div(4095);
		}
		const modelInputsShape = model.inputs[0].shape;
		for (let i = 0; i < modelInputsShape.length; i++) {
			if (modelInputsShape[i] == null) {
				modelInputsShape[i] = 1;
			}
		}
		const preProcessedImage = imageTensor.reshape(modelInputsShape);
		let modelPrediction = model.predict(preProcessedImage);
		if (modelPrediction.size !== imageSize) {
			if (
				modelPrediction.shape[modelPrediction.shape.length - 1] !== 1 &&
				modelPrediction.shape[modelPrediction.shape.length - 1] !==
					modelConfigurationSelected.classNames.length
			) {
				modelPrediction = modelPrediction.reshape([512, 512, 1]);
			}
			modelPrediction = tf.image.resizeNearestNeighbor(modelPrediction, [
				imageCanvas.height,
				imageCanvas.width,
			]);
		}
		if (modelPrediction.shape[modelPrediction.shape.length - 1] === 1) {
			modelPrediction = modelPrediction.dataSync();
			for (let i = 0; i < modelPrediction.length; i++) {
				if (modelPrediction[i] > 0.5) {
					masks[imageOffset + i] = 1;
				}
			}
		} else {
			modelPrediction = modelPrediction.argMax(-1).squeeze(0).dataSync();
			for (let i = 0; i < modelPrediction.length; i++) {
				masks[imageOffset + i] = modelPrediction[i];
			}
		}
	});
	drawCanvas();
}

function predictImagesAllButtonOnClick() {
	imageIndexCurrent = 0;
	disableUI(true);
	const interval = setInterval(() => {
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

function readFileNifti(file) {
	const fileReader = new FileReader();
	fileReader.readAsArrayBuffer(file);
	fileReader.onloadend = (event) => {
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
					images = images.filter((value, index) => {
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
			maskCanvas.height = imageCanvas.height;
			maskCanvas.width = imageCanvas.width;
			brushCanvas.height = imageCanvas.height;
			brushCanvas.width = imageCanvas.width;
			masks = new Uint8Array(images.length);
			imageHeightWidthSpan.textContent = `${imageCanvas.height}\u00D7${imageCanvas.width}`;
			imageIndexSpan.textContent = `${imageIndexCurrent}/${imagesNum}`;
			resetImageValue();
		}
		disableUI(false);
	};
}

function resetData() {
	brushContext.clearRect(0, 0, brushCanvas.width, brushCanvas.height);
	fileDecompressed = null;
	imageContext.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
	imageHeightWidthSpan.textContent = "0\u00D70";
	imageIndexCurrent = 0;
	imageIndexInputRange.value = 0;
	imageIndexSpan.textContent = "0/0";
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

function saveData(data, fileName) {
	const a = document.createElement("a");
	document.body.appendChild(a);
	const blob = new Blob(data);
	const url = window.URL.createObjectURL(blob);
	a.href = url;
	a.download = fileName;
	a.click();
	window.URL.revokeObjectURL(url);
}

async function savePredictionsToDiskButtonOnClick() {
	if (files === undefined) {
		return;
	}
	const niftiHeaderTmp = fileDecompressed.slice(0, 352);
	const tmp = new Uint16Array(niftiHeaderTmp, 0, niftiHeaderTmp.length);
	tmp[35] = 2;
	const data = [tmp, new Uint16Array(masks.buffer, 0, masks.buffer.length)];
	saveData(data, "masks.nii");
}

async function selectModelName() {
	labelListDiv.textContent = "";
	resetData();
	modelConfigurationSelected = modelConfigurationArray.find(
		(modelConfiguration) =>
			modelConfiguration.modelDownloadUrl === modelSelect.value,
	);
	let loadModelFunction = tf.loadLayersModel;
	if (modelConfigurationSelected.format === "graph-model") {
		loadModelFunction = tf.loadGraphModel;
	} else if (modelConfigurationSelected.format === "layers-model") {
		loadModelFunction = tf.loadLayersModel;
	}
	model = await loadModelFunction(modelConfigurationSelected.modelDownloadUrl, {
		onProgress: (fraction) => {
			modelDownloadProgress.value = fraction;
			if (fraction === 1) {
				modelDownloadDiv.style.display = "none";
			}
			modelInputShapeSpan.textContent = "NaN";
			modelPredictionShapeSpan.textContent = "NaN";
			disableUI(true);
		},
	});
	modelInputShape = model.inputs[0].shape.filter((x) => x > 3);
	imageCanvas.width = modelInputShape[0];
	imageCanvas.height = modelInputShape[1];
	maskCanvas.width = modelInputShape[0];
	maskCanvas.height = modelInputShape[1];
	brushCanvas.width = modelInputShape[0];
	brushCanvas.height = modelInputShape[1];
	modelInputShapeSpan.textContent = modelInputShape;
	modelPredictionShapeSpan.textContent = model.outputs[0].shape;
	for (const [
		i,
		labelText,
	] of modelConfigurationSelected.classNames.entries()) {
		const labelDiv = document.createElement("div");
		labelDiv.id = `labelDiv${i}`;
		labelListDiv.appendChild(labelDiv);
		const labelPredictionDiv = document.createElement("text");
		labelPredictionDiv.id = `labelPredictionDiv${i}`;
		labelPredictionDiv.style.float = "left";
		labelPredictionDiv.style.width = "50px";
		labelPredictionDiv.style.height = "15px";
		labelPredictionDiv.textContent = "NaN";
		labelDiv.appendChild(labelPredictionDiv);
		const labelColorDiv = document.createElement("div");
		labelColorDiv.id = `label-color-div-${i}`;
		labelColorDiv.style.backgroundColor = `rgb(${labelColorArray[i]})`;
		labelColorDiv.style.float = "left";
		labelColorDiv.style.height = "15px";
		labelColorDiv.style.opacity = 0.3;
		labelColorDiv.style.width = "15px";
		labelColorDiv.onclick = (event) => {
			const nodeList = document.querySelectorAll('*[id^="label-color-div-"]');
			for (let i = 0; i < nodeList.length; i++) {
				nodeList[i].style.opacity = 0.3;
			}
			event.currentTarget.style.opacity = 1;
			labelCurrent = parseInt(event.currentTarget.id.match(/\d+/)[0]);
		};
		labelDiv.appendChild(labelColorDiv);
		const labelTextDiv = document.createElement("text");
		labelTextDiv.id = `labelTextDiv${i}`;
		labelTextDiv.style.float = "left";
		labelTextDiv.style.height = "15px";
		labelTextDiv.textContent = labelText;
		labelDiv.appendChild(labelTextDiv);
		const br = document.createElement("br");
		labelDiv.appendChild(br);
	}
	const nodeList = document.querySelectorAll('*[id^="labelPredictionDiv"]');
	for (let i = 0; i < nodeList.length; i++) {
		nodeList[i].style.display = "none";
	}
	maskCanvas.style.display = "";
	brushCanvas.style.display = "";
	brushSizeDiv.style.display = "";
	loadFilesInputFile.disabled = false;
	modelSelect.disabled = false;
}

for (const modelConfiguration of modelConfigurationArray) {
	const option = document.createElement("option");
	option.value = modelConfiguration.modelDownloadUrl;
	fetch(option.value)
		.then((response) => response.text())
		.then((text) => {
			modelConfiguration.format = JSON.parse(text).format;
		});
	option.textContent = modelConfiguration.name;
	modelSelect.appendChild(option);
}
selectModelName();
window.onload = () => {
	document.body.style.display = "";
};
