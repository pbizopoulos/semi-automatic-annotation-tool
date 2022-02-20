const modelDetailsArray = [
	{ 
		URL: 'https://raw.githubusercontent.com/pbizopoulos/signal2image-modules-in-deep-neural-networks-for-eeg-classification/master/wbml/resnet34-1D/model.json',
		classNames: ['Open', 'Closed', 'Healthy', 'Tumor', 'Epilepsy'],
		exampleData: 'https://raw.githubusercontent.com/pbizopoulos/signal2image-modules-in-deep-neural-networks-for-eeg-classification/master/wbml/eeg-classification-example-data.txt',
		name: 'eeg-classification-resnet34-1D',
		type: 'signal-classification',
	},
	{ 
		URL: 'https://raw.githubusercontent.com/pbizopoulos/tmp/master/wbml/mobilenet_v2.imagenet/model.json',
		classNames: ['No Findings', 'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia'],
		exampleData: 'https://raw.githubusercontent.com/pbizopoulos/tmp/master/wbml/lung-classification-example-data.jpg',
		name: 'lung-classification-mobilenet_v2.imagenet',
		type: 'image-classification',
	},
	{ 
		URL: 'https://raw.githubusercontent.com/pbizopoulos/comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-in-ct/master/wbml/lesion-segmentation-a.FPN.mobilenet_v2.imagenet/model.json',
		classNames: ['None', 'Covid-19'],
		exampleData: 'https://raw.githubusercontent.com/pbizopoulos/comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-in-ct/master/wbml/lesion-segmentation-example-data.png',
		name: 'lesion-segmentation-a.FPN.mobilenet_v2.imagenet',
		type: 'image-segmentation',
	},
	{
		URL: 'https://raw.githubusercontent.com/pbizopoulos/comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-in-ct/master/wbml/lung-segmentation.FPN.mobilenet_v2.imagenet/model.json',
		classNames: ['None', 'Lung'],
		exampleData: 'https://raw.githubusercontent.com/pbizopoulos/comprehensive-comparison-of-deep-learning-models-for-lung-and-covid-19-lesion-segmentation-in-ct/master/wbml/lung-segmentation-example-data.jpg',
		name: 'lung-segmentation.FPN.mobilenet_v2.imagenet',
		type: 'image-segmentation',
	},
]
