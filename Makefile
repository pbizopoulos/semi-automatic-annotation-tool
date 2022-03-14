tmp/N2D_0003.dcm:
	mkdir -p tmp/
	curl -L -o tmp/chest_xray.jpg "https://upload.wikimedia.org/wikipedia/commons/a/a1/Normal_posteroanterior_%28PA%29_chest_radiograph_%28X-ray%29.jpg"
	curl -L -o tmp/val_im.nii.gz "https://drive.google.com/uc?export=download&id=1Tl5PTS2rmajWKJMrYcZ2Na5DURvbbpit"
	curl -L -o tmp/N2D_0001.dcm https://github.com/datalad/example-dicom-structural/raw/master/dicoms/N2D_0001.dcm
	curl -L -o tmp/N2D_0002.dcm https://github.com/datalad/example-dicom-structural/raw/master/dicoms/N2D_0002.dcm
	curl -L -o tmp/N2D_0003.dcm https://github.com/datalad/example-dicom-structural/raw/master/dicoms/N2D_0003.dcm

clean:
	rm -rf tmp/
