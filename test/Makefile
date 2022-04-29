.POSIX:

artifactsdir=artifacts

.PHONY: test clean update

test: package-lock.json $(artifactsdir)/val_im.nii.gz
	node test.js

update:
	npm update

package-lock.json:
	npm install

$(artifactsdir)/val_im.nii.gz:
	mkdir -p $(artifactsdir)/
	curl -L -o $(artifactsdir)/chest_xray.jpg "https://upload.wikimedia.org/wikipedia/commons/a/a1/Normal_posteroanterior_%28PA%29_chest_radiograph_%28X-ray%29.jpg"
	curl -L -o $(artifactsdir)/val_im.nii.gz "https://drive.google.com/uc?export=download&id=1Tl5PTS2rmajWKJMrYcZ2Na5DURvbbpit"
	curl -L -o $(artifactsdir)/N2D_0001.dcm https://github.com/datalad/example-dicom-structural/raw/master/dicoms/N2D_0001.dcm
	curl -L -o $(artifactsdir)/N2D_0002.dcm https://github.com/datalad/example-dicom-structural/raw/master/dicoms/N2D_0002.dcm
	curl -L -o $(artifactsdir)/N2D_0003.dcm https://github.com/datalad/example-dicom-structural/raw/master/dicoms/N2D_0003.dcm

clean:
	rm -rf $(artifactsdir) node_modules/ package-lock.json
