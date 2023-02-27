# DAMS
Domain Adaptation of MRI Scanners as an alternative to MRI harmonization

## Abstract



## Requirements
PyTorch  
nibabel  
scipy  
scikit-image  


## Datasets
ADNI1, ADNI2, and AIBL dataset can be downloaded from [ADNI](http://adni.loni.usc.edu/) (Alzheimer’s Disease Neuroimaging Initiative)

MIRIAD dataset can be downloaded from [MIRIAD](http://miriad.drc.ion.ucl.ac.uk) (Minimal Interval Resonance Imaging in Alzheimer's Disease)

CALSNIC dataset can be requested from [CALSNIC](https://calsnic.org/) (Canadian ALS Neuroimaging Consortium)

## Preprocessing
### Skull stripping using Freesurfer v7.3.2
Command ``mri_synthstrip -i input -o stripped``

Details can be found [SynthStrip](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/) (SynthStrip: Skull-Stripping for Any Brain Image)


### Registration to MNI-152 using FSL FLIRT function
Details can be found [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FLIRT)

One implementation can be found [here](https://github.com/vkola-lab/brain2020/tree/master/Data_Preprocess). After registration, the image dimension will be $182\times218\times182$ and the voxel dimension will be $1\times1\times1$ $mm^3$.



## Training
Run `python train.py` to train the network. It will generate `dataset_source1_source2_to_target_max_accuracy.pth` in `Results` folder


## Testing
Run `python test.py`. It will load the pre-trained model `dataset_source1_source2_to_target_max_accuracy.pth` and generate the classification results based on the given target dataset

## Contact
Email at: 

## Acknowledgement
This basic structure of the code relies on the project of [Deep Transfer Learning in PyTorch](https://github.com/easezyc/deep-transfer-learning/tree/master/MUDA/MFSAN)

[Aligning Domain-specific Distribution and Classifier for Cross-domain Classification from Multiple Sources](https://dl.acm.org/doi/pdf/10.1609/aaai.v33i01.33015989)

[Deep CORAL: Correlation Alignment for Deep Domain Adaptation](https://link.springer.com/chapter/10.1007/978-3-319-49409-8_35)
