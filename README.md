# DAMS
**Domain Adaptation of MRI Scanners as an alternative to MRI harmonization**

The paper has been accepted for presentation at the 5th MICCAI Workshop on Domain Adaptation and Representation Transfer (DART). [Paper link](https://link.springer.com/chapter/10.1007/978-3-031-45857-6_1)

Download the slide of the presentation [DART_Rafsanjany_Kushol.pdf](https://github.com/rkushol/DAMS/files/12877285/DART_Rafsanjany_Kushol.pdf)

```
@inproceedings{kushol2023domain,
  title={Domain adaptation of MRI scanners as an alternative to MRI harmonization},
  author={Kushol, Rafsanjany and Frayne, Richard and Graham, Simon J and Wilman, Alan H and Kalra, Sanjay and Yang, Yee-Hong},
  booktitle={MICCAI Workshop on Domain Adaptation and Representation Transfer},
  pages={1--11},
  year={2023},
  organization={Springer}
}
```

## Abstract
Combining large multi-center datasets can enhance statistical power, particularly in the field of neurology, where data can be scarce. However, applying a deep learning model trained on existing neuroimaging data often leads to inconsistent results when tested on new data due to domain shift caused by differences between the training (source domain) and testing (target domain) data. Existing literature offers several solutions based on domain adaptation (DA) techniques, which ignore complex practical scenarios where heterogeneity may exist in the source or target domain. This study proposes a new perspective in solving the domain shift issue for MRI data by identifying and addressing the dominant factor causing heterogeneity in the dataset. We design an unsupervised DA method leveraging the maximum mean discrepancy and correlation alignment loss in order to align domain-invariant features. Instead of regarding the entire dataset as a source or target domain, the dataset is processed based on the dominant factor of data variations, which is the scanner manufacturer. Afterwards, the target domain's feature space is aligned pairwise with respect to each source domain's feature map. Experimental results demonstrate significant performance gain for multiple inter- and intra-study neurodegenerative disease classification tasks.

![Proposed_architecture](https://github.com/rkushol/DAMS/assets/76894940/35c6283a-8a8c-472c-b480-2eb8027b9678)



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
Email at: kushol@ualberta.ca

## Acknowledgement
This basic structure of the code relies on the project of [Deep Transfer Learning in PyTorch](https://github.com/easezyc/deep-transfer-learning/tree/master/MUDA/MFSAN)

[Aligning Domain-specific Distribution and Classifier for Cross-domain Classification from Multiple Sources](https://dl.acm.org/doi/pdf/10.1609/aaai.v33i01.33015989)

[Deep CORAL: Correlation Alignment for Deep Domain Adaptation](https://link.springer.com/chapter/10.1007/978-3-319-49409-8_35)
