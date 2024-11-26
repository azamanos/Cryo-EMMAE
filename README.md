# Cryo-EMMAE : Self-Supervised particle picking in Cryo-EM imaging

![header](./params/Cryo-EMMAE.png)

The Cryo-EMMAE pipeline starts with an input micrograph and follows these steps:
a. **Pre-processing**: The micrograph undergoes normalization of background noise to minimize correlation with experimental parameters and is filtered to enhance particle contrast.
b. **Micrograph Representation**: Patches are extracted from the pre-processed micrograph and used to map it onto the MAE representation space.
c. **Denoising**: The resulting embeddings form a smaller image where a k-means trained on the train set identifies pixels with the lowest noise levels. These images undergo further denoising through micrograph-specific hierarchical clustering.
d. **Post-processing**: Convolution-based smoothing is applied on the predictions of the particle centres with greater accuracy.

## Table of Contents

- [Description](#description)
- [Installation (Linux)](#Installation-(Linux))
- [Usage](#usage)
- [License](#license)
- [Contributing](#contributing)
- [Contact](#contact)
- [Citing this work](#citing-this-work)

<a id="description"></a>
## Description

Cryo-Electron Microscopy Masked AutoEncoder Cryo-EMMAE a self-supervised method designed to mitigate the need for such manually annotated data.
Cryo-EMMAE leverages the representation space of a masked autoencoder to sequentially denoise an input micrograph based on clusters of datapoints with different noise levels.
<a id="features"></a>

<a id="Installation-(Linux)"></a>
## Installation (Linux)

1\. Create the emmae environment by running the following command in your terminal:
```bash
conda env create -f ./emmae.yaml
```
2\. Whenever you want to work on the project, activate the emmae environment by executing the following command in the terminal:

```bash
conda activate emmae
```
3\. Download from zenodo [https://doi.org/10.5281/zenodo.11659477](https://doi.org/10.5281/zenodo.11659477 ""), checkpoints of the models and the datasets used in this work.
Extract the files in the main directory of the project.

<a id="usage"></a>
## Usage

### Preprocess data
To preprocess your data (supporting image format and MRC format),
give micrograph image directory,
the particle diameter of your protein in pixels of the original micrograph shape,
and the output directory to save the preprocessed micrographs.

```bash
python preprocess.py --micrographs-directory 'input_directory_path'
                     --mrc-type 'True if mrc file type, False if image file type'
                     --particle-diameter 'particle diameter as integer,
                                          in pixel size of the original image, e.g. 224'
                     --output-directory 'output_directory_path/'
```
### Train
To train the model prepare your yaml file from the given example.

```bash
python train.py -c runs/example.yaml
```
### Predict
Pick particles, given the yaml file, the experiment number, the according epoch for the checkpoint and the path to the list of the images to be predicted in npy format file.

```bash
python predict.py -c runs/example.yaml
                  --experiment 'example'
                  --epoch 400 --prediction-description 'example_prediction_set'
                  --prediction-set-path 'path to prediction list in npy format,
                                         e.g. ./datasets/data_lists/10291_validation.npy'
```
### Evaluate
Evaluate the picked particles based on the ground truth data, uploaded at the zenodo link under the results.zip file.
Give the same experiment, prediction-description, and prediction-set-path as in prediction step, the ground-truth-path is in default at "./results/target_512_20_npy/"

```bash
python evaluate.py  --experiment 'example'
                    --prediction-description 'example_prediction_set'
                    --prediction-set-path 'path to prediction list in npy format,
                                           e.g. ./datasets/data_lists/10291_validation.npy'
                    --ground-truth-path './results/target_512_20_npy/'
```

<a id="license"></a>
## License

This project is licensed under the MIT License - see the LICENSE file for details.

<a id="contributing"></a>
## Contributing

If you encounter any issues or have suggestions for improvement, please create an issue on GitHub. We appreciate your contribution!

<a id="contact"></a>
## Contact

For queries and suggestions, please contact: andreas.zamanos@athenarc.gr

<a id="citing-this-work"></a>
## Citing this work

If you use this code in your research, please cite the repository.
