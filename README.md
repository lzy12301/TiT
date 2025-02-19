Efficient simulation and assimilation of turbulent flow using diffusion transformer
===================================================================================

This is the source code for paper "Efficient simulation and assimilation of turbulent flow using diffusion transformer" submitted to the 1st International Symposium on AI and Fluid Mechanics (AIFLUIDs), containing the training code for the diffusion transformer.

## 1. Install dependencies

You can install the dependencies by using this command

```
pip install -r requirements.txt
```

## 2. Prepare dataset

The dataset are created by [Shu et al.](https://doi.org/10.1016/j.jcp.2023.111972), and you can download it from their [Github repository](https://github.com/BaratiLab/Diffusion-based-Fluid-Super-resolution). After downloading, please run the following command to obtain the preprocessed data:

```
python data_preprocess.py --data_path path/to/your/dataset
```

Remenber to change the "path/to/your/dataset" to the actural path for saving the dataset.

## 3. Train the DiT model (optional)

To train the DiT, run the script by the following command:

```
bash run.sh
```

Of course you can change the options in the script according to your demand.

## 4. Download and test the pretrained models

You can directly download the pretrained checkpoints from our [Zenodo repository](https://zenodo.org/records/14890459), which contains two .zip files. Please unzip these two files to the '\_\_results\_\_' folder.

We provide a notebook named 'demo.ipynb' to present the basic usage of our model. You can open it and run block-by-block.
