# Model Explanability

[![python](https://img.shields.io/badge/-Python_3.7_%7C_3.8_%7C_3.9_%7C_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_1.8+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_1.6+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.2-89b8cd)](https://hydra.cc/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![pre-commit](https://img.shields.io/badge/Pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![tests](https://github.com/ashleve/lightning-hydra-template/actions/workflows/test.yml/badge.svg)](https://github.com/ashleve/lightning-hydra-template/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/ashleve/lightning-hydra-template/branch/main/graph/badge.svg)](https://codecov.io/gh/ashleve/lightning-hydra-template)
[![code-quality](https://github.com/ashleve/lightning-hydra-template/actions/workflows/code-quality-main.yaml/badge.svg)](https://github.com/ashleve/lightning-hydra-template/actions/workflows/code-quality-main.yaml)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/ashleve/lightning-hydra-template#license)
[![contributors](https://img.shields.io/github/contributors/ashleve/lightning-hydra-template.svg)](https://github.com/ashleve/lightning-hydra-template/graphs/contributors)

<!-- <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.7+-blue?style=for-the-badge&logo=python&logoColor=white"></a> -->

<!-- <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 1.8+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning 1.6+-792ee5?style=for-the-badge&logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: hydra" src="https://img.shields.io/badge/config-hydra 1.2-89b8cd?style=for-the-badge&labelColor=gray"></a>
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge&labelColor=gray"></a> -->

A clean and scalable template to kickstart your deep learning project 🚀⚡🔥<br>
Click on [<kbd>Use this template</kbd>](https://github.com/ashleve/lightning-hydra-template/generate) to initialize new repository.

_Suggestions are always welcome!_

</div>

<br>

## 📌  TSAI Assignment-7 by Sparsh Kedia

This is built on top of Assignment-3(in the next section).

**Objectives**

- To do model explainability on 10 images from imagenet dataset on a heavy timm model using multiple techniques.

Techniques implemented in this - 

**Model Explanation with**- 

- IG
- IG w/ Noise Tunnel
- Saliency
- Occlusion
- SHAP
- GradCAM
- GradCAM++

**Model Robustness with** -

- Pixel Dropout
- FGSM

**Steps to run:**

- Git clone the repository - 
        ``` git clone https://github.com/sparshk/emlov2-session-07.git ```
- Change working directory to the cloned repo - 
        ``` cd emlov2-session-07/ModelExplanability ```
- Create conda environment - 
        ``` conda create --name <env_name> ```
- Activate conda environment - 
        ``` conda activate <env_name> ```
- Install requirements - 
        ``` pip install -r requirements.txt ```       
- Remove output directory - 
        ``` rm -r images/output ```   
- Run the hypermater sweep - 
        ``` python explain.py ```

### Some images and their outputs have been plotted below. All the outputs are in ```ModelExplanability/images/output/``` path.

## 1. Dog Image 
**Original Image**

![](images/dog-2.jpg)
### Output:
![](images/output/_6/_dog-2.jpg_gradcam.png)![](images/output/_6/_dog-2.jpg_gradcam_plus_plus.png)![](images/output/_6/_dog-2.jpg_gradient_shap.png)![](images/output/_6/_dog-2.jpg_ig.png)![](images/output/_6/_dog-2.jpg_min_robust_perturbation.png)![](images/output/_6/_dog-2.jpg_occlusion.png)![](images/output/_6/_dog-2.jpg_org_img.png)![](images/output/_6/_dog-2.jpg_perturbed_image.png)![](images/output/_6/_dog-2.jpg_saliency1.png)![](images/output/_6/_dog-2.jpg_saliency2.png)![](images/output/_6/_dog-2.jpg_shap.png)![](images/output/_6/_dog-2.jpgperturbed_pgd.png)


## 2. Boat Image
**Original Image**

![](images/boat-8.jpg)
### Output:

![](images/output/_3/_boat-8.jpg_gradcam.png)![](images/output/_3/_boat-8.jpg_gradcam_plus_plus.png)![](images/output/_3/_boat-8.jpg_gradient_shap.png)![](images/output/_3/_boat-8.jpg_ig.png)![](images/output/_3/_boat-8.jpg_min_robust_perturbation.png)![](images/output/_3/_boat-8.jpg_occlusion.png)![](images/output/_3/_boat-8.jpg_org_img.png)![](images/output/_3/_boat-8.jpg_perturbed_image.png)![](images/output/_3/_boat-8.jpg_saliency1.png)![](images/output/_3/_boat-8.jpg_saliency2.png)![](images/output/_3/_boat-8.jpg_shap.png)![](images/output/_3/_boat-8.jpgperturbed_pgd.png)


## 3. Car Image
**Original Image**

![](images/car-4.jpg)
### Output:

![](images/output/_9/_car-4.jpg_gradcam_plus_plus.png)![](images/output/_9/_car-4.jpg_gradcam.png)![](images/output/_9/_car-4.jpg_gradient_shap.png)![](images/output/_9/_car-4.jpg_ig.png)![](images/output/_9/_car-4.jpg_min_robust_perturbation.png)![](images/output/_9/_car-4.jpg_occlusion.png)![](images/output/_9/_car-4.jpg_org_img.png)![](images/output/_9/_car-4.jpg_saliency1.png)![](images/output/_9/_car-4.jpg_perturbed_image.png)![](images/output/_9/_car-4.jpg_saliency2.png)![](images/output/_9/_car-4.jpg_shap.png)![](images/output/_9/_car-4.jpgperturbed_pgd.png)

## 4. Hen Image
**Original Image**

![](images/hen-5.jpg)
### Output:

![](images/output/_1/_hen-5.jpg_gradcam.png)![](images/output/_1/_hen-5.jpg_gradcam_plus_plus.png)![](images/output/_1/_hen-5.jpg_gradient_shap.png)![](images/output/_1/_hen-5.jpg_min_robust_perturbation.png)![](images/output/_1/_hen-5.jpg_occlusion.png)![](images/output/_1/_hen-5.jpg_ig.png)![](images/output/_1/_hen-5.jpg_org_img.png)![](images/output/_1/_hen-5.jpg_saliency1.png)![](images/output/_1/_hen-5.jpg_perturbed_image.png)![](images/output/_1/_hen-5.jpg_saliency2.png)![](images/output/_1/_hen-5.jpg_shap.png)![](images/output/_1/_hen-5.jpgperturbed_pgd.png)