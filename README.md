# PORTAL: Scalable Tabular Foundation Models via Content-Specific Tokenization
[![REUSE status](https://api.reuse.software/badge/github.com/SAP-samples/portal)](https://api.reuse.software/info/github.com/SAP-samples/portal)

## Description
Implementation of the deep learning models with training and evaluation pipelines described in the paper "PORTAL: Scalable Tabular Foundation Models via Content-Specific Tokenization" published at 3rd Table Representation Learning Workshop at NeurIPS 2024. Link to the paper: https://arxiv.org/pdf/2410.13516. Poster: [link](portal_poster_portrait.pdf)

![logo](https://github.com/SAP-samples/portal/blob/main/model_architecture.png)

## Abstract

Self-supervised learning on tabular data seeks to apply advances from natural language and image domains to the diverse domain of tables. However, current techniques often struggle with integrating multi-domain data and require data cleaning or specific structural requirements, limiting the scalability of pre-training datasets. We introduce PORTAL (Pretraining One-Row-at-a-Time for All tabLes), a framework that handles various data modalities without the need for cleaning or preprocessing. This simple yet powerful approach can be effectively pre-trained on online-collected datasets and fine-tuned to match state-of-the-art methods on complex classification and regression tasks. This work offers a practical advancement in self-supervised learning for large-scale tabular data.

## Requirements

The requirements are detailed in the `requirements.txt` file

## Download and Installation

To run the model finetuning on the `carte` or `numeric` datasets:
```
python3 -m portal.portal YOUR_RUN_NAME --patience=20 --max_epochs 100 --dataset=carte --regression_loss=l2 --regression_target_normalization standard -c checkpoints/pretrained_ckpt.pth
```

For the detailed description of the parameters, please check the the `parse_args` function in the `portal/portal.py` file

## Citations

If you use this dataset in your research or want to refer to our work, please cite:
```
@inproceedings{
spinaci2024portal,
title={{PORTAL}: Scalable Tabular Foundation Models via Content-Specific Tokenization},
author={Marco Spinaci and Marek Polewczyk and Johannes Hoffart and Markus C. Kohler and Sam Thelin and Tassilo Klein},
booktitle={NeurIPS 2024 Third Table Representation Learning Workshop},
year={2024},
url={https://openreview.net/forum?id=TSZQvknbLO}
}
```

## Known Issues
No known issues

## How to obtain support
[Create an issue](https://github.com/SAP-samples/portal/issues) in this repository if you find a bug or have questions about the content.
 
For additional support, [ask a question in SAP Community](https://answers.sap.com/questions/ask.html).

## Contributing
If you wish to contribute code, offer fixes or improvements, please send a pull request. Due to legal reasons, contributors will be asked to accept a DCO when they create the first pull request to this project. This happens in an automated fashion during the submission process. SAP uses [the standard DCO text of the Linux Foundation](https://developercertificate.org/).

## License
Copyright (c) 2024 SAP SE or an SAP affiliate company. All rights reserved. This project is licensed under the Apache Software License, version 2.0 except as noted otherwise in the [LICENSE](LICENSE) file.
