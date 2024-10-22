# PORTAL: Scalable Tabular Foundation Models via Content-Specific Tokenization
[![REUSE status](https://api.reuse.software/badge/github.com/SAP-samples/portal)](https://api.reuse.software/info/github.com/SAP-samples/portal)

## Description
Implementation of the deep learning models with training and evaluation pipelines described in the paper "PORTAL: Scalable Tabular Foundation Models via Content-Specific Tokenization" published at 3rd Table Representation Learning Workshop at NeurIPS 2024

![logo](https://github.com/SAP-samples/portal/model_architecture.pdf)

## Requirements

The requirements are detailed in the `requirements.txt` file

## Download and Installation

To run the model finetuning on the `carte` or `50k_subsample` datasets:
```
python3 -m portal.portal YOUR_RUN_NAME --regression_loss=mixed --patience=20 --max_epochs 100 --dataset=carte
```

For the detailed description of the parameters, please check the the `parse_args` function in the `portal/portal.py` file


## Known Issues
No known issues

## How to obtain support
[Create an issue](https://github.com/SAP-samples/portal/issues) in this repository if you find a bug or have questions about the content.
 
For additional support, [ask a question in SAP Community](https://answers.sap.com/questions/ask.html).

## Contributing
If you wish to contribute code, offer fixes or improvements, please send a pull request. Due to legal reasons, contributors will be asked to accept a DCO when they create the first pull request to this project. This happens in an automated fashion during the submission process. SAP uses [the standard DCO text of the Linux Foundation](https://developercertificate.org/).

## License
Copyright (c) 2024 SAP SE or an SAP affiliate company. All rights reserved. This project is licensed under the Apache Software License, version 2.0 except as noted otherwise in the [LICENSE](LICENSE) file.
