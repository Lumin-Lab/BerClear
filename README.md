# README

## Description

BER-Clear is a project that focuses on curating Building Energy Rating (BER) datasets using contrastive learning and confident learning techniques. This repository contains the implementation of methods described in the paper titled "Model Failure or Data Corruption? Exploring Inconsistencies in Building Energy Ratings with Contrastive Learning and Confident Learning".

## Features

- ETL for Irish BER dataset: Our project includes an Extract, Transform, and Load (ETL) process specifically designed for the Irish Building Energy Rating (BER) dataset. This allows for efficient data extraction, transformation, and loading into the project.

- Integration of Scarf for Self-supervised Contrastive Learning: We have integrated the [Scarf](https://github.com/clabrugere/pytorch-scarf) repository into our project to leverage its self-supervised contrastive learning capabilities. This integration enables us to extract meaningful embeddings for the BER dataset for visualization.

- Merging overlapping class with Confident Learning (use [CleanLab](https://github.com/cleanlab/cleanlab)): To address the issue of overlapping classes in the BER dataset, we have implemented a merging technique using Confident Learning with CleanLab. This approach helps to improve the accuracy and reliability of the dataset by resolving inconsistencies.


## Installation

To install BER-Clear, follow these steps:

1. Clone the repository: `git clone https://github.com/luminlab/BerClear.git`
2. Navigate to the project root directory
3. Install dependencies: `pip install -r requirements.txt`

## BER Dataset
- Download [BER dataset](https://ndber.seai.ie/BERResearchTool/ber/search.aspx)

- Put the dataset under the input folder (./input)

## Usage
Follow the example notebooks under the `examples` folder.

## License

BER-Clear is open source software licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Integration with Scarf

A portion of the codebase in BER-Clear uses modules from the Scarf repository. This integration allows BER-Clear to leverage self-supervised contrastive learning provided by Scarf to extract embeddings for the BER dataset.

Please note that the copyright and license for the code from Scarf belong to the Scarf repository. Any issues or inquiries related to that specific part of the code should be directed to the Scarf repository.

For more information about Scarf and its licensing terms, please visit the [Scarf repository](https://github.com/clabrugere/pytorch-scarf).



