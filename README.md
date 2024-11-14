# README


## Description

BER-Clear is a project that aims for curating Building Energy Rating datasets using contrastive learning and confident learning.

## Features

- ETL for Irish BER dataset
- Integration of Scarf for Self-supervised Contrastive Learning
- Merging overlapping class with Confident Learning (use CleanLab)

## Installation

To install Clear, follow these steps:

1. Clone the repository: `git clone https://github.com/luminlab/BerClear.git`
2. Navigate to the project root directory
3. Install dependencies: `pip install -r requirements.txt`

## BER Dataset

## Usage

Once the application is running, you can access it by opening your web browser and navigating to `http://localhost:3000`. From there, you can create an account or log in if you already have one. Once logged in, you can start creating and managing your tasks.


## License

Clear is open source software licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Integration with Scarf

A portion of the codebase in BerClear use the modules from the Scarf repository. This integration allows BerClear to leverage self-supervised contrastive learning provided by Scarf to extract the embeddings for the BER dataset.

Please note that the copyright and license for the code from Scarf belong to the Scarf repository. Any issues or inquiries related to that specific part of the code should be directed to the Scarf repository.

For more information about Scarf and its licensing terms, please visit the [Scarf repository](https://github.com/clabrugere/pytorch-scarf).
