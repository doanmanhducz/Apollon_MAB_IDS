# Apollon_MAB_IDS

## Overview
Apollon_MAB_IDS is an advanced Intrusion Detection System (IDS) designed to defend against adversarial machine learning (AML) attacks. The system leverages a Multi-Armed Bandit (MAB) approach with Thompson Sampling to dynamically select the optimal classifier for each input, thereby enhancing the resilience of the IDS in adversarial environments.

This research-grade system is developed and evaluated on real-world intrusion detection datasets, incorporating both standard and adversarially generated traffic data.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Key Features
* **Adversarial Robustness:** Mitigates adversarial attacks by diversifying classifier selection using the MAB framework.
* **Multi-Classifier Strategy:** Includes a set of ML classifiers, increasing unpredictability and reducing vulnerability.
* **Multi-Dataset Evaluation:** Validated on CIC-IDS-2017, CSE-CIC-IDS-2018, and CIC-DDoS-2019 datasets.
* **Adversarial Data Integration:** Supports generation of adversarial traffic data using RelevaGAN and ADVGAN.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Installation
### Clone this repository
```bash
git clone https://github.com/doanmanhducz/Apollon_MAB_IDS.git
cd Apollon_MAB_IDS
```
```bash
#(Optional but recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Usage
### Run Apollon IDS with MAB
To train and evaluate the IDS using Multi-Armed Bandit (Thompson Sampling):
```bash
Open and run: Apollon_MAB_Detection.ipynb
```
This notebook performs:
* Dataset preprocessing
* Classifier training
* MAB-based model selection
* Performance evaluation (including under adversarial conditions)

### Generating Adversarial Samples
The repository includes two methods for generating adversarial network traffic using GANs.
#### Using RelevaGAN
1. Clone the official RelevaGAN repository:
```bash
git clone https://github.com/rhr407/relevagan.git
```
2. Copy the contents from:
```bash
fixing_version/RelevanGAN/
``` 
3. Open and run:
```bash
RelevaGAN.ipynb
```
This notebook will generate adversarial network traffic using the integrated RelevaGAN model.

#### Using ADVGAN-TrafficManipulator
1. Clone the official ADVGAN repository:
```bash
git clone https://github.com/dongtsi/TrafficManipulator.git
```
2. Copy the contents from:
```bash
fixing_version/ADVGAN_Trafficmanipulator/
``` 
3. Open and run:
```bash
advgan-trafficmanipulator-ml4secids.ipynb
```
This notebook will generate adversarial traffic samples using the adjusted ADVGAN framework.


