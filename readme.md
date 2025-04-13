# Fire Damage Detection Project

This project is designed to detect fire damage using machine learning models. Below are the instructions to set up the project, prepare the data, and run the training and validation scripts.

## Setup Instructions

1. Clone the repository to your local machine:
    ```
    git clone <repository_url>
    cd 743_Fire_Damage_Detection
    ```

2. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```

3. Set up the data folder:
    - Create a folder named `data` in the project directory.
    - Place your dataset inside the `data` folder. The folder structure should look like this:
    - Place the 
      ```
      743_Fire_Damage_Detection/
      ├── data/
      │   ├── Images/
      │   ├── labels.csv
      ├── Results/ResNet18_rgb_NIR_spectra_and_blur.pth
      ```

## Running the Scripts

## 1 Validating the Model
Run `validate.py`. The default options will generate the classification images in a image_predictions folder. 
You can turn this off by running `validate.py --no-save_images`

### 2. Training the Model
To train the model, run the `train.py`. The default settings will train with no augmentations.
Run `train.py --weighted_loss --augment --augment_RGB --augment_NIR` for the augmented training.

