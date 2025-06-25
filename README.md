# Weather Nowcasting

This repository contains the work I produced for a technical assessment on short-term weather forecasting (nowcasting). The main objective was to implement and train two models:

- A simple **persistence baseline**, where the forecast at time *t+1* is simply the input image at time *t*.
- A more advanced **deep learning model**, such as a U-Net.

## Workflow

To reproduce the full pipeline, please follow the steps below:

1. **Preprocessing**  
   Start by running all the cells in the notebook `01_preprocessing.ipynb`, located in the `notebooks` folder. This notebook handles the loading, formatting, and normalization of the input data.

2. **Model Definition**  
   The models are defined and initialized in `02_models.ipynb`. This includes both the baseline and the U-Net model.

3. **Evaluation**  
   Once the data is preprocessed and the models are defined, you can run `03_evaluation.ipynb` to evaluate the performance of both models using the chosen metrics and visualization tools.

## Alternative 

Run the `all_in_one.py` script that regroups everything from the preprocessing to the evaluationof each models on the test set. 
