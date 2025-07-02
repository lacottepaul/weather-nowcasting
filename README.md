# Nowcasting Radar Forecasting

Author: Paul Lacotte  
Date: 2025-07-02

---

## Description

This project implements a script for forecasting future radar reflectivity images based on past frames. It provides two models:

- A **persistence baseline model** that predicts the next image will be the same as the last input frame.  
- A **deep learning model** based on a simple U-Net architecture to learn spatiotemporal dynamics.

---

You can find everything functionnal in the `Nowcasting.py` script and the raw work including research, wrong turns etc. in the `notebooks`. 

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/nowcasting-radar.git
   cd nowcasting-radar
   
2. Install dependecies:

   ```bash
   pip install -r requirements.txt

## Dataset 

The script expects a NetCDF (.nc) file containing radar reflectivity data over time.
The main variable should be the reflectivity array with dimensions:
(time, height, width). All of this is found in `radar_data.nc` 

### Usage 

1. Baseline model 

You can run the baseline model using: 

   ```bash
   python3 Nowcasting.py --data_path XXX --model baseline
   ```
In your console you should see the RMSE score on the test set. 
That will also gives you 4 images saved as you can see in the `Images` folder. That allows you to visualise the error maps and check if the prediction has a physical sense. 

2. Unet model

Following the baseline model you can launch the Unet training and evaluation with: 

   ```bash
   python3 Nowcasting.py --data_path XXX --model unet --epochs X
   ```

## Contact 

For questions or contributions, feel free to reach out. 
This work was developed as part of a technical test on radar nocasting. 
