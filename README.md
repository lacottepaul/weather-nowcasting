# Nowcasting Radar Forecasting

**Author:** Paul Lacotte  
**Date:** 2025-07-02  

---

## Description

This project implements a script for forecasting future radar reflectivity images based on past frames. It includes two models:

- A **persistence baseline model** that predicts the next image will be the same as the last input frame.  
- A **deep learning model** based on a simple U-Net architecture to learn spatiotemporal dynamics.

---

The core logic is encapsulated in the `Nowcasting.py` script, which is clean, functional, and ready to use.  
You’ll also find a set of messy but informative notebooks in the `notebooks/` directory. These contain early explorations, visual tests, and failed model attempts — they’re not meant to be reused as-is but can help understand the design decisions and development path. The report for this project is available at [`forecasting_report.pdf`](./forecasting_report.pdf).

If you're interested in understanding the reasoning behind the choices made and the steps taken to reach the final script, check out the [`research.txt`](./research.txt) file.

---

You can find everything functionnal in the `Nowcasting.py` script and the raw work including research, wrong turns etc. in the `notebooks`. 

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/lacottepaul/weather-nowcasting.git
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

## Research Notes 
The file `research.txt` summarizes the development process: from early baseline attempts to deep learning explorations and final implementation choices.

It explains why some decisions were made and what didn't work — particularly useful if you want to extend or adapt the project.

## Contact 

For questions or contributions, feel free to reach out. 
This work was developed as part of a technical test on radar nowcasting. 
