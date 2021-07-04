# Sintec_project
Hypertension (the ‘silent killer’) is one of the main risk factors for cardiovascular diseases (CVDs), main cause of death worldwide. Its continuous monitoring can offer a valid tool for patients care, as blood pressure (BP) is a significant indicator of health and putting it together with other parameters, such as heart and breath rates, could strongly improve prevention of CVDs. In this work we investigate the cuff-less estimation of continuous BP through pulse transit time (PTT) and heart rate (HR) using regression techniques. Our approach is intended as the first step towards continuous BP estimation with a low error according to AAMI guidelines. The novelties introduced in this work are represented by the implementation of pre-processing and by the innovative method for features research and features processing to continuously monitor blood pressure in a non-invasive way. In fact, invasive methods are the only reliable methods for continuous monitoring, while non-invasive techniques recover the values in a discreet way. 
This approach can be considered the first step for the integration of this type of algorithms on wearable devices, in particular on the devices developed for SINTEC project.

## Folder organization
Folders are organized according the following description:
* **Patients**: folder containing all the patients data collected from [MIMIC III](https://archive.physionet.org/cgi-bin/atm/atm).
* **Dataset**: data selected and processed from previous folder are collected here; in here, the structure of data is standardized.
   * **\\Regression**: after a further selection and feature extraction, data related to each patient are stored here to be used for the regression. 
* **Plots**: contains all plots of signals to show which signals contain enough information and can be used for next steps.
   * **\\Peaks**: contains ABP, ECG and PPG signals for each patient highlighting the position of the peak and the KDE distribution of the points. 
   * **\\HR and PTT**: extraction of the HR and PTT features that will be used for the regression; if necessary, based on the standard deviation evaluated within small time windows, signal was interpolated to remove outliers. 
   * **\\interpolation**: given the different frequency sampling for each patient, every 0.1 seconds the values of each feature was interpolated and resampled.
   * **\\Regression**: DBP and SBP are shown with their prediction respectively; an error for each of the algorithms tested is also present.

## Usage

```python
from SintecProj import SintecProj

SP = SintecProj()

#read and plot data
SP.data_reader()

#find peaks and extract the HR and PTT from ECG and PPG signals
SP.peak_finder()

#Regress data 
SP.regression_process()

#(optional) gives an insight about best algorithm to use
SP.best_fz()
```

## Workflow
[data_processing.pdf](https://github.com/DanieleRussoGH/Sintec_project/files/6760265/data_processing.pdf)
[Train_Vs_Test.pdf](https://github.com/DanieleRussoGH/Sintec_project/files/6760266/Train_Vs_Test.pdf)

