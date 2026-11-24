# Low-Cost Underwater Robotic Fish-Biometric Measurement

Code and supporting materials for the manuscript:

**“A Low-Cost Underwater Robotic Platform for Non-Contact Fish Biometric Measurement with Externally Assisted Yaw Control”**

**Authors:** Cheuk Yiu Cheng and Condon Lau  
**Affiliation:** Department of Physics, City University of Hong Kong  
**Manuscript status:** In preparation

## Overview

This project investigates a low-cost laboratory-scale underwater robotic platform
for non-contact fish-biometric measurement.

The system integrates two underwater endoscope cameras, an external top-view
camera, a Raspberry Pi 5, and an ESP32-controlled propulsion assembly. External
fish tracking supports bounded yaw adjustments during recording, followed by
anatomical keypoint estimation and physical measurement of fish standard length
and body height.

The Raspberry Pi 5 is located outside the tank, and the cameras remain connected
by physical cables. The system therefore demonstrates externally assisted
robotic acquisition rather than fully untethered operation or autonomous fish
following.

## Main Features

- **Lightweight pose estimation:** A customized YOLOv11n-pose model using
  α-CIoU and Wing loss for fish detection and anatomical landmark estimation.

- **Edge deployment:** Comparison of the local-computer PyTorch/GPU configuration
  with the Raspberry Pi 5 NCNN/CPU configuration.

- **Externally assisted yaw control:** Top-view tracking supplies fish-position
  information for bounded motor commands transmitted to an ESP32 over Wi-Fi/UDP.

- **Physical dimensional estimation:** Distance-dependent calibration converts
  underwater image measurements into standard length and body height.

- **Fish-level aggregation:** Eligible observations from both underwater cameras
  are pooled by dimension before representative percentile calculation.

- **Exploratory mass regression:** Candidate equations relate platform-derived
  length and height to reference body mass, with all-fish fitting and
  leave-one-fish-out evaluation reported separately.

---

## Repository Structure

The following organization is proposed for the paper release. Update this list
to match the uploaded files.

- `/calibration_result/` — Images and Results of the top-view homography and pixel-to-cm conversion fitting and validations
- `/code/` — Codes for data analysis, model performance evaluation, calibration, and end-to-end measurement system
- `/end_to_end_measurement/` — Pooled lengths and heights of each fish from the end-to-end measurement, and corresponding UDP info
- `/mass_result/` — Data and fitting results of an exploratory fish mass estimation according to the the end-to-end measurement
- `/model_training_evaluation/` — All trained models in 6 configurations, with evaluation results
- `/reference_benchmark_study (Pi5)/` — Benchmark summaries of measurement-based results done by the Raspberry Pi 5 CPU
- `/reference_benchmark_study (local)/` — Benchmark summaries of measurement-based results done by the local workstation
- `/yaw_ack_results/` — Analysis results of ACK command acknowledgment


File availability and any separately hosted materials should be documented
within the corresponding directories.

---

## Data Availability

### Annotated Image Dataset

The image dataset is available on Roboflow Universe:

[Fish Tracking Dataset — Version 55](https://universe.roboflow.com/fishbiometric/fish-tracking-z9jdq/dataset/55)

The manuscript reports an image dataset containing 2,127 images, including
top-view and side-view imagery. A model-selection evaluation set containing
130 images and 687 annotated fish instances was used to compare training
configurations.


## Software and Hardware

### Local Training and Evaluation

- Python 3.12.
- PyTorch 2.9.1.
- Ultralytics 8.3.78.
- Windows 11.
- Intel Core i5-13400F.
- NVIDIA GeForce RTX 3070 Ti with 8 GB video memory.

### Embedded Processing and Control

- Raspberry Pi 5 with 8 GB RAM.
- NCNN inference backend running on the CPU.
- ESP32 microcontroller.
- TB6612FNG motor driver.
- Two independently controlled brushed DC motors.
- Two underwater endoscope cameras.
- One external top-view camera.

Exact package versions, calibration settings, and runtime parameters should be
taken from the configuration files associated with the archived paper release.

## Contact

For questions about the code, dataset, or experimental workflow, contact:

Cheuk Yiu Cheng  
Department of Physics, City University of Hong Kong  
Email: cycheng248-c@my.cityu.edu.hk

---

## Citation

If you use this work, please cite the associated manuscript. Publication details
and the archived software/data DOI will be added when available.

Cheuk Yiu Cheng and Condon Lau.
“A Low-Cost Underwater Robotic Platform for Non-Contact Fish Biometric
Measurement with Externally Assisted Yaw Control.”
Manuscript in preparation.

Please also follow the citation requirements of the original image sources
identified in the dataset documentation.
