# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 13:33:29 2026

@author: user
"""

import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd

#%%

array_l = np.load(r"D:\2nd_project_AUV_deepwater_fish_size_estimation\for_paper\reference_benchmark_study (local)\fish_3_3_measurement_arrays.npz")

length_combined_array_l = array_l['standard_length_combined_cm']
# thickness_top_array = array['top_width_thickness_proxy_cm']
height_combined_array_l = array_l['body_height_combined_cm']


array_p = np.load(r"D:\2nd_project_AUV_deepwater_fish_size_estimation\for_paper\reference_benchmark_study (Pi5)\fish_3_3\measurement_arrays.npz")

length_combined_array_p = array_p['standard_length_combined_cm']
# thickness_top_array = array['top_width_thickness_proxy_cm']
height_combined_array_p = array_p['body_height_combined_cm']



#%%

actual_length = 5.8
# actual_thickness = 1.7
actual_height = 4.5
# actual_mass = 5.7

fig, ax = plt.subplots(1,2,figsize=(15, 5))


# length_iqr = np.array(length_combined_array_l)
# q1 = np.percentile(length_iqr, 25)
# q3 = np.percentile(length_iqr, 75)
# lower_fence = q1 - 1.5 * (q3 - q1)
# upper_fence = q3 + 1.5 * (q3 - q1)
# inlier_mask = (
#     (length_iqr >= lower_fence) &
#     (length_iqr <= upper_fence)
# )

# length_iqr = length_iqr[inlier_mask]
# length_iqr = length_iqr[length_iqr > 0]

ax[0].hist(length_combined_array_l ,bins=np.linspace(0,20,200))
ax[0].hist(length_combined_array_p ,bins=np.linspace(0,20,200), alpha=0.5)
ax[0].axvline(x=actual_length, color='red', linestyle='--')
ax[0].set_xlabel('length (cm)')
ax[0].set_ylabel('frequency')
ax[0].set_title('Fish_3 Length (AI estimated)')

print(np.percentile(np.array(length_combined_array_l),48))




# ax[1].hist(thickness_top_array,bins=np.linspace(0,5,100))
# ax[1].axvline(x=actual_thickness, color='red', linestyle='--')
# ax[1].set_xlabel('thickness (cm)')
# ax[1].set_ylabel('frequency')
# ax[1].set_title('Thickness (AI estimated)')
# print(np.percentile(np.array(thickness_top_array),95))


# height_iqr = np.array(height_combined_array_l)
# q1 = np.percentile(height_iqr, 25)
# q3 = np.percentile(height_iqr, 75)
# lower_fence = q1 - 1.5 * (q3 - q1)
# upper_fence = q3 + 1.5 * (q3 - q1)
# inlier_mask = (
#     (height_iqr >= lower_fence) &
#     (height_iqr <= upper_fence)
# )

# height_iqr = height_iqr[inlier_mask]
# height_iqr = height_iqr[height_iqr > 0]

ax[1].hist(height_combined_array_l,bins=np.linspace(0,10,200))
ax[1].hist(height_combined_array_p,bins=np.linspace(0,10,200), alpha=0.5)
ax[1].axvline(x=actual_height, color='red', linestyle='--')
ax[1].set_xlabel('height (cm)')
ax[1].set_ylabel('frequency')
ax[1].set_title('Fish_3 Height (AI estimated)')
ax[1].legend(['Reference value','Local Computer', 'Raspberry Pi 5'],loc='upper right')

print(np.percentile(np.array(height_combined_array_l),30))
plt.savefig('FigureS3c.png', dpi=300, format='png', bbox_inches="tight")
