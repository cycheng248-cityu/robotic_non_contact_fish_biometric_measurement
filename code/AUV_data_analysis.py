import os
import numpy as np
import matplotlib.pyplot as plt

#%%

result_array = np.load(r"D:\2nd_project_AUV_deepwater_fish_size_estimation\for_paper\end_to_end_measurement\fish_9\measurement_results.npz")
standard_length = result_array['standard_length_cm']
height = result_array['height_cm']
#%%

actual_length = 3.0
# actual_thickness = 1.0
actual_height = 0.9

fig, ax = plt.subplots(1,2,figsize=(15, 5))

ax[0].hist(standard_length ,bins=np.linspace(0,20,100))
ax[0].axvline(x=actual_length, color='red', linestyle='--')
ax[0].axvline(x=np.percentile(np.array(standard_length),48), color='green', linestyle='--')
ax[0].set_xlabel('length (cm)')
ax[0].set_ylabel('frequency')
ax[0].set_title('Fish_9 end-to-end Length (AI estimated)')
print(np.percentile(np.array(standard_length),48))

# ax[1].hist(thickness,bins=np.linspace(0,10,20))
# ax[1].axvline(x=actual_thickness, color='red', linestyle='--')
# ax[1].set_xlabel('thickness (cm)')
# ax[1].set_ylabel('frequency')
# ax[1].set_title('Thickness (AI estimated)')

ax[1].hist(height,bins=np.linspace(0,10,100))
ax[1].axvline(x=actual_height, color='red', linestyle='--')
ax[1].axvline(x=np.percentile(np.array(height),20), color='green', linestyle='--')
ax[1].set_xlabel('height (cm)')
ax[1].set_ylabel('frequency')
ax[1].set_title('Fish_9 end-to-end Height (AI estimated)')
print(np.percentile(np.array(height),20))
ax[1].legend(['Reference value','end-to-end estimated value'],loc='upper right')


plt.savefig('FigureS12.png', dpi=300, format='png', bbox_inches="tight")

