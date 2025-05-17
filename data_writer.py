

import os
import pandas as pd

input_dir = './input/'
os.makedirs(input_dir, exist_ok=True)

# Create a DataFrame matching your orbit data
data = {
    'ID': ['B1'],
    'x_in': [736.2575],
    'y_in': [-7555.9584],
    'z_in': [-6308.0627],
    'vx_in': [4.8810],
    'vy_in': [2.3350],
    'vz_in': [-2.4600],
    'a_fin': [15230.0000],
    'e_fin': [0.2232],
    'i_fin': [1.0770],
    'Omega_fin': [2.6360],
    'omega_fin': [1.3510],
    'theta_fin': [1.9600]
}

df = pd.DataFrame(data)

# Save to CSV
csv_path = os.path.join(input_dir, 'orbit_data.csv')
df.to_csv(csv_path, index=False)# Creating a .csv file with key/value pairs for orbit input

print(f'Input data written in {csv_path}')