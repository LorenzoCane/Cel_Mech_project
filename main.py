

import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import pandas as pd

from utils import kep2car, car2kep, plotOrbit, earth_3D
from Orbitchanges import * 

#****************************************************************************************************
#Define output dirs
output_dir = './output/'
os.makedirs(output_dir, exist_ok=True)

# Read the input CSV
input_path = './input/orbit_data.csv'
df = pd.read_csv(input_path)

#Select orbid ID
orbit_id = 'B1'

# select the row
row_df = df.loc[df['ID'] == orbit_id]
if row_df.empty:
    raise KeyError(f"No orbit with ID='{orbit_id}' found in {input_path}")
row = row_df.iloc[0]   # <-- now a Series
#import input data

# initial orbit
r_i_ECI = row[['x_in','y_in','z_in']].to_numpy(dtype=float).flatten()      # km
v_i_ECI = row[['vx_in','vy_in','vz_in']].to_numpy(dtype=float).flatten()   # km/s

a_i, e_i, i_i, OM_i, om_i, theta_i = car2kep(r_i_ECI, v_i_ECI)

# Final orbit
a_f = row['a_fin'] # semi-major axis [km]
e_f = row['e_fin'] # eccentricity [-]
i_f = row['i_fin'] # inclination [rad]
OM_f = row['Omega_fin'] # RAAN [rad]
om_f = row['omega_fin'] # Pericenter anomaly [rad]
theta_f = row['theta_fin'] # True anomaly [rad]

#check and print
print("Initial Orbit:")
print(f"Initial pos: {r_i_ECI} km")
print(f"Initial vel: {v_i_ECI} km/s")
print(f"a_i = {a_i} km")
print(f"e_i = {e_i}")
print(f"i_i = {i_i} rad")
print(f"OM_i = {OM_i} rad")
print(f"om_i = {om_i} rad")
print(f"theta_i = {theta_i} rad")
print("\nTarget Orbit:")
print(f"a_f = {a_f} km")
print(f"e_f = {e_f}")
print(f"i_f = {i_f} rad")
print(f"OM_f = {OM_f} rad")
print(f"om_f = {om_f} rad")
print(f"theta_f ={theta_f} rad")
print('=====================================================================================')
#****************************************************************************************************
#Naive strategy = indipendent and consecutive maneuvers

#1st maneuvers : Change orbital plane
img_name = 'first_man.pdf'
out_file = 'first_man_data.csv'

Delta_v_1, om_transition, theta_maneuver_1, Delta_t_1 = changeOrbitalPlane(a_i, e_i, i_i, OM_i, om_i, theta_i, i_f, OM_f)

'''
print("Cost of the Maneuver:", Delta_v_1, "km/s")
print("Time to perform the maneuver:", Delta_t_1, "s")
print("Final inclination:", i_f, "rad")
print("Final RAAN: ", OM_f, "rad")
print("Current argument of periapsis:", om_transition, "rad")
print("Current true anomaly:", theta_maneuver_1, "rad")
'''

#Plot  initial orbit and the FIRST transition orbit 

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([1, 1, 1])
ax.set_xlabel('X [km]', fontsize=12)
ax.set_ylabel('Y [km]', fontsize=12)
ax.set_zlabel('Z [km]', fontsize=12)
ax.set_title('3D Orbit around Earth', fontsize=14)

# Axis Ticks and Limits
ticks = [-1e4, -0.5e4, 0, 0.5e4, 1e4]
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_zticks(ticks)

ax.set_xlim(-1.2e4, 1.2e4)
ax.set_ylim(-1.2e4, 1.2e4)
ax.set_zlim(-1.2e4, 1.2e4)

# Plot the Earth
earth_3D(ax)

# -----------------------------
# Plot initial orbit (before maneuver)
initial_orbit = np.array([a_i, e_i, i_i, OM_i, om_i, theta_i])
X_i, Y_i, Z_i = plotOrbit(initial_orbit, theta_mark=True, ax=ax, mark_color = "red")

# Plot Fisrt transition orbit (after maneuver)
current_orbit = np.array([a_i, e_i, i_f, OM_f, om_transition, theta_maneuver_1])
X_trans_1, Y_trans_1, Z_trans_1 = plotOrbit(current_orbit, theta_mark=True, ax=ax, mark_color = "green")

# Add Orbit Label
ax.plot(X_i, Y_i, Z_i, color='orange', linewidth=1.5, label='Initial orbit')
ax.plot(X_trans_1, Y_trans_1, Z_trans_1, color = "blue", linewidth = 1.5, label ='First transition orbit')
ax.legend()

#Save img 
save_path = os.path.join(output_dir, img_name)
plt.savefig(save_path)
print(f'First Maneuver plot saved in {save_path}')

#Save data for late
maneuver_data = {
    'Delta_v_1_km_s':      Delta_v_1,
    'Delta_t_1_s':         Delta_t_1,
    'final_inclination_rad': i_f,
    'final_RAAN_rad':        OM_f,
    'om_transition_rad':     om_transition,
    'theta_maneuver_1_rad':  theta_maneuver_1
}
df_out = pd.DataFrame([maneuver_data])
out_path = os.path.join(output_dir, out_file)
df_out.to_csv(out_path, index=False)

print(f"First Maneuver data saved to {out_path}")
#2nd maneuvers : Change argument of pericenter


#3rd maneuvers : Change shape



