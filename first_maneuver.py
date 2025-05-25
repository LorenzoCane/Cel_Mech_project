

import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import pandas as pd

from utils import *
from Orbitchanges import * 

#****************************************************************************************************
#Define output dirs
output_dir = './output/StandardStrategy1'
os.makedirs(output_dir, exist_ok=True)

# Read the input CSV
input_path = r'./input/orbit_data.csv'
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

#****************************************************************************************************
# Standard strategy 1 = indipendent and consecutive maneuvers

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

"""
#Plot  initial orbit and the FIRST transition orbit 

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([1, 1, 1])
ax.set_xlabel('X [km]', fontsize=12)
ax.set_ylabel('Y [km]', fontsize=12)
ax.set_zlabel('Z [km]', fontsize=12)
ax.set_title('3D Orbit around Earth', fontsize=14)

ticks = [-1e4, -0.5e4, 0, 0.5e4, 1e4]
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_zticks(ticks)

ax.set_xlim(-1.2e4, 1.2e4)
ax.set_ylim(-1.2e4, 1.2e4)
ax.set_zlim(-1.2e4, 1.2e4)

earth_3D(ax)
"""

# -----------------------------
"""
# Plot initial orbit (before maneuver)
initial_orbit = np.array([a_i, e_i, i_i, OM_i, om_i, theta_i])
X_i, Y_i, Z_i = plotOrbit(initial_orbit, theta_mark=True, ax=ax, mark_color = "red")

# Plot Fisrt transition orbit (after maneuver)
current_orbit = np.array([a_i, e_i, i_f, OM_f, om_transition, theta_maneuver_1])
X_trans_1, Y_trans_1, Z_trans_1 = plotOrbit(current_orbit, theta_mark=True, ax=ax, mark_color = "green")

ax.plot(X_i, Y_i, Z_i, color='orange', linewidth=1.5, label='Initial orbit')
ax.plot(X_trans_1, Y_trans_1, Z_trans_1, color = "blue", linewidth = 1.5, label ='First transition orbit')
ax.legend()

#Save img 
save_path = os.path.join(output_dir, img_name)
plt.savefig(save_path)
print(f'First Maneuver plot saved in {save_path}')

#Save data for late
maneuver_data = {
    'cost1_km_s':      Delta_v_1,
    'Delta_t_1_s':         Delta_t_1,
    'theta_before_maneuver_1_rad': theta_maneuver_1,
    'final_inclination_rad': i_f,
    'final_RAAN_rad':        OM_f,
    'om_transition_rad':     om_transition,
}

df_out = pd.DataFrame([maneuver_data])
out_path = os.path.join(output_dir, out_file)
df_out.to_csv(out_path, index=False)

print(f"First Maneuver data saved to {out_path}")
"""
#------------------------------------------------------------------------------------

#2nd maneuvers : Change argument of pericenter
img_name = 'second_man.pdf'
out_file = 'second_man_data.csv'

# Change the argument of pericenter
Delta_om = (om_f - om_transition) % (2. * np.pi)  # Ensure Delta_om is in the range [0, 2*pi]

# Need to perform the maneuver at theta_a = Delta_om/2 or theta = np.pi + Delta_om/2
theta_A = Delta_om / 2.  # Maneuver point A [rad]
theta_B = np.pi + (Delta_om / 2.)  # Maneuver point B [rad]

# Compute the time of flight
Delta_t_A = timeOfFlight(a_i, e_i, theta_maneuver_1, theta_A)
Delta_t_B = timeOfFlight(a_i, e_i, theta_maneuver_1, theta_B)

# Select the maneuver point
if Delta_t_A < Delta_t_B:
    Delta_v_2, om_final, theta_maneuver_2 = changePeriapsisArg(a_i, e_i, om_transition, Delta_om, theta_A)
    Delta_t_2 = Delta_t_A
    theta_change_periapsis = theta_A
else:
    Delta_v_2, om_final, theta_maneuver_2 = changePeriapsisArg(a_i, e_i, om_transition, Delta_om, theta_B)
    Delta_t_2 = Delta_t_B
    theta_change_periapsis = theta_B
'''
print("Cost of the Maneuver:", Delta_v_2, "km/s")
print("Time to perform the maneuver:", Delta_t_2, "s")
print("Final argument of periapsis:", om_final, "rad")
print("Final true anomaly:", theta_maneuver_2, "rad")
'''


# Plot the FIRST transition orbit and the SECOND transition orbit 
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([1, 1, 1])
ax.set_xlabel('X [km]', fontsize=12)
ax.set_ylabel('Y [km]', fontsize=12)
ax.set_zlabel('Z [km]', fontsize=12)
ax.set_title('3D Orbit around Earth', fontsize=14)

ticks = [-1e4, -0.5e4, 0, 0.5e4, 1e4]
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_zticks(ticks)

ax.set_xlim(-1.2e4, 1.2e4)
ax.set_ylim(-1.2e4, 1.2e4)
ax.set_zlim(-1.2e4, 1.2e4)

earth_3D(ax)

# Plot FIRST transition orbit (after maneuver)
current_orbit = np.array([a_i, e_i, i_f, OM_f, om_transition, theta_maneuver_1])
X_trans_1, Y_trans_1, Z_trans_1 = plotOrbit(current_orbit, theta_mark=True, ax=ax, mark_color = "red")
# Plot SECOND transition orbit (after maneuver)
current_orbit = np.array([a_i, e_i, i_f, OM_f, om_final, theta_maneuver_2])
X_trans_2, Y_trans_2, Z_trans_2 = plotOrbit(current_orbit, theta_mark=True, ax=ax, mark_color = "green")

ax.plot(X_trans_1, Y_trans_1, Z_trans_1, color='orange', linewidth=1.5, label='First transition orbit')
ax.plot(X_trans_2, Y_trans_2, Z_trans_2, color = "blue", linewidth = 1.5, label ='Second transition orbit')

ax.legend()

#Save img 
save_path = os.path.join(output_dir, img_name)
plt.savefig(save_path)
print(f'Second Maneuver plot saved in {save_path}')

#Save data for late
maneuver_data = {
    'cost2_km_s':      Delta_v_2,
    'Delta_t_2_s':     Delta_t_2,
    'om_final':        om_final,
    'theta_maneuver_2_rad':  theta_maneuver_2
}
df_out = pd.DataFrame([maneuver_data])
out_path = os.path.join(output_dir, out_file)
df_out.to_csv(out_path, index=False)

print(f"Second Maneuver data saved to {out_path}")

#------------------------------------------------------------------------------------
#3rd maneuvers : Change shape with Hohmann strategy
img_name = 'third_man.pdf'
out_file = 'third_man_data.csv'

# Hohmann transfer to arrive to the final orbit where om_final = om_f, hence aligned pericenter
Delta_v_Hohmann1, Delta_v_Hohmann2, Delta_t_Hohmann, theta_after_Hohmann = changeOrbitShape(a_i, e_i, om_final, a_f, e_f, om_f) # Return the optimal Hohman transfer cost between the two orbits

# Compute the position where the maneuver must be performed
if np.isclose(theta_after_Hohmann, np.pi, 1e-16):
  Delta_t_preHohmann = timeOfFlight(a_i, e_i, theta_maneuver_2, 0) # Pericenter transition orbit -> Apocenter final orbit
  theta_pre_Hohmann = 0
else:
  Delta_t_preHohmann = timeOfFlight(a_i, e_i, theta_maneuver_2, np.pi) # Apocenter transition orbit -> Pericenter transition orbit
  theta_pre_Hohmann = np.pi

Delta_v_3 = Delta_v_Hohmann1 + Delta_v_Hohmann2 # Total Cost of the Hohmann transfer
Delta_t_3 = Delta_t_preHohmann + Delta_t_Hohmann # Total time cost for the Hohmann transfer

# Wait until it reaches theta_f
Delta_t_4 = timeOfFlight(a_f, e_f, theta_after_Hohmann, theta_f) # Time to reach the final position [s]
'''
print("Cost of the Maneuver:", Delta_v_3, "km/s")
print("Time to perform the maneuver:", Delta_t_3, "s")
print("Final semi-major axis:", a_f, "km")
print("Final eccentricity:", e_f)
print("Final true anomaly:", theta_after_Hohmann, "rad")
'''

#Plot the SECOND transition orbit and the FINAL orbit
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([1, 1, 1])
ax.set_xlabel('X [km]', fontsize=12)
ax.set_ylabel('Y [km]', fontsize=12)
ax.set_zlabel('Z [km]', fontsize=12)
ax.set_title('3D Orbit around Earth', fontsize=14)

ticks = [-1e4, -0.5e4, 0, 0.5e4, 1e4]
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_zticks(ticks)

ax.set_xlim(-1.2e4, 1.2e4)
ax.set_ylim(-1.2e4, 1.2e4)
ax.set_zlim(-1.2e4, 1.2e4)

earth_3D(ax)

# -----------------------------
# Plot SECOND transition orbit (with initial Hohmann maneuver point)
current_orbit = np.array([a_i, e_i, i_f, OM_f, om_final, theta_pre_Hohmann])
X_trans_2, Y_trans_2, Z_trans_2 = plotOrbit(current_orbit, theta_mark=True, ax=ax, mark_color = "red")
# Plot FINAL orbit (with second Hohmann maneuver point)
current_orbit = np.array([a_f, e_f, i_f, OM_f, om_f, theta_after_Hohmann])
X_f_0, Y_f_0, Z_f_0 = plotOrbit(current_orbit, theta_mark=True, ax=ax, mark_color = "green")

ax.plot(X_trans_2, Y_trans_2, Z_trans_2, color='orange', linewidth=1.5, label='Second transition orbit')
ax.plot(X_f_0, Y_f_0, Z_f_0, color = "blue", linewidth = 1.5, label ='Final orbit')

ax.legend()

#Save img 
save_path = os.path.join(output_dir, img_name)
plt.savefig(save_path)
print(f'Third Maneuver plot saved in {save_path}')

#Save data for late
maneuver_data = {
    'cost3_km_s':      Delta_v_3,
    'Delta_t_3_s':     Delta_t_3,
    'a_final':         a_f,
    'e_final':         e_f,
    'final_theta_anom': theta_after_Hohmann
}
df_out = pd.DataFrame([maneuver_data])
out_path = os.path.join(output_dir, out_file)
df_out.to_csv(out_path, index=False)

print(f"Third Maneuver data saved to {out_path}")

#*********************************************************************************************
#Print and save cost and time 
print('=====================================================================================')

#summary print
print("Total cost Maneuver:")
print(f"First maneuver: {Delta_v_1:.3f} km/s")
print(f"Second maneuver: {Delta_v_2:.3f} km/s")
print(f"Hohmann transfer: {Delta_v_3:.3f} km/s")
print(f"Total cost: {Delta_v_1 + Delta_v_2 + Delta_v_3:.3f} km/s")

print("\nTotal time cost Maneuver:")
print(f"First maneuver: {Delta_t_1:.3f} s")
print(f"Second maneuver: {Delta_t_2:.3f} s")
print(f"Hohmann transfer (reach pericenter + transfer orbit): {Delta_t_3:.3f} s")
print(f"Reach the final position: {Delta_t_4:.3f} s")
print(f"Total time cost: {Delta_t_1 + Delta_t_2 + Delta_t_3 + Delta_t_4:.3f} s")

#collect summury data as a dict
total_maneuver_data = {
    'orbit_id': orbit_id,
    'Delta_v_1_km_s': Delta_v_1,
    'Delta_v_2_km_s': Delta_v_2,
    'Delta_v_3_km_s': Delta_v_3,
    'Total_Delta_v_km_s': Delta_v_1 + Delta_v_2 + Delta_v_3,
    
    'Delta_t_1_s': Delta_t_1,
    'Delta_t_2_s': Delta_t_2,
    'Delta_t_3_s': Delta_t_3,
    'Delta_t_4_s': Delta_t_4,
    'Total_Delta_t_s': Delta_t_1 + Delta_t_2 + Delta_t_3 + Delta_t_4
}

#Save to unique CSV per orbit ID
summary_outfile = os.path.join(output_dir, f"maneuver_summary_{orbit_id}.csv")
#save as dataframe as .csv file
df_summary = pd.DataFrame([total_maneuver_data])
df_summary.to_csv(summary_outfile, index=False)

print(f"Summary for orbit '{orbit_id}' saved to: {summary_outfile}")

# -----------------------------

# Plot complete Maneuver
img_name = 'Standard_strategy_maneuver_1.pdf'
 
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([1, 1, 1])
ax.set_xlabel('X [km]', fontsize=12)
ax.set_ylabel('Y [km]', fontsize=12)
ax.set_zlabel('Z [km]', fontsize=12)
ax.set_title('3D Orbit around Earth', fontsize=14)

ticks = [-1e4, -0.5e4, 0, 0.5e4, 1e4]
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_zticks(ticks)

ax.set_xlim(-1.2e4, 1.2e4)
ax.set_ylim(-1.2e4, 1.2e4)
ax.set_zlim(-1.2e4, 1.2e4)

earth_3D(ax)

#Plot  initial orbit and the FIRST transition orbit
# Plot initial orbit (before maneuver)
initial_orbit = np.array([a_i, e_i, i_i, OM_i, om_i, theta_i])
X_i, Y_i, Z_i = plotOrbit(initial_orbit, deltaTh =  (theta_maneuver_1 - theta_i) % (2 * np.pi),
                        theta_mark=True, ax=ax, mark_color = "red", line_color = "blue",
                        mark_label="Initial position")

# Plot Fisrt transition orbit (Change plane)
transition_orbit_1 = np.array([a_i, e_i, i_f, OM_f, om_transition, theta_maneuver_1])
X_trans_1, Y_trans_1, Z_trans_1 = plotOrbit(transition_orbit_1, deltaTh = (theta_change_periapsis - theta_maneuver_1) % (2 * np.pi),
                                        theta_mark=True, ax=ax, mark_color = "Gold", line_color = "limegreen", mark_type=".",
                                        mark_label = "")

# Plot Second transition orbit (Change periapsi arg) 
transition_orbit_2 = np.array([a_i, e_i, i_f, OM_f, om_f, theta_maneuver_2])
X_trans_2, Y_trans_2, Z_trans_2 = plotOrbit(transition_orbit_2, deltaTh = (theta_pre_Hohmann - theta_maneuver_2) % (2 * np.pi),
                                        theta_mark = True, ax = ax, mark_color="Gold", line_color="aqua", mark_type=".",
                                        mark_label = "")

# Plot Third transition orbit (Hohmann transfer)
# Determine the orbital elements of the Hohmann transition orbit
r_p_initial = a_i * (1 - e_i) # Periapsis of the initial orbit
r_a_final = a_f * (1 + e_f) # Apoapsis final orbit
a_Hohmann = (r_p_initial + r_a_final) / 2
e_Hohmann = (r_a_final - r_p_initial) / (r_p_initial + r_a_final)

# Plot Hohmann transition orbit, mark on the first impulse
transition_orbit_3 = np.array([a_Hohmann, e_Hohmann, i_f, OM_f, om_f, theta_pre_Hohmann])
X_trans_3, Y_trans_3, Z_trans_3 = plotOrbit(transition_orbit_3, deltaTh = (theta_after_Hohmann - theta_pre_Hohmann) % (2 * np.pi),
                                        theta_mark = True, ax = ax, mark_color = "Gold", line_color="magenta", mark_type=".",
                                        mark_label = "")

# Plot Hohmann transition orbit, mark on the second impulse
transition_orbit_3_bis = np.array([a_Hohmann, e_Hohmann, i_f, OM_f, om_f, theta_after_Hohmann])
X_trans_3_bis, Y_trans_3_bis, Z_trans_3_bis = plotOrbit(transition_orbit_3_bis, deltaTh = 0,
                                                    theta_mark = True, ax = ax, mark_color = "Gold", line_color="magenta", mark_type=".",
                                                    mark_label = "")

# Plot final orbit
current_orbit = np.array([a_f, e_f, i_f, OM_f, om_f, theta_f])
X_f, Y_f, Z_f = plotOrbit(current_orbit, deltaTh = np.abs(theta_after_Hohmann - theta_f)% (2 * np.pi),
                        theta_mark=True, ax=ax, mark_color = "green", 
                        mark_label="Final position")

ax.plot(X_i, Y_i, Z_i, color='blue', linewidth=1.5, label='Initial orbit')
ax.plot(X_trans_1, Y_trans_1, Z_trans_1, color = "limegreen", linewidth = 1.5, label ='Change orbital plane')
ax.plot(X_trans_2, Y_trans_2, Z_trans_2, color = "aqua", linewidth = 1.5, label = "Change argument of periapsis")
ax.plot(X_trans_3, Y_trans_3, Z_trans_3, color = "magenta", linewidth = 1.5, label = "Hohmann transfer orbit")
ax.plot(X_trans_3_bis, Y_trans_3_bis, Z_trans_3_bis, color = "magenta", linewidth = 0)
ax.plot(X_f, Y_f, Z_f, color = "orange", linewidth = 1.5, label = "Final orbit")
ax.plot
ax.legend()

#Save img 
save_path = os.path.join(output_dir, img_name)
plt.savefig(save_path)
print(f'Plot of full maneuver saved in {save_path}')

# Plot complete Maneuver in 2D
img_name_2D = 'Standard_strategy_maneuver_1_2D.pdf'

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlabel('X [km]', fontsize=12)
ax.set_ylabel('Y [km]', fontsize=12)
ax.set_title('2D Orbit around Earth', fontsize=14)

ticks = [-1e4, -0.5e4, 0, 0.5e4, 1e4]
ax.set_xticks(ticks)
ax.set_yticks(ticks)

earth_2D(ax, color='blue', marker='o', size=200)  # Add Earth as a point in the center

# Plot initial orbit (before maneuver)
initial_orbit = np.array([a_i, e_i, i_i, OM_i, om_i, theta_i])
X_i, Y_i = plotOrbit2D(initial_orbit, deltaTh=(theta_maneuver_1 - theta_i) % (2 * np.pi),
                       theta_mark=True, ax=ax, mark_color="red", line_color="blue",
                       mark_label="Initial position")

# Plot First transition orbit (Change plane)
transition_orbit_1 = np.array([a_i, e_i, i_f, OM_f, om_transition, theta_maneuver_1])
X_trans_1, Y_trans_1 = plotOrbit2D(transition_orbit_1, deltaTh=(theta_change_periapsis - theta_maneuver_1) % (2 * np.pi),
                                   theta_mark=True, ax=ax, mark_color="Gold", line_color="limegreen", mark_type=".",
                                   mark_label="")

# Plot Second transition orbit (Change periapsis arg)
transition_orbit_2 = np.array([a_i, e_i, i_f, OM_f, om_f, theta_maneuver_2])
X_trans_2, Y_trans_2 = plotOrbit2D(transition_orbit_2, deltaTh=(theta_pre_Hohmann - theta_maneuver_2) % (2 * np.pi),
                                   theta_mark=True, ax=ax, mark_color="Gold", line_color="aqua", mark_type=".",
                                   mark_label="")

# Plot Third transition orbit (Hohmann transfer)
# Determine the orbital elements of the Hohmann transition orbit
r_p_initial = a_i * (1 - e_i)  # Periapsis of the initial orbit
r_a_final = a_f * (1 + e_f)  # Apoapsis of the final orbit
a_Hohmann = (r_p_initial + r_a_final) / 2
e_Hohmann = (r_a_final - r_p_initial) / (r_p_initial + r_a_final)

# Hohmann transition orbit, first impulse
transition_orbit_3 = np.array([a_Hohmann, e_Hohmann, i_f, OM_f, om_f, theta_pre_Hohmann])
X_trans_3, Y_trans_3 = plotOrbit2D(transition_orbit_3, deltaTh=(theta_after_Hohmann - theta_pre_Hohmann) % (2 * np.pi),
                                   theta_mark=True, ax=ax, mark_color="Gold", line_color="magenta", mark_type=".",
                                   mark_label="")

# Second impulse of the Hohmann maneuver (on final orbit)
transition_orbit_3_bis = np.array([a_f, e_f, i_f, OM_f, om_f, theta_after_Hohmann])
X_trans_3_bis, Y_trans_3_bis = plotOrbit2D(transition_orbit_3_bis, deltaTh=(theta_f-theta_after_Hohmann) % (2 * np.pi),
                                           theta_mark=True, ax=ax, mark_color="Gold", line_color="orange", mark_type=".",
                                           mark_label="")

# Plot Final position
final_orbit = np.array([a_f, e_f, i_f, OM_f, om_f, theta_f])
X_f, Y_f = plotOrbit2D(final_orbit, deltaTh=0,
                       theta_mark=True, ax=ax, mark_color="green", line_color="orange",
                       mark_label="Final position")

# Add all lines and labels to the 2D plot
ax.plot(X_i, Y_i, color='blue', linewidth=1.5, label='Initial orbit')
ax.plot(X_trans_1, Y_trans_1, color="limegreen", linewidth=1.5, label='Change orbital plane')
ax.plot(X_trans_2, Y_trans_2, color="aqua", linewidth=1.5, label='Change argument of periapsis')
ax.plot(X_trans_3, Y_trans_3, color="magenta", linewidth=1.5, label='Hohmann transfer orbit')
ax.plot(X_trans_3_bis, Y_trans_3_bis, color="magenta", linewidth=0)
ax.plot(X_f, Y_f, color="orange", linewidth=1.5, label='Final orbit')
ax.grid(True)
ax.legend()

# Save the 2D plot image
save_path_2D = os.path.join(output_dir, img_name_2D)
plt.savefig(save_path_2D)
print(f'2D Plot of full maneuver saved in {save_path_2D}')