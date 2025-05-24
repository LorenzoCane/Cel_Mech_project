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
output_dir = './output/StandardStrategy2'
os.makedirs(output_dir, exist_ok=True)

# Read the input CSV
input_path = './input/orbit_data.csv'
df = pd.read_csv(input_path)

plot_3d_title = '3D Satellite Orbit (ECI Frame)'
plot_2d_title = '2D Proj. of Sat Orbit in the EP (ECI Frame)'

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
# Naive strategy = indipendent and consecutive maneuvers

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
ax.set_title(plot_3d_title, fontsize=14)

ticks = [-1e4, -0.5e4, 0, 0.5e4, 1e4]
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_zticks(ticks)

ax.set_xlim(-1.2e4, 1.2e4)
ax.set_ylim(-1.2e4, 1.2e4)
ax.set_zlim(-1.2e4, 1.2e4)

earth_3D(ax)

# -----------------------------
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
    'final_inclination_rad': i_f,
    'final_RAAN_rad':        OM_f,
    'om_transition_rad':     om_transition,
    'theta_maneuver_1_rad':  theta_maneuver_1
}
df_out = pd.DataFrame([maneuver_data])
out_path = os.path.join(output_dir, out_file)
df_out.to_csv(out_path, index=False)

print(f"First Maneuver data saved to {out_path}")

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
ax.set_title(plot_3d_title, fontsize=14)

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
# 3rd maneuvers : Change shape using bielliptic strategy
img_name = 'third_man.pdf'
out_file = 'third_man_data.csv'

# Wait to arrive at the pericenter
theta_pre_bielliptic = 0
Delta_t_pre_Bielliptic = timeOfFlight(a_i, e_i, theta_maneuver_2, theta_pre_bielliptic)

# Load the Hohmann transfer cost from the output folder
hohmann_data_path = './output/StandardStrategy1/third_man_data.csv'
hohmann_data = pd.read_csv(hohmann_data_path)
Delta_v_Hohmann = hohmann_data['cost3_km_s'].iloc[0]  # Extract the Hohmann transfer cost
Delta_t_Hohmann = hohmann_data['Delta_t_3_s'].iloc[0] # Extract the Hohmann time cost

# Plot Bielliptic cost depending on the choice of r_b
r_b_array = np.linspace(a_f * (1-e_f), 11 * a_f, 5000)  # from r_f to 10*r_f (wide range)
total_delta_v_bielliptic = []  # Store total delta-v
total_delta_t_bielliptic = []  # Store total delta-t 

for r_b in r_b_array:
    Delta_v1_bielliptic, Delta_v2_bielliptic, Delta_v3_bielliptic, Delta_t_bielliptic, _ = changeOrbitShape_bielliptic(
        a_i, e_i, om_final, a_f, e_f, om_f, r_b
    )
    total_delta_v_bielliptic.append(
        np.abs(Delta_v1_bielliptic) + np.abs(Delta_v2_bielliptic) + np.abs(Delta_v3_bielliptic)
    )

    total_delta_t_bielliptic.append(Delta_t_bielliptic)

total_delta_v_bielliptic = np.array(total_delta_v_bielliptic)
total_delta_t_bielliptic = np.array(total_delta_t_bielliptic)

# Create a single figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1: Delta-v Comparison
axes[0].plot(r_b_array, total_delta_v_bielliptic, label=r'Total $\Delta v$ for Bi-elliptic', color='blue')
axes[0].axhline(y=Delta_v_Hohmann, color='red', linestyle='--', label=r'Hohmann Total $\Delta v$')
axes[0].set_title('Delta-v Comparison')
axes[0].set_xlabel('Intermediate radius $r_b$ [km]')
axes[0].set_ylabel('Total $\Delta v$ [km/s]')
axes[0].legend()
axes[0].grid(True)

# Subplot 2: Time Cost Comparison
axes[1].plot(r_b_array, total_delta_t_bielliptic, label=r'Total $\Delta t$ for Bi-elliptic', color='blue')
axes[1].axhline(y=Delta_t_Hohmann, color='red', linestyle='--', label=r'Hohmann Total $\Delta t$')
axes[1].set_title('Time Cost Comparison')
axes[1].set_xlabel('Intermediate radius $r_b$ [km]')
axes[1].set_ylabel('Total $\Delta t$ [s]')
axes[1].legend()
axes[1].grid(True)

# Adjust layout and save the figure
plt.tight_layout()
comparison_plot_path = os.path.join(output_dir, 'bielliptic_vs_hohmann_comparison.png')
plt.savefig(comparison_plot_path)
print(f'Comparison plot saved to {comparison_plot_path}')

# Find the r_b that minimizes the bielliptic maneuver cost
optimal_r_b_index = np.argmin(total_delta_v_bielliptic)
optimal_r_b = r_b_array[optimal_r_b_index]
min_delta_v_bielliptic = total_delta_v_bielliptic[optimal_r_b_index]

# Perform the bielliptic maneuver with the optimal r_b
Delta_v1_bielliptic, Delta_v2_bielliptic, Delta_v3_bielliptic, Delta_t_bielliptic, theta_after_bielliptic = changeOrbitShape_bielliptic(
    a_i, e_i, om_final, a_f, e_f, om_f, optimal_r_b
)

# Save the bielliptic maneuver data
bielliptic_maneuver_data = {
    'optimal_r_b_km': optimal_r_b,
    'Delta_v1_km_s': Delta_v1_bielliptic,
    'Delta_v2_km_s': Delta_v2_bielliptic,
    'Delta_v3_km_s': Delta_v3_bielliptic,
    'Total_Delta_v_km_s': min_delta_v_bielliptic,
    'Delta_t_s': Delta_t_bielliptic,
    'theta_f_rad': theta_after_bielliptic,
}

bielliptic_data_path = os.path.join(output_dir, 'optimal_bielliptic_maneuver.csv')
pd.DataFrame([bielliptic_maneuver_data]).to_csv(bielliptic_data_path, index=False)
print(f'Optimal bielliptic maneuver data saved to {bielliptic_data_path}')

# Save the comparison data between Bielliptic and Hohmann
comparison_data = {
    'r_b_km': r_b_array,
    'Bielliptic_Total_Delta_v_km_s': total_delta_v_bielliptic,
    'Hohmann_Total_Delta_v_km_s': [Delta_v_Hohmann] * len(r_b_array),
}

comparison_data_path = os.path.join(output_dir, 'bielliptic_vs_hohmann_data.csv')
pd.DataFrame(comparison_data).to_csv(comparison_data_path, index=False)
print(f'Comparison data saved to {comparison_data_path}')

# Plot the SECOND transition orbit and the FINAL orbit for the bielliptic strategy
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([1, 1, 1])
ax.set_xlabel('X [km]', fontsize=12)
ax.set_ylabel('Y [km]', fontsize=12)
ax.set_zlabel('Z [km]', fontsize=12)
ax.set_title('3D Orbit around Earth (Bielliptic Strategy)', fontsize=14)

ticks = [-1e4, -0.5e4, 0, 0.5e4, 1e4]
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_zticks(ticks)

ax.set_xlim(-1.2e4, 1.2e4)
ax.set_ylim(-1.2e4, 1.2e4)
ax.set_zlim(-1.2e4, 1.2e4)

earth_3D(ax)

# Plot SECOND transition orbit (after first bielliptic maneuver)
current_orbit = np.array([a_i, e_i, i_f, OM_f, om_final, 0])  # Assuming theta = 0 for the first maneuver
X_trans_2, Y_trans_2, Z_trans_2 = plotOrbit(current_orbit, theta_mark=True, ax=ax, mark_color="red")

# Plot FINAL orbit (after second bielliptic maneuver)
current_orbit = np.array([a_f, e_f, i_f, OM_f, om_f, theta_after_bielliptic])
X_f, Y_f, Z_f = plotOrbit(current_orbit, theta_mark=True, ax=ax, mark_color="green")

ax.plot(X_trans_2, Y_trans_2, Z_trans_2, color='orange', linewidth=1.5, label='Second transition orbit')
ax.plot(X_f, Y_f, Z_f, color="blue", linewidth=1.5, label='Final orbit')
ax.legend()

# Save the plot
bielliptic_plot_path = os.path.join(output_dir, 'bielliptic_transition_orbits.png')
plt.savefig(bielliptic_plot_path)
print(f'Bielliptic transition orbits plot saved in {bielliptic_plot_path}')

Delta_v_3 = Delta_v1_bielliptic + Delta_v2_bielliptic # Total Cost of the Bielliptic transfer
Delta_t_3 = Delta_t_pre_Bielliptic + Delta_t_bielliptic # Total time cost for the Bielliptic transfer

# Wait until it reaches theta_f
Delta_t_4 = timeOfFlight(a_f, e_f, theta_after_bielliptic, theta_f) # Time to reach the final position [s]
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
ax.set_title(plot_3d_title, fontsize=14)

ticks = [-1e4, -0.5e4, 0, 0.5e4, 1e4]
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_zticks(ticks)

ax.set_xlim(-1.2e4, 1.2e4)
ax.set_ylim(-1.2e4, 1.2e4)
ax.set_zlim(-1.2e4, 1.2e4)

earth_3D(ax)

# -----------------------------
# Plot SECOND transition orbit (with initial bielliptic maneuver point)
current_orbit = np.array([a_i, e_i, i_f, OM_f, om_final, theta_pre_bielliptic])
X_trans_2, Y_trans_2, Z_trans_2 = plotOrbit(current_orbit, theta_mark=True, ax=ax, mark_color = "red")
# Plot FINAL orbit (with second bielliptic maneuver point)
current_orbit = np.array([a_f, e_f, i_f, OM_f, om_f, theta_after_bielliptic])
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
    'final_theta_anom': theta_after_bielliptic
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

# Plot complete Maneuver in 3D
img_name = 'Standard_strategy_maneuver_2.pdf'

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([1, 1, 1])
ax.set_xlabel('X [km]', fontsize=12)
ax.set_ylabel('Y [km]', fontsize=12)
ax.set_zlabel('Z [km]', fontsize=12)
ax.set_title(plot_3d_title, fontsize=14)

ticks = [-1.5e4, -1e4, -0.5e4, 0, 0.5e4, 1e4, 1.5e4, 2e4]
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_zticks(ticks)

earth_3D(ax)

# Plot initial orbit (before maneuver)
initial_orbit = np.array([a_i, e_i, i_i, OM_i, om_i, theta_i])
X_i, Y_i, Z_i = plotOrbit(initial_orbit, deltaTh=(theta_maneuver_1 - theta_i) % (2 * np.pi),
                          theta_mark=True, ax=ax, mark_color="red", line_color="blue",
                          mark_label="Initial position")

# Plot First transition orbit (Change plane)
transition_orbit_1 = np.array([a_i, e_i, i_f, OM_f, om_transition, theta_maneuver_1])
X_trans_1, Y_trans_1, Z_trans_1 = plotOrbit(transition_orbit_1, deltaTh=(theta_change_periapsis - theta_maneuver_1) % (2 * np.pi),
                                            theta_mark=True, ax=ax, mark_color="Gold", line_color="limegreen", mark_type=".",
                                            mark_label="")

# Plot Second transition orbit (Change periapsis argument)
transition_orbit_2 = np.array([a_i, e_i, i_f, OM_f, om_final, theta_maneuver_2])
X_trans_2, Y_trans_2, Z_trans_2 = plotOrbit(transition_orbit_2, deltaTh=(theta_pre_bielliptic - theta_maneuver_2) % (2 * np.pi),
                                            theta_mark=True, ax=ax, mark_color="Gold", line_color="aqua", mark_type=".",
                                            mark_label="")

# Plot Bielliptic transfer orbits

# Orbital element of first transfer ellipse
# Radius at periapsis of initial orbit
r_p1 = a_i * (1 - e_i)
# First transfer ellipse: from r_p1 to r_b
a_t1 = (r_p1 + optimal_r_b) / 2
e_t1 = (optimal_r_b - r_p1) / (optimal_r_b + r_p1)

# First transfer orbit (to intermediate radius r_b)
transfer_orbit_1 = np.array([a_t1, e_t1, i_f, OM_f, om_final, theta_pre_bielliptic])
X_trans_3, Y_trans_3, Z_trans_3 = plotOrbit(transfer_orbit_1, deltaTh=np.pi,  # Half orbit to r_b
                                            theta_mark=True, ax=ax, mark_color="Gold", line_color="magenta", mark_type=".",
                                            mark_label="")

# Orbital element of second transfer ellipse
# Radius at periapsis of final orbit
r_p2 = a_f * (1 - e_f)
# Second transfer ellipse: from r_b to r_p2
a_t2 = (optimal_r_b + r_p2) / 2
e_t2 = (optimal_r_b - r_p2) / (r_p2 + optimal_r_b)

# Second transfer orbit (from r_b to final orbit)
transfer_orbit_2 = np.array([a_t2, e_t2, i_f, OM_f, om_f, np.pi])
X_trans_4, Y_trans_4, Z_trans_4 = plotOrbit(transfer_orbit_2, deltaTh=np.pi,  # Half orbit from r_b to final orbit
                                            theta_mark=True, ax=ax, mark_color="Gold", line_color="darkviolet", mark_type=".",
                                            mark_label="")

# Plot the third impulse in the bielliptic maneuver (in the final orbit)
transfer_orbit_2_bis = np.array([a_f, e_f, i_f, OM_f, om_f, theta_after_bielliptic])
X_trans_4_bis, Y_trans_4_bis, Z_trans_4_bis = plotOrbit(transfer_orbit_2_bis, deltaTh= (theta_f - theta_after_bielliptic) % (2 * np.pi),  
                                            theta_mark=True, ax=ax, mark_color="Gold", line_color="orange", mark_type=".",
                                            mark_label="")

# Plot Final position 
final_orbit = np.array([a_f, e_f, i_f, OM_f, om_f, theta_f])
X_f, Y_f, Z_f = plotOrbit(final_orbit, deltaTh= 0 % (2 * np.pi), # Just the mark of the final position 
                          theta_mark=True, ax=ax, mark_color="green", line_color="orange",
                          mark_label="Final position")

# Add all lines and labels to the 3D plot
ax.plot(X_i, Y_i, Z_i, color='blue', linewidth=1.5, label='Initial orbit')
ax.plot(X_trans_1, Y_trans_1, Z_trans_1, color="limegreen", linewidth=1.5, label='Change orbital plane')
ax.plot(X_trans_2, Y_trans_2, Z_trans_2, color="aqua", linewidth=1.5, label='Change argument of periapsis')
ax.plot(X_trans_3, Y_trans_3, Z_trans_3, color="magenta", linewidth=1.5, label='Bielliptic transfer (1st)')
ax.plot(X_trans_4, Y_trans_4, Z_trans_4, color="darkviolet", linewidth=1.5, label='Bielliptic transfer (2nd)')
ax.plot(X_trans_4_bis, Y_trans_4_bis, Z_trans_4_bis, color="orange", linewidth=1.5)
ax.plot(X_f, Y_f, Z_f, color="orange", linewidth=1.5, label='Final orbit')
ax.legend()

ax.set_xlim(-1.5e4, 1.5e4)
ax.set_ylim(-1.5e4, 1.5e4)
ax.set_zlim(-1.5e4, 1.5e4)

# Save the 3D plot image
save_path = os.path.join(output_dir, img_name)
plt.savefig(save_path)
print(f'3D Plot of full maneuver saved in {save_path}')

# Plot complete Maneuver in 2D
img_name_2D = 'Standard_strategy_maneuver_2_2D.pdf'

# Plot complete Maneuver in 2D
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlabel('X [km]', fontsize=12)
ax.set_ylabel('Y [km]', fontsize=12)
ax.set_title(plot_2d_title, fontsize=14)

ticks = [-1e4, -0.5e4, 0, 0.5e4, 1e4]
ax.set_xticks(ticks)
ax.set_yticks(ticks)

earth_2D(ax, color='blue', marker='o', size=200)  # Add Earth as a point in the center

# Plot initial orbit (before maneuver)
X_i, Y_i = plotOrbit2D(initial_orbit, deltaTh=(theta_maneuver_1 - theta_i) % (2 * np.pi),
                       theta_mark=True, ax=ax, mark_color="red", line_color="blue",
                       mark_label="Initial position")

# Plot First transition orbit (Change plane)
X_trans_1, Y_trans_1 = plotOrbit2D(transition_orbit_1, deltaTh=(theta_change_periapsis - theta_maneuver_1) % (2 * np.pi),
                                   theta_mark=True, ax=ax, mark_color="Gold", line_color="limegreen", mark_type=".",
                                   mark_label="")

# Plot Second transition orbit (Change periapsis argument)
X_trans_2, Y_trans_2 = plotOrbit2D(transition_orbit_2, deltaTh=(theta_pre_bielliptic - theta_maneuver_2) % (2 * np.pi),
                                   theta_mark=True, ax=ax, mark_color="Gold", line_color="aqua", mark_type=".",
                                   mark_label="")

# Plot Bielliptic transfer orbits

# Orbital element of first transfer ellipse
# Radius at periapsis of initial orbit
r_p1 = a_i * (1 - e_i)
# First transfer ellipse: from r_p1 to r_b
a_t1 = (r_p1 + optimal_r_b) / 2
e_t1 = (optimal_r_b - r_p1) / (optimal_r_b + r_p1)

# First transfer orbit (to intermediate radius r_b)
transfer_orbit_1 = np.array([a_t1, e_t1, i_f, OM_f, om_final, theta_pre_bielliptic])
X_trans_3, Y_trans_3 = plotOrbit2D(transfer_orbit_1, deltaTh=np.pi,  # Half orbit to r_b
                                   theta_mark=True, ax=ax, mark_color="Gold", line_color="magenta", mark_type=".",
                                   mark_label="")

# Orbital element of second transfer ellipse
# Radius at periapsis of final orbit
r_p2 = a_f * (1 - e_f)
# Second transfer ellipse: from r_b to r_p2
a_t2 = (optimal_r_b + r_p2) / 2
e_t2 = (optimal_r_b - r_p2) / (r_p2 + optimal_r_b)

# Second transfer orbit (from r_b to final orbit)
transfer_orbit_2 = np.array([a_t2, e_t2, i_f, OM_f, om_f, np.pi])
X_trans_4, Y_trans_4 = plotOrbit2D(transfer_orbit_2, deltaTh=np.pi,  # Half orbit from r_b to final orbit
                                   theta_mark=True, ax=ax, mark_color="Gold", line_color="darkviolet", mark_type=".",
                                   mark_label="")
print(a_t2, e_t2, a_f, e_f)
# Plot the third impulse in the bielliptic maneuver (in the final orbit)
transfer_orbit_2_bis = np.array([a_f, e_f, i_f, OM_f, om_f, theta_after_bielliptic])
X_trans_4_bis, Y_trans_4_bis = plotOrbit2D(transfer_orbit_2_bis, deltaTh= (theta_f - theta_after_bielliptic) % (2 * np.pi),  
                                           theta_mark=True, ax=ax, mark_color="Gold", line_color="magenta", mark_type=".",
                                           mark_label="")

# Plot Final position
X_f, Y_f = plotOrbit2D(final_orbit, deltaTh=0,
                       theta_mark=True, ax=ax, mark_color="green", line_color="orange",
                       mark_label="Final position")

# Add all lines and labels to the 2D plot
ax.plot(X_i, Y_i, color='blue', linewidth=1.5, label='Initial orbit')
ax.plot(X_trans_1, Y_trans_1, color="limegreen", linewidth=1.5, label='Change orbital plane')
ax.plot(X_trans_2, Y_trans_2, color="aqua", linewidth=1.5, label='Change argument of periapsis')
ax.plot(X_trans_3, Y_trans_3, color="magenta", linewidth=1.5, label='Bielliptic transfer (1st)')
ax.plot(X_trans_4, Y_trans_4, color="darkviolet", linewidth=1.5, label='Bielliptic transfer (2nd)')
ax.plot(X_trans_4_bis, Y_trans_4_bis, color="orange", linewidth=1.5)
ax.plot(X_f, Y_f, color="orange", linewidth=1.5, label='Final orbit')
ax.grid(True)
ax.legend()

# Save the 2D plot image
save_path_2D = os.path.join(output_dir, img_name_2D)
plt.savefig(save_path_2D)
print(f'2D Plot of full maneuver saved in {save_path_2D}')