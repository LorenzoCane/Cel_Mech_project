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
output_dir = './output/BiellipticStrategy'
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
# Bielliptic strategy = bielliptic transfer and change of plane at higher distance

#1st maneuvers : Move to higher altitude orbit
out_file = 'first_man_data.csv'

# Change the argument of periapsis in order to have it align along the line of nodes
om_transition_1, _, _ = findLineofNodes(i_i, OM_i, i_f, OM_f) # Argument of periapsis to perform the bielliptic strategy

# Change the argument of pericenter
Delta_om = (om_transition_1 - om_i) % (2. * np.pi)  # Ensure Delta_om is in the range [0, 2*pi]

# Need to perform the maneuver at theta_A = Delta_om/2 or theta_B = np.pi + Delta_om/2
theta_A = Delta_om / 2.  # Maneuver point A [rad]
theta_B = np.pi + (Delta_om / 2.)  # Maneuver point B [rad]

# Compute the time of flight
Delta_t_A = timeOfFlight(a_i, e_i, theta_i, theta_A)
Delta_t_B = timeOfFlight(a_i, e_i, theta_i, theta_B)

# Select the maneuver point
if Delta_t_A < Delta_t_B:
    Delta_v_1, om_final, theta_maneuver_1 = changePeriapsisArg(a_i, e_i, om_i, Delta_om, theta_A)
    Delta_t_1 = Delta_t_A
    theta_change_periapsis1 = theta_A
else:
    Delta_v_1, om_final, theta_maneuver_1 = changePeriapsisArg(a_i, e_i, om_i, Delta_om, theta_B)
    Delta_t_1 = Delta_t_B
    theta_change_periapsis1 = theta_B

#Save data for late
maneuver_data = {
    'cost2_km_s':      Delta_v_1,
    'Delta_t_1_s':     Delta_t_1,
    'theta_before_maneuver_1_rad':  theta_change_periapsis1,
    'om_initial_rad':  om_i,
    'om_transition_1_rad':        om_transition_1,
    'theta_after_maneuver_1_rad':     theta_maneuver_1,  
}
df_out = pd.DataFrame([maneuver_data])
out_path = os.path.join(output_dir, out_file)
df_out.to_csv(out_path, index=False)

print(f"First Maneuver data saved to {out_path}")

#------------------------------------------------------------------------------------
#2nd maneuvers : Perform the bielliptic strategy
"""
Values now a_i, e_i, i_f, OM_f, om_transition_1, theta_maneuver_1"""
out_file = 'second_man_data.csv'

# Wait to arrive at the pericenter
theta_before_bielliptic = 0
Delta_t_pre_bielliptic = timeOfFlight(a_i, e_i, theta_maneuver_1, theta_before_bielliptic)
print(f"Time to reach the pericenter before bielliptic transfer: {Delta_t_pre_bielliptic:.3f} s")

# Choose the optimal r_b to minimize the cost to perform the bielliptic strategy
# Plot Bielliptic cost depending on the choice of r_b

r_b_array = np.linspace(a_f * (1-e_f), 8 * a_f, 500)  # from r_f to 4*r_f (wide range)
total_delta_v_bielliptic = []  # Store total delta-v
total_delta_t_bielliptic = []  # Store total delta-t 

for r_b in r_b_array:
    Delta_v, _, _, _, Delta_t, _, _,  _, _, _ = changeOrbitalPlane_bielliptic(a_i, e_i, om_transition_1, i_i, OM_i, a_f, e_f, i_f, OM_f, r_b)
    total_delta_v_bielliptic.append(Delta_v)
    total_delta_t_bielliptic.append(Delta_t)

total_delta_v_bielliptic = np.array(total_delta_v_bielliptic)
total_delta_t_bielliptic = np.array(total_delta_t_bielliptic)

# Create a single figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1: Delta-v 
axes[0].plot(r_b_array, total_delta_v_bielliptic, label=r'Total $\Delta v$ for Bi-elliptic change of plane', color='blue')
axes[0].set_title('Delta-v for Bi-elliptic change of plane')
axes[0].set_xlabel('Intermediate radius $r_b$ [km]')
axes[0].set_ylabel('Total $\Delta v$ [km/s]')
axes[0].legend()
axes[0].grid(True)

# Subplot 2: Time Cost 
axes[1].plot(r_b_array, total_delta_t_bielliptic, label=r'Total $\Delta t$ for Bi-elliptic change of plane', color='blue')
axes[1].set_title('Time Cost for Bi-elliptic change of plane')
axes[1].set_xlabel('Intermediate radius $r_b$ [km]')
axes[1].set_ylabel('Total $\Delta t$ [s]')
axes[1].legend()
axes[1].grid(True)

# Adjust layout and save the figure
plt.tight_layout()
biellipticstrategy_plot_path = os.path.join(output_dir, 'bielliptic_changeOfPlane.pdf')
plt.savefig(biellipticstrategy_plot_path)
print(f'Bielliptic change of plane plot saved to {biellipticstrategy_plot_path}')

# Find the r_b that minimizes the bielliptic maneuver cost (obvoiusly should be the last one of the vector)
optimal_r_b_index = np.argmin(total_delta_v_bielliptic)
optimal_r_b = r_b_array[optimal_r_b_index]
min_delta_v_bielliptic = total_delta_v_bielliptic[optimal_r_b_index]

# Perform the maneuver with the optimal r_b
Delta_v_2, Delta_v_bielliptic1, Delta_v_bielliptic2, Delta_v_bielliptic3, Delta_t_bielliptic, Delta_t_bielliptic1, Delta_t_bielliptic2, om_transition_2, theta_second_burn, theta_after_bielliptic = changeOrbitalPlane_bielliptic(a_i, e_i, om_transition_1, i_i, OM_i,
                                                                                                    a_f, e_f, i_f, OM_f, optimal_r_b)


# Save the optimal bielliptic change of plane maneuver data
bielliptic_maneuver_data = {
    'optimal_r_b_km': optimal_r_b,
    'Delta_v1_km_s': Delta_v_bielliptic1,
    'Delta_v2_km_s': Delta_v_bielliptic2,
    'Delta_v3_km_s': Delta_v_bielliptic3,
    'Total_Delta_v_km_s': min_delta_v_bielliptic,
    'Delta_t_s': Delta_t_bielliptic,
    'Delta_t_bielliptic1_s': Delta_t_bielliptic1,
    'Delta_t_bielliptic2_s': Delta_t_bielliptic2,
}

bielliptic_data_path = os.path.join(output_dir, 'optimal_bielliptic_change _of_plane_maneuver.csv')
pd.DataFrame([bielliptic_maneuver_data]).to_csv(bielliptic_data_path, index=False)
print(f'Optimal bielliptic change of plane maneuver data saved to {bielliptic_data_path}')


Delta_t_2 = Delta_t_bielliptic + Delta_t_pre_bielliptic  # Total time cost for the bielliptic transfer

# First bielliptic transfer orbit to r_b
r_p1 = a_i * (1 - e_i)
a_t1 = (r_p1 + optimal_r_b) / 2
e_t1 = (optimal_r_b - r_p1) / (optimal_r_b + r_p1)
start_bielliptic_maneuver = theta_before_bielliptic  # Start of the bielliptic maneuver at r_p1
theta_apoapsis = theta_second_burn  # Half orbit to r_b

# Second bielliptic transfer orbit from r_b to final orbit 
rp_f = a_f * (1 - e_f)  # Final periapsis
e_t2 = (optimal_r_b - rp_f)/(optimal_r_b + rp_f) # Eccentricity of the second transfer orbit
a_t2 = (rp_f + optimal_r_b) / 2 # Semi-major axis of the second transfer orbit
end_bielliptic_maneuver = theta_after_bielliptic  # Half orbit from r_b to r_p2

#Save data for orbit details
# Save the bielliptic maneuver data
bielliptic_maneuver_data = {
    'optimal_r_b_km': optimal_r_b,
    'cost2_km_s':Delta_v_2,
    'cost_bielliptic1_km_s': Delta_v_bielliptic1,
    'cost_bielliptic2_km_s': Delta_v_bielliptic2,
    'cost_bielliptic3_km_s': Delta_v_bielliptic3,
    'Total_Delta_t_km_s': Delta_t_2,
    'Delta_t_before_bielliptic_s': Delta_t_pre_bielliptic,
    'Delta_t_bielliptic1_s': Delta_t_bielliptic1,
    'Delta_t_bielliptic2_s': Delta_t_bielliptic2,
    'startbielliptic_maneuver_rad': start_bielliptic_maneuver,
    'a_t1_km': a_t1,
    'e_t1': e_t1,
    'theta_apoapsis_rad': theta_apoapsis,
    'a_t2_km': a_t2,
    'e_t2': e_t2,
    'om_transition_2_rad': om_transition_2,
    'Om_f': OM_f, 
    'i_f': i_f, 
    'theta_second_burn_rad': theta_second_burn,
    'a_f': a_f, 
    'e_f': e_f,
    'om_transition_2_rad': om_transition_2,
    'Om_f_rad': OM_f,
    'i_f_rad': i_f,
}

df_out = pd.DataFrame([bielliptic_maneuver_data])
out_path = os.path.join(output_dir, out_file)
df_out.to_csv(out_path, index=False)

print(f"Second Maneuver data saved to {out_path}")

#------------------------------------------------------------------------------------
# 3rd maneuvers : Change periapsis argument to reach the final orbit
out_file = 'third_man_data.csv'

# Change the argument of pericenter
Delta_om = (om_f - om_transition_2) % (2. * np.pi)  # Ensure Delta_om is in the range [0, 2*pi]

# Need to perform the maneuver at theta_A2 = Delta_om/2 or theta_B2 = np.pi + Delta_om/2
theta_A = Delta_om / 2.  # Maneuver point A [rad]
theta_B = np.pi + (Delta_om / 2.)  # Maneuver point B [rad]

# Compute the time of flight
Delta_t_A = timeOfFlight(a_f, e_f, theta_after_bielliptic, theta_A)
Delta_t_B = timeOfFlight(a_f, e_f, theta_after_bielliptic, theta_B)

# Select the maneuver point
if Delta_t_A < Delta_t_B:
    Delta_v_3, om_f, theta_maneuver_2 = changePeriapsisArg(a_f, e_f, om_transition_2, Delta_om, theta_A)
    Delta_t_3 = Delta_t_A
    theta_change_periapsis2 = theta_A
else:
    Delta_v_3, om_f, theta_maneuver_2 = changePeriapsisArg(a_f, e_f, om_transition_2, Delta_om, theta_B)
    Delta_t_3 = Delta_t_B
    theta_change_periapsis2 = theta_B

# Wait until it reaches theta_f
Delta_t_4 = timeOfFlight(a_f, e_f, theta_maneuver_2, theta_f) # Time to reach the final position [s]

#Save data for late
maneuver_data = {
    'cost3_km_s':      Delta_v_3,
    'Delta_t_3_s':     Delta_t_3,
    'theta_before_maneuver_2_rad':  theta_change_periapsis2,
    'om_transition_2_rad': om_transition_2,
    'om_final':        om_f,
    'theta_after_maneuver_2_rad':  theta_maneuver_2,  
    'Delta_t_4_s':     Delta_t_4,
    'theta_final_rad': theta_f
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
print(f"Bielliptic change of plane maneuver: {Delta_v_2:.3f} km/s")
print(f"Third Maneuver: {Delta_v_3:.3f} km/s")
print(f"Total cost: {Delta_v_1 + Delta_v_2 + Delta_v_3:.3f} km/s")

print("\nTotal time cost Maneuver:")
print(f"First maneuver: {Delta_t_1:.3f} s")
print(f"Bielliptic change of plane maneuver: {Delta_t_2:.3f} s")
print(f"Third Maneuver: {Delta_t_3:.3f} s")
print(f"Reach the final position: {Delta_t_4:.3f} s")
print(f"Total time cost: {Delta_t_1 + Delta_t_2 + Delta_t_3 + Delta_t_4:.3f} s")

# collect summury data as a dict
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
img_name = 'Bielliptic_strategy_maneuver.pdf'

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([1, 1, 1])
ax.set_xlabel('X [km]', fontsize=12)
ax.set_ylabel('Y [km]', fontsize=12)
ax.set_zlabel('Z [km]', fontsize=12)
ax.set_title(plot_3d_title, fontsize=14)

ax.set_xlim(-9e4, 9e4)
ax.set_ylim(-9e4, 9e4)
ax.set_zlim(-9e4, 9e4)
'''
ticks = np.linspace(-3.5e4, 3.5e4, 6)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_zticks(ticks)
'''
earth_3D(ax)

# Plot initial orbit (before maneuver)
initial_orbit = np.array([a_i, e_i, i_i, OM_i, om_i, theta_i])
X_i, Y_i, Z_i = plotOrbit(initial_orbit, deltaTh=(theta_change_periapsis1 - theta_i) % (2 * np.pi),
                          theta_mark=True, ax=ax, mark_color="red", line_color="blue",
                          mark_label="Initial position")

# Plot periapsis argument change (first maneuver)
orbit_after_periapsis_change = np.array([a_i, e_i, i_i, OM_i, om_transition_1, theta_maneuver_1])
X_periapsis, Y_periapsis, Z_periapsis = plotOrbit(orbit_after_periapsis_change, deltaTh=(theta_before_bielliptic - theta_maneuver_1) % (2 * np.pi),
                                                  theta_mark=True, ax=ax, mark_color="gold", line_color="limegreen", mark_type=".",
                                                  mark_label="")

# Plot first bielliptic transfer (to r_b, with plane change at apoapsis)
bielliptic_1 = np.array([a_t1, e_t1, i_i, OM_i, om_transition_1, start_bielliptic_maneuver])
X_bielliptic1, Y_bielliptic1, Z_bielliptic1 = plotOrbit(bielliptic_1, deltaTh=(theta_apoapsis - start_bielliptic_maneuver) % (2 * np.pi),  # Half orbit to r_b
                                                        theta_mark=True, ax=ax, mark_color="gold", line_color="magenta", mark_type=".",
                                                        mark_label="")

# Plot second bielliptic transfer (from r_b to final orbit, with new inclination and RAAN)
bielliptic_2 = np.array([a_t2, e_t2, i_f, OM_f, om_transition_2, theta_second_burn])
X_bielliptic2, Y_bielliptic2, Z_bielliptic2 = plotOrbit(bielliptic_2, deltaTh=(end_bielliptic_maneuver - theta_second_burn) % (2 * np.pi),  # Half orbit from r_b to r_p2
                                                    theta_mark=True, ax=ax, mark_color="gold", line_color="darkviolet", mark_type=".",
                                                    mark_label="")

# Plot orbit after bielliptic transfer, before final periapsis argument change
orbit_before_final_periapsis = np.array([a_f, e_f, i_f, OM_f, om_transition_2, theta_after_bielliptic])
X_before_final, Y_before_final, Z_before_final = plotOrbit(orbit_before_final_periapsis, deltaTh=(theta_change_periapsis2 - theta_after_bielliptic) % (2 * np.pi),
                                                          theta_mark=True, ax=ax, mark_color="gold", line_color="aqua", mark_type=".",
                                                          mark_label="")

# Plot final orbit (after last periapsis argument change)
final_orbit = np.array([a_f, e_f, i_f, OM_f, om_f, theta_maneuver_2])
X_f, Y_f, Z_f = plotOrbit(final_orbit, deltaTh=(theta_f - theta_maneuver_2) % (2 * np.pi),
                          theta_mark=True, ax=ax, mark_color="gold", line_color="orange", mark_type=".",
                          mark_label="")

# Plot final position
final_poisition = np.array([a_f, e_f, i_f, OM_f, om_f, theta_f])
X_final, Y_final, Z_final = plotOrbit(final_poisition, deltaTh=0,
                                     theta_mark=True, ax=ax, mark_color="green", line_color="orange",
                                     mark_label="Final position")

# Add all lines and labels to the 3D plot
ax.plot(X_i, Y_i, Z_i, color='blue', linewidth=1.5, label='Initial orbit')
ax.plot(X_periapsis, Y_periapsis, Z_periapsis, color="limegreen", linewidth=1.5, label='Periapsis argument change')
ax.plot(X_bielliptic1, Y_bielliptic1, Z_bielliptic1, color="magenta", linewidth=1.5, label='Bielliptic transfer (1st)')
ax.plot(X_bielliptic2, Y_bielliptic2, Z_bielliptic2, color="darkviolet", linewidth=1.5, label='Bielliptic transfer (2nd)')
ax.plot(X_before_final, Y_before_final, Z_before_final, color="aqua", linewidth=1.5, label='Before final periapsis change')
ax.plot(X_f, Y_f, Z_f, color="orange", linewidth=1.5, label='Final orbit')
ax.plot(X_final, Y_final, Z_final, color="green", linewidth=1.5, label='')
ax.legend()
plt.tight_layout()

# Save the 3D plot image
save_path = os.path.join(output_dir, img_name)
plt.savefig(save_path)
print(f'3D Plot of full maneuver saved in {save_path}')


# -----------------------------
# Plot complete Maneuver in 2D
img_name_2D = 'Bielliptic_strategy_maneuver_2D.pdf'

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlabel('X [km]', fontsize=12)
ax.set_ylabel('Y [km]', fontsize=12)
ax.set_title(plot_2d_title, fontsize=14)
ax.set_aspect('equal')
'''
ticks = [-1.5e4, -1e4, -0.5e4, 0, 0.5e4, 1e4, 1.5e4, 2e4]
ax.set_xticks(ticks)
ax.set_yticks(ticks)
'''
ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
earth_2D(ax, color='blue', marker='o', size=200)

# Plot initial orbit (before maneuver)
X_i, Y_i = plotOrbit2D(initial_orbit, deltaTh=(theta_change_periapsis1 - theta_i) % (2 * np.pi),
                       theta_mark=True, ax=ax, mark_color="red", line_color="blue",
                       mark_label="Initial position")

# Plot periapsis argument change (first maneuver)
X_periapsis, Y_periapsis = plotOrbit2D(orbit_after_periapsis_change, deltaTh=(theta_before_bielliptic - theta_maneuver_1) % (2 * np.pi),
                                       theta_mark=True, ax=ax, mark_color="gold", line_color="limegreen", mark_type=".",
                                       mark_label="")

# Plot first bielliptic transfer (to r_b, with plane change at apoapsis)
X_bielliptic1, Y_bielliptic1 = plotOrbit2D(bielliptic_1, deltaTh=np.pi, stepTh=np.pi/2000,
                                           theta_mark=True, ax=ax, mark_color="gold", line_color="magenta", mark_type=".",
                                           mark_label="")

# Plot second bielliptic transfer (from r_b to final orbit, with new inclination and RAAN)
X_bielliptic2, Y_bielliptic2 = plotOrbit2D(bielliptic_2, deltaTh=(end_bielliptic_maneuver - theta_second_burn) % (2 * np.pi), stepTh=np.pi/5000,
                                           theta_mark=True, ax=ax, mark_color="gold", line_color="darkviolet", mark_type=".",
                                           mark_label="")

# Plot orbit after bielliptic transfer, before final periapsis argument change
X_before_final, Y_before_final = plotOrbit2D(orbit_before_final_periapsis, deltaTh=(theta_change_periapsis2 - theta_after_bielliptic) % (2 * np.pi),
                                             theta_mark=True, ax=ax, mark_color="gold", line_color="aqua", mark_type=".",
                                             mark_label="")

# Plot final orbit (after last periapsis argument change)
X_f, Y_f = plotOrbit2D(final_orbit, deltaTh=(theta_f - theta_maneuver_2) % (2 * np.pi),
                       theta_mark=True, ax=ax, mark_color="gold", line_color="orange", mark_type=".",
                       mark_label="")

# Plot final position
X_final, Y_final = plotOrbit2D(final_poisition, deltaTh=0,
                               theta_mark=True, ax=ax, mark_color="green", line_color="orange",
                               mark_label="Final position")

# Add all lines and labels to the 2D plot
ax.plot(X_i, Y_i, color='blue', linewidth=1.5, label='Initial orbit')
ax.plot(X_periapsis, Y_periapsis, color="limegreen", linewidth=1.5, label='Periapsis argument change')
ax.plot(X_bielliptic1, Y_bielliptic1, color="magenta", linewidth=1.5, label='Bielliptic transfer (1st)')
ax.plot(X_bielliptic2, Y_bielliptic2, color="darkviolet", linewidth=1.5, label='Bielliptic transfer (2nd)')
ax.plot(X_before_final, Y_before_final, color="aqua", linewidth=1.5, label='Before final periapsis change')
ax.plot(X_f, Y_f, color="orange", linewidth=1.5, label='Final orbit')
ax.plot(X_final, Y_final, color="orange", linewidth=1.5, label='')
ax.legend()
plt.tight_layout()

# Save the 2D plot image
save_path_2D = os.path.join(output_dir, img_name_2D)
plt.savefig(save_path_2D)
print(f'2D Plot of full maneuver saved in {save_path_2D}')
