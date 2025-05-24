import os
import pandas as pd
import matplotlib.pyplot as plt
from utils import * 

# Directory where store all the plot
output_dir = './output'

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

r_f_ECI, v_f_ECI = kep2car(a_f, e_f, i_f, OM_f, om_f, theta_f)

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
print(f"Final pos: {r_f_ECI} km")
print(f"Final vel: {v_f_ECI} km/s")
print(f"a_f = {a_f} km")
print(f"e_f = {e_f}")
print(f"i_f = {i_f} rad")
print(f"OM_f = {OM_f} rad")
print(f"om_f = {om_f} rad")
print(f"theta_f ={theta_f} rad")
print('=====================================================================================')


# Plot the initial orbit in 3D and 2D
fig, axes = plt.subplots(1, 2, figsize=(16, 8), subplot_kw={'projection': None})
ax_3d = fig.add_subplot(121, projection='3d')  # 3D plot
ax_2d = fig.add_subplot(122)  # 2D plot

# 3D Plot of the initial orbit
ax_3d.set_box_aspect([1, 1, 1])
ax_3d.set_xlabel('X [km]', fontsize=12)
ax_3d.set_ylabel('Y [km]', fontsize=12)
ax_3d.set_zlabel('Z [km]', fontsize=12)
ax_3d.set_title('Initial Orbit (3D)', fontsize=14)
ticks = [-1e4, -0.5e4, 0, 0.5e4, 1e4]
ax_3d.set_xticks(ticks)
ax_3d.set_yticks(ticks)
ax_3d.set_zticks(ticks)
ax_3d.set_xlim(-1.2e4, 1.2e4)
ax_3d.set_ylim(-1.2e4, 1.2e4)
ax_3d.set_zlim(-1.2e4, 1.2e4)
earth_3D(ax_3d)
initial_orbit = np.array([a_i, e_i, i_i, OM_i, om_i, theta_i])
plotOrbit(initial_orbit, theta_mark=True, ax=ax_3d, mark_color="red", line_color="blue")

# 2D Plot of the initial orbit
ax_2d.grid(True)
ax_2d.set_xlabel('X [km]', fontsize=12)
ax_2d.set_ylabel('Y [km]', fontsize=12)
ax_2d.set_title('Initial Orbit (2D)', fontsize=14)
ticks = [-1e4, -0.5e4, 0, 0.5e4, 1e4]
ax_2d.set_xticks(ticks)
ax_2d.set_yticks(ticks)
earth_2D(ax_2d)
plotOrbit2D(initial_orbit, theta_mark=True, ax=ax_2d, mark_color="red", line_color="blue")
ax_2d.set_title('Initial Orbit (2D)', fontsize=14)

# Save the initial orbit figure
initial_orbit_path = os.path.join(output_dir, 'initial_orbit.pdf')
plt.tight_layout()
plt.savefig(initial_orbit_path)
print(f"Initial orbit plot saved to {initial_orbit_path}")
plt.close(fig)

# Plot the final orbit in 3D and 2D
fig, axes = plt.subplots(1, 2, figsize=(16, 8), subplot_kw={'projection': None})
ax_3d = fig.add_subplot(121, projection='3d')  # 3D plot
ax_2d = fig.add_subplot(122)  # 2D plot

# 3D Plot of the final orbit
ax_3d.set_box_aspect([1, 1, 1])
ax_3d.set_xlabel('X [km]', fontsize=12)
ax_3d.set_ylabel('Y [km]', fontsize=12)
ax_3d.set_zlabel('Z [km]', fontsize=12)
ax_3d.set_title('Final Orbit (3D)', fontsize=14)
ax_3d.set_xticks(ticks)
ax_3d.set_yticks(ticks)
ax_3d.set_zticks(ticks)
ax_3d.set_xlim(-1.2e4, 1.2e4)
ax_3d.set_ylim(-1.2e4, 1.2e4)
ax_3d.set_zlim(-1.2e4, 1.2e4)
earth_3D(ax_3d)
final_orbit = np.array([a_f, e_f, i_f, OM_f, om_f, theta_f])
plotOrbit(final_orbit, theta_mark=True, ax=ax_3d, mark_color="green", line_color="orange")

# 2D Plot of the final orbit
ax_2d.grid(True)
ax_2d.set_xlabel('X [km]', fontsize=12)
ax_2d.set_ylabel('Y [km]', fontsize=12)
ax_2d.set_title('Initial Orbit (2D)', fontsize=14)
ticks = [-1e4, -0.5e4, 0, 0.5e4, 1e4]
ax_2d.set_xticks(ticks)
ax_2d.set_yticks(ticks)
earth_2D(ax_2d)
earth_2D(ax_2d)
plotOrbit2D(final_orbit, theta_mark=True, ax=ax_2d, mark_color="green", line_color="orange")
ax_2d.set_title('Final Orbit (2D)', fontsize=14)

# Save the final orbit figure
final_orbit_path = os.path.join(output_dir, 'final_orbit.pdf')
plt.tight_layout()
plt.savefig(final_orbit_path)
print(f"Final orbit plot saved to {final_orbit_path}")
plt.close(fig)


# Comparison of the three strategies

# Define output directories for all strategies
strategy1_dir = './output/StandardStrategy1'
strategy2_dir = './output/StandardStrategy2'
bielliptic_dir = './output/BiellipticStrategy'

# Load summary data for all strategies
strategy1_summary_path = os.path.join(strategy1_dir, 'maneuver_summary_B1.csv')
strategy2_summary_path = os.path.join(strategy2_dir, 'maneuver_summary_B1.csv')
bielliptic_summary_path = os.path.join(bielliptic_dir, 'maneuver_summary_B1.csv')

if not os.path.exists(strategy1_summary_path):
    raise FileNotFoundError(f"Summary file for Strategy 1 not found: {strategy1_summary_path}")
if not os.path.exists(strategy2_summary_path):
    raise FileNotFoundError(f"Summary file for Strategy 2 not found: {strategy2_summary_path}")
if not os.path.exists(bielliptic_summary_path):
    raise FileNotFoundError(f"Summary file for Bielliptic Strategy not found: {bielliptic_summary_path}")

strategy1_data = pd.read_csv(strategy1_summary_path)
strategy2_data = pd.read_csv(strategy2_summary_path)
bielliptic_data = pd.read_csv(bielliptic_summary_path)

# Extract total delta-v and total time cost
strategy1_total_delta_v = strategy1_data['Total_Delta_v_km_s'].iloc[0]
strategy2_total_delta_v = strategy2_data['Total_Delta_v_km_s'].iloc[0]
bielliptic_total_delta_v = bielliptic_data['Total_Delta_v_km_s'].iloc[0]

strategy1_total_time = strategy1_data['Total_Delta_t_s'].iloc[0]
strategy2_total_time = strategy2_data['Total_Delta_t_s'].iloc[0]
bielliptic_total_time = bielliptic_data['Total_Delta_t_s'].iloc[0]

# Create bar plots
os.makedirs(output_dir, exist_ok=True)

# Plot 1: Total Delta-v Comparison
plt.figure(figsize=(8, 6))
strategies = ['Strategy 1', 'Strategy 2', 'Bielliptic']
delta_v_values = [strategy1_total_delta_v, strategy2_total_delta_v, bielliptic_total_delta_v]
plt.bar(strategies, delta_v_values, color=['blue', 'green', 'orange'])
plt.title('Total Delta-v Comparison')
plt.ylabel('Total Delta-v [km/s]')
plt.savefig(os.path.join(output_dir, 'total_delta_v_comparison.png'))
print(f"Total Delta-v comparison plot saved to {os.path.join(output_dir, 'total_delta_v_comparison.png')}")

# Plot 2: Total Time Cost Comparison
plt.figure(figsize=(8, 6))
time_values = [strategy1_total_time, strategy2_total_time, bielliptic_total_time]
plt.bar(strategies, time_values, color=['blue', 'green', 'orange'])
plt.title('Total Time Cost Comparison')
plt.ylabel('Total Time Cost [s]')
plt.savefig(os.path.join(output_dir, 'total_time_cost_comparison.png'))
print(f"Total Time Cost comparison plot saved to {os.path.join(output_dir, 'total_time_cost_comparison.png')}")


#TO DO : 
# Check of final orbit + general check
# Try a strategy with bielliptic + change orbital plane + bielliptic + changeperiapsisArg