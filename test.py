
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import pandas as pdiowejfiweoj

from utils import kep2car, car2kep, plotOrbit, earth_3D

#Test results folder
os.makedirs('./test_img/', exist_ok=True)
img_dir = './test_img/'

#****************************************************************************************************
# Test Function car2kep
# Input data
r_in = np.array([10000.0, 20000.0, 10000.0]) # Radial Position [km]
v_in = np.array([-2.5, -2.5, 3.0])  # Velocity [km/s]
mu = 398600.433 # Earth planetary constant [km^3/s^2]

# Keplerian elements
a_out, e_out, i_out, OM_out, om_out, theta_out = car2kep(r_in, v_in, mu)
print(f'a = {a_out} km')
print(f'e = {e_out}')
print(f'i = {i_out} rad')
print(f'OM = {OM_out} rad')
print(f'om = {om_out} rad')
print(f'theta = {theta_out} rad')

print('==========================================================================')

#****************************************************************************************************
# Test function kep2car
# Input data
a_in = 15000 # semi-major axis [km]
e_in = 0.1 # eccentricity [-]
i_in = 15. # inclination [deg]
OM_in = 45.0 # RAAN [deg]
om_in = 30. # pericenter anomaly [deg]
theta_in = 180. # true anomaly [deg]
mu = 398600.433 # Earth planetary constant [km^3/s^2]

# Cartesian coordinates ECI reference frame
r_out_ECI, v_out_ECI = kep2car(a_in, e_in, np.deg2rad(i_in), np.deg2rad(OM_in), np.deg2rad(om_in), np.deg2rad(theta_in), mu)
print(f'r_out_ECI = {r_out_ECI} km')
print(f'v_out_ECI = {v_out_ECI} km/s')

# Cartesian coordinates PF reference frame
r_out_PF, v_out_PF = kep2car(a_in, e_in, i_in, OM_in, om_in, theta_in, mu, PF = 1)
print(f'r_out_PF = {r_out_PF} km')
print(f'v_out_PF = {v_out_PF} km/s')

print('==========================================================================')
# Check results
a_check, e_check, i_check, OM_check, om_check, theta_check = car2kep(r_out_ECI, v_out_ECI, mu)
print('Check:')
print('Input:')
print(f'a_in = {a_in:.2f} km')
print(f'e_in = {e_in:.2f}')
print(f'i_in = {(i_in):.2f} deg')
print(f'OM_in = {(OM_in):.2f} deg')
print(f'om_in = {(om_in):.2f} deg')
print(f'theta_in = {(theta_in):.2f} deg')
print('Output:')
print(f'a_check = {a_check:.2f} km')
print(f'e_check = {e_check:.2f}')
print(f'i_check = {np.rad2deg(i_check):.2f} deg')
print(f'OM_check = {np.rad2deg(OM_check):.2f} deg')
print(f'om_check = {np.rad2deg(om_check):.2f} deg')
print(f'theta_check = {np.rad2deg(theta_check):.2f} deg')

#****************************************************************************************************
#Check plot ficntion and figure setup

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

# Initialize the vector of keplerian elements
kepEl = np.array([a_in, e_in, np.deg2rad(i_in), np.deg2rad(OM_in), np.deg2rad(om_in), np.deg2rad(theta_in)])

# Plot the orbit and mark a the position on the orbit
X, Y, Z = plotOrbit(kepEl, mu, theta_mark=True, ax=ax)

# Add Orbit Label
ax.plot(X, Y, Z, color='orange', linewidth=1.5, label='Satellite orbit')

# Legend and show
ax.legend()
plt.savefig(os.path.join(img_dir, 'plot_Earth_test.pdf'))
print(f'Earth img save as: {os.path.join(img_dir, 'plot_Earth_test.pdf')}')
