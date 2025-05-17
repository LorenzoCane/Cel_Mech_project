

import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import pandas as pd

#****************************************************************************************************
def timeOfFlight(a, e, theta_i, theta_f, mu = 398600.433):
  """
  Compute the time of flight between two different true anomaly.

  INPUT:
    a, e -> orbital parameters
    theta_i -> initial true anomaly
    theta_f -> final true anomaly

  OUTPUT:
    Delta_t -> Time of flight between theta_i and theta_f

  PARAMETER:
    mu -> Earth planetary constant [km^3/s^2]
  """

  # Compute the eccentric anomaly at theta_i
  if np.isclose(theta_i, np.pi , atol=1e-16):
    E_i = np.pi # Eccentric anomaly when theta_i = np.pi [rad]
  else:
    E_i = 2 * np.arctan(np.sqrt((1 - e)/(1 + e)) * np.tan(theta_i/2)) % (2 * np.pi) # Eccentric anomaly [rad]

  # Compute the eccentric anomaly at theta_f
  if np.isclose(theta_f, np.pi , atol=1e-16):
    E_f = np.pi # Eccentric anomaly when theta_f = np.pi [rad]
  else:
    E_f = 2 * np.arctan(np.sqrt((1 - e)/(1 + e)) * np.tan(theta_f/2)) % (2 * np.pi) # Eccentric anomaly [rad]

  sinE_i = np.sqrt(1 - e**2) * np.sin(E_i) / (1 + e * np.cos(theta_i)) # sin(E) at theta_i
  sinE_f = np.sqrt(1 - e**2) * np.sin(E_f) / (1 + e * np.cos(theta_i)) # sin(E) at theta_f

  Delta_M = (E_f - E_i) - e * (sinE_f - sinE_i) # Kepler's equation
  T = 2 * np.pi * np.sqrt(a**3 / mu) # orbital period

  # Compute the time of flight
  if theta_f >= theta_i:
    Delta_t = Delta_M / np.sqrt(mu/a ** 3) # Time of flight [s]

  else:
    Delta_t = Delta_M / np.sqrt(mu/a ** 3) + T # Time of flight [s]

  return Delta_t

#****************************************************************************************************
def changeOrbitShape(a_i, e_i, om_i, a_f, e_f, om_f, mu = 398600.433):
  """
  Perform an Hohmann transfer between with two impulsive velocity changes for orbit with aligned or anti aligned pericenters
  Choose the most efficient transfer cost wise in each case

  INPUT:
    a_i, e_i, om_1 -> orbital parameters of initial orbit
    a_f, e_f, om_f -> orbital parameters of target orbit

  OUTPUT:
    Delta_v1 -> Delta v to perform the first maneuver (N.B. with sign)
    Delta_v2 -> Delta v to perform the second maneuver (N.B. with sign)
    Delta_t -> time to perform the maneuver [s]
    theta_f -> true anomaly in the target orbit [rad]

  PARAMETER:
    mu -> Earth planetary constant [km^3/s^2]
  """

  rp_i = a_i * (1 - e_i) # Pericenter initial orbit
  ra_i = a_i * (1 + e_i) # Apocente initial orbit
  rp_f = a_f * (1 - e_f) # Pericenter target orbit
  ra_f = a_f * (1 + e_f) # Apocenter of target orbit

  # Check if orbits are coaxial with aligned pericenters
  if np.isclose(np.abs(om_i - om_f), 0, atol=1e-16): # Avoid numerical errors

    # == Pericenter initial orbit -> Apocenter target orbit ==
    a_T_1 = (rp_i + ra_f)/2 # Semi-major axis of the transfer orbit
    Delta_v1_1 = (
       np.sqrt(2 * mu * (1 / rp_i - 1 / (2 * a_T_1))) -
       np.sqrt(2 * mu * (1 / rp_i - 1 / (2 * a_i)))
    ) # First maneuver cost
    Delta_v2_1 = (
       np.sqrt(2 * mu * (1 / ra_f - 1 / (2 * a_f))) -
       np.sqrt(2 * mu * (1 / ra_f - 1 / (2 * a_T_1)))
    ) # Second maneuver cost
    Delta_t_1 = np.pi * np.sqrt(a_T_1**3 /mu) # Time cost of the Maneuver [s]
    theta_f_1 = np.pi # True anomaly on final orbit [rad]


    # == Apocenter inital orbit -> Pericenter target orbit ==
    a_T_2 = (rp_f + ra_i)/2 # Semi-major axis of the transfer orbit
    Delta_v1_2 = (
        np.sqrt(2 * mu * (1 / ra_i - 1 / (2 * a_T_2))) -
        np.sqrt(2 * mu * (1 / ra_i - 1 / (2 * a_i)))
    )
    Delta_v2_2 = (
        np.sqrt(2 * mu * (1 / rp_f - 1 / (2 * a_f))) -
        np.sqrt(2 * mu * (1 / rp_f - 1 / (2 * a_T_2)))
    )
    Delta_t_2 = np.pi * np.sqrt(a_T_2**3 /mu) # Time cost of the Maneuver [s]
    theta_f_2 = 0 # True anomaly on final orbit [rad]

  # Choose the most cost-convenient transfer for aligned orbits
    if (abs(Delta_v1_1) + abs(Delta_v2_1)) < (abs(Delta_v1_2) + abs(Delta_v2_2)):
      return Delta_v1_1, Delta_v2_1, Delta_t_1, theta_f_1
    else:
      return Delta_v1_2, Delta_v2_2, Delta_t_2, theta_f_2

  else:
    # Coaxial orbit with anti-aligned pericenters
    if np.isclose(np.abs(om_i - om_f), np.pi, atol=1e-16): # Avoid numerical errors

      # == Pericenter initial orbit -> Pericenter target orbit ==
      a_T_1 = (rp_i + rp_f)/2 # Semi-major axis of the transfer orbit
      Delta_v1_1 = (
          np.sqrt(2 * mu * (1 / rp_i - 1 / (2 * a_T_1))) -
          np.sqrt(2 * mu * (1 / rp_i - 1 / (2 * a_i)))
      )
      Delta_v2_1 = (
          np.sqrt(2 * mu * (1 / rp_f - 1 / (2 * a_f))) -
          np.sqrt(2 * mu * (1 / rp_f - 1 / (2 * a_T_1)))
      )
      Delta_t_1 = np.pi * np.sqrt(a_T_1**3 /mu) # Time cost of the Maneuver
      theta_f_1 = 0 # True anomaly on final orbit [rad]

      # == Apocenter initial orbit -> Apocenter target orbit ==
      a_T_2 = (ra_i + ra_f)/2 # Semi-major axis of the transfer orbit
      Delta_v1_2 = (
          np.sqrt(2 * mu * (1 / ra_i - 1 / (2 * a_T_2))) -
          np.sqrt(2 * mu * (1 / ra_i - 1 / (2 * a_i)))
      )
      Delta_v2_2 = (
          np.sqrt(2 * mu * (1 / ra_f - 1 / (2 * a_f))) -
          np.sqrt(2 * mu * (1 / ra_f - 1 / (2 * a_T_2)))
      )
      Delta_t_2 = np.pi * np.sqrt(a_T_2**3 /mu) # Time cost of the Maneuver
      theta_f_2 = np.pi # True anomaly on final orbit [rad]

      # Choose the most cost-convenient transfer for anti-aligned orbit
      if (abs(Delta_v1_1) + abs(Delta_v2_1)) < (abs(Delta_v1_2) + abs(Delta_v2_2)):
        return Delta_v1_1, Delta_v2_1, Delta_t_1, theta_f_1
      else:
        return Delta_v1_2, Delta_v2_2, Delta_t_2, theta_f_2
      
#****************************************************************************************************
def changeOrbitShape_bielliptic(a_i, e_i, om_i, a_f, e_f, om_f, r_b, mu = 398600.433):
  """
  Perform a bi-elliptic transfer between with three impulsive velocity changes

  INPUT:
    a_i, e_i, om_1 -> orbital parameters of initial orbit
    a_f, e_f, om_f -> orbital parameters of target orbit
    r_b -> Tangency position of two elliptic transfer orbits (grater than r_f)

  OUTPUT:
    Delta_v1 -> Delta v to perform the first maneuver (N.B. with sign)
    Delta_v2 -> Delta v to perform the second maneuver (N.B. with sign)
    Delta_v3 -> Delta v to perform the third maneuver (N.B. with sign)
    Delta_t -> time to perform the maneuver [s]
    theta_f -> true anomaly in the target orbit [rad]

  PARAMETER:
    mu -> Earth planetary constant [km^3/s^2]
  """

  rp_i = a_i * (1 - e_i) # Pericenter initial orbit
  ra_i = a_i * (1 + e_i) # Apocenter initial orbit
  rp_f = a_f * (1 - e_f) # Pericenter target orbit
  ra_f = a_f * (1 + e_f) # Apocenter of target orbit

  a_T_1 = (rp_i + r_b)/2 # Semi-major axis of the first transfer orbit
  a_T_2 = (r_b + rp_f)/2 # Semi-major axis of the second transfer orbit

  Delta_v1 = (
      np.sqrt(2 * mu * (1 / rp_i - 1 / (2 * a_T_1))) -
      np.sqrt(2 * mu * (1 / rp_i - 1/ (2 * a_i)))
  )
  Delta_v2 = (
      np.sqrt(2 * mu * (1 / r_b - 1 / (2 * a_T_2))) -
      np.sqrt(2 * mu * (1 / r_b - 1 / (2 * a_T_1)))
  )
  Delta_v3 = (
      np.sqrt(2 * mu * (1 / rp_f - 1 / (2 * a_f))) -
      np.sqrt(2 * mu * (1 / rp_f - 1 / (2 * a_T_2)))
  )  

  Delta_t = np.pi * np.sqrt(a_T_1**3 /mu) + np.pi * np.sqrt(a_T_2 ** 3/mu)   # Time cost of the Maneuver [s]
  theta_f = 0 # True anomaly on final orbit [rad]

  return Delta_v1, Delta_v2, Delta_v3, Delta_t, theta_f

#****************************************************************************************************
def changePeriapsisArg(a_i, e_i, om_i, Delta_om, theta_0, mu = 398600.433):
  """
  Perform a change of periapsi arg by an assigned Delta_om -> Modified to automatically chose the closest theta_0?

  INPUT:
    a_i, e_i, om_1 -> initial orbital parameters
    Delta_om -> desired change in pericenter anomaly
    theta_0 -> Initial true anomaly (Choose between Delta_om/2 and np.pi + Delta_om/2)

  OUTPUT:
    Delta_v -> Delta v to perform the maneuver (N.B. with sign)
    om_f -> final pericenter anomaly [rad]
    theta_f -> final true anomaly [rad]

  PARAMETER:
    mu -> Earth planetary constant [km^3/s^2]
  """

  Delta_v = 2 * np.sqrt(mu / (a_i * (1 - e_i **2))) * e_i * np.sin(Delta_om/2) # Delta_v to perform the maneuver (independent of the position)
  om_f = (om_i + Delta_om) % (2 * np.pi) # Final pericenter anomaly between o and 2.np.pi [rad]

  # Case a: theta_0 = Delta_om/2
  if theta_0 == Delta_om/2:
    theta_f = 2 * np.pi - Delta_om/2 # New true anomaly [rad]
  else:
    # Case b: theta_0 = np.pi + Delta_om/2
    if theta_0 == np.pi + Delta_om/2:
      theta_f = np.pi - Delta_om/2 # New true anomaly [rad]
    else:
      # Print error message
      print(f"ERRORE: 'theta_0' deve essere uguale a Delta_om / 2 ({Delta_om/2:.4f}) oppure π + Delta_om / 2 ({Delta_om/2:.4f}).")

  return Delta_v, om_f, theta_f

#****************************************************************************************************
def changeOrbitalPlane(a_i, e_i, i_i, Om_i, om_i, theta_0, i_f, Om_f, mu = 398600.433):
  """
  Perform a change of orbital plane.

  INPUT:
    a_i, e_i, i_i, Om_i, om_i, theta_0 -> initial orbital parameters:
      a_i: semi-major axis [km]
      e_i: eccentricity
      i_i: inclination [rad]
      Om_i: RAAN [rad]
      om_i: argument of periapsis [rad]
      theta_0: initial true anomaly [rad]
    i_f, Om_f -> desired inclination [rad] and RAAN [rad] for target orbit

  OUTPUT:
    Delta_v -> cost of the maneuver [km/s]
    Delta_t -> time of flight to get to the maneuver point [s]
    om_f -> final argument of periapsis [rad]
    theta_f -> final true anomaly [rad]

  PARAMETER:
    mu -> Earth planetary constant [km^3/s^2]
  """

  # Compute changes in RAAN and inclination
  Delta_Om = Om_f - Om_i  # Change in RAAN [rad]
  Delta_i = i_f - i_i  # Change in inclination [rad]

  # Compute the rotation angle alpha between the orbital planes
  alpha = np.arccos(np.cos(i_i) * np.cos(i_f) + np.sin(i_i) * np.sin(i_f) * np.cos(Delta_Om))

  # Handle the four cases for Delta_Om and Delta_i
  if Delta_Om > 0 and Delta_i >= 0:  # FIRST CASE
      cos_ui = (np.cos(alpha) * np.cos(i_i) - np.cos(i_f)) / (np.sin(alpha) * np.sin(i_i))
      sin_ui = np.sin(Delta_Om) * np.sin(i_f) / np.sin(alpha)

      cos_uf = (np.cos(i_i) - np.cos(alpha) * np.cos(i_f)) / (np.sin(alpha) * np.sin(i_f))
      sin_uf = np.sin(i_i) * np.sin(Delta_Om) / np.sin(alpha)

  elif Delta_Om >= 0 and Delta_i < 0:  # SECOND CASE
      cos_ui = (np.cos(i_f) - np.cos(alpha) * np.cos(i_i)) / (np.sin(alpha) * np.sin(i_i))
      sin_ui = -np.sin(i_f) * np.sin(Delta_Om) / np.sin(alpha)

      cos_uf = (np.cos(alpha) * np.cos(i_f) - np.cos(i_i)) / (np.sin(alpha) * np.sin(i_f))
      sin_uf = -np.sin(i_i) * np.sin(Delta_Om) / np.sin(alpha)

  elif Delta_Om <= 0 and Delta_i > 0:  # THIRD CASE
      cos_ui = (np.cos(alpha) * np.cos(i_i) - np.cos(i_f)) / (np.sin(alpha) * np.sin(i_i))
      sin_ui = np.sin(Delta_Om) * np.sin(i_f) / np.sin(alpha)

      cos_uf = (np.cos(i_i) - np.cos(alpha) * np.cos(i_f)) / (np.sin(alpha) * np.sin(i_f))
      sin_uf = np.sin(i_i) * np.sin(Delta_Om) / np.sin(alpha)

  elif Delta_Om < 0 and Delta_i <= 0:  # FOURTH CASE
      cos_ui = (np.cos(i_f) - np.cos(alpha) * np.cos(i_i)) / (np.sin(alpha) * np.sin(i_i))
      sin_ui = -np.sin(i_f) * np.sin(Delta_Om) / np.sin(alpha)

      cos_uf = (np.cos(alpha) * np.cos(i_f) - np.cos(i_i)) / (np.sin(alpha) * np.sin(i_f))
      sin_uf = -np.sin(i_i) * np.sin(Delta_Om) / np.sin(alpha)

  # Compute the initial and final arguments of latitude
  u_i = np.arctan2(sin_ui, cos_ui) % (2 * np.pi)  # Initial argument of latitude [rad]
  u_f = np.arctan2(sin_uf, cos_uf) % (2 * np.pi)  # Final argument of latitude [rad]

  # Compute the true anomaly at the maneuver point
  if u_i >= om_i:
      theta_i = u_i - om_i
  else:
      theta_i = u_i - om_i + 2 * np.pi

  theta_i = theta_i % (2 * np.pi)  # Ensure theta_i is in the range [0, 2*pi]

  # Compute the final argument of periapsis
  if u_f >= theta_i:
      om_f = u_f - theta_i
  else:
      om_f = u_f - theta_i + 2 * np.pi

  om_f = om_f % (2 * np.pi) # Ensure om_f is in the rane [0, 2*pi]

  # Choose in which nodal point perform the maneuver
  # Compute the maneuver cost at theta_i
  r_theta_i = a_i * (1 - e_i**2) / (1 + e_i * np.cos(theta_i))
  v_theta_i = np.sqrt(mu * (2 / r_theta_i - 1 / a_i))
  Delta_v_theta_i = 2 * v_theta_i * np.sin(alpha / 2)

  # Compute the maneuver cost at the antipodal point (theta_i + pi)
  r_theta_i_antipodal = a_i * (1 - e_i**2) / (1 + e_i * np.cos(np.pi + theta_i))
  v_theta_i_antipodal = np.sqrt(mu * (2 / r_theta_i_antipodal - 1 / a_i))
  Delta_v_theta_i_antipodal = 2 * v_theta_i_antipodal * np.sin(alpha / 2)


  # Choose where to perform the maneuver
  if Delta_v_theta_i < Delta_v_theta_i_antipodal:
      Delta_v = Delta_v_theta_i  # Maneuver cost
      theta_f = theta_i  # New true anomaly [rad]
      Delta_t = timeOfFlight(a_i, e_i, theta_0, theta_i)  # Time cost of the maneuver [s]
  else:
      Delta_v = Delta_v_theta_i_antipodal  # Maneuver cost
      theta_f = np.pi + theta_i  # New true anomaly [rad]
      Delta_t = timeOfFlight(a_i, e_i, theta_0, np.pi + theta_i)  # Time cost of the maneuver [s]

  theta_f = theta_f % (2 * np.pi)  # Ensure theta_f is in the range [0, 2*pi]

  return Delta_v, om_f, theta_f, Delta_t