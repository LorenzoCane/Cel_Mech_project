



import numpy as np
import matplotlib.pyplot as plt


#****************************************************************************************************
#Conversion functions

def car2kep(r, v, mu):
  """
  car2kep: Conversion from Cartesian coordinates to Keplerian elements.
  N.B np.clip() avoid numerical erros in the computation of np.arccos() to be between -1 and 1

  INPUT:
    r -> Position vector [km]
    v -> Velocity vector [km/s]
    mu -> Gravitational parameter [km^3/s^2]

  OUTPUT:
    a -> Semi-major axis [km]
    e -> Eccentricity [-]
    i -> Inclination [rad]
    OM -> RAAN [rad]
    om -> Pericenter anomaly [rad]
    theta -> True anomaly [rad]
  """

  # Position and velocity norm
  norm_r = np.linalg.norm(r)  # norm of r
  norm_v = np.linalg.norm(v)  # norm of v

  # Angular momentum [km^2/s]
  h = np.cross(r, v)  # Angular momentum vector
  norm_h = np.linalg.norm(h)

  # Compute the keplerian elements

  # INCLINATION
  i = np.arccos(np.clip(h[2] / norm_h, -1.0, 1.0))  # [rad]

  # ECCENTRICITY AND ECCENTRICITY VECTOR:
  e_vec = (1 / mu) * ((norm_v**2 - mu / norm_r) * r - np.dot(r, v) * v)  # [-]
  e = np.linalg.norm(e_vec)  # [-]

  # SEMI-MAJOR AXIS
  E_mech = 1/2 * norm_v**2 - mu / norm_r  # [km^2/s^2]
  a = -mu / (2 * E_mech)  # [km]

  # RIGHT ASCENSION ASCENDING NODE (RAAN)
  # Line of nodes
  N = np.cross(np.array([0, 0, 1]), h)
  norm_N = np.linalg.norm(N)

  # RAAN
  if N[1] >= 0:
      OM = np.arccos(np.clip(N[0] / norm_N, -1.0, 1.0))  # [rad]
  else:
      OM = 2 * np.pi - np.arccos(np.clip(N[0] / norm_N, -1.0, 1.0))  # [rad]

  # PERICENTER ANOMALY
  if e_vec[2] >= 0:
      om = np.arccos(np.clip(np.dot(e_vec, N) / (e * norm_N), -1.0, 1.0))  # [rad]
  else:
      om = 2 * np.pi - np.arccos(np.clip(np.dot(e_vec, N) / (e * norm_N), -1.0, 1.0))  # [rad]

  # TRUE ANOMALY
  # Radial velocity
  v_r = np.dot(r, v) / norm_r

  # True anomaly
  if v_r >= 0:
      theta = np.arccos(np.clip(np.dot(r, e_vec) / (norm_r * e), -1.0, 1.0))  # [rad]
  else:
      theta = 2 * np.pi - np.arccos(np.clip(np.dot(r, e_vec) / (norm_r * e), -1.0, 1.0))  # [rad]

  return a, e, i, OM, om, theta

#-----------------------------------------------------------------------------------------------------------

def kep2car(a, e, i, OM, om, theta, mu, PF = 0):
  """
  kep2car: Conversion from keplerian elements to cartesian coordinates

  INPUT:
    a -> Semi-major axis [km]
    e -> Eccentricity [-]
    i -> Inclination [rad]
    OM -> RAAN [rad]
    om -> Pericenter anomaly [rad]
    theta -> True anomaly [rad]
    mu -> Gravitational parameter [km^3/s^2]
    PF (Optional) -> If PF == 0 (Default) output in ECI coordinates,
                     If PF == 1 output in PF coordinates

  OUTPUT:
    PF = 0 (Default):
    r -> Position vector [km] ECI reference frame
    v -> Velocity vector [km/s] ECI reference frame
    PF = 1:
    r -> Position vector [km] PF reference frame
    v -> Velocity vector [km/s] PF reference frame
  """

  # Semi-latus rectum
  p = a * (1-e**2)
  # Radius
  r = p / (1 + e * np.cos(theta))

  # Radius in PF reference frame
  r_PF = r * np.array([np.cos(theta), np.sin(theta), 0])

  # Velocity in PF reference frame
  v_PF = np.sqrt(mu/p) * np.array([-np.sin(theta), e + np.cos(theta), 0])

  # Rotate r_PF and v_PF from PF to ECI reference frame
  # Define the rotation Matrices
  R3_om = np.array([[np.cos(om), np.sin(om), 0], [-np.sin(om), np.cos(om), 0], [0, 0, 1]])
  R1_i = np.array([[1, 0, 0], [0, np.cos(i), np.sin(i)], [0, -np.sin(i), np.cos(i)]])
  R3_OM = np.array([[np.cos(OM), np.sin(OM), 0], [-np.sin(OM), np.cos(OM), 0], [0, 0, 1]])

  # Transpose the rotation matrices
  R3_om_T = R3_om.T
  R1_i_T = R1_i.T
  R3_OM_T = R3_OM.T

  # Radius in ECI reference frame
  r_ECI = np.dot(R3_OM_T, np.dot(R1_i_T, np.dot(R3_om_T, r_PF)))

  # Velocity in ECI reference frame
  v_ECI = np.dot(R3_OM_T, np.dot(R1_i_T, np.dot(R3_om_T, v_PF)))

  # Choose desired output
  if PF == 0:
    return r_ECI, v_ECI
  elif PF == 1:
    return r_PF, v_PF
  
#****************************************************************************************************
#Plot functions

def plotOrbit(kepEl, mu, deltaTh=2*np.pi, stepTh=np.pi/180, theta_mark=False, ax=None, mark_color = "red"):
  """
  plotOrbit: Plot the arc length deltaTh of the orbit described by kepEl

  INPUT:
    kepEl -> Orbital elements vector [a, e, i, OM, om, theta]
    mu -> Gravitational constant
    deltaTh -> arc length [rad]
    stepTh -> arc step [rad]
    theta_mark -> (Default = False) if True, place a mark at theta = kepEl[5]
    mark_color -> (Default = "red") color of the mark
    ax -> (optional) matplotlib axis to plot on
  OUTPUT:
    X -> X position [km]
    Y -> Y position [km]
    Z -> Z position [km]
  """

  # Create theta vector (from 0 to deltaTh)
  theta_vector = np.arange(0, deltaTh, stepTh)

  # Initialize output vectors
  X = np.array([])
  Y = np.array([])
  Z = np.array([])

  # Loop over theta vector and compute positions
  for theta in theta_vector:
      r_ECI, _ = kep2car(kepEl[0], kepEl[1], kepEl[2], kepEl[3], kepEl[4], theta, mu)
      X = np.append(X, r_ECI[0])
      Y = np.append(Y, r_ECI[1])
      Z = np.append(Z, r_ECI[2])

  # Setup plot if no axis provided
  if ax is None:
      fig = plt.figure()
      ax = fig.add_subplot(111, projection='3d')

  # Plot the orbit
  ax.plot(X, Y, Z, linewidth=1.5)

  # If theta_mark is True, place a red X at kepEl[5]
  if theta_mark:
      r_mark, _ = kep2car(kepEl[0], kepEl[1], kepEl[2], kepEl[3], kepEl[4], kepEl[5], mu)
      ax.scatter(r_mark[0], r_mark[1], r_mark[2], color=mark_color, marker='x', s=100, label=f'Mark at θ = {np.degrees(kepEl[5]):.1f}°')

  return X, Y, Z

#-----------------------------------------------------------------------------------------------------------

def earth_3D(ax):
  """
  Print a three dimensional sphere with earth dimension

  INPUT:
    ax: 3D axes where the earth will be plotted
  OUTPUT:
    ---
  """

  u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:25j]
  x = 6371 * np.cos(u) * np.sin(v)
  y = 6371 * np.sin(u) * np.sin(v)
  z = 6371 * np.cos(v)
  ax.plot_surface(x, y, z, color='blue', alpha=0.3)