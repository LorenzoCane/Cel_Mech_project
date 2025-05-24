# Orbital Maneuvers simulator

A Python-based tool for simulating and visualizing satellite orbital maneuvers using classical orbital mechanics principles.

## Overview

This project provides functionalities to:

* Simulate orbital maneuvers (e.g., Hohmann transfers, inclination changes).
* Visualize orbits in 2D and 3D.
* Calculate delta-v requirements for various maneuvers.
* Save generated data for further analysis.

## Features

* **Orbital Mechanics Utilities**: Functions to convert between Keplerian and Cartesian coordinates.
* **Maneuver Simulations**: Scripts for simulating first, second, and third orbital maneuvers.
* **Visualization**: Plotting tools for 2D and 3D representations of orbits around the Earth.
* **Data Handling**: Utilities for writing simulation data to files.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/LorenzoCane/Cel_Mech_project.git
   cd Cel_Mech_project
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Add Input Data**  
   Add your desired transfer data to `input/orbit_data.csv`.

2. **Prepare Data**  
   Run the following script to process input data:

   ```bash
   python data_writer.py
   ```

3. **Perform Maneuvers**
    Run the maneuver scripts individually, selecting the correct Orbit ID when prompted or configured:
    ```bash 
    python first_maneuver.py
    python second_maneuver.py
    python third_maneuver.py
    ```

4. **Compare Strategies**
Run the main script to perform a final comparison of the results:
    ```bash
    python main.py
    ```

## Academic Context

This project was developed as part of the academic exam for the Celestial Mechanics and Astrodynamics course within the [MPMSSIA Master's Program] (https://www.mpmssia.unito.it/do/home.pl) at the University of Turin.

Further improvements and extensions may be implemented in the future, such as:

    - GUI interface for input selection

    - More advanced transfer modeling

    - Inclusion of perturbation effects

    - Automatic animation generation

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
