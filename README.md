# FAABP
A multi-agent based simulation code combining a physics engine and logic and communication of a heterogenous swarm
This engine contains everything needed to simulate curvity-driven swarms in 2D boxes.

## What this is

These Python scripts simulate a swarm of particles that self-propel and align depending on curvity. You can run them in either circular or rectangular environments, and compare homogeneous vs heterogeneous setups.

greens.py - sets the "green function" for the interaction (choice of function sets boundary condition, type of interaction, potential etc.)
timepropogation.py - has the runge kutta integrator
ecoSystem.py - the class the contains the ecosystem (coordinates and various other parameters of particles)

circle_homo_phasespace.py — the code to run and set the various parameters for the simulation (Number of particles, size of system, speed, curvity etc), this case for homogeneous FAABPs in a circular confinement.
circle_hetero_phasespace.py — same thing, but heterogeneous
rectangle_homo_phasespace.py — rectangular box, homogeneous
rectangle_hetero_phasespace.py — rectangular box, heterogeneous

circle_homo_vs_hetero.py — analyzes and plots results from both circle runs
rectangle_homo_vs_hetero.py — same but for rectangular runs
