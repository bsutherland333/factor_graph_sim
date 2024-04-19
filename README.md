# Factor Graph Book Sim

This repo contains a basic python simulator for following along with this [factor graph book](https://www.cs.cmu.edu/~kaess/pub/Dellaert17fnt.pdf).

The model used for this simulator is a 2D robot moving in a 2D world, one that is constrained to moving only forward and backward but able to turn in place and while in motion.

Poses are in inertial frame, where the (0, 0) point of the field is the datum center, and a heading of 0 is along the positive x-axis. Measurements are in body frame, where the datum center is at the robot and a heading of 0 points directly in front of the robot and parallel with its direction of motion.

## Usage

Create a python virtual environment, source the environment, and install the necessary dependencies with this:
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run any of the files in the launch directory. Details on what each launch file does is found in the file itself. Supporting files are in the src directory.
