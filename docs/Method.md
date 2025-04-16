# Methodology
In our project, we propose to use a vision based system to develop perception algorithms for DSO. We introduce the different hardware and software aspects of the proposed perception algorithm.
## Sensors
There are many different types of cameras available in the market that can be used for our project. We mainly focus on Monocular cameras and stereo cameras.
### Monocular camera
Monocular cameras use a single sensor to percieve the environment around them. They are widely used in robotic applications because they are cheap and can provide a rich data about the environment. Monocular cameras produce a 2D representation of the 3D world and because of this, important depth information about the environment can be lost. There are however, algorithms to estimate the depth information from monocular camera such as: ==structure from motion, depth from defocus== etc. These methods can however be less effective than stereo cameras.

The image formation in a monocular camera can be modelled by the perspective projection model which is given by:
\\[\mathbf{p} = \begin{bmatrix}f_x & 0 & c_x\\ 0 & f_y & c_y\\ 0 &0& 1\end{bmatrix}\\]

### Stereo camera
Stereo cameras make use of multiple sensors to measure the depth of different objects with respect to the camera. 
