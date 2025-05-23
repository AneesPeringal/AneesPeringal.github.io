<h1 align="center">Occlusion-aware segmentation of deformable slender objects</h1>
<h2> Abstract</h2>

We present an algorithm for detecting the shape of deformable slender objects (DSOs), such as cables and ropes, even in the presence of occlusions. DSOs are common in real-world environments, yet robotic manipulation has largely focused on rigid objects due to the complexities involved in perceiving deformable ones. The main challenges arise from the lack of a fixed shape and frequent occlusions, either from the manipulator or due to self-occlusion. To address this, we propose a two-stage method: first, the DSO is segmented from the scene using off-the-shelf segmentation techniques; second, the occluded sections are reconstructed in 3D space. We validate our method by comparing estimated shapes with and without occlusion, achieving an RMSE of 7 mm. This approach enables more natural and robust robotic interaction with deformable linear objects (DLOs), with implications for applications such as cable routing, construction, and surgical knot tying, where shape estimation through occlusions is important.

<figure markdown="span">
    ![Image title](Abstract_picture.png)
    <figcaption style="text-align: center; font-style: italic;">**Figure 1:** Vision based perception of DSO manipulation. During manipulation, there may be different types of occlusions which the proposed segmentation algorithm must be robust to.</figcaption>
</figure>


<h2 align='center'>Introduction</h2>

Deformable objects are all around us and humans are able to naturally interact with them to accomplish various tasks that are not possible with rigid objects. There are several challenges that limit the use of robots for the manipulation of deformable objects. Unlike rigid objects, deformable objects exhibit significant flexibility and elasticity that make it difficult to model them effectively. Especially in the case of active manipulation, the configuration of the object is dynamic and is subject to change throughout the manipulation. Without accurate perception algorithms for deformable objects, robots would struggle to find the current state of the deformable object and therefore cause large errors in manipulation. Moreover, we need perception methods that do not impose additional forces and moments on the deformable objects. In this project, we propose to use a vision based method to perceive a DSO that will be manipulated by a robot. 

There are many different vision based methods that we can adopt to develop algorithms for the perception of DSO in terms of the hardware and the software. The different hardware solutions available to be used are:

### Monocular camera
Monocular cameras use a single sensor to perceive the environment around them. They are widely used in robotic applications because they are cheap and can provide a rich data about the environment. Monocular cameras produce a 2D representation of the 3D world and because of this, important depth information about the environment can be lost. 

<figure markdown="span">
    ![alt text](cam_model_fig41.jpg)
    <figcaption style="text-align: center; font-style: italic;">**Figure 2:** Perspective projection model of a camera [1]</figcaption>
</figure>

The image formation in a monocular camera can be modelled by the perspective projection model which is given by:

$$
\mathbf{p} = \frac{1}{Z}\begin{bmatrix}
f_x & 0 & c_x  \\
 0 & f_y & c_y \\ 
 0 &0& 1
 \end{bmatrix} \mathbf{P}
$$

Where $\mathbf{p}$ is the position of the image on the image plane in the homogenous coordinates, $f_x$, $f_y$, $c_x$ and $c_y$ are the focal lengths and the principle point coordinates respectively. Moreover, $\mathbf{P}$ is the position of the object in the 3D scene. It must be noted that a monocular camera produces a 2D projection of the 3D scene, therefore, it cannot measure directly the depth data that is contained in the scene. There are however, algorithms to estimate the depth information from monocular camera. The two main classes of algorithms are:

 - Triangulation based methods: Ex. structure from motion [2]
 - Depth of field methods: Eg. depth from defocus [3]

A commonly used triangulation based technique is called structure from motion. This involves capturing images from the same camera at two different camera viewpoints, and using certain feature points that are same on both pictures to reconstruct a 3D structure of the scene. 

<figure markdown="span">
    ![Structure from motion](Structure-from-Motion-SfM-process-is-illustrated-The-structure-in-the.png)
    <figcaption style="text-align: center; font-style: italic;">**Figure 3:** By detecting the change of some feature points in the image plane, the motion of the camera can be estimated. And using this, the 3D representation of the scene can be obtained from pictures at different viewpoints.[4]</figcaption>
</figure>


The key steps for estimating the 3d structure from a monocular camera using structure from motion are:

 - Capture multiple images from different viewpoints
 - Detect features such as corners that can be easily distinguished in different images
 - Match the detected features in one image to the other.
 - Estimate the motion of the camera between the two images. This can be done by solving the perspective-n-point problem.
 - The 3D position of the feature points are found by triangulation.

Another class of methods make use of the focus of the image produced to estimate the depth information. A monocular camera has a limited range of positions where it can capture a sharp image. This can be seen from:
<figure markdown="span">
    ![alt text](Depth-defocus-relationship-The-same-object-point-placed-at-different-distances-will-be.png)
    <figcaption style="text-align: center; font-style: italic;">**Figure 4:** Objects at a distance $u$ from the lens creates a sharp image at the image plane. However, other objects will result in a blurred image.[5]</figcaption>
</figure>

USing the blur circle radius $r_b$, we can have an estimate of the depth by the equation: 

\[r_b(z) = \frac{A(|z-z_f|)}{z}\frac{f}{z_f-f}\]

Where $A$ is the aperture size, $f$ is the focal length, $z_f$ is the distance to the focal plane and finally $z$ is the depth of the object. If we know the blur circle radius, we can solve for $z$ to estimate the depth from defocus.

  
### Stereo camera
Stereo cameras use a triangulation method to find the 3D data of the scene. The major difference is that rather than using a single camera to obtain multiple pictures of the scene, stereo cameras make use of multiple cameras that have a known relative transformation to obtain multiple images of the scene at the same time. 

<figure markdown="span">
    ![Stereo camera](Principle-of-stereo-cameras.png)
    <figcaption style="text-align: center; font-style: italic;">**Figure 5:** Using cameras that are a known fixed transformation apart, depth in the scene can be estimated using the difference in the image produced by the two sensors.[6]</figcaption>
</figure>

The projection matrix of the camera that we are using in this project is found to be:

$$\mathbf{K} = \begin{bmatrix} 595.998 & 0 & 320.825 \\
0 & 595.998 &239.252 \\
0 & 0 &1\end{bmatrix} $$


<h2 align='center'> Methodology</h2>
In our project, we propose to use a vision based system to develop perception algorithms for DSO. We introduce the different hardware and software aspects of the proposed perception algorithm.


<figure markdown="span">
    ![Software methodology](Methodology.png)
    <figcaption style="text-align: center; font-style: italic;">**Figure 6:** The steps taken in this methodology</figcaption>
</figure>


## I. Preprocessing
The RGB-D data obtained from the camera is preprocessed to downsample it to a standard size of $640\times 480$. Further the image is filtered with gaussian blurring to smooth out the image for noise.


<figure markdown="span">
    ![Preprocessed_image](example_image.png)
    <figcaption style="text-align: center; font-style: italic;">**Figure 7:** A typical example of an image of the DSO being manipulated</figcaption>
</figure>



## II. Segmentation
The stereo camera will give us dense data about all the pixels in the image. However, in most cases, we only need certain sparse data that pertains to our problem. In this case, we only need RGB-D data of the deformable slender object that we will be manipulating. To achieve this, we use segmentation to find which pixels in the image correspond to the DSO. Image segmentation is the process of splitting an image into different sets of pixels based on some condition. There are different methods of doing image segmentation:

- **Semantic segmentation**: Classifies the pixels based on the meaning of the object. Employs deep learning methods that can learn the pixel based classification problem. Example U-net [7]
- **Region-based segmentation**: Classifies the pixels based on similarities between nearby pixels. The criteria for similarity can be color, texture etc. 

### Semantic segmentation
Semantic segmentation is typically done using deep-learning models. They have the ability to learn the meaning behind the image and therefore successfully classify the pixels that belong to a certain kind of object. The deep learning models that do this have an autoencoder architecture. U-net is a highly successful model that employs skip connections that ensures that finer details of the image can be successfully classified. To train the network for semantic segmentations, we need to have pixel wise labels for each image in the training data. The training is done to minimize the focal loss for each pixel. 

<figure markdown="span">
    ![alt text](u-net-architecture.png)
    <figcaption style="text-align: center; font-style: italic;">**Figure 8:** Deep-learning based semantic segmentation model. [7]</figcaption>
</figure>


Semantic segmentation requires large amounts of data to give accurate results. There are other methods of segmentation that can be used to segment the deformable slender object from the image.

### Region-based segmentation
Segmentation can also be performed based on other aspects of the image. For example, a group of pixels having the same color or texture can be segmented. In this project, we adopt color based segmentation to find the pixels that are associated with the DSO. For color-based segmentation, we convert the RGB image into the HSV color space because it is less dependent on lighting conditions. In our case, the DSO is blue in color, so we set the limits in the HSV Space as:

\\[\mathrm{Lower limit} = [100,150,50]\\]
\\[\mathrm{Upper limit} = [140, 255, 255]\\]

All the pixels that do not belong in this interval are equated to zero. 

<figure markdown="span">
    ![HSV_color_space](hsvcone.gif)
    <figcaption style="text-align: center; font-style: italic;">**Figure 9:** HSV color spaced used for color based segmentation</figcaption>
</figure>

<figure markdown="span">
    ![Segmented_image](segmented_image.png)
    <figcaption style="text-align: center; font-style: italic;">**Figure 10:** Example image after color based segmentation</figcaption>
</figure>

## III. Our method
To obtain the topology of the DSO from the image, we construct the skeleton of the image using Zhang's method [8]. Zhang's Method for skeletonization is an efficient algorithm for thinning a binary image to obtain a skeleton representation of shapes. The method is based on iteratively removing pixels from the boundaries of the objects in the binary image while preserving the topology and structure of the shapes. Zhang’s algorithm works by applying a series of conditional rules that allow the removal of boundary pixels in a way that retains the essential structure of the object. Specifically, it works by iterating through the image and checking each pixel's neighborhood for continuity, and then removing pixels that satisfy the continuity. This process continues until no further pixels can be removed, resulting in a skeleton that represents the object as a thin, one-pixel-wide line.

<figure markdown="span">
    ![Skelotonized](skeleton.png)
    <figcaption style="text-align: center; font-style: italic;">**Figure 11:** 1 pixel width representation of the rope</figcaption>
</figure>

Now that we have a one dimensional representation of the DSO in the image plane, we can find the contours that represent the topology of the DSO. Contours are a sequence of points on the skeleton that are continuous. When occlusions occur, the image of the rope might not be continuous and we may have several disjoint contours. Moreover, when there are self-occlusions, the contour might not follow the actual direction of the DSO an example of this is shown in:

<figure markdown="span">
    ![alt text](Contours.png)
    <figcaption style="text-align: center; font-style: italic;">**Figure 12:** The sequence of points in the contour does not match the actual sequence on the DSO. This should be corrected.</figcaption>
</figure>

To overcome this problem, we design an algorithm that operates on smaller segments of each contour to identify locally linear chains of connected points. For each contours, it traverses the points sequentially and splits it into smaller segments based on the local curvature. The traversal begins by initializing a starting point for the current segment. As points are added, the algorithm checks whether the Euclidean distance between the segment starting point and the current point exceeds a fixed segment length $l_{thresh}$. If this length threshold is reached, the direction of the current segment is calculated using $\theta_i = (x_s - x_i,y_s - y_i)$ where $(x_s,y_s)$ and $(x_i, y_i)$ are the positions of the start of the segment and the current point in the image plane. If the angle between the current and the previous segment is below a predefined threshold $\theta_{thresh}>= \theta_i \cdot \theta_{i-1}$, this segment is added to the chain. Otherwise, a new chain is initialized. Through this method, we obtain a set of chains that are spatially coherent.

<figure markdown="span">
    ![image](chains.png)
    <figcaption style="text-align: center; font-style: italic;">**Figure 13:** The different chains created from the contours of the DSO skeleton.</figcaption>
</figure>

The disjoint chains should now be rejoined to form the sequence of points that form the skeleton of the DSO. Each chain has two ends, so for an image that has $n$ chains, the total number of ends will be $2n$. The number of possible combinations are $\frac{n(n-2)}{2}$. To decide how the combination of chains should be, we use the Hungarian algorithm [9] which requires a cost matrix $\mathbf{C}$ that determines the cost of every possible assignment. The Hungarian algorithm tries to minimize the cost function:

\[\min \sum_{k=1}^{2n} \mathbf{C}(k,\sigma(k))\]

where $\sigma(i)$ is the end of the chain that is assigned to the end at $i$. The cost associated with connecting end $k$ with end $l$ is computed as a combination of the curvature $C_c(k,l)$ and the Euclidean distance $C_e(k,l).

\[C(k,l) = \lambda_e C_e(k,l) +\lambda_c C_c(k,l)\]

\[C_{k,l} = \lambda_e \sqrt{(p_k - p_{l})^2} + \lambda_c \left(\cos^{-1}\left(\frac{(p_k-p_l)\cdot p'_k}{\|(p_k-p_l)\| \|p'_k\|}\right) + \cos^{-1}\left(\frac{(p_k-p_l)\cdot p'_l}{\|(p_k-p_l)\| \|p'_l\|}\right)\right)\]

The Euclidean norm ensures that the two ends are close to each other and the curvature cost will ensure that joining them together will not cause a sharp change in the direction of the rope. We also have a few special cases that must be accounted for such as one end of a chain is not allowed to be joined with itself. To discourage this, we populate the diagonal elements of the cost matrix with large numbers, in this case it is chosen to be $10^5$. The values of the diagonal elements and the weightage for Euclidean and curvature cost is found through trial and error to see which provides the best performance.  The final weightage that we used are $\lambda_e = 0.001$ and $\lambda_c = 1$



<figure markdown="span">
    ![alt text](image-1.png)
    <figcaption style="text-align: center; font-style: italic;">**Figure 14:** The disjoint chains are connected based on the solution of the Hungarian algorithm.</figcaption>
</figure>

  

<figure markdown="span">
    ![type:video](Algorithm_video.mp4){'width: 40%'}
    <figcaption style="text-align: center; font-style: italic;">**Video 1:** Overview of the image processing</figcaption>
</figure>

## 3D reconstruction
Using the depth data at the points where the rope exists, we can reconstruct the shape of the rope in 3D. This is done by first inverting the perspective projection through:

\[\mathbf{P}_x = \frac{(\mathbf{p}_x - c_x)}{f_x} \mathbf{P}_z\]

\[\mathbf{P}_y = \frac{(\mathbf{p}_y - c_y)}{f_y} \mathbf{P}_z\]

The z-coordinate can be obtained directly from the depth image corresponding to $p_x$ and $p_y$. Using this, we can approximate the 3D shape of the DSO.
<div style="text-align: center;">
  <img src="3d_reconstruction.png" alt="alt text" style="width: 70%; display: block; margin: 0 auto;" />
  <figcaption style="font-style: italic; margin-top: 8px;">
    <strong>Figure 15:</strong> 3D reconstruction of the DSO using depth data.
  </figcaption>
</div>

Given an image of the rope taken from an RGB-D camera, we can use the spatial relationship between different parts of the DSO to obtain a coherent 3D representation of the DSO. However, during occlusions, we have missing parts of the DSO and disjoint sections of the DSO in the image. To account for the occlusions, we also make use of some temporal dependencies. 

# Temporal dependencies
The evolution of the DSO under manipulation will follow the dynamics of the DSO. This depends on the material parameters, the external and internal forces on the DSO. The external forces include gravity, manipulation forces and contact with other objects in the environment. On the other hand, the internal forces could arise from the objects elasticity, damping etc. Some of the forces, like gravity, can be easy to measure, however, internal forces or forces from contact with the environment may be difficult to measure from an experiment. Therefore, we do not use the full dynamics of the DSO, rather we make the assumption that the dynamics of the DSO is a smooth and continuous. 
We can use the assumption of smoothness to identify the shape of the DSO under occlusion where we cannot directly measure the DSO position using the camera. There can be many different approaches to leveraging this assumption. In this project, we make use of smoothing splines to obtain a smooth representation of the DSO shape that remove outliers and fill missing information.

To leverage the temporal dependencies, we use a video obtained of the manipulation of the DSO. Using the spatial dependence, we obtain $(x(t), y(t), z(t))$ for various points on the DSO throughout the manipulation time. We fit 3rd order splines with a smoothing factor of 0.5 for $x(t)$, $y(t)$ and $z(t)$. The spline is then used to find missing data due to occlusions. Using this method, we are able to estimate the shape of the DSO even when it is occluded by a manipulator.

<h2 align='center'> Results</h2>

To test our methodology of making segmentation of DSOs aware of occlusions that may be unavoidable during manipulation, we choose a slender object that can be used as the object for manipulation. To simplify our experiments, we chose to manipulate the DSO with our hands rather than using a robotic manipulator. Although our end-goal is for this method to be used for robotic manipulation, we can use hand manipulation because problems like occlusions will still be present in hand manipulation.


We initialise the experiment by placing the DSO on a table such that there are no occlusions of the DSO. We record a video of the DSO while we move the DSO such that there are occlusions due to our hand as well as due to other parts of the DSO itself. We provide both qualitative as well as quantitative validation for our method at estimating the shape of the DSO under occlusion. The video of the DSO manipulation conducted:
<figure markdown="span">
    ![type:video](experiment.mp4){'width: 40%'}
    <figcaption style="text-align: center; font-style: italic;">**Video 2:** Manipulation of the DSO</figcaption>
</figure>

We can see in that our method is able to estimate the shape of the DSO under the occlusion. and that the shape reconstructed in 3D matches the actual shape of the DSO. 
<figure markdown="span">
    ![type:video](temporal.mp4){'width: 40%'}
    <figcaption style="text-align: center; font-style: italic;">**Video 3:** Leveraging temporal consistency, we can obtain an approximation of the occluded DSO.</figcaption>
</figure>

However, to get a quantitative evaluation of our method at estimating the shape of the DSO under evaluation, we consider a video where we do not have any occlusion. We find the shape of the DSO in the absence of occlusions. Later, the same video is augmented to have an occlusion that is added in the form of a rectangle, as shown in figure. The shape estimated in the presence of the occlusion is compared with that in the absence of occlusions, and we conclude that we have an RMSE error of 7mm.

<div style="text-align: center;">
  <img src="artificial_occlusion.png" alt="alt text" style="width: 70%; display: block; margin: 0 auto;" />
  <figcaption style="font-style: italic; margin-top: 8px;">
    <strong>Figure 16:</strong> Artificial occlusion is added. For this frame the RMSE between occluded and non-occluded is only 7mm.
  </figcaption>
</div>

# Conclusion
In this project, we investigate the problem of deformable slender objects (DSO) segmentation in the presence of occlusion events. During manipulation of DSO, it is inevitable that the manipulator may come between the camera and the DSO causing occlusions. We need a method that can estimate the shape and position of the DSO even during occlusion events. To achieve this, we develop a computer vision algorithm that takes the segmented image of the DSO being manipulated and by leveraging some spatio-temporal assumptions on the evolution of DSO dynamics during manipulation, we are able to estimate the shape and position of the DLO with respect to the camera. Specifically, we assume that the spatial and temporal change in the DSO will be smooth. Currently, we check the performance of the algorithm qualitatively, however, we need further testing under different conditions and quantitative measures before the algorithm can be implemented in the real-world. Further work from this project will be mainly focused on quantitative validation of the method. This could be done using an external motion capture system as ground-truth to compare against.


# References

[1] https://jordicenzano.name/front-test/2d-3d-paradigm-overview-2011/camera-model/

[2] J. L. Schonberger and J.-M. Frahm, “Structure-from-motion revisited,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 4104–4113

[3] M. Subbarao and G. Surya, “Depth from defocus: A spatial domain approach,” International Journal of Computer Vision, vol. 13, no. 3, p. 271–294, Dec. 1994. [Online]. Available: http://dx.doi.org/10.1007/BF02028349

[4] ¨O. YILMAZ and F. KARAKUS¸ , “Stereo and kinect fusion for continuous 3d reconstruction and visual odometry,” Turkish Journal of Electrical Engineering and Computer Sciences, vol. 24, no. 4, pp. 2756–2770, 2016

[5] https://www.researchgate.net/publication/273947737/figure/fig8/AS:650806445473802@1532175756521/Depth-defocus-relationship-The-same-object-point-placed-at-different-distances-will-be.png

[6] https://blog.cometlabs.io/teaching-robots-presence-what-you-need-to-know-about-slam-9bf0ca037553#7851

[7] O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional networks for biomedical image segmentation,” 2015. [Online]. Available: https://arxiv.org/abs/1505.04597

[8] T. Y. Zhang and C. Y. Suen, “A fast parallel algorithm for thinning digital patterns,” Communications of the ACM, vol. 27, no. 3, p. 236–239, Mar. 1984. [Online]. Available: http://dx.doi.org/10.1145/357994.358023