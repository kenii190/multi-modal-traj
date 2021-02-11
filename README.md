# Multi-modal Trajectory Generation Using Imitation Learning with Contrastive Loss and Its Application to Motion Synthesis

The source code for the paper "Multi-modal Trajectory Generation Using Imitation Learning with Contrastive Loss and Its Application to Motion Synthesis".

[video](https://drive.google.com/file/d/102VbyrXSb3-pCCZVnx8eKm7DmM7ASjrW/view?usp=sharing)

<br>

## Environments:

- 2D circle trajectories: **"circle"** directory
- 2D traffic trajectories: **"traffic"** directory
- Motion synthesis: **"motion"** directory

<br>

## Models:

- Behavior Cloning (BC): **"bc"** directory
- Conditional VAE Behavior Cloning (cVAE-BC): **"vae_bc"** directory
- Generative Adversarial Imitation Learning (GAIL): **"gail"** directory
- InfoGAIL: **"infogail"** directory
- Our Model: **"ours"** directory

<br>

## Basic Requirement:

For training, testing, and visualization in 2D synthetic environments, please install the following.

1. Install [Python >= 3.6](https://www.python.org/)

2. Install the following python packages:

- [pytorch >= 1.7.1](https://pytorch.org/)
- numpy >= 1.19.5
- matplotlib == 3.1.0
- tqdm >= 4.56.0

e.g.
```
pip install numpy
pip install matplotlib==3.1.0
pip install tqdm
```

## Advanced Requirement:

For visualization in the motion synthesis environment, please first build the gx rendering library.

### Linux (Test on Ubuntu 16.04 LTS)

1. Install GCC >= 5.4.0:

```
sudo apt install build-essential
```

2. Install CMake >= 3.5.1:

```
sudo apt install cmake
```

3. Install header files and a static library for Python:

```
sudo apt-get install libpython[your version]-dev
```

4. Install OpenGL:

```
sudo apt-get install libx11-dev xorg-dev libglu1-mesa-dev
```

5. Install OpenCV >= 3.4:

Please follow the instruction in the [official website](https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html)

6. Download the third-party repositories:

- [pybind11](https://github.com/pybind/pybind11.git)
- [glfw](https://github.com/glfw/glfw.git)
- [glm](https://github.com/g-truc/glm.git)
- [tinyobjloader](https://github.com/tinyobjloader/tinyobjloader.git)

```
cd gx
git clone https://github.com/pybind/pybind11.git
cd lib
git clone https://github.com/glfw/glfw.git
git clone https://github.com/g-truc/glm.git
git clone https://github.com/tinyobjloader/tinyobjloader.git
```

7. Download [glad (GL version 4.5)](https://glad.dav1d.de/) and put it in **"./gx/lib"** directory.

8. Build the **"gx.cpxx-gcc_xxxxxx.so"** file:

```
cd ..
mkdir build
cd build
cmake ..
make -j4
```

After building successfully, please put the file into **"./motion/visualization"** directory.

<br>

### Windows (Test on Windows 10)

1. Install [Visual C++ 15 2017](https://visualstudio.microsoft.com/zh-hant/vs/older-downloads/) or later.

2. Install [CMake >= 3.5.1](https://cmake.org/download/)

3. Install [OpenCV >= 3.4](https://opencv.org/releases/)

6. Download the third-party repositories:

- [pybind11](https://github.com/pybind/pybind11.git)
- [glfw](https://github.com/glfw/glfw.git)
- [glm](https://github.com/g-truc/glm.git)
- [tinyobjloader](https://github.com/tinyobjloader/tinyobjloader.git)

```
cd gx
git clone https://github.com/pybind/pybind11.git
cd lib
git clone https://github.com/glfw/glfw.git
git clone https://github.com/g-truc/glm.git
git clone https://github.com/tinyobjloader/tinyobjloader.git
```

7. Download [glad (GL version 4.5)](https://glad.dav1d.de/) and put it in **"./gx/lib"** directory.

8. Build the **"gx.cpxx-win_xxxxx.pyd"** file:

```
cd ..
mkdir build
cd build
cmake -G "Visual Studio 15 2017 Win64" ..
```
Then open **"gx.sln"** and build the project.

After building successfully, please put the file into **"./motion/visualization"** directory.

<br>

## Execution

### 1. Dataset

- For **circle** and **traffic**, generate the synthetic expert trajectories first:

```
python gen_expert_traj.py
```

- For **motion**, please first download the expert trajectories [here](https://drive.google.com/drive/folders/1eNTMgG6WfV-LqrKlZuxibDpn-sKDJom8?usp=sharing) and put it in **"./motion/data"** directory.

<br>

### 2. Training

For each model in each environment:

```
python train.py
```

The trained model will be saved in **"save"** directory.
(Note that we already provide trained weights in **"save"**)

<br>

### 3. Testing

- For each model in **circle** and **traffic**, run the following to see the reconstruction/interpolation/latent space results:

```
python test.py
```

- And run the following to see the process of trajectory generation:

```
python play.py
```

- For each model in **motion**, run the following to see the latent space:

```
python test.py
```

- For visualizing the motion synthesis results, go to **"./motion/visualization"** and run:

```
python gui_manifold.py
```

for visualization of expert trajectories in the VAE manifold.

```
python gui_policy.py
```

for visualization of motion generated by the trained policy.

```
python gui_recon.py
```

for visualization of motion reconstruction results.