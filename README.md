# Unsupervised Anomaly Detection on Spatio-Temporal Multi-way Data
This repository will be a python library for unsupervised anomaly detection algorithms which will include Graph Signal Processing tools, Tucker Decomposition tools, algorithms, datasets, visualization tools, sample experiments, utility functions and results. Overall contents of this repository can be categorized as follows:

1. **Seperation Algorithms:**
    - Smooth and Sparse Seperation (SSS)
    - Higher order Robust Principal Component Analysis (HoRPCA)
    - Graph Regularized Higher-order Robust Principal Component Analysis (GRHoRPCA)
    - Low-rank and temporally smooth-sparse seperation (LOSS)
    - Low-rank and spatially smooth-sparse seperation (LOSSS)
    - Low-rank and spatio-temporally smooth-sparse seperation (LOTSSS)
2. **Synthetic Anomaly Generation Functions:**
    - Point anomalies
    - Temporally smooth anomalies
    - Spatialy smooth anomalies
    - Spatio-temporally smooth anomalies
3. **Sythetic Data Generation Functions:**
    - Low-rank data 
    - Smooth data
    - Low-rank and smooth data
    - Spatio-temporally smooth data
4. **Utility Functions:**
    - Projection depth function(s)
    - Tensor manipulation functions
    - Graph Signal Processing Functions
        - Random graph generator
        - Product graph generator
        - Random Graph process and Graph filtering
5. **Experiments**
    - 5.1. Point anomalies
    - 5.2. Temporally smooth anomalies
    - 5.3. Locally and temporally smooth anomalies
6. **Unit Tests**
7. **Datasets**
    - NYC Taxi hourly number of arrivals/departures for taxi zones
    - NYC Citi Bike hourly number of arrivals for bike zones

## <center> __1. Seperation Algorithms:__ </center>
Seperation algorithms model the data or the observation $\mathcal{B}$ as a superposition of anomalous component $\mathcal{S}$ and normal component $\mathcal{X}$ i.e. $\mathcal{B = X + S}$. Each algorithm has different assumptions on the nature of each of the components. For example, temporally smooth sparse part asumption assumes that the anomaly is sparse (rarely occuring) and slowly varying in time. 

### <span style='color:green'> __1.1. Smooth and Sparse seperation (SSS)__  </span>
The underlying assumption for the normal component $\mathcal{X}$ is smooth with respect to a graph and the anomaly is rarely occuring in other words it is sparse.

<div align="center">

**Solves the following optimization problem:**

$\begin{aligned}
\mathbf{minimize} &\;\; \alpha_i tr(X_{(i)}^TL_iX_{(i)}) +\lambda_1\|\mathcal{S}\|_1 \;\;\\
\mathbf{such\:that} & \;\;\mathcal{X+S=B}
\end{aligned}$
</div>

Where,
- $\alpha_i$ controls the smoothness of the seperated normal component with respect to the $i$'th mode of the tensor. 
- $\lambda_1$ controls the sparsity of seperated anomaly component.
- $L_i$ is the laplacian matrix of the graph structure for the $i$'th mode. 

<div align='right'>

**Located in:** `src.algos.gsp_smooth_tc.py`

**STATUS:** ✅ *Done*
</div>

### <span style='color:green'> **1.2. Higher-order Robust Principal Component Analysis** __(HoRPCA)__ </span>
Normal component of the data $\mathcal{X}$ is assumed to be low-rank and anomaly component $\mathcal{S}$ is assumed to be sparse. <span style="color:red">CITATION NEEDED </span>

<div align="center"> 

**Solves the following optimization problem:**

$\begin{aligned}
\mathbf{minimize} &\;\; \sum_{i=1}^N \|X_{(i)}\|_* +\lambda_1\|\mathcal{S}\|_1 \;\;\\
\mathbf{such\:that} & \;\;\mathcal{X+S=B}
\end{aligned}$
</div>

<div align='right'>

**Located in:** `src.algos.horpca_singleton.py`  

**Status:** ✅ *Done*
</div>

### <span style='color:green'> **1.3. Graph Regularized Higher-order Robust Principal Component Analysis** __(GRHoRPCA)__ </span>
Normal component of the data $\mathcal{X}$ is assumed to be low-rank and smooth with respect to the laplacian $L_i$ in the $i$'th mode and anomaly component $\mathcal{S}$ is assumed to be sparse.

<div align="center"> 

**Solves the following optimization problem:**

$\begin{aligned}
\mathbf{minimize} &\;\; \sum_{i=1}^N (\|X_{(i)}\|_*+\alpha_i tr(X_{(i)}^TL_iX_{(i)})) +\lambda_1\|\mathcal{S}\|_1 \;\;\\
\mathbf{such\:that} & \;\;\mathcal{X+S=B}
\end{aligned}$

</div>

Where,

$\alpha_i$ controls the smoothness of the seperated normal component with respect to the $i$'th mode of the tensor. 

$\lambda_1$ controls the sparsity of seperated anomaly component.

$L_i$ is the laplacian matrix of the graph structure for the $i$'th mode.

<div align='right'>

**Located in** `src.algos.grhorpca.py`

**STATUS:** ✅ *Done*

</div>

### <span style='color:green'> __1.4. Low-rank and temporally smooth-sparse seperation__ **(LOSS)** </span>
Normal component of the data $\mathcal{X}$ is assumed to be low-rank and anomaly component $\mathcal{S}$ is assumed to be sparse and slowly varying with respect to time or in other words it's temporally smooth. <span style="color:red">CITATION NEEDED </span>

<div align="center"> 

**Solves the following optimization problem:**

$\begin{aligned}
\mathbf{minimize} &\;\; \sum_{i=1}^N (\lambda_{*_i}\|X_{(i)}\|_* +\lambda_1\|\mathcal{S}\|_1 + \lambda_\Delta\|\mathcal{S}\times_1\Delta\|_1 \;\;\\
\mathbf{such\:that} & \;\;\mathcal{X+S=B}
\end{aligned}$
</div>


Where,
 
- $\lambda_\Delta$ controls the temporal smoothness of the seperated anomaly component. 
- $\lambda_1$ controls the sparsity of seperated anomaly component.
- $\times_1 \Delta$ difference operation in the time mode.   

<div align='right'>

**Located in** `src.algos.loss.py`

**Status:**  ❌ *Stub function*

</div>


### <span style='color:green'> __1.5. Low-rank and spatially smooth-sparse seperation__ **(LOSSS)** </span>
Normal component of the data $\mathcal{X}$ is assumed to be low-rank and anomaly component $\mathcal{S}$ is assumed to be sparse and slowly varying in the neighboring locations or in other words it's locally smooth.

<div align="center"> 

**Solves the following optimization problem:**

$\begin{aligned}
\mathbf{minimize} &\;\; \sum_{i=1}^N (\lambda_{*_i}\|X_{(i)}\|_*  + \alpha_i\|\mathcal{S}\times_i \sqrt{\Lambda_i} V_i^T\|_1)+ \lambda_\Delta\|\mathcal{S}\times_1\Delta\|_1 +\lambda_1\|\mathcal{S}\|_1\;\;\\
\mathbf{such\:that} & \;\;\mathcal{X+S=B}
\end{aligned}$
</div>

Where,
 
- $\lambda_1$ controls the sparsity of seperated anomaly component.
- $\times_i \sqrt{\Lambda_i} V_i^T $ is the Graph Fourier Transform operation in the $i$'th mode.   
- $\alpha_i$ controls the local smoothness of the seperated anomaly.

<div align='right'>

**Located in** `src.algos.losss.py`

**Status:**  ❌ *Stub function*

</div>

### <span style='color:green'> __1.6. Low-rank and spatio-temporally smooth-sparse seperation__ **(LOTSSS)**
Normal component of the data $\mathcal{X}$ is assumed to be low-rank and anomaly component $\mathcal{S}$ is assumed to be sparse and slowly varying in the neighboring locations and in time or in other words it's spatio-temporally smooth.

<div align="center"> 

**Solves the following optimization problem:**

$\begin{aligned}
\mathbf{minimize} &\;\; \sum_{i=1}^N (\lambda_{*_i}\|X_{(i)}\|_* +\lambda_1\|\mathcal{S}\|_1 + \alpha_i\|\mathcal{S}\times_i \sqrt{\Lambda_i} V_i^T\|_1) + \lambda_\Delta\|\mathcal{S}\times_1\Delta\|_1 \;\;\\
\mathbf{such\:that} & \;\;\mathcal{X+S=B}
\end{aligned}$
</div>

Where,
 
- $\lambda_1$ controls the sparsity of seperated anomaly component.
- $\times_i \sqrt{\Lambda_i} V_i^T $ is the Graph Fourier Transform operation in the $i$'th mode.   
- $\alpha_i$ controls the local smoothness of the seperated anomaly.
- $\times_1 \Delta$ difference operation in the time mode. 
- $\lambda_\Delta$ controls the temporal smoothness of the seperated anomaly component. 
<div align='right'>

**Located in** `src.algos.losss.py`

**Status:**  ❌ *Stub function*

</div>

## <center>__2. Synthetic Anomaly Generation Functions:__</center>
These functions generate $\mathcal{S}$ which represents the additive anomalies to be superposed to the normal component $\mathcal{X}$ to synthesize anomalous data. These functions also return tensor $\mathcal{M}$ that takes on the value $1$ in the support of non-zero anomalies to act as an indicator mask.
### <span style="color:green"> **2.1. Point Anomaly**</span>
Generates $\mathcal{S}$ which comprises of impulses at random locations.

Currently Located in `src.util.contaminate_data.py` will be moved to `src.util.synthetic_anomaly.py`
<div align='right'>

**Status:**  ✅ *Done*
</div>

### <span style="color:green"> **2.2. Temporally smooth anomalies**</span>
Generates $\mathcal{S}$ which may be comprised of one of two different types of temporally smooth anomalies at random locations. The type is specified.
- Rectangular steps of specified length and magnitude.
- Anomalies in the shape of gaussian function with specified scale and magnitude.

<div align='right'>

**Located in** `src.util.synthetic_anomaly.py`

**Status:**  ❌ *Stub function*
</div>

### <span style="color:green"> **2.3. Spatialy smooth anomalies** </span>
Generate $\mathcal{S}$ which is comprised of locally smooth anomalies at random entries of the data. The anomaly takes on similar values at its local neighbours. The neighbors are specified with the inputed graph structure. Diameter and the magnitude of the anomaly must also specified.

<div align='right'>

**Located in** `src.util.synthetic_anomaly.py`

**Status:**  ❌ *Stub function*
</div>

### <span style="color:green"> **2.4. Spatio-temporally smooth anomalies** </span>
Generate $\mathcal{S}$ which is comprised of spatio-temporally smooth anomalies at random entries of the data. The anomaly takes on similar values at its local neighbours and is slowly varying in time. The neighbors are specified with the inputed graph structure. Diameter and the magnitude of the anomaly must also specified.

<div align='right'>

**Located in** `src.util.synthetic_anomaly.py`

**Status:**  ❌ *Stub function*
</div>

## <center> **3. Sythetic Data Generation Functions:** </center>
These functions are to generate synthetic data that will act as ground truth information when we do experiments to evaluate the performance of the above mentioned algorithms. Functions 3.1-3.3 generate  purely synthetic data, function 3.4 generates data that is similar to empirical ones provided.
### <span style="color:green">**3.1. Low-rank data**</span>
Generate low-rank data with specified dimensions and tucker rank. Refer to the documentation of the function for more details. <span style="color:red">CITATION NEEDED </span>
<div align='right'>

**Located in** `src.util.generate_lr_data.py`

**Status:**  ✅ *Done*

**Unit Tested:**  ✅ *Passed*
</div>

### <span style="color:green">**3.2. Smooth data**</span>
Generate random data that is smooth with respect to the specified graph structure. Can use three types of graph filters. Namely; Heat, Tikhonov, Gaussian. The data takes on similar values across neighbors. Refer to the documentation of the function for more details. <span style="color:red">CITATION NEEDED </span>
<div align='right'>

**Located in** `src.util.graph.py`

**Status:**  ✅ *Done*
</div>

### <span style="color:green">**3.3. Low-rank and smooth data**</span>
Generate random data that is smooth with respect to the specified graph structure and is approximately low-rank. Can use three types of graph filters. Namely; Heat, Tikhonov, Gaussian. The data takes on similar values across neighbors. Refer to the documentation of the function for more details.

<div align='right'>

**Currently Located in** `src.util.gen_lr_smooth_data.py`

**Status:**  ✅ *Done*
</div>


### <span style="color:green">**3.4. Spatio-temporally smooth data**</span>
Generates random data that is similar the given real spatio-temporal data. This is done by calculating the mean values of the temporal data for each week and multiplying it with gaussian noise with low standard deviation for each location. Will be implemented as a method of STData class under the name `gen_similar_st_data`. 

<div align='right'>

**Located in** `src.util.spatio_temporal_data.py`

**Status:**  ❌ *Stub function*
</div>

## <center> __4. Utility Functions__ </center>
These functions or classes are more primitive units for the algorithms or used in the data generation or experiments.

### <span style="color:green">**4.1. Tensor manipulation functions**</span>
Operations on multi-way data that are required for the seperation algorithms such as unfolding the tensor in a mode, tensorizing a matrix, soft-thresholding the singular values of an unfolded tensor, folding a matrix into tensor.

#### **4.1.1 Tensor to matrix (t2m)**
Given a multi-way data $\mathcal{X}\in\mathbb{R}^{n_1\times n_2 \times ...\times n_N}$ indexed by indices $i_1,i_2,...,i_N$. `t2m(X,k)` matricizes the tensor $\mathcal{X}$ into $X \in \mathbb{R}^{n_k\times n_{k+1}n_{k+2}...n_Nn_1...n_{k-1}}$. The fibers $\mathcal{X_{i_1,i_2,...,i_{k-1},:,i_{k+1},...,i_N}}$ that are obtained by varying the $k$'th index and keeping other indices fixed are stacked as columns to obtain the matrix $X$.

<div align='right'>

**Located in** `src.util.t2m.py`

**Status:**  ✅ *Done*

**Unit Tested:**  ✅ *Passed*
</div>

#### **4.1.2 Matrix to tensor (m2t)**
Given a matrix $X\in\mathbb{R}^{n_k\times n_{k+1}n_{k+2}...n_Nn_1...n_{k-1}}$ which is unfolded $\mathcal{X}\in\mathbb{R}^{n_1\times n_2 \times ...\times n_N}$ in the $k$'th mode, `m2t(X,dims,k)` folds the matrix into its original form. The definition follows from the folding operation done by `t2m` function.

<div align='right'>

**Located in** `src.util.m2t.py`

**Status:**  ✅ *Done*

**Unit Tested:**  ✅ *Passed*
</div>


#### **4.1.3 Soft threshold singular values (soft_moden)**
Unfolds the tensor in the $n$'th mode and calculates its singular value decomposition to soft treshold its singular values. Required by the algorithms that minimize the nuclear norm $\|\cdot \|_*$ with its proximal operator.

<div align='right'>

**Located in** `src.util.soft_hosvd.py`

**Status:**  ✅ *Done*

**Unit Tested:**  ❌ *Done manually*
</div>

#### **4.1.4 Soft threshold (soft_threshold)**
Applies the soft tresholding operation on the entries of a given tensor. Required by the algorithms that minimize the $l_1$ norm with its proximal operator.

<div align='right'>

**Located in** `src.util.soft_treshold.py`

**Status:**  ✅ *Done*

**Unit Tested:**  ✅ *Passed*
</div>


#### **4.1.5 Q-Mult (qmult)**
Generates a random orthonormal array Q drawn from the Haar distribution with the specified dimension. This function is used in generating random low-rank tensors. <span style="color:red">CITATION NEEDED </span>

<div align='right'>

**Located in** `src.util.qmult.py`

**Status:**  ✅ *Done*

**Unit Tested:**  ❌ *No*
</div>

### <span style="color:green"> **4.2. Graph Signal Processing Functions** </span>
These functions are located within `src.util.graph.py` module as methods of `Graph`, `ProductGraph` and `GraphProcess` class objects.


#### **4.2.1. Random graph generator**
`Graph` class is a wrapper for `networkx` graph objects to initialize graphs either randomly with provided parameters of models such as erdos-renyi or barabasi-albert models or directly using graph laplacian matrices, adjacency matrices with its `__init__` method.

<div align='right'>

**Located in** `src.util.graph.py`

**Status:**  ✅ *Done*

</div>

#### **4.2.2. Product graph generator**
`ProductGraph` class is a wrapper for `networkx` graph classes to initialize product graphs either randomly similar to the random graph generator or calculate the product graph of given factor graphs. Currently only cartesian product graph is implemented. 

<div align='right'>

**Located in** `src.util.graph.py`

**Status:**  ✅ *Done*

</div>


#### **4.2.3. Random Graph Process**
`ProductGraph` is used to generate graph signals that are smooth with respect to a graph structure (signal values on each node is similar to its neighboring nodes indicated by the graph structure.) Currently Heat, Tikhonov, Gaussian filters are implemented. <span style="color:red">CITATION NEEDED </span>

<div align='right'>

**Located in** `src.util.graph.py`

**Status:**  ✅ *Done*

</div>

### <span style="color:green">**4.3. Projection depth function(s)**</span>
If time permits, rayleigh projection depth for vector, matrix, higher-order tensor data will be implemented. Projection depth is used to discriminate anomalous points of interest from normal ones.

### <span style="color:green">**4.4. Read spatio-temporal data from cvs file**</span>
Reads a cvs file of spatio-temporal data and parses it into `np.array` with specified resolution. 

<div align='right'>

**Will be located in** `src.util.spatio_temporal_data.py`

**Status:**  ❌ 
</div>

## <center> **5. Experiments** </center>
The performance evaluation of different algorithms will be done via experiments. Combination of different types of anomalies and different types of data will be superposed to eachother and different various algorithms will be evaluated. The results will be saved into `testbench.results` and the experiments will be plotted in `testbench.experiments.ipynb` file.

- **Experiment 1. Point anomalies** 
Temporally smooth anomalies of varying duration will be superposed to different types of synthetic data to evaluate seperation algorithms performance.<p align='right'> **Status:**  ❌</p>
- **Experiment 2. Temporally smooth anomalies**
Temporally smooth anomalies of varying duration will be superposed to different types of synthetic data to evaluate seperation algorithms performance.<p align='right'> **Status:**  ❌</p>
- **Experiment 3. Locally and temporally smooth anomalies**
Locally and temporally smooth anomalies of varying duration will be superposed to different types of synthetic data to evaluate seperation algorithms performance. <p align='right'> **Status:**  ❌</p>

## <center> **6. Unit Tests** </center>
Most of the functions used in this project are very hard to unit-test robustly. They require manual tests, experiments or they are just wrappers for well known packages. Most primitive and fundamental functions in this project will be unit-tested. These include, tensor manipulation functions, some data and anomaly generation functions, real data parsing functions. 

**The list of functions unit-tested are as follows:**

- **TestTensor** Located in: `src.tests.test_tensor.py`
    - **Tensor to matrix (t2m)**
    - **Matrix to tensor (m2t)**
    - **Soft threshold**
- **TestSyntheticData** Located in `src.tests.test_synthetic_data.py`
    - **Q-Mult**
    - **Low-rank data**

<div align='right'>

**Located in** `src.tests` folder

</div>

## <center> **7. Datasets** </center>
- NYC Taxi hourly number of arrivals/departures for taxi zones
<div align='right'>

**Status:**  ❌
</div>

- NYC Citi Bike hourly number of arrivals for bike zones
<div align='right'>

**Status:**  ❌
</div>


## Contributing
Mert Indibi

## License


