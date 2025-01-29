# Activity 3

## Due: 9am on January 31, 2025

## Objectives

- Understand the structure and function of linear regression models.
- Learn how to implement linear regression using gradient descent.
- Gain hands-on experience by enhancing a basic linear regression implementation.

## Tasks

1. **Enhance the Linear Regression Model**:
   - Implement three loss functions:
     - **Mean Squared Error (MSE)**: 
       ![MSE](https://latex.codecogs.com/png.latex?\text{MSE}=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2)
     - **Mean Absolute Error (MAE)**: 
       ![MAE](https://latex.codecogs.com/png.latex?\text{MAE}=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|)
     - **Huber Loss**: 
       ![Huber Loss](https://latex.codecogs.com/png.latex?L_\delta(a)=\begin{cases}\frac{1}{2}a^2&\text{for}|a|\leq\delta\\\delta(|a|-\frac{1}{2}\delta)&\text{for}|a|>\delta\end{cases})
  
   - Modify the provided code to include these loss functions and observe their impact on the model's performance.
2. **Experiment with Learning Rates**:
   - Experiment with different values of the learning rate (e.g., 0.001, 0.01, 0.1, 1) and observe how it impacts the loss function.
   - Look at the plots of the loss function values over epochs for each learning rate to understand the convergence behavior.
   - Analyze and share the results, noting which learning rates lead to faster convergence and which ones cause instability or slow learning.
