%% Fisher Information Matrix: Multidimensional Stimulus (2D Position Encoding)
% This example demonstrates how to calculate the Fisher Information Matrix
% for multidimensional stimuli, using 2D position encoding as an example
% Shows spatial distribution of FIM and error ellipsoids
clear; clc; close all;

%% 1. Parameter Settings
N = 30;                    % Number of neurons
k = 2;                     % Stimulus dimensionality (x, y position)

% Stimulus grid
x_range = linspace(-10, 10, 40);
y_range = linspace(-10, 10, 40);
[X, Y] = meshgrid(x_range, y_range);

% Receptive field centers of neurons (randomly distributed)
rng(42);
RF_centers = randn(N, 2) * 5;  % N×2 matrix

%% 2. Define 2D Gaussian Tuning Curves
sigma_RF = 3;              % Receptive field width
max_firing = 50;

% Tuning curve: f_n(x, y)
tuning_func = @(x, y, n) max_firing * exp(-((x - RF_centers(n,1)).^2 + ...
                                              (y - RF_centers(n,2)).^2) / (2*sigma_RF^2));

% Partial derivative with respect to x
f_dx = @(x, y, n) tuning_func(x, y, n) .* (-(x - RF_centers(n,1)) / sigma_RF^2);

% Partial derivative with respect to y
f_dy = @(x, y, n) tuning_func(x, y, n) .* (-(y - RF_centers(n,2)) / sigma_RF^2);

%% 3. Construct Covariance Matrix (simplified as independent Poisson noise)
% Q = diag(f_1, f_2, ..., f_N)
% Use max to avoid division by zero
Q_inv = @(x, y) diag(1 ./ max(tuning_func(x, y, 1:N), 0.1));

%% 4. Calculate Fisher Information Matrix (2×2)
FIM_det = zeros(size(X));   % Determinant (measure of total information)
FIM_trace = zeros(size(X)); % Trace (average information)
FIM_eig_max = zeros(size(X));  % Maximum eigenvalue
FIM_eig_min = zeros(size(X));  % Minimum eigenvalue

for i = 1:size(X, 1)
    for j = 1:size(X, 2)
        x = X(i, j);
        y = Y(i, j);
        
        % Construct Jacobian matrix F'(s) [N×2]
        F_jacobian = zeros(N, k);
        for n = 1:N
            F_jacobian(n, 1) = f_dx(x, y, n);
            F_jacobian(n, 2) = f_dy(x, y, n);
        end
        
        % Fisher Information Matrix: F'^T * Q^(-1) * F'
        Q_inv_mat = Q_inv(x, y);
        FIM = F_jacobian' * Q_inv_mat * F_jacobian;  % 2×2 matrix
        
        % Extract features
        FIM_det(i, j) = det(FIM);           % Determinant
        FIM_trace(i, j) = trace(FIM);       % Trace
        
        % Eigenvalues
        eigs_FIM = eig(FIM);
        FIM_eig_max(i, j) = max(eigs_FIM);
        FIM_eig_min(i, j) = min(eigs_FIM);
    end
end

%% 5. Visualization
figure('Position', [100, 100, 1400, 800]);

% Subplot 1: FIM Determinant (total information)
subplot(2,3,1);
contourf(X, Y, log10(FIM_det + 1), 20, 'LineColor', 'none');
colorbar;
hold on;
plot(RF_centers(:,1), RF_centers(:,2), 'r.', 'MarkerSize', 15);
xlabel('X Position');
ylabel('Y Position');
title('log_{10}(det(FIM) + 1)');
axis equal tight;

% Subplot 2: FIM Trace
subplot(2,3,2);
contourf(X, Y, log10(FIM_trace + 1), 20, 'LineColor', 'none');
colorbar;
hold on;
plot(RF_centers(:,1), RF_centers(:,2), 'r.', 'MarkerSize', 15);
xlabel('X Position');
ylabel('Y Position');
title('log_{10}(Trace(FIM) + 1)');
axis equal tight;

% Subplot 3: Maximum Eigenvalue
subplot(2,3,3);
contourf(X, Y, log10(FIM_eig_max + 1), 20, 'LineColor', 'none');
colorbar;
hold on;
plot(RF_centers(:,1), RF_centers(:,2), 'r.', 'MarkerSize', 15);
xlabel('X Position');
ylabel('Y Position');
title('log_{10}(\lambda_{max} + 1)');
axis equal tight;

% Subplot 4: Minimum Eigenvalue
subplot(2,3,4);
contourf(X, Y, log10(FIM_eig_min + 1), 20, 'LineColor', 'none');
colorbar;
hold on;
plot(RF_centers(:,1), RF_centers(:,2), 'r.', 'MarkerSize', 15);
xlabel('X Position');
ylabel('Y Position');
title('log_{10}(\lambda_{min} + 1)');
axis equal tight;

% Subplot 5: Condition Number (anisotropy)
subplot(2,3,5);
condition_number = FIM_eig_max ./ (FIM_eig_min + 1e-10);
contourf(X, Y, log10(condition_number), 20, 'LineColor', 'none');
colorbar;
hold on;
plot(RF_centers(:,1), RF_centers(:,2), 'r.', 'MarkerSize', 15);
xlabel('X Position');
ylabel('Y Position');
title('log_{10}(Condition Number: \lambda_{max}/\lambda_{min})');
axis equal tight;

% Subplot 6: Error Ellipse Example at a Specific Point
subplot(2,3,6);
test_x = 0;
test_y = 0;

% Calculate FIM at this point
F_jac_test = zeros(N, k);
for n = 1:N
    F_jac_test(n, 1) = f_dx(test_x, test_y, n);
    F_jac_test(n, 2) = f_dy(test_x, test_y, n);
end
Q_inv_test = Q_inv(test_x, test_y);
FIM_test = F_jac_test' * Q_inv_test * F_jac_test;

% Cramér-Rao lower bound: error covariance matrix
Cov_CR = inv(FIM_test);

% Plot error ellipse
[V, D] = eig(Cov_CR);
theta = linspace(0, 2*pi, 100);
ellipse = [cos(theta); sin(theta)];
ellipse_transformed = V * sqrt(D) * ellipse;

plot(test_x + ellipse_transformed(1,:), test_y + ellipse_transformed(2,:), ...
    'LineWidth', 2);
hold on;
plot(RF_centers(:,1), RF_centers(:,2), 'r.', 'MarkerSize', 15);
plot(test_x, test_y, 'k*', 'MarkerSize', 15, 'LineWidth', 2);
xlabel('X Position');
ylabel('Y Position');
title(['Error Ellipse @ (', num2str(test_x), ', ', num2str(test_y), ')']);
legend('Cramer-Rao Error Ellipse', 'Neuron RF Centers', 'Test Point', 'Location', 'best');
axis equal;
grid on;

%% 6. Output Statistical Information
fprintf('=== Fisher Information Matrix Analysis ===\n');
fprintf('Number of neurons: %d\n', N);
fprintf('Stimulus dimensionality: %d\n', k);
fprintf('Average FIM determinant: %.2e\n', mean(FIM_det(:)));
fprintf('Average FIM trace: %.2e\n', mean(FIM_trace(:)));
fprintf('Average condition number: %.2f\n', mean(condition_number(:)));
fprintf('\nCramer-Rao lower bound at test point (%.1f, %.1f):\n', test_x, test_y);
fprintf('  sigma_x >= %.4f\n', sqrt(Cov_CR(1,1)));
fprintf('  sigma_y >= %.4f\n', sqrt(Cov_CR(2,2)));
fprintf('  Correlation coefficient: %.4f\n', Cov_CR(1,2) / sqrt(Cov_CR(1,1)*Cov_CR(2,2)));
