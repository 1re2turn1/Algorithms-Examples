%% Fisher Information: Single-Dimensional Stimulus with Population Coding (with Noise Correlations)
% This example demonstrates how to calculate Fisher Information 
% for a neural population responding to a single-dimensional stimulus
% and analyzes the impact of noise correlations
clear; clc; close all;

%% 1. Parameter Settings
N = 128;                    % Number of neurons
s_range = linspace(0, 180, 100);  % Stimulus range (e.g., orientation angle)
sigma_noise = 5;           % Poisson noise parameter

% Preferred directions of neurons (uniformly distributed)
preferred_dirs = linspace(0, 180, N);

%% 2. Define Tuning Curves (Von Mises / Gaussian type)
tuning_width = 30;         % Tuning width
max_firing = 40;           % Maximum firing rate

% Tuning curve matrix: f(s) for all neurons
% Add minimum firing rate to avoid zero variance
min_firing = 0.1;  % Minimum baseline firing rate
f = @(s) max(min_firing, max_firing * exp(-((s - preferred_dirs').^2) / (2*tuning_width^2)));

% Derivative of tuning curves
f_prime = @(s) f(s) .* (-(s - preferred_dirs') / tuning_width^2);

%% 3. Construct Noise Covariance Matrix Q
% Assume noise correlation decays exponentially with difference in preferred direction
corr_length = 40;  % Correlation length
rho = 0.3;         % Base correlation coefficient

% Distance matrix
dist_matrix = abs(preferred_dirs' - preferred_dirs);
dist_matrix = min(dist_matrix, 180 - dist_matrix);  % Handle periodicity

% Correlation coefficient matrix
R = rho * exp(-dist_matrix / corr_length);
R(1:N+1:end) = 1;  % Diagonal elements = 1

% Covariance matrix (assume variance proportional to mean firing rate, Poisson approximation)
Q = @(s) diag(f(s)) * R * diag(f(s));

%% 4. Calculate Fisher Information
FI = zeros(size(s_range));

% Regularization parameter for numerical stability
reg_param = 1e-6;  % Ridge regularization

for i = 1:length(s_range)
    s = s_range(i);
    
    % Derivative of tuning curves at current stimulus
    fp = f_prime(s);  % N×1 vector
    
    % Covariance matrix with regularization
    Q_s = Q(s);
    Q_s_reg = Q_s + reg_param * eye(N);  % Add regularization to diagonal
    
    % Fisher Information: f'(s)^T * Q^(-1) * f'(s)
    FI(i) = fp' * (Q_s_reg \ fp);  % Use left division with regularized matrix
end

%% 5. Calculate Independent Case for Comparison
FI_independent = zeros(size(s_range));
for i = 1:length(s_range)
    s = s_range(i);
    fp = f_prime(s);
    f_s = f(s);
    
    % Independent case: simple summation
    FI_independent(i) = sum(fp.^2 ./ f_s);
end

%% 6. Visualization
figure('Position', [100, 100, 1200, 400]);

% Subplot 1: Tuning Curves
subplot(1,3,1);hold on
s_plot = 90;  % Choose a stimulus value for display
plot(preferred_dirs, f(s_plot), 'LineWidth', 2);
xlabel('Neuron Preferred Direction (deg)');
ylabel('Firing Rate (Hz)');
title(['Tuning Curves (s = ', num2str(s_plot), ' deg)']);

% Subplot 2: Fisher Information Curve
subplot(1,3,2);hold on;
plot(s_range, FI, 'LineWidth', 2, 'DisplayName', 'With Noise Correlation');
plot(s_range, FI_independent, '--', 'LineWidth', 2, 'DisplayName', 'Independent');
xlabel('Stimulus Value (deg)');
ylabel('Fisher Information');
title('Fisher Information vs Stimulus');
legend('Location', 'best');

% Subplot 3: Correlation Matrix
subplot(1,3,3);
imagesc(preferred_dirs, preferred_dirs, R);
colorbar;
xlabel('Neuron Preferred Direction (deg)');
ylabel('Neuron Preferred Direction (deg)');
title('Noise Correlation Coefficient Matrix');
axis square;

%% 7. Output Key Results
fprintf('=== Fisher Information Analysis Results ===\n');
fprintf('Number of neurons: %d\n', N);
fprintf('Max FI (with correlation): %.2f\n', max(FI));
fprintf('Max FI (independent): %.2f\n', max(FI_independent));
fprintf('Information loss due to correlation: %.2f%%\n', ...
    (1 - max(FI)/max(FI_independent)) * 100);
