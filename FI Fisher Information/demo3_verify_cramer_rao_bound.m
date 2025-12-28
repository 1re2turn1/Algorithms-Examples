%% Verify Cramér-Rao Lower Bound: Monte Carlo Simulation
% This example verifies whether actual decoding performance reaches 
% the theoretical lower bound through Monte Carlo simulation
% Uses maximum likelihood decoder to estimate stimulus from neural responses
clear; clc; close all;

%% 1. Parameter Settings
N = 20;                     % Number of neurons
s_true = 90;                % True stimulus value (angle in degrees)
n_trials = 1000;            % Number of simulation trials

% Preferred directions of neurons
preferred_dirs = linspace(0, 180, N)';

%% 2. Tuning Curves
tuning_width = 30;
max_firing = 40;
f = @(s) max_firing * exp(-((s - preferred_dirs).^2) / (2*tuning_width^2));
f_prime = @(s) f(s) .* (-(s - preferred_dirs) / tuning_width^2);

%% 3. Calculate Fisher Information
fp = f_prime(s_true);
f_s = f(s_true);

% Assume independent Poisson noise
FI = sum(fp.^2 ./ f_s);
CR_bound = sqrt(1 / FI);  % Cramér-Rao lower bound (standard deviation)

%% 4. Monte Carlo Simulation
rng(123);
s_estimates = zeros(n_trials, 1);

for trial = 1:n_trials
    % Generate neural responses (Poisson noise)
    r = poissrnd(f(s_true));
    
    % Maximum likelihood decoding (simplified: grid search)
    s_grid = linspace(0, 180, 500);
    log_likelihood = zeros(size(s_grid));
    
    for i = 1:length(s_grid)
        f_grid = f(s_grid(i));
        % Poisson log-likelihood
        log_likelihood(i) = sum(r .* log(f_grid + 1e-10) - f_grid);
    end
    
    [~, idx] = max(log_likelihood);
    s_estimates(trial) = s_grid(idx);
end

%% 5. Statistical Analysis
empirical_std = std(s_estimates);
empirical_bias = mean(s_estimates) - s_true;

%% 6. Visualization
figure('Position', [100, 100, 1200, 400]);

% Subplot 1: Distribution of Estimates
subplot(1,3,1);
histogram(s_estimates, 30, 'Normalization', 'pdf', 'DisplayName', '');
hold on;
h1 = xline(s_true, 'r--', 'LineWidth', 2, 'DisplayName', 'True Value');
h2 = xline(mean(s_estimates), 'b--', 'LineWidth', 2, 'DisplayName', 'Estimated Mean');
xlabel('Estimated Stimulus Value (deg)');
ylabel('Probability Density');
title('Decoding Distribution');
legend([h1, h2], 'Location', 'best');
grid on;

% Subplot 2: Comparison with Cramér-Rao Lower Bound
subplot(1,3,2);
bar([empirical_std, CR_bound]);
set(gca, 'XTickLabel', {'Empirical Std', 'CR Bound'});
ylabel('Standard Deviation (deg)');
title('Decoding Precision vs Cramer-Rao Bound');
grid on;

% Subplot 3: Time Series
subplot(1,3,3);
plot(1:min(100, n_trials), s_estimates(1:min(100, n_trials)), 'o-');
hold on;
yline(s_true, 'r--', 'LineWidth', 2);
xlabel('Trial');
ylabel('Estimated Value (deg)');
title('Decoding Results (First 100 Trials)');
ylim([s_true - 3*empirical_std, s_true + 3*empirical_std]);
grid on;

%% 7. Output Results
fprintf('=== Cramer-Rao Lower Bound Verification ===\n');
fprintf('True stimulus: %.2f deg\n', s_true);
fprintf('Number of trials: %d\n', n_trials);
fprintf('\nFisher Information: %.4f\n', FI);
fprintf('Cramer-Rao lower bound: %.4f deg\n', CR_bound);
fprintf('\nEmpirical statistics:\n');
fprintf('  Estimated mean: %.4f deg (bias: %.4f deg)\n', mean(s_estimates), empirical_bias);
fprintf('  Estimated std: %.4f deg\n', empirical_std);

% Check if empirical performance achieves CR bound
if empirical_std >= CR_bound * 0.95
    achieve_flag = 'Yes';
else
    achieve_flag = 'No';
end
fprintf('  Achieves CR bound (within 95%%): %s (ratio: %.2f)\n', ...
    achieve_flag, empirical_std / CR_bound);
