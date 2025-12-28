function result = compute_fim_gaussian(stim, responses, opts)
% Computes Fisher Information Matrix under Gaussian approximation with
% stimulus-invariant covariance.
% Inputs:
%   stim:       (n_sample, k)
%   responses:  (n_sample, N)
%   opts: struct with fields (all optional)
%       .bandwidth        scalar >0 for local linear weights (default auto)
%       .cov_reg          ridge for covariance (default 1e-6)
%       .ridge_beta       ridge for local regression slopes (default 1e-3)
%       .min_weight_frac  minimum total weight fraction (default 1e-3)
%       .use_global_linear if true use global linear regression for J (default false)
%       .group_tol        tolerance to group duplicate stimuli (default [])
%
% Outputs (in struct):
%   result.unique_stimuli  (m, k)
%   result.FIM             (k, k, m)
%   result.det             (m, 1)
%   result.trace           (m, 1)
%   result.eigs            (m, k)
%   result.cond            (m, 1)
%   result.J               (N, k, m)
%   result.Q               (N, N)
%   result.options         opts used

if nargin < 3, opts = struct(); end
[n_sample, k] = size(stim);
[ns2, N] = size(responses);
if ns2 ~= n_sample
    error('responses must have same number of rows as stim');
end

cov_reg   = get_opt(opts, 'cov_reg', 1e-6);
ridge_beta = get_opt(opts, 'ridge_beta', 1e-3);
min_w_frac = get_opt(opts, 'min_weight_frac', 1e-3);
use_global = get_opt(opts, 'use_global_linear', false);

% Group duplicate stimuli
if isfield(opts, 'group_tol') && ~isempty(opts.group_tol)
    [unique_stimuli, group_idx] = group_by_tol(stim, opts.group_tol);
else
    [unique_stimuli, ~, group_idx] = unique(stim, 'rows', 'stable');
end
m = size(unique_stimuli, 1);

% Mean response per condition
mean_resp = zeros(m, N);
for g = 1:m
    idx = (group_idx == g);
    if ~any(idx)
        error('Empty group encountered');
    end
    mean_resp(g, :) = mean(responses(idx, :), 1);
end

% Pooled residual covariance (stimulus-invariant)
residuals = zeros(n_sample, N);
for j = 1:n_sample
    g = group_idx(j);
    residuals(j, :) = responses(j, :) - mean_resp(g, :);
end
Q_hat = cov(residuals); % (N,N)
Q_reg = Q_hat + cov_reg * eye(N);

% Bandwidth for local linear regression
if isfield(opts, 'bandwidth') && ~isempty(opts.bandwidth)
    h = opts.bandwidth;
else
    sstd = std(stim, 0, 1);
    h = 0.5 * mean(sstd);
    if h <= 0
        h = 1.0;
    end
end

FIM = zeros(k, k, m);
J_all = zeros(N, k, m);

total_w_target = n_sample * min_w_frac;

% Precompute for global linear regression if requested
if use_global
    Xg = [ones(n_sample,1), stim];
    Rg = responses; % (n_sample, N)
    Reg = diag([0, ridge_beta*ones(1,k)]);
    Bg = (Xg' * Xg + Reg) \ (Xg' * Rg); % (k+1, N)
    J_global = Bg(2:end, :)'; % (N, k)
end

% Compute J and FIM per condition
for g = 1:m
    s0 = unique_stimuli(g, :);
    if use_global
        J = J_global;
    else
        w = exp(-sum((stim - s0).^2, 2) / (2*h^2));
        sw = sum(w);
        if sw < total_w_target
            % Fallback to global linear if weights too small
            Xg = [ones(n_sample,1), stim];
            Rg = responses;
            Reg = diag([0, ridge_beta*ones(1,k)]);
            Bg = (Xg' * (Xg) + Reg) \ (Xg' * Rg);
            J = Bg(2:end, :)';
        else
            X = [ones(n_sample,1), stim];
            W = diag(w);
            Reg = diag([0, ridge_beta*ones(1,k)]);
            B = (X' * W * X + Reg) \ (X' * W * responses); % (k+1, N)
            J = B(2:end, :)'; % (N, k)
        end
    end
    J_all(:, :, g) = J;
    A = Q_reg \ J;            % (N,k)
    F = J' * A;               % (k,k)
    FIM(:, :, g) = F;
end

% Summaries
detF = zeros(m,1);
trF  = zeros(m,1);
eigsF = zeros(m,k);
condF = zeros(m,1);
for g = 1:m
    F = FIM(:, :, g);
    detF(g) = det(F);
    trF(g)  = trace(F);
    eigsF(g, :) = sort(eig(F), 'descend');
    if k >= 2
        condF(g) = eigsF(g,1) / max(eigsF(g,end), eps);
    else
        condF(g) = 1.0;
    end
end

result = struct();
result.unique_stimuli = unique_stimuli;
result.FIM = FIM;
result.det = detF;
result.trace = trF;
result.eigs = eigsF;
result.cond = condF;
result.J = J_all;
result.Q = Q_hat;
result.options = struct('bandwidth', h, 'cov_reg', cov_reg, 'ridge_beta', ridge_beta, 'min_weight_frac', min_w_frac, 'use_global_linear', use_global);

end

function val = get_opt(opts, name, default)
if isfield(opts, name) && ~isempty(opts.(name))
    val = opts.(name);
else
    val = default;
end
end

function [U, group_idx] = group_by_tol(stim, tol)
[n, ~] = size(stim);
U = [];
group_idx = zeros(n,1);
cur = 0;
for i = 1:n
    si = stim(i, :);
    if i == 1
        U = si; cur = 1; group_idx(i) = 1; continue;
    end
    d = sqrt(sum((U - si).^2, 2));
    j = find(d <= tol, 1, 'first');
    if isempty(j)
        U = [U; si]; 
        cur = cur + 1;
        group_idx(i) = cur;
    else
        group_idx(i) = j;
    end
end
end
