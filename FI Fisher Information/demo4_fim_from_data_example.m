%% FIM from real-like data: Gaussian approx (stimulus-invariant covariance)
clear; clc; close all;

% Synthetic data generation
N = 32; k = 2;
mx = 20; my = 20;  % grid size for unique stimuli
rep = 5;           % repeats per condition
xv = linspace(-10, 10, mx);
yv = linspace(-10, 10, my);
[Xg, Yg] = meshgrid(xv, yv);
Suniq = [Xg(:), Yg(:)];
M = size(Suniq,1);
S = repelem(Suniq, rep, 1);
ns = size(S,1);

rng(7);
RF = randn(N,2) * 5;    % neuron RF centers
sigmaRF = 3;
maxRate = 50;

% Mean tuning
Fmean = zeros(ns, N);
for i = 1:ns
    dx = S(i,1) - RF(:,1);
    dy = S(i,2) - RF(:,2);
    Fmean(i,:) = maxRate * exp(-(dx.^2 + dy.^2) / (2*sigmaRF^2));
end

% Correlated Gaussian noise (stimulus-invariant)
rho = 0.2;
idx = (1:N)';
D = abs(idx - idx');
R = rho * exp(-D/10);
R(1:N+1:end) = 1;
noiseVar = 5;
Qtrue = noiseVar * R;
L = chol(Qtrue + 1e-8*eye(N), 'lower');

Resp = Fmean + randn(ns, N) * L';

% Compute FIM
opts = struct('bandwidth', 1.0, 'cov_reg', 1e-6, 'ridge_beta', 1e-3);
res = compute_fim_gaussian(S, Resp, opts);

% Reshape summaries to grid
DET = reshape(res.det, my, mx);
TRC = reshape(res.trace, my, mx);
EIG1 = reshape(res.eigs(:,1), my, mx);
EIG2 = reshape(res.eigs(:,end), my, mx);
COND = reshape(res.cond, my, mx);

figure('Position', [100, 100, 1400, 500]);
subplot(1,3,1); imagesc(xv, yv, log10(DET + 1)); axis xy; colorbar;
xlabel('x'); ylabel('y'); title('log_{10}(det(FIM)+1)');
subplot(1,3,2); imagesc(xv, yv, log10(TRC + 1)); axis xy; colorbar;
xlabel('x'); ylabel('y'); title('log_{10}(trace(FIM)+1)');
subplot(1,3,3); imagesc(xv, yv, log10(COND + 1)); axis xy; colorbar;
xlabel('x'); ylabel('y'); title('log_{10}(cond(FIM)+1)');

% Error ellipse at a chosen point
ix = round(mx/2); iy = round(my/2);
idx0 = (iy-1)*mx + ix;
F = res.FIM(:,:,idx0);
CovCR = inv(F + 1e-6*eye(k));
[V, Ld] = eig(CovCR);
theta = linspace(0, 2*pi, 200);
E = (V * sqrt(Ld)) * [cos(theta); sin(theta)];
Ex = E(1,:) + xv(ix);
Ey = E(2,:) + yv(iy);
figure('Position', [100, 100, 600, 500]);
imagesc(xv, yv, log10(DET + 1)); axis xy; colorbar; hold on;
plot(Ex, Ey, 'r', 'LineWidth', 2);
xlabel('x'); ylabel('y'); title('Error ellipse (approx 1\sigma) overlay');
