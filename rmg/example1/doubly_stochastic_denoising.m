function doubly_stochastic_denoising()
% Find a doubly stochastic matrix closest to a given matrix, in Frobenius norm.
%
% This example demonstrates how to use the geometry factories for the
% doubly stochastic multinomial manifold:
%  multinomialdoublystochasticfactory and
%  multinomialsymmetricfactory (for the symmetric case.)
% 
% The file is based on developments in the research paper
% A. Douik and B. Hassibi, "Manifold Optimization Over the Set 
% of Doubly Stochastic Matrices: A Second-Order Geometry"
% ArXiv:1802.02628, 2018.
%
% Link to the paper: https://arxiv.org/abs/1802.02628.
%
% Please cite the Manopt paper as well as the research paper:
% @Techreport{Douik2018Manifold,
%   Title   = {Manifold Optimization Over the Set of Doubly Stochastic 
%              Matrices: {A} Second-Order Geometry},
%   Author  = {Douik, A. and Hassibi, B.},
%   Journal = {Arxiv preprint ArXiv:1802.02628},
%   Year    = {2018}
% }
% 
% This can be a starting point for many optimization problems of the form:
%
% minimize f(X) such that X is a doubly stochastic matrix (symmetric or not)
%
% Input:  None. This example file generates random data.
% 
% Output: None.
%
% This file is part of Manopt: www.manopt.org.
% Original author: Ahmed Douik, March 15, 2018.
% Contributors:
% Change log:
%
%    Xiaowen Jiang Aug. 31, 2021
%       Added AD to compute the egrad and the ehess  

% Generate input data
n = 100;
sigma = 1/n^2;
% Generate a doubly stochastic matrix using the Sinkhorn algorithm
% Generate a doubly stochastic matrix using the Sinkhorn algorithm.
B = doubly_stochastic(abs(randn(n, n)));
A = max(B + sigma*randn(n, n), 0.01);

% Choose symmetric or non-symmetric case.
symmetric_case = true;
if symmetric_case
    % Symmetrize A and select symmetric manifold.
    A = (A+A')/2;
    manifold = multinomialsymmetricfactory(n);
else
    manifold = multinomialdoublystochasticfactory(n);
end

% Define the manifold optimization problem.
problem.M = manifold;
problem.cost  = @(X) 0.5*norm(A-X, 'fro')^2;
problem.egrad = @(X) X-A;
problem.ehess = @(X, U) U;

% Initial guess.
x0 = problem.M.rand();

% Common options.
options.m          = 8;
options.l          = 1000;
options.gamma_eps  = 1e-4;
options.linesearch = @linesearch_wolfe;
options.gamma_type = 'gamma_3';

% Run conjugate gradient (CG) methods for four β-update variants.
% Define the list of CG β-update types.
cg_beta = {'D-Y', 'F-R', 'H-S', 'P-R'};
nCG = numel(cg_beta);
cg_infos = cell(nCG,1);
% The labels will be: RDY, RFR, RHS, RPR.
cg_labels = cellfun(@(s) ['R', strrep(s, '-', '')], cg_beta, 'UniformOutput', false);

for i = 1:nCG
    options.beta_type = cg_beta{i};
    [~, ~, cg_infos{i}] = conjugategradient(problem, x0, options);
end

% Run memorygradient.
[x_mem, cost_mem, info_mem] = memorygradient(problem, x0, options);
% info_mem corresponds to "RMG" in the legend.

% Plot: Gradient norm versus Time.
figure;
% First plot memorygradient, then the CG results.
semilogy([info_mem.time], [info_mem.gradnorm], 'LineWidth', 1.5);
hold on;
for i = 1:nCG
    semilogy([cg_infos{i}.time], [cg_infos{i}.gradnorm], 'LineWidth', 1.5);
end
xlabel('Time (s)', 'FontSize', 16);
ylabel('Gradient norm', 'FontSize', 16);
set(gca, 'FontSize', 14, 'LineWidth', 1.5, 'XMinorTick', 'off', 'YMinorTick', 'off');
legend({'RMG', cg_labels{:}}, 'Location', 'NorthEast', 'FontSize', 16);
hold off;
print('figure3sub1.eps', '-depsc', '-r300');

% Plot: Gradient norm versus Iteration number.
figure;
semilogy([info_mem.iter], [info_mem.gradnorm], 'LineWidth', 1.5);
hold on;
for i = 1:nCG
    semilogy([cg_infos{i}.iter], [cg_infos{i}.gradnorm], 'LineWidth', 1.5);
end
xlabel('Iteration number', 'FontSize', 16);
ylabel('Gradient norm', 'FontSize', 16);
set(gca, 'FontSize', 14, 'LineWidth', 1.5, 'XMinorTick', 'off', 'YMinorTick', 'off');
legend({'RMG', cg_labels{:}}, 'Location', 'NorthEast', 'FontSize', 16);
hold off;
print('figure3sub2.eps', '-depsc', '-r300');
    
end
