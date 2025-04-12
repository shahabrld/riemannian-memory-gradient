function [X, info] = dominant_invariant_subspace(A, p)
% Returns an orthonormal basis of the dominant invariant p-subspace of A.
%
% function X = dominant_invariant_subspace(A, p)
%
% Input: A real, symmetric matrix A of size nxn and an integer p < n.
% Output: A real, orthonormal matrix X of size nxp such that trace(X'*A*X)
%         is maximized. That is, the columns of X form an orthonormal basis
%         of a dominant subspace of dimension p of A. These are thus
%         eigenvectors associated with the largest eigenvalues of A (in no
%         particular order). Sign is important: 2 is deemed a larger
%         eigenvalue than -5.
%
% The optimization is performed on the Grassmann manifold, since only the
% space spanned by the columns of X matters. The implementation is short to
% show how Manopt can be used to quickly obtain a prototype. To make the
% implementation more efficient, one might first try to use the caching
% system, that is, use the optional 'store' arguments in the cost, grad and
% hess functions. Furthermore, using egrad2rgrad and ehess2rhess is quick
% and easy, but not always efficient. Having a look at the formulas
% implemented in these functions can help rewrite the code without them,
% possibly more efficiently.
%
% See also: dominant_invariant_subspace_complex

% This file is part of Manopt and is copyrighted. See the license file.
%
% Main author: Nicolas Boumal, July 5, 2013
% Contributors:
%
% Change log:
%
%   NB Dec. 6, 2013:
%       We specify a max and initial trust region radius in the options.
%   NB Jan. 20, 2018:
%       Added a few comments regarding implementation of the cost.
%   XJ Aug. 31, 2021
%       Added AD to compute the grad and the hess

    % Generate some random data to test the function
    if ~exist('A', 'var') || isempty(A)
        A = randn(100);
        A = (A+A')/2;
    end
    if ~exist('p', 'var') || isempty(p)
        p = 4;
    end
    
    % Make sure the input matrix is square and symmetric
    n = size(A, 1);
	assert(isreal(A), 'A must be real.')
    assert(size(A, 2) == n, 'A must be square.');
    assert(norm(A-A', 'fro') < n*eps, 'A must be symmetric.');
	assert(p<=n, 'p must be smaller than n.');
    
    % Define the cost and its derivatives on the Grassmann manifold
    Gr = grassmannfactory(n, p);
    problem.M = Gr;
    problem.cost = @(X)    -.5*trace(X'*A*X);
    problem.grad = @(X)    -Gr.egrad2rgrad(X, A*X);
    
    % Notice that it would be more efficient to compute trace(X'*A*X) via
    % the formula sum(sum(X .* (A*X))) -- the code above is written so as
    % to be as close as possible to the familiar mathematical formulas, for
    % ease of interpretation. Also, the product A*X is needed for both the
    % cost and the gradient, as well as for the Hessian: one can use the
    % caching capabilities of Manopt (the store structures) to save on
    % redundant computations.
    
    % An alternative way to compute the gradient and the hessian is to use 
    % automatic differentiation provided in the deep learning toolbox (slower).
    % Notice that the function trace is not supported for AD so far.
    % Replace it with ctrace described in manoptADhelp
    % problem.cost = @(X)    -.5*ctrace(X'*A*X);
    % It's also feasible to specify the cost in a more efficient way
    % problem.cost = @(X)    -.5*sum(sum(X .* (A*X))); 
    % Call manoptAD to prepare AD for the problem structure
    % problem = manoptAD(problem);
    
    % Execute some checks on the derivatives for early debugging.
    % These can be commented out.
    % checkgradient(problem);
    % pause;
    % checkhessian(problem);
    % pause;
    
    % Issue a call to a solver. A random initial guess will be chosen and
    % default options are selected except for the ones we specify here.
    x0 = problem.M.rand();
    options.m = 3;
    options.n = 10000;
    options.gamma_eps = 1e-4;
    options.linesearch = @linesearch_wolfe;

    options.beta_type = 'D-Y';
    options.maxiter = 600;
    
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
