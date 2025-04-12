function essential_svd
% Sample solution of an optimization problem on the essential manifold.
%
% Solves the problem \sum_{i=1}^N ||E_i-A_i||^2, where E_i are essential
% matrices. Essential matrices are used in computer vision to represent the
% epipolar constraint between projected points in two perspective views.
%
% Note: the essentialfactory file uses a quotient R1/R2 representation to
% work with essential matrices. On the other hand, from a user point of 
% view, it is convenient to use the E representation  (a matrix of size
% 3-by-3) to give cost, gradient, and Hessian  information. To this end, we
% provide auxiliary files essential_costE2cost, essential_egradE2egrad, and
% essential_ehessE2ehess that convert these ingredients to their R1/R2
% counterparts.
%
% See also: essentialfactory essential_costE2cost essential_egradE2egrad
% essential_ehessE2ehess
 
% This file is part of Manopt: www.manopt.org.
% Original author: Roberto Tron, Aug. 8, 2014
% Contributors: Bamdev Mishra, May 15, 2015.
% Change log:
%
%    Xiaowen Jiang Aug. 20, 2021
%       Added AD to compute the egrad and the ehess 

    % Make data for the test
    N = 2;    % Number of matrices to process in parallel.
    A = multiprod(multiprod(randrot(3, N), essential_hat3([0; 0; 1])), randrot(3, N));
    
    % The essential manifold
    M = essentialfactory(N);
    problem.M = M;
    
    % Function handles of the essential matrix E and Euclidean gradient and Hessian
    costE  = @(E) 0.5*sum(multisqnorm(E-A));
    egradE = @(E) E - A;
    ehessE = @(E, U) U;

    
    % Manopt descriptions
    problem.cost = @cost;
    function val = cost(X)
        val = essential_costE2cost(X, costE); % Cost
    end
    
    problem.egrad = @egrad;
    function g = egrad(X)
        g = essential_egradE2egrad(X, egradE); % Converts gradient in E to X.
    end
    
    problem.ehess = @ehess;
    function gdot = ehess(X, S)
        gdot = essential_ehessE2ehess(X, egradE, ehessE, S); % Converts Hessian in E to X.
    end
    
    % An alternative way to compute the egrad and the ehess is to use 
    % automatic differentiation provided in the deep learning toolbox (slower)
    % call manoptAD to automatically obtain the egrad and the ehess
    % problem = manoptAD(problem);
    
    % Numerically check the differentials.
    % checkgradient(problem); pause;
    % checkhessian(problem); pause;
    
    %Solve the problem
    x0 = problem.M.rand();
    options.m = 5;
    options.l = 1000;
    options.gamma_eps = 1e-4;
    options.linesearch = @linesearch_wolfe;
    options.gamma_type = 'gamma_2';

    % Run conjugate gradient (CG) methods for four β-update variants.
    % Define the list of CG β-update types.
    cg_beta = {'D-Y', 'F-R', 'H-S', 'P-R'};
    nCG = numel(cg_beta);
    cg_infos = cell(nCG,1);
    % The labels will be: RDY, RFR, RHS, RPR.
    cg_labels = cellfun(@(s) ['R', strrep(s, '-', '')], cg_beta, 'UniformOutput', false);
    
    for i = 1:nCG
        options.beta_type = cg_beta{i};
        if strcmp(options.beta_type, 'H-S') || strcmp(options.beta_type, 'P-R')
            options.ls_c1 = 1e-4;
            options.ls_c2 = 0.9;
        else
            options.ls_c1 = 1e-4;
            options.ls_c2 = 0.1;
        end
        [~, ~, cg_infos{i}] = conjugategradient(problem, x0, options);
    end
    
    % Run memorygradient.
    options.ls_c1 = 1e-4;
    options.ls_c2 = 0.1;
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
