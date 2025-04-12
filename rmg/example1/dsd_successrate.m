function dsd_successrate

tol = 1e-8;          % Tolerance for declaring success.
n_values = 25:25:500; 

% Method names for the table columns.
method_names = {'RMG', 'RDY', 'RFR', 'RHS', 'RPR'};
num_methods = numel(method_names);

% Preallocate a matrix to store success (1) or failure (0) for each experiment.
successes = zeros(length(n_values), num_methods);

% Loop over each problem size.
for k = 1:length(n_values)
    n = n_values(k);
    sigma = 1/n^2;
    
    % Generate input data.
    % Create a doubly stochastic matrix via the Sinkhorn algorithm.
    B = doubly_stochastic(abs(randn(n, n)));
    % Add noise.
    A = max(B + sigma * randn(n, n), 0.01);
    
    % We choose the symmetric case:
    A = (A + A')/2;
    manifold = multinomialsymmetricfactory(n);
    
    % Setup the optimization problem.
    problem.M = manifold;
    problem.cost  = @(X) 0.5 * norm(A - X, 'fro')^2;
    problem.egrad = @(X) X - A;
    
    % Get an initial guess.
    x0 = problem.M.rand();
    
    % Common solver options.
    options.m = 8;
    options.l = n^2;
    options.linesearch = @linesearch_wolfe;
    options.gamma_type = 'gamma_3';
    
    % 1. Run memorygradient (MG).
    [~, ~, info_mg] = memorygradient(problem, x0, options);
    if any([info_mg.gradnorm] <= tol)
        successes(k, 1) = 1;
    end
    
    % 2. Run conjugate gradient (CG) with different beta update rules.
    cg_beta = {'D-Y', 'F-R', 'H-S', 'P-R'};
    for j = 1:length(cg_beta)
        options.beta_type = cg_beta{j};
        [~, ~, info_cg] = conjugategradient(problem, x0, options);
        if any([info_cg.gradnorm] <= tol)
            % Column 1 corresponds to RMG, so store CG result at column j+1.
            successes(k, j+1) = 1;
        end
    end
end

% Compute the success rate for each method.
success_rates = sum(successes, 1) / length(n_values);

% Create a table with the results.
T = array2table(success_rates, 'VariableNames', method_names);

disp('Success rate for each method:');
disp(T);

end