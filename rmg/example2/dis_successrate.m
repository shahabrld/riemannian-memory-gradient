function dis_successrate

tol = 1e-4;            % Tolerance for declaring success.
n_values = 25:25:500;

% Method names.
method_names = {'RMG', 'RDY', 'RFR', 'RHS', 'RPR'};
num_methods = numel(method_names);

% Preallocate success count vector.
success_counts = zeros(length(n_values), num_methods);

% Loop over each problem dimension.
for i = 1:length(n_values)
    n = n_values(i);
    p = 4;
    
    % Generate a random symmetric matrix A.
    A = randn(n, n);
    A = (A + A') / 2;
    
    % Define the Grassmann manifold for subspaces of dimension p.
    Gr = grassmannfactory(n, p);
    
    % Set up the optimization problem.
    % We want to maximize trace(X'*A*X) (i.e., capture the dominant subspace)
    % so we minimize the cost: -0.5 * trace(X'*(A*X)).
    problem.M = Gr;
    problem.cost  = @(X) -0.5 * trace(X'*(A*X));
    problem.grad  = @(X) -Gr.egrad2rgrad(X, A*X);

    % Get an initial guess.
    x0 = Gr.rand();
    
    % Set common solver options.
    options.m = 3;
    options.l = n^2;
    options.linesearch = @linesearch_wolfe;
    
    % 1. Run memorygradient (MG).
    [~, ~, info_mg] = memorygradient(problem, x0, options);
    if any([info_mg.gradnorm] <= tol)
        success_counts(i, 1) = 1;
    end
    
    % 2. Run conjugate gradient (CG) methods with different beta updates.
    cg_beta = {'D-Y', 'F-R', 'H-S', 'P-R'};
    for j = 1:length(cg_beta)
        options.beta_type = cg_beta{j};
        [~, ~, info_cg] = conjugategradient(problem, x0, options);
        if any([info_cg.gradnorm] <= tol)
            % Column 1 is MG; add offset.
            success_counts(i, j+1) = 1;
        end
    end
end

success_rates = sum(success_counts, 1) / length(n_values);

% Create a table. The row shows the success rate for each method.
T = array2table(success_rates, 'VariableNames', method_names);

disp('Success rate for each method:');
disp(T);

end