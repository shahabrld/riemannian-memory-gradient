function es_successrate

tol = 1e-8;
N_values = 50:50:1000;
num_exps = numel(N_values);

% Define method names.
method_names = {'RMG', 'RDY', 'RFR', 'RHS', 'RPR'};
num_methods = numel(method_names);

% Preallocate success matrix.
successes = zeros(num_exps, num_methods);

for k = 1:num_exps
    N = N_values(k);
    
    % --- Generate test data ---
    % A is built as in the original essential_svd example:
    A = multiprod(multiprod(randrot(3, N), essential_hat3([0; 0; 1])), randrot(3, N));
    
    % --- Set up the optimization problem ---
    % Create the essential manifold for N matrices.
    M = essentialfactory(N);
    problem.M = M;
    
    % Define the cost and its Euclidean derivatives in the E-space.
    costE  = @(E) 0.5 * sum(multisqnorm(E - A));
    egradE = @(E) E - A;
    ehessE = @(E, U) U;
    
    % Use nested functions to convert to the R1/R2 representation.
    problem.cost  = @costfun;
    problem.egrad = @egradfun;
    problem.ehess = @ehessfun;
    
    
    
    % (Optional: If desired, enable automatic differentiation)
    % problem = manoptAD(problem);
    
    % Get an initial guess.
    x0 = M.rand();
    
    % --- Common solver options ---
    options.l = 1000;
    options.linesearch = @linesearch_wolfe;
    options.m = 5;
    options.gamma_type = 'gamma_2';
    
    % --- Run memorygradient (MG) ---
    [~, ~, info_mg] = memorygradient(problem, x0, options);
    if any([info_mg.gradnorm] <= tol)
        successes(k, 1) = 1;
    end
    
    % --- Run conjugate gradient (CG) with different beta-update rules ---
    cg_beta = {'D-Y', 'F-R', 'H-S', 'P-R'};
    for j = 1:length(cg_beta)
        options.beta_type = cg_beta{j};
        [~, ~, info_cg] = conjugategradient(problem, x0, options);
        if any([info_cg.gradnorm] <= tol)
            successes(k, j+1) = 1;
        end
    end
end

% Compute the success rate for each method as a fraction of experiments.
success_rates = sum(successes, 1) / num_exps;

% Create a table with the results.
T = array2table(success_rates, 'VariableNames', method_names);
disp('Success rate for each method:');
disp(T);

function val = costfun(X)
    val = essential_costE2cost(X, costE);
end
function g = egradfun(X)
    g = essential_egradE2egrad(X, egradE);
end
function h = ehessfun(X, S)
    h = essential_ehessE2ehess(X, egradE, ehessE, S);
end

end