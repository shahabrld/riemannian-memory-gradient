function es_gamma_test

% Define the essential manifold using the quotient R1/R2 representation.
N = 100;    % Number of matrices to process in parallel.
A = multiprod(multiprod(randrot(3, N), essential_hat3([0; 0; 1])), randrot(3, N));
M = essentialfactory(N);
problem.M = M;

% In the "E representation" (i.e., as 3-by-3 matrices) we define
% cost, gradient, and Hessian functions.
costE  = @(E) 0.5 * sum(multisqnorm(E - A));
egradE = @(E) E - A;
ehessE = @(E, U) U;

% Wrap the functions using the conversion functions provided by Manopt,
% converting from the E representation to the R1/R2 representation.
problem.cost = @cost;
function val = cost(X)
     val = essential_costE2cost(X, costE);
end
problem.egrad = @egrad;
function g = egrad(X)
     g = essential_egradE2egrad(X, egradE);
end
problem.ehess = @ehess;
function gdot = ehess(X, S)
     gdot = essential_ehessE2ehess(X, egradE, ehessE, S);
end

% Choose an initial point on the manifold.
x0 = problem.M.rand();

% Define Parameter Arrays for the Study
m_vals = [3, 5, 8];
l_vals = [100, 1000, 10000];
% Three choices for gamma_type.
gamma_types = {'gamma_1', 'gamma_2', 'gamma_3'};

% Preallocate a structure array to store results.
results = struct('m', [], 'l', [], 'gamma_type', [], 'iterations', [], 'time', []);
idx = 1;

% Loop Over All Combinations of Parameters
for m_val = m_vals
    for l_val = l_vals
        for g = 1:numel(gamma_types)
            % Set options specific to the memorygradient solver.
            options.m = m_val;
            options.l = l_val;
            options.gamma_type = gamma_types{g};
            options.linesearch = @linesearch_wolfe;
            options.tolgradnorm = 1e-8;
            
            % Run memorygradient for the given parameter configuration.
            [~, ~, info] = memorygradient(problem, x0, options);
            
            % Record performance using the final iteration's info.
            finalInfo = info(end);
            results(idx).m = m_val;
            results(idx).l = l_val;
            results(idx).gamma_type = options.gamma_type;
            results(idx).iterations = finalInfo.iter;
            results(idx).time = finalInfo.time;

            % If the final gradient norm did not fall below 1e-8, mark as "failed".
            if finalInfo.gradnorm > 1e-6
                results(idx).iterations = "failed";
                results(idx).time = "failed";
            else
                results(idx).iterations = finalInfo.iter;
                results(idx).time = finalInfo.time;
            end
            
            idx = idx + 1;
        end
    end
end

% Convert Results to a Table and Display
T = struct2table(results);

% Print the table in the Command Window.
disp('Performance study for memorygradient on the essential manifold (varying m, l, gamma_type):');
disp(T);

end