function dsd_gamma_test

% Generate a doubly stochastic matrix using the Sinkhorn algorithm
n = 100;
sigma = 1/n^2;
B = doubly_stochastic(abs(randn(n, n)));
A = max(B + sigma*randn(n, n), 0.01);

% Select symmetric case. (Set symmetric_case=false to use the non-symmetric version.)
symmetric_case = true;
if symmetric_case
    % Ensure symmetry of A.
    A = (A+A')/2;
    manifold = multinomialsymmetricfactory(n);
else
    manifold = multinomialdoublystochasticfactory(n);
end

% Define the manifold optimization problem.
problem.M = manifold;
problem.cost  = @(X) 0.5 * norm(A-X, 'fro')^2;
problem.egrad = @(X) X-A;

% Get an initial guess on the manifold.
x0 = problem.M.rand();

% Parameter Arrays for the Study
% Three different values for m and l.
m_vals = [2, 4, 8];
l_vals = [100, 1000, 10000];
% Three gamma_type options.
gamma_types = {'gamma_1', 'gamma_2', 'gamma_3'};

% Preallocate a structure array for results.
results = struct('m', [], 'l', [], 'gamma_type', [], 'iterations', [], 'time', []);
idx = 1;

% Loop over all combinations of parameters
for m_val = m_vals
    for l_val = l_vals
        for g = 1:numel(gamma_types)
            % Set options.
            options.m = m_val;
            options.l = l_val;
            options.gamma_type = gamma_types{g};
            options.linesearch = @linesearch_wolfe;
            options.tolgradnorm = 1e-8;
            
            % Run memorygradient for the given parameter configuration.
            [x, cost, info] = memorygradient(problem, x0, options);
            
            % Record performance using the final iteration's data.
            finalInfo = info(end);
            results(idx).m = m_val;
            results(idx).l = l_val;
            results(idx).gamma_type = options.gamma_type;
            results(idx).iterations = finalInfo.iter;
            results(idx).time = finalInfo.time;

            % If the final gradient norm did not fall below 1e-8, mark as "failed".
            if finalInfo.gradnorm > 1e-8
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

% Convert results to a table and display
T = struct2table(results);

% Print the table in the command window.
disp('Performance study for memorygradient method (varying m, l, gamma_type):');
disp(T);

end