function dis_gamma_test

if ~exist('A', 'var') || isempty(A)
    A = randn(100);
    A = (A+A')/2;
end
if ~exist('p', 'var') || isempty(p)
    p = 4;
end
n = size(A, 1);
assert(isreal(A), 'A must be real.')
assert(size(A, 2) == n, 'A must be square.');
assert(norm(A-A', 'fro') < n*eps, 'A must be symmetric.');
assert(p<=n, 'p must be smaller than n.');

% Define the Grassmann manifold.
Gr = grassmannfactory(n, p);
problem.M = Gr;

% The goal is to find an orthonormal matrix X (n-by-p) that maximizes trace(X'*A*X),
% so that its columns span the dominant invariant subspace of A.
% Equivalently, one can minimize:
%      f(X) = -0.5 * trace(X'*A*X).
% Note that for convenience, the gradient and Hessian are computed 
% via the manifold's conversion functions.
problem.cost = @(X) -0.5 * trace(X' * A * X);
problem.grad = @(X) -Gr.egrad2rgrad(X, A * X);

% Generate an initial point on the Grassmann manifold.
x0 = problem.M.rand();

% Parameter Study Setup
m_vals = [3, 5, 8];
l_vals = [1000, 10000, 50000];
% Three choices for gamma_type.
gamma_types = {'gamma_1', 'gamma_2', 'gamma_3'};

% Preallocate a structure array to store the results.
results = struct('m', [], 'l', [], 'gamma_type', [], 'iterations', [], 'time', []);
idx = 1;

% Loop Over All Combinations of Parameters and Run memorygradient
for m_val = m_vals
    for l_val = l_vals
        for g = 1:numel(gamma_types)
            % Set the options for memorygradient.
            options.m = m_val;
            options.l = l_val;
            options.gamma_type = gamma_types{g};
            options.linesearch = @linesearch_wolfe;
            options.tolgradnorm = 1e-6;
            
            % Run memorygradient with the current parameter configuration.
            [~, ~, info] = memorygradient(problem, x0, options);
            
            % Record performance using the final iteration's information.
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

% Convert the Results to a Table and Display
T = struct2table(results);

% Print the performance table in the Command Window.
disp('Performance study for memorygradient on dominant_invariant_subspace (varying m, l, gamma_type):');
disp(T);

end