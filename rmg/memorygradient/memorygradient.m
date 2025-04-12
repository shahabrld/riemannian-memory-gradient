function [x, cost, info, options] = memorygradient(problem, x, options)
% MEMORYGRADIENT  Memory gradient minimization algorithm for Manopt.
%
% This function implements a Riemannian memory gradient method to
% minimize a cost function f on a manifold. It uses stored past search 
% directions to build a memory term in the update of the search direction.
%
% Note: This code was adapted from a Manopt conjugate gradient template.
%       The update of the scaling parameter gammak allows three options:
%       'default', 'gamma_star', and 'constant'.

M = problem.M;

% Verify that the problem description is sufficient for the solver.
if ~canGetCost(problem)
    warning('manopt:getCost', ...
        'No cost provided. The algorithm will likely abort.');
end
if ~canGetGradient(problem) && ~canGetApproxGradient(problem)
    warning('manopt:getGradient:approx', ...
           ['No gradient provided. Using an FD approximation instead (slow).\n' ...
            'It may be necessary to increase options.tolgradnorm.\n' ...
            'To disable this warning: warning(''off'', ''manopt:getGradient:approx'')']);
    problem.approxgrad = approxgradientFD(problem);
end

% Set local defaults here
localdefaults.minstepsize = 1e-10;
localdefaults.maxiter = 1000;
localdefaults.tolgradnorm = 1e-16;
localdefaults.storedepth = 20;
localdefaults.orth_value = Inf; % As suggested in Nocedal and Wright

if ~isfield(options, 'gamma_eps')
    options.gamma_eps = 1e-4;
end
if ~isfield(options, 'm')
    options.m = 4;
end
if ~isfield(options, 'l')
    options.l = 1000;
end

% Assign the parameters.
gamma_eps = options.gamma_eps;
m = options.m;
l = options.l;

% Depending on whether the problem structure specifies a hint for
% line-search algorithms, choose a default line-search routine.
if ~canGetLinesearch(problem)
    localdefaults.linesearch = @linesearch_adaptive;
else
    localdefaults.linesearch = @linesearch_hint;
end

% Merge global and local defaults, then merge with user options, if any.
localdefaults = mergeOptions(getGlobalDefaults(), localdefaults);
if ~exist('options', 'var') || isempty(options)
    options = struct();
end
options = mergeOptions(localdefaults, options);

% Default gamma_type if not provided.
if ~isfield(options, 'gamma_type')
    options.gamma_type = 'gamma_1';
end

timetic = tic();

% If no initial point x is given by the user, generate one at random.
if ~exist('x', 'var') || isempty(x)
    x = M.rand();
end

% Create a store database and generate a key for the current x.
storedb = StoreDB(options.storedepth);
key = storedb.getNewKey();

% Compute cost-related quantities for x.
[cost, grad] = getCostGrad(problem, x, storedb, key);
gradnorm = M.norm(x, grad);
Pgrad = getPrecon(problem, x, grad, storedb, key);
gradPgrad = M.inner(x, grad, Pgrad);

% Iteration counter (here, iter is the number of fully executed iterations).
iter = 0;

% Allocate memory for storing past iterates and directions.
xHistory = cell(1, m);
etaHistory = cell(1, m);

% Save stats in a struct array info and preallocate.
stats = savestats();
info(1) = stats;
info(min(10000, options.maxiter+1)).iter = [];

if options.verbosity >= 2
    fprintf(' iter\t               cost val\t    grad. norm\n');
end

% Compute a first descent direction (not normalized).
desc_dir = M.lincomb(x, -1, Pgrad);

% Start iterating until the stopping criterion triggers.
while true
    
    % Display iteration information.
    if options.verbosity >= 2
        fprintf('%5d\t%+.16e\t%.8e\n', iter, cost, gradnorm);
    end
    
    % Start timing this iteration.
    timetic = tic();
    
    % Run standard stopping criterion checks.
    [stop, reason] = stoppingcriterion(problem, x, options, info, iter+1);
    
    % Check a specific stopping criterion on the step size.
    if ~stop && abs(stats.stepsize) < options.minstepsize
        stop = true;
        reason = sprintf(['Last stepsize smaller than minimum '  ...
                          'allowed; options.minstepsize = %g.'], ...
                          options.minstepsize);
    end
    
    if stop
        if options.verbosity >= 1
            fprintf([reason '\n']);
        end
        break;
    end
    
    % Compute the directional derivative of the cost at x along desc_dir.
    df0 = M.inner(x, grad, desc_dir);
        
    % If df0 is nonnegative, reset the search direction to the (preconditioned)
    % steepest descent direction (this resets the memory).
    if df0 >= 0
        if options.verbosity >= 3
            fprintf(['Info: got an ascent direction (df0 = %2e), '...
                     'resetting to the (preconditioned) steepest descent direction.\n'], df0);
        end
        desc_dir = M.lincomb(x, -1, Pgrad);
        df0 = -gradPgrad;
    end
    
    % Execute line search along the search direction.
    [stepsize, newx, newkey, lsstats] = options.linesearch( ...
                   problem, x, desc_dir, cost, df0, options, storedb, key);
               
    % Compute the new cost-related quantities for newx.
    [newcost, newgrad] = getCostGrad(problem, newx, storedb, newkey);
    newgradnorm = M.norm(newx, newgrad);
    Pnewgrad = getPrecon(problem, newx, newgrad, storedb, newkey);
    newgradPnewgrad = M.inner(newx, newgrad, Pnewgrad);
    
    % Update the search direction.
    %
    % For the first few iterations, use a simple gradient descent step.
    if iter < m
        desc_dir = M.lincomb(newx, -1, Pnewgrad);
        xHistory{iter+1} = newx;
        etaHistory{iter+1} = desc_dir;
        
    else
        % When sufficient iterations are available, compute a memory-based update.
        % Transport the previous gradient to the new point.
        oldgrad = M.transp(x, newx, grad);
        orth_grads = M.inner(newx, oldgrad, Pnewgrad) / newgradPnewgrad;

        if abs(orth_grads) >= options.orth_value
            desc_dir = M.lincomb(x, -1, Pnewgrad);
        else 
            % Compute s_{k-1} and y_{k-1}.
            sk = M.transp(x, newx, M.lincomb(x, stepsize, desc_dir));
            yk = M.lincomb(newx, 1, newgrad, -1, M.transp(x, newx, grad));
            
            % Select gamma using one of the three strategies.
            switch options.gamma_type
                case 'gamma_1'
                    inner_sk_yk = M.inner(newx, sk, yk);
                    gammak = inner_sk_yk / M.norm(newx, yk)^2;
                    if gammak < gamma_eps
                        gammak = 1;
                    end
                case 'gamma_2'
                    % temp corresponds to T_{alpha*eta}(g_{k-1}) + g_k.
                    temp = M.lincomb(newx, 1, M.transp(x, newx, grad), 1, newgrad);
                    % theta = 6*(f(x_{k-1}) - f(x_k)) + 3< T_{alpha*eta}(g_{k-1})+g_k, s_{k-1} >.
                    theta = 6*(cost - newcost) + 3*M.inner(newx, temp, sk);
                    norm_s_sq = M.norm(newx, sk)^2;
                    if norm_s_sq < 1e-10
                        norm_s_sq = 1e-10;
                    end
                    % Compute z_{k-1} = yk + (theta / ||sk||^2) * sk.
                    z = M.lincomb(newx, 1, yk, theta/norm_s_sq, sk);
                    gammak = M.inner(newx, z, sk) / M.inner(newx, z, z);
                    if gammak < gamma_eps
                        gammak = 1;
                    end
                case 'gamma_3'
                    % Compute temp = T_{alpha_{k-1}eta_{k-1}}(g_{k-1}) + g_k.
                    temp = M.lincomb(newx, 1, M.transp(x, newx, grad), 1, newgrad);
                    % theta = 6*(f(x_{k-1}) - f(x_k)) + 3 <temp, sk>.
                    theta = 6*(cost - newcost) + 3*M.inner(newx, temp, sk);
                    % Use the inner product <sk, yk> instead of ||sk||^2.
                    denom = M.inner(newx, sk, yk);
                    if abs(denom) < 1e-10
                        denom = 1e-10;
                    end
                    % Compute z = yk + (theta/denom)*yk = (1+theta/denom)*yk.
                    z = M.lincomb(newx, (1 + theta/denom), yk);
                    gammak = M.inner(newx, z, sk) / M.inner(newx, z, z);
                    if gammak < gamma_eps
                        gammak = 1;
                    end
                otherwise
                    error('Unknown gamma_type option.');
            end
            
            % Accumulate the memory contributions from the past m search directions.
            desc_dir_sum = M.zerovec(newx);
            for i = 1 : m
                eta_ki = M.transp(xHistory{i}, newx, etaHistory{i});
                psi_ki = (newgradnorm * M.norm(newx, eta_ki) + M.inner(newx, newgrad, eta_ki) + l) / gammak;
                psi_dagger = 1 / psi_ki;
                if psi_dagger < 1e-8
                    psi_dagger = 0;
                end
                beta_ki = newgradnorm^2 * psi_dagger;
                desc_dir_sum = M.lincomb(newx, 1, desc_dir_sum, 1, M.lincomb(newx, beta_ki, eta_ki));
            end
            
            % Update the search direction by combining the negative gradient and the memory term.
            desc_dir = M.lincomb(newx, -gammak, Pnewgrad, 1/m, desc_dir_sum);
        end
        
        % Rotate stored iterates and directions, keeping the most recent m values.
        xHistory = xHistory([2:end, 1]);
        etaHistory = etaHistory([2:end, 1]);
        xHistory{m} = newx;
        etaHistory{m} = desc_dir;
        
    end
    
    % Transfer iterate information.
    storedb.removefirstifdifferent(key, newkey);
    x = newx;
    key = newkey;
    cost = newcost;
    grad = newgrad;
    Pgrad = Pnewgrad;
    gradnorm = newgradnorm;
    gradPgrad = newgradPnewgrad;
    
    % Increment iteration counter.
    iter = iter + 1;
    
    % Purge old entries from the store database to control memory usage.
    storedb.purge();
    
    % Log statistics for the current iteration.
    stats = savestats();
    info(iter+1) = stats;
    
end

info = info(1:iter+1);

if options.verbosity >= 1
    fprintf('Total time is %f [s] (excludes statsfun)\n', info(end).time);
end

% Routine to collect current iteration statistics.
function stats = savestats()
    stats.iter = iter;
    stats.cost = cost;
    stats.gradnorm = gradnorm;
    if iter == 0
        stats.stepsize = nan;
        stats.time = toc(timetic);
        stats.linesearch = [];
    else
        stats.stepsize = stepsize;
        stats.time = info(iter).time + toc(timetic);
        stats.linesearch = lsstats;
    end
    stats = applyStatsfun(problem, x, storedb, key, options, stats);
end

end
