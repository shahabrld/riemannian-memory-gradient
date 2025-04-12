function [stepsize, newx, newkey, lsstats] = ...
             linesearch_wolfe(problem, x, d, f0, ~, options, storedb, key)
% Wolfe line-search based on the line-search hint in the problem structure.
%
% function [stepsize, newx, newkey, lsstats] = 
%            linesearch_wolfe(problem, x, d, f0, df0, options, storedb, key)
%
% The algorithm obtains an initial step size candidate from the problem
% structure, typically through the problem.linesearch function. If that
% step does not fulfill the Armijo sufficient decrease criterion, that step
% size is reduced geometrically until a satisfactory step size is obtained
% or until a failure criterion triggers. If the problem structure does not
% provide an initial alpha, then alpha = 1 is tried first.
% 
% Below, the step is constructed as alpha*d, and the step size is the norm
% of that vector, thus: stepsize = alpha*norm_d. The step is executed by
% retracting the vector alpha*d from the current point x, giving newx.

    % Allow omission of the key, and even of storedb.
    if ~exist('key', 'var')
        if ~exist('storedb', 'var')
            storedb = StoreDB();
        end
        key = storedb.getNewKey();
    end

    % Backtracking default parameters. These can be overwritten in the
    % options structure which is passed to the solver.
    default_options.ls_contraction_factor = .5;
    default_options.ls_c1 = 1e-4;
    default_options.ls_c2 = .1;
    default_options.ls_max_steps = 25;
    default_options.ls_backtrack = true;
    default_options.ls_force_decrease = true;
    
    if ~exist('options', 'var') || isempty(options)
        options = struct();
    end
    options = mergeOptions(default_options, options);
    
    c1 = options.ls_c1;
    c2 = options.ls_c2;
    max_ls_steps = options.ls_max_steps;
    
    % Make the chosen step and compute the cost there.
    newkey = storedb.getNewKey();
    cost_evaluations = 1;

    phi = @phi_fun;
    derphi = @derphi_fun;

    phi0 = phi(0);
    derphi0 = derphi(0);
    alpha0 = 0;
    alpha1 = 1;
    phi_a1 = phi(alpha1);
    phi_a0 = phi0;
    derphi_a0 = derphi0;
    
    % Check the Wolfe conditions.
    for e=1:max_ls_steps

        derphi_a1 = derphi(alpha1);

        if (phi_a1 > phi0 + c1 * alpha1 * derphi0) || ((phi_a1 >= phi_a0) && (e > 1))
            alpha_star = zoom(alpha0, alpha1, phi_a0, phi_a1, derphi_a0, phi, derphi, phi0, derphi0);
            break;
        end

        if derphi_a1 >= c2 * derphi0
            alpha_star = alpha1;
            break;
        end

        if (derphi_a1 >= 0)
            alpha_star = zoom(alpha1, alpha0, phi_a1, phi_a0, derphi_a1, phi, derphi, phi0, derphi0);
            break;
        end

        alpha2 = 2 * alpha1;
        alpha0 = alpha1;
        alpha1 = alpha2;
        phi_a0 = phi_a1;
        phi_a1 = phi(alpha1);
        derphi_a0 = derphi_a1;

        cost_evaluations = cost_evaluations + 1;
        
        % Make sure we don't run out of budget.
        if cost_evaluations > max_ls_steps
            alpha_star = alpha1;
            break;
        end
        
    end

    % Reduce the step size,
    alpha = alpha_star;
    
    % and look closer down the line.
    storedb.remove(newkey);              % we no longer need this cache
    newx = problem.M.retr(x, d, alpha);
    newkey = storedb.getNewKey();
    newf = getCost(problem, newx, storedb, newkey);
    
    % If we got here without obtaining a decrease, we reject the step.
    if options.ls_force_decrease && newf > f0
        alpha = 0;
        newx = x;
        newkey = key;
        newf = f0; %#ok<NASGU>
    end
    
    % As seen outside this function, stepsize is the size of the vector we
    % retract to make the step from x to newx. Since the step is alpha*d:
    norm_d = problem.M.norm(x, d);
    stepsize = alpha * norm_d;
    
    % Return some statistics also, for possible analysis.
    lsstats.costevals = cost_evaluations;
    lsstats.stepsize = stepsize;
    lsstats.alpha = alpha;

    function val = phi_fun(alpha)
        updated_x = problem.M.retr(x, d, alpha);
        val = problem.cost(updated_x);
    end

    function der = derphi_fun(alpha)
        updated_x = problem.M.retr(x, d, alpha);
        updated_direction = problem.M.transp(x, updated_x, d);
        g = getGradient(problem, updated_x);
        der = problem.M.inner(updated_x, g, updated_direction);
    end

    function a_star = zoom(a_lo, a_hi, phi_lo, phi_hi, derphi_lo, phi, derphi, phi0, derphi0)
        i = 0;
        delta1 = .2;
        delta2 = .1;
        phi_rec = phi0;
        a_rec = 0;

        while true

            dalpha = a_hi - a_lo;
            if dalpha < 0
                a = a_hi;
                b = a_lo;
            else
                a = a_lo;
                b = a_hi;
            end
            if (i > 0)
                cchk = delta1 * dalpha;
                a_j = cubicmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi, a_rec, phi_rec);
            end
            if (i == 0) || (isempty(a_j)) || (a_j > b-cchk) || (a_j < a+cchk)
                qchk = delta2 * dalpha;
                a_j = quadmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi);
                if (isempty(a_j)) || (a_j > b-qchk) || (a_j < a+qchk)
                    a_j = a_lo + 0.5*dalpha;
                end
            end
            phi_aj = phi(a_j);
            if (phi_aj > phi0 + c1*a_j*derphi0) || (phi_aj >= phi_lo)
                phi_rec = phi_hi;
                a_rec = a_hi;
                a_hi = a_j;
                phi_hi = phi_aj;
            else
                derphi_aj = derphi(a_j);
                if derphi_aj >= c2*derphi0
                    a_star = a_j;
                    break;
                end
                if derphi_aj*(a_hi - a_lo) >= 0
                    phi_rec = phi_hi;
                    a_rec = a_hi;
                    a_hi = a_lo;
                    phi_hi = phi_lo;
                else
                    phi_rec = phi_lo;
                    a_rec = a_lo;
                end
                a_lo = a_j;
                phi_lo = phi_aj;
                derphi_lo = derphi_aj;
            end
            i = i + 1;
            if (i > 10)
                a_star = a_j;
                break;
            end
        end
    end
    
    function xmin = cubicmin(a, fa, fpa, b, fb, c, fc)
        try
            db = b - a;
            dc = c - a;
            d1 = zeros(2);
            d1(1, 1) = dc ^ 2;
            d1(1, 2) = -db ^ 2;
            d1(2, 1) = -dc ^ 3;
            d1(2, 2) = db ^ 3;
            D = d1 * [fb - fa - fpa * db; fc - fa - fpa * dc];
            denom = (db * dc) ^ 2 * (db - dc);
            D(1) = D(1) / denom;
            D(2) = D(2) / denom;
            radical = D(2) * D(2) - 3 * D(1) * fpa;
            xmin = a + (-D(2) + sqrt(radical)) / (3 * D(1));
        catch
            xmin = [];
        end
    end

    function xmin = quadmin(a, fa, fpa, b, fb)
        try
            db = b - a * 1;
            B = (fb - fa - fpa * db) / (db * db);
            xmin = a - fpa / (2 * B);
        catch
            xmin = [];
        end
    end

end
