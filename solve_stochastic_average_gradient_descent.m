function [weights, cost, cost_test] = solve_stochastic_average_gradient_descent(X,Y,X_test,Y_test,lambda,alpha,epsilon, number_of_iterations)


%% Weight initialization
    D = size(X,1);
    N = size(X,2);
    D_y = size(Y,2);
    weights = rand(D,D_y)*1e-2;

%% First iteration to get the gradients
gradient = calculate_logistic_gradient(X,Y,weights,lambda);
weights = weights - alpha.*gradient;
v_value = 0;    
%% Gradient descent iterations
err=1e2;
for iteration = 1:number_of_iterations
    if err > epsilon
        data_index = randi([1,N],1,1);
        x_selected = X(:,data_index);
        y_selected = Y(data_index,:);
        gradient_new = calculate_logistic_gradient(x_selected,y_selected,weights,lambda);
        gradient = gradient - v_value + gradient_new;
        v_value = gradient_new;
        weights = weights - alpha.*gradient;
        err = norm(gradient);
        cost(iteration) = logistic_cost_function(X, Y, weights, lambda);
        cost_test(iteration) = logistic_cost_function(X_test, Y_test, weights, lambda);
        iteration;
    end
    
end
    


end


