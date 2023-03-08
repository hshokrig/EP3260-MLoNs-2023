function gradient_with_regul = calculate_logistic_gradient(X,Y,weights,lambda)
    D = size(X,1);
    N = size(X,2);
    gradient=0;
    for cnt_sample=1:N
        x_sample = X(:,cnt_sample);
        y_sample = Y(cnt_sample,:);
        gradient_of_sample = - (x_sample*y_sample)...
            ./(1+exp(y_sample*weights'*x_sample));
        gradient = gradient+gradient_of_sample;
    end
    
    gradient = gradient./N;
    gradient_with_regul = gradient + 2*lambda.*weights;
end