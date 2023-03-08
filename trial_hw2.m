clc; clear; close all;

% %% Data Import
% 
% list = dir('ghg_data/*.dat');
% 
% for k=1:length(list)
%    FileNames=list(k).name;
%    A=readtable(['ghg_data/',FileNames]);
%    Z = table2array(A(1:15,:));
%    X(:,k) = Z(:);
%    Y(k,:)=table2array(A(16,:));   
% end

load('dataset.mat');

%% Preprocess
A = [X; Y'];

A = A ./ std(A(:));
%A = A - mean(A(:));

X_data = A(1:4905,:);
Y_data = A(4906:end,:);

%Y_data = Y_data./max(max(Y_data));


X_train = X_data(:,1:2500);
Y_train = Y_data(:,1:2500).';

X_test = X_data(:,2501:end);
Y_test = Y_data(:,2501:end).';

lambda = 0.01;
alpha = 1;
epsilon = 1e-3;
number_of_iterations = 1;

%tic
%[weights_SAG,cost_SAG,cost_test_SAG] = solve_stochastic_average_gradient_descent(X_train,Y_train,X_test,Y_test,lambda,alpha,epsilon, number_of_iterations);
%toc

tic
[weights_GD,cost_GD,cost_test_GD] = solve_gradient_descent(X_train,Y_train,X_test,Y_test,lambda,alpha,epsilon, 100);
toc

%tic
%[weights_SGD, cost_SGD, cost_test_SGD] = solve_stochastic_gradient_descent(X,Y,X_test,Y_test,lambda,alpha,epsilon, number_of_iterations);
%toc

