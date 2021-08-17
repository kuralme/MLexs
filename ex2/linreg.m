%% Initialize
clear; close all; clc;

%% Read training data
e2=load('ex2data1.txt');

%% initialize matrices and variables
X = e2(:,(1:2));
y = e2(:, 3);
m = length(y);
initheta = zeros(3,1);
alpha = 0.01;

%% plot the data
% plotData(X,y);

%% adding x0 to X
X = [ones(m,1) X];

%% Compute cost J
[J, grad] = costFunction(initheta, X, y);

%% Run gradient descent
options = optimset('GradObj', 'on', 'MaxIter', 400);
theta = fminunc(@(t)costFunction(t, X, y), initheta, options);
% [theta, J_history] = gradientDescent(X, y, initheta, alpha, iterations);

%% Use theta and do prediction
[pred,acc] = predict(theta, X, y);
acc