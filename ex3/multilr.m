% Multi-class logistic regression classification
%% Initialize
clear; close all; clc;

%% Read training data
load('ex3data1.mat');

%% Display training data
% displayData(X,20)

%% Multi-class logistic regression
lambda = 0.2;
K = 10;
[all_theta] = oneVsAll(X, y, K, lambda);
classprediction = predictOneVsAll(all_theta, X);
[M, I]= max(classprediction,[],2);
accuracy = mean( double(I ==y) );

