% Function to compute scenario-based transition probabilities for the
% proposed iMDP abstraction.
%
% Code based on the procedure proposed in:
%
%   S. Garatti and M. Campi. Risk and complexity in scenario optimization. 
%   Mathematical Programming, pages 1–37, 2019.
% -------------------------------------------------------------------------

addpath('input/');

% Confidence level to create the table for
beta = 0.005;

% List of values for N (number of samples) to create table for
N_list = [25,50,100,200,400,800,1600,3200,6400];

% Loop over createTable function to create it
for N = 1:length(N_list)    
    createTable(N_list(N), beta)
end