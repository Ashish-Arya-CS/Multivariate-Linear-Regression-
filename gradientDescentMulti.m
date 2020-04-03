function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
temp1=X*theta;
temp2=temp1-y;
t0=temp2'*X(:,1);
t1=temp2'*X(:,2);
t3=temp2'*X(:,3);
temp3=theta(1,1)-(alpha/m)*t0;
temp4=theta(2,1)-(alpha/m)*t1;
temp5=theta(3,1)-(alpha/m)*t3;
theta(1,1)=temp3;
theta(2,1)=temp4;
theta(3,1)=temp5;


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
