function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = sigmoid((X)*(theta));
matr = -y.*(log(h)) - (1 - y).*(log(1 - h));
matri = theta.^2 ;
J = J + (sum(matr))/m + ((lambda)*(sum(matri) - (theta(1,1))^2))/(2*m);

grad(1,1) = grad(1,1) + (sum(h-y))/m;

for i=2:length(theta)
    jk = 0;
    for j=1:m
        jk = jk + (h(j,1)-y(j,1))*(X(j,i));
    end
    jk = jk/m;
    jk = jk + ((lambda)*(theta(i,1)))/m;
    grad(i,1) = grad(i,1) + jk;
end
% =============================================================

end
