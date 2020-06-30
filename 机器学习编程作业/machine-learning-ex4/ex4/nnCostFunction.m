function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network

%取出参数
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


% Setup some useful variables
m = size(X, 1); % m = 5000
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
%%%%%%no for-loop版本%%%%%%%%%
%Part 1: implement cost function computation
%------------------------------------------------------------------------------------

%把y初始化为高维向量ylabel
ylable = zeros(num_labels, m);   %10x5000
for i = 1:m
    ylable(y(i),i) = 1;
end

%第一层输入 
a1 = [ones(m,1) X];    %5000x401

%第二层输入为z2，输出a2
z2 = a1 * Theta1';     %5000x25
a2 = sigmoid(z2);     

%a2加上偏差作为第三层输入，输出a3 
a2 = [ones(m,1) a2];      %5000x26
a3 = sigmoid(a2 * Theta2');        %5000x10


%计算J
J = 1 / m * sum( sum( -ylable'.* log(a3) -  (1-ylable').*log(1-a3) ));

regularized = lambda/(2*m) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)) ); 

J = J + regularized;

%------------------------------------------------------------------------------------

%Part 2: Implement the backpropagation algorithm
%------------------------------------------------------------------------------------
delta3 = a3-ylable';            %5000x10 

delta2 = delta3 * Theta2 ;  %5000x26
delta2 = delta2(:,2:end);    %5000x25
delta2 = delta2 .* sigmoidGradient(z2);  %5000x25 

Delta_1 = zeros(size(Theta1));        %25x401
Delta_2 = zeros(size(Theta2));         %10x26

Delta_1 = Delta_1 + delta2' * a1 ;    %25x401
Delta_2 = Delta_2 + delta3' * a2 ;     %10x26

Theta1_grad = 1/m * Delta_1;   
Theta2_grad = 1/m * Delta_2;  
%------------------------------------------------------------------------------------ 


% Part3: Implement regularization
%------------------------------------------------------------------------------------ 

regularized_1 = lambda/m * Theta1;  
regularized_2 = lambda/m * Theta2;  
% j = 0 is not need to do the regularization
regularized_1(:,1) = zeros(size(regularized_1,1),1);  
regularized_2(:,1) = zeros(size(regularized_2,1),1);  

Theta1_grad = Theta1_grad + regularized_1;  
Theta2_grad = Theta2_grad + regularized_2; 
% -----------------------------------------------------------------------------------

% =========================================================================


%%%%%%for-loop版本%%%%%%%%%
ylable = zeros(num_labels, m);   %10x5000
for i = 1:m
    ylable(y(i),i) = 1;
end


a1 = [ones(m,1) X]; %5000x401

z2 = Theta1 * a1'; %25x5000
a2 = sigmoid(z2); %25x5000

a2 = [ones(1,m); a2]; %26x5000
z3 = Theta2 * a2;  %10x5000
a3 = sigmoid(z3); %10x5000

for i=1:m,    % m = 5000
	%J += sum( -1 * ylable(:,i) .* log(a3(:,i)) - (1-ylable(:,i)) .* log(1-a3(:,i)));
 	J += sum(-1*ylable(:,i).*log(a3(:,i))-(1-ylable(:,i)).*log(1-a3(:,i)));
end

J = J/m;

J = J + lambda*(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)))/2/m;



%Implement the backpropagation algorithm

Delta_1 = zeros(size(Theta1));
Delta_2 = zeros(size(Theta2));


for i=1:m,
	delta3 = a3(:,i) - ylable(:,i); %10x5000


	%26x10 * 10x5000 = 26x5000，从第二行开始计算
	delta2 = (Theta2' * delta3)(2:end,:).*sigmoidGradient(z2(:,i));

	Delta_2 += delta3 * a2(:,i)'; %10x26
	Delta_1 += delta2 * a1(i,:);
end

Theta1_grad  = Delta_1/m;
Theta2_grad = Delta_2/m;

%Implement regularization


Theta2_grad(:,2:end) = Theta2_grad(:,2:end) .+ lambda * Theta2(:,2:end) / m;
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) .+ lambda * Theta1(:,2:end) / m;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
