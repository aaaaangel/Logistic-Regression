%% Load data
%  X=[x; 1]  (attributes_num+1, samples_num)
%  Y: 1=malignant, 0=benign    (1, samples_num)
dataset = importdata('wdbc.data');
samples_num = size(dataset.data, 1);
attributes_num = size(dataset.data, 2);
X = [dataset.data, ones(samples_num, 1)]';
Y = double(cell2mat(dataset.textdata(:,2)) == 'M')';

%% Normalization & SMOTE
%  X is converted to have a mean of 0 and a standard deviation of 1.
%  SMOTE is used to solve class imbalance problems.
X = zscore(X')';
X1 = X(:, Y==1);
X0 = X(:, Y==0);
% Divide the test set.
temp_cnt = floor(samples_num*0.1);
test_samples_num = 2*temp_cnt;
X_test = [X1(:, 1:temp_cnt), X0(:, 1:temp_cnt)];
Y_test = [ones(1, temp_cnt), zeros(1, temp_cnt)];
% Apply SMOTE to class with fewer samples.
X1_train = X1(:, temp_cnt+1:end);
X0_train = X0(:, temp_cnt+1:end);
instances_per_class = max(size(X1_train, 2), size(X0_train, 2));
train_samples_num = 2*instances_per_class;
if size(X1_train, 2)<instances_per_class
    X1_train = SMOTE(X1_train, instances_per_class-size(X1_train, 2));
elseif size(X0_train, 2)<instances_per_class
    X0_train = SMOTE(X0_train, instances_per_class-size(X0_train, 2));
end
X_train = [X1_train, X0_train];
Y_train = [ones(1, instances_per_class), zeros(1, instances_per_class)];

%% Logistic calculation formula
P1 = inline('exp(W''*X)./(exp(W''*X)+1)', 'W', 'X');
P0 = inline('1./(exp(W''*X)+1)', 'W', 'X');
loss_func = inline('sum(-Y.*(W''*X)+log(1+exp(W''*X)))', 'W', 'X', 'Y');
loss_gradient1 = inline('-sum(X.*(Y-exp(W''*X)./(exp(W''*X)+1)), 2)', 'W', 'X', 'Y');

%% Gradient descent
% W is initialized as rand value in [0, 1]
learning_rate = 0.001;
itrations_num = 100;
W = rand(attributes_num+1, 1); 
for i=0:itrations_num
    loss = loss_func(W, X_train, Y_train);
    train_acc = sum(P1(W,X_train)>0.5==Y_train)/train_samples_num;
    test_acc = sum(P1(W,X_test)>0.5==Y_test)/test_samples_num;
    if rem(i, 10) == 0
        fprintf("[%d/%d]:\tloss: %.4f   train accuracy: %.4f   test accuracy: %.4f\n", ... 
        i, itrations_num, loss, train_acc, test_acc);
    end
    W = W - learning_rate*loss_gradient1(W, X_train, Y_train);
end

%% Performance metric
TP = sum(P1(W,X1(:, 1:temp_cnt))>0.5==ones(1, temp_cnt));
FN = temp_cnt - TP;
TN = sum(P0(W,X0(:, 1:temp_cnt))>0.5==ones(1, temp_cnt));
FP = temp_cnt - TN;
precision = TP/(TP+FP);
recall = TP/(TP+FN);
beta = 1.2;
F_measure = (1+beta*beta)*precision*recall/(beta*beta*precision+recall);
fprintf("Precision: %.4f   Recall: %.4f   F measure: %.4f(beta=1.2)\n", ...
precision, recall, F_measure);

%% Visualization
%  Choose the first 3 attributes to perform visualization
x = X(1, :);
y = X(2, :);
z = X(3, :);
figure;
[wx,wy]=meshgrid(-3:5,-3:5);
wz = (0.5-W(1,1)*wx-W(2,1)*wy)./W(3,1);
mesh(wx, wy, wz);
hold on;
for i = 1:samples_num
    if Y(i)==1
        color = 'r*';
    else
        color = 'bo';
    end
    plot3(x(i), y(i), z(i), color);
end
grid on;
view(3);

