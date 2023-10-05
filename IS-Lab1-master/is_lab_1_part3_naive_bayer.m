% Naive Bayes Classifier


% Setting up data

% Reading apple images
A1=imread('apple_04.jpg');
A2=imread('apple_05.jpg');
A3=imread('apple_06.jpg');
A4=imread('apple_07.jpg');
A5=imread('apple_11.jpg');
A6=imread('apple_12.jpg');
A7=imread('apple_13.jpg');
A8=imread('apple_17.jpg');
A9=imread('apple_19.jpg');

% Reading pears images
P1=imread('pear_01.jpg');
P2=imread('pear_02.jpg');
P3=imread('pear_03.jpg');
P4=imread('pear_09.jpg');

% Calculate for each image, colour and roundness
% For Apples
% 1st apple image(A1)
hsv_value_A1=spalva_color(A1); %color
metric_A1=apvalumas_roundness(A1); %roundness
% 2nd apple image(A2)
hsv_value_A2=spalva_color(A2); %color
metric_A2=apvalumas_roundness(A2); %roundness
% 3rd apple image(A3)
hsv_value_A3=spalva_color(A3); %color
metric_A3=apvalumas_roundness(A3); %roundness
% 4th apple image(A4)
hsv_value_A4=spalva_color(A4); %color
metric_A4=apvalumas_roundness(A4); %roundness
% 5th apple image(A5)hsv_value_A5=spalva_color(A5); %color
metric_A5=apvalumas_roundness(A5); %roundness
% 6th apple image(A6)
hsv_value_A6=spalva_color(A6); %color
metric_A6=apvalumas_roundness(A6); %roundness
% 7th apple image(A7)
hsv_value_A7=spalva_color(A7); %color
metric_A7=apvalumas_roundness(A7); %roundness
% 8th apple image(A8)
hsv_value_A8=spalva_color(A8); %color
metric_A8=apvalumas_roundness(A8); %roundness
% 9th apple image(A9)
hsv_value_A9=spalva_color(A9); %color
metric_A9=apvalumas_roundness(A9); %roundness

%For Pears
%1st pear image(P1)
hsv_value_P1=spalva_color(P1); %color
metric_P1=apvalumas_roundness(P1); %roundness
%2nd pear image(P2)
hsv_value_P2=spalva_color(P2); %color
metric_P2=apvalumas_roundness(P2); %roundness
%3rd pear image(P3)
hsv_value_P3=spalva_color(P3); %color
metric_P3=apvalumas_roundness(P3); %roundness
%2nd pear image(P4)
hsv_value_P4=spalva_color(P4); %color
metric_P4=apvalumas_roundness(P4); %roundness

%selecting features(color, roundness, 3 apples and 2 pears)
%A1,A2,A3,P1,P2
%building matrix 2x5
x1=[hsv_value_A1 hsv_value_A2 hsv_value_A3 hsv_value_P1 hsv_value_P2];
x2=[metric_A1 metric_A2 metric_A3 metric_P1 metric_P2];


% Split the data into a training set and a testing set (e.g., 80% training, 20% testing)
rng(42); % Set a random seed for reproducibility
X = [x1, x2]; % Combine your features into a single matrix
Y = T; % Your labels (1 for apple, 0 for pear)
[trainX, testX, trainY, testY] = split_data(X, Y, 0.8);

% Train a Naive Bayes Classifier
nb = fitcnb(trainX, trainY);

% Predict on the test data
predictedY = predict(nb, testX);

% Evaluate the classifier's performance
confusionMatrix = confusionmat(testY, predictedY);
accuracy = sum(diag(confusionMatrix)) / sum(confusionMatrix(:));
precision = confusionMatrix(1,1) / sum(confusionMatrix(:,1));
recall = confusionMatrix(1,1) / sum(confusionMatrix(1,:));
f1_score = 2 * (precision * recall) / (precision + recall);

fprintf('Accuracy: %.2f%%\n', accuracy * 100);
fprintf('Precision: %.2f%%\n', precision * 100);
fprintf('Recall: %.2f%%\n', recall * 100);
fprintf('F1 Score: %.2f\n', f1_score);

% Helper function to split data into training and testing sets
function [trainX, testX, trainY, testY] = split_data(X, Y, train_ratio)
    num_samples = size(X, 1);
    num_train = round(num_samples * train_ratio);

    indices = randperm(num_samples);
    train_indices = indices(1:num_train);
    test_indices = indices(num_train+1:end);

    trainX = X(train_indices, :);
    testX = X(test_indices, :);
    trainY = Y(train_indices);
    testY = Y(test_indices);
end