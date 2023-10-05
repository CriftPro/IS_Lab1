% Classification using perceptron

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
% estimated features are stored in matrix P:
P=[x1;x2];

%Desired output vector
T=[1;1;1;0;0]; % <- ČIA ANKSČIAU BUVO KLAIDA!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

%% train single perceptron with two inputs and one output

% generate random initial values of w1, w2 and b
w1 = randn(1);
w2 = randn(1);
b = randn(1);

disp( "Generated weights : w1 = " + w1 + ", w2 = " + w2 + " , b = " + b);



eta = 0.1;
sum_er = 1;
n = 1;

w1(n) = w1;
w2(n) = w2;
b(n) = b;

max_iteration = 1000;
iteration = 0;

while iteration <= max_iteration
    sum_er = 0;
    for i = 1 : 5
        v = x1(i)*w1(n) + x2(i)*w2(n) + b(n);
        if v > 0
	    y = 1;
        else
	    y = 0;
        end
        e = T(i) - y;
        sum_er = sum_er + e;

        % Update weights
        w1(n+1) = w1(n) + eta * e * x1(i);
        w2(n+1) = w1(n) + eta * e * x2(i);
        b(n+1) = b(n) + eta * e; 

        n = n + 1;
    end

    if sum_er == 0
%   Test how good are updated parameters (weights) on all examples used for training
%   calculate outputs and errors for all 5 examples using current values of the parameter set {w1, w2, b}
%   calculate 'v1', 'v2', 'v3',... 'v5'
    
        s = 5;
        v = size(s);
        e2 = size(s);

        for i = 1:s 
            v(i) = x1(i)*w1(n) + x2(i)*w2(n) + b(n);
            if v(i) > 0
	            y = 1;
            else
	            y = 0;
            end
            e2(i) = T(i) - y;
        end

    
	    % calculate the total error for these 5 inputs 
	    et = abs(e2(1)) + abs(e2(2)) + abs(e2(3)) + abs(e2(4)) + abs(e2(5))
        if et == 0
           disp("Solution found, iterations took " + iteration + ", Current errors : " + et);
           break;
        end
    end 
    iteration = iteration + 1;
    
    disp("Iteration " + iteration + " Errors : " + sum_er);
end

disp( "Chosen weights : w1 = " + w1(n) + ", w2 = " + w2(n) + " , b = " + b(n));

%   Test how good are updated parameters (weights) on all examples used for training
%   calculate outputs and errors for all 5 examples using current values of the parameter set {w1, w2, b}
%   calculate 'v1', 'v2', 'v3',... 'v5'
    
    s = 5;
    v = size(s);
    e2 = size(s);

    for i = 1:s 
        v(i) = x1(i)*w1(n) + x2(i)*w2(n) + b(n);
        if v(i) > 0
	    y = 1;
        else
	    y = 0;
        end
        e2(i) = T(i) - y;
    end

    
	% calculate the total error for these 5 inputs 
	et = abs(e2(1)) + abs(e2(2)) + abs(e2(3)) + abs(e2(4)) + abs(e2(5));
    
    disp( "Testing weights, total error = " + et);