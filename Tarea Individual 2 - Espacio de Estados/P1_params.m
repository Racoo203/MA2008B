%% Problema 1
A = [0, 1, 0; 0, 0, 1; 0, -2, -3];
B = [0; 0; 1];
C = [1, 0, 0];
D = [0];

%% Problema 2
%A = [0, 1, 0; 0, 0, 1; -2, -5, -4];
%B = [-7; 19; -43];
%C = [1, 0, 0];
%D = [2];

%% Problema 3
%A = [0 1; 3 -1];
%B = [1; 1];
%C = [1 0];
%D = 2;

[num, den] = ss2tf(A, B, C, D);
G = tf(num, den);
disp('Transfer function G(s):');
display(G);

%% Problema 4
%syms z p s

%A = [0 1 0; 0 0 1; -z -1 -p];
%B = [0; 1; z-p];
%C = [1 0 0];
%D = 0;

%I = eye(size(A));
%G_s = simplify(C * inv(s * I - A) * B + D);

%disp('Transfer function G(s):');
%pretty(G_s)