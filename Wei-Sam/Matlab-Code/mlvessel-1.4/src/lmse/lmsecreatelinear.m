function lmsecreatelinear(processedfilename, linearfilename)
% function lmsecreatelinear(processedfilename, linearfilename)
%
% Creates linear classifier.
%
% Creates the linear classifier function based on samples from
% "processedfilename". Saves the classifier in "linearfilename". The
% file "processedfilename" should have been created using
% "createprocessed" and contains normalized labeled training samples.
%
% The function minimizes the sum of squared errors from the desired
% output and real output on the training set. The training is the
% using the pseudo-inverse method and assigning labels 1 and -1 for
% desired output of classes 0 and 1.
%
% See also: testmixed, lmseapply, createprocessed.

%
% Copyright (C) 2006  João Vitor Baldini Soares
%
% This program is free software; you can redistribute it and/or
% modify it under the terms of the GNU General Public License
% as published by the Free Software Foundation; either version 2
% of the License, or (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program; if not, write to the Free Software
% Foundation, Inc., 51 Franklin Street, Fifth Floor,
% Boston, MA 02110-1301, USA.
%

T = load(processedfilename);

description = T.description;

samplemean = T.samplemean;

samplestd = T.samplestd;

vessels = (T.featurematrix(:,1) == 1);

% Extending feature matrix with ones.
T.featurematrix = [T.featurematrix ones(size(vessels))];

% Creating desired output vector.
y = ones(size(vessels));
y(~vessels) = -1;

% Pseudo-inverse to find best vector Y.
P = pinv(T.featurematrix(:, 2:end));
Y = P * y;

% Normalizing the projection vector.
Y = Y / norm(Y);

% Decomposing Y = [X, th].
X = Y(1:(end - 1));
th = - Y(end);

% Saves some info on parameters.
info = [' File used for training: ' processedfilename];

% Saves in "linearfilename" in matlab format.
save(linearfilename, 'description', 'info', 'X', 'th', 'samplemean', ...
     'samplestd', '-MAT');