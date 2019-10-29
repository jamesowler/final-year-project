function [class, classgray] = lmseapply(featurefilename, linearfilename, ...
                                        normalizeflag)
% [class, classgray] = lmseapply(featurefilename, linearfilename, ...
%                                normalizeflag)
%
% [class, classgray] = lmseapply(featurefilename, linearfilename)
%
% Applies the linear classifier in "linearfilename" created using
% "lmsecreatelinear". Returns the classification result in "class" and
% the result of the linear function in "classgray". "featurefilename"
% is the name of the file with the image features. 
%
% The threshold is subtracted from "classgray", so that classgray > 0
% produces "class"."normalizeflag" is optional and is 0 if the same
% normalization is to be used for the training samples and the ones
% being classified (default is different normalization). 
%
% See also: testmixed, lmsecreatelinear.

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

F = load(featurefilename);

L = load(linearfilename);

% Normalizes features before classification.
[rows, columns, pages] = size(F.features);
F.features = reshape(F.features, [rows * columns, pages]);
samplematrix = F.features(F.mask(:), :);

% Verifies normalizeflag to see if same normalization should be used
% for the training samples and the ones being classified.
if (nargin > 2 & normalizeflag == 0)
  disp('Using same normalization for training and classification.');
  samplematrix = normfeats(samplematrix, L.samplemean, L.samplestd);
else
  samplematrix = normfeats(samplematrix);
end

% If projectionmatrix is present, projects the features.
if (isfield(L, 'projectionmatrix'))
  ndims = L.ndims;
  samplematrix = samplematrix * L.projectionmatrix(:, 1:ndims);
else
  ndims = pages;
end

% Applies the linear function.
classgray = zeros(size(F.mask));
classgray(F.mask) = samplematrix * L.X;

% Associates the lowest probability with the region outside the
% aperture.
low = min(classgray(F.mask(:)));
classgray(~F.mask) = low;

% Subtracting the threshold from classgray and producing class.
classgray = classgray - L.th;
class = classgray > 0;