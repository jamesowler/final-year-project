function [class, classgray] = knnclassify(featurefilename, ...
                                          processedfilename, k, normalizeflag)
% [class, classgray] = knnclassify(featurefilename,
%                                  processedfilename, k, normalizeflag)
%
% [class, classgray] = knnclassify(featurefilename, processedfilename, k)
%
% Classifies image pixels using k-nearest neighbors.
%
% Receives:
%
% "featurefilename": name of file with pixel image features to be
% classified.
%
% "processedfilename": normalized labeled processed training
% samples file.
%
% "k": Number of neighbors used for classification.
%
% "normalizeflag": optional parameter. Is 0 if the same normalization
% is to be used for the training samples and the ones being
% classified.
%
% Returns:
%
% "class": The final classification.
%
% "classgray": grayscale representation of votes for vessel.
%
% See also: textmixed.

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

F = load(featurefilename);

% Normalizes features before classification.
[rows, columns, pages] = size(F.features);
F.features = reshape(F.features, [rows * columns, pages]);
samplematrix = F.features(F.mask(:), :);

% Verifies normalizeflag to see if same normalization should be used
% for the training samples and the ones being classified.
if (nargin > 3 & normalizeflag == 0)
  disp('Using same normalization for training and classification.');
  samplematrix = normfeats(samplematrix, T.samplemean, T.samplestd);
else
  samplematrix = normfeats(samplematrix);
end

% Classifies each pixel, calculating its distande to each training
% sample.
trainmatrix = T.featurematrix(:, 2:end);
s = size(trainmatrix, 1);

gray = zeros(1, size(samplematrix, 1));

for i = 1:size(samplematrix, 1)
  sample = samplematrix(i, :);
  tempmatrix = trainmatrix -  sample(ones(1, s), :);
  dist = sum(tempmatrix.^2, 2);
  [ignore, index] = sort(dist);
  nvesselneighbors = sum(T.featurematrix(index(1:k), 1));
  gray(i) = nvesselneighbors;
  if(mod(i, 1000) == 0)
    disp(['Completed i = ' num2str(i)]);
  end
end

gray = gray / k;

classgray = zeros(size(F.mask));
classgray(F.mask(:)) = gray;

class = classgray > 0.5;
