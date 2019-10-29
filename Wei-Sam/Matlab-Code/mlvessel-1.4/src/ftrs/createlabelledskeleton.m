function mask = createlabelledskeleton(featurefilename, skeletonfilename,...
                                       outputfilename, roisize)
% mask = createlabelled(featurefilename, skeletonfilename,...
%                       outputfilename, roisize)
%
% Opens the features from an image from "featurefilename" and writes
% each pixel's features in file "outputfilename", along with the
% pixel's labels. "skeletonfilename" is the image with a skeleton from
% which labels are produced, which will be used for classifier
% training. The pixels forming the labelled file are determined by a
% rectangle in random position of size (x * "roisize", y * "roisize"),
% where (x, y) is the original image size. All other pixels will be
% ignored. The region used is returned in mask.
%
% mask = createlabelled(featurefilename, skeletonfilename, outputfilename)
%		         
% Opens the features from an image from "featurefilename" and writes
% each pixel's features in file "outputfilename", along with the
% pixel's labels. "skeletonfilename" is the image with the skeleton
% for the generation of labels, which will be used for classifier
% training.

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

% Creating the matrix with the labels.
class = imread(skeletonfilename);

if ( size(class, 3) == 3 )
  class = rgb2gray(class);
end  
class = class > 128;
% The vessels are a region of one pixel around the segmentation.
vessels = imdilate(class, strel('disk', 1, 0));
% The rest is 15 pixels away from the vessels.
rest = ~imdilate(class, strel('disk', 15, 0));

% Creates the mask with a random part of the image.
if ( nargin == 4 )
  rectmask = logical(zeros(size(class)));
  width = round(size(class, 2) * roisize) - 1;
  height = round(size(class, 1) * roisize) - 1;

  x0 = floor( rand(1) * (size(class, 2) - width) ) + 1;
  y0 = floor( rand(1) * (size(class, 1) - height) ) + 1;
  rectmask(y0:y0 + height, x0:x0 + width ) = logical(1);

  mask = F.mask & rectmask;
else % Normal mask.
  mask = F.mask;
end

% Adds labels to features.
features = cat(3, double(vessels), F.features);
description = strvcat('class', F.description);

% Arranges features in a matrix, taking only pixels from the region of
% interest.
[rows, columns, pages] = size(features);
featurematrix = reshape(features, [rows * columns, pages]);
featurematrix = featurematrix(mask(:) & (rest(:) | vessels(:)), :);

% Saves in "outputfilename" in matlab format.
save(outputfilename, 'description', 'featurematrix', '-MAT');
