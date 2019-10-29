function class = gmmclassify(featurefilename, modelfilename)
% class = gmmclassify(featurefilename, modelfilename)
%
% Classifies the pixels in "featurefilename" using the gaussian
% mixture models in "modelfilename". Returns "class", the image
% with each pixel's class.
%
% See also: gmmclassifygray, gmmlikelihoods, gmmcreatemodel.

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

classgray =  gmmclassifygray(featurefilename, modelfilename);

class = classgray >= 0.5;