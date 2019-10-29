function mask = createretinamaskangiogram(img)
% mask = createretinamaskangiogram(img)
%
% Creates a region of interest indicating the area inside the camera's
% aperture. This function is for use with angiographies.

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

% Empirical threshold.
th = 20;  
mask = img >= th;

% Removes spurious regions.
mask = bwareaopen(mask, 200000);

% Fills in holes.
mask = bwareaclose(mask, 100);

% A minor erosion.
mask = imerode(mask, strel('diamond', 2));

function bw2 = bwareaclose(bw1, n)

bw2 = ~bwareaopen(~bw1, n);