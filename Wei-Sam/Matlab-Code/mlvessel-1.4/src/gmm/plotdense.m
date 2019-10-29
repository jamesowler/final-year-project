function gray = plotdense(X, Y, scale, n)
% gray = plotdense(X, Y, scale, n)
%
% Plots the points in "X" and "Y", creating the density as the number
% of points in each square region. "X" and "Y" are column vectors and
% should have the same size. "scale" is [XMIN XMAX YMIN YMAX] and
% fixes the axes for the image. "n" is [nx ny], the resolution of the
% image created.

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

nx = n(1);
ny = n(2);

gray = zeros([ny, nx]);

XMIN = scale(1);
XMAX = scale(2);
YMIN = scale(3);
YMAX = scale(4);

DX = XMAX - XMIN;
DY = YMAX - YMIN;

x = 1 + floor( (X - XMIN) / DX * (nx - 1) );
y = 1 + floor( (Y - YMIN) / DY * (ny - 1) );

for i = 1:size(X, 1)
  if (x(i) > 0 & x(i) <= nx & y(i) > 0 & y(i) <= ny)
    gray(y(i), x(i)) = gray(y(i), x(i)) + 1;
  end
end