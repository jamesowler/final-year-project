function c = int2symbol(ind)
% C = INT2SYMBOL(IND)
%
%  INT2SYMBOL looks up and returns a string representation for a
%  symbol corresponding to integer IND. For use with PLOT.
%
% See also PLOT

%
% Copyright (C) 2002  Roberto Marcondes Cesar Junior
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

switch (ind)
case 1,
 c= 'ko';
case 2,
 c= 'rs';
case 3,
 c= 'g^';
case 4,
 c= 'b*';
case 5,
 c= 'mx';
case 6,
 c= 'yh';
case 7,
 c= 'cv';
case 8,
  c= 'o';
case 9,
  c= '+';
otherwise,
 c= 'p';
end
