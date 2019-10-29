function htwhr(fp)
%HTWHR Insert an horizontal bar in the file pointed by FP.
%  HTWHR(FP) inserts an horizontal bar in HTML format on a file
%  pointed by FP.
%
%  Example:
%  htwhr(fp)

%  See also HTOPEN, HTCLOSE, HTWMATRIX, HTWTEXT, HTWPAR, HTWHANDLE
%
%  Further Information: 
%       http://www.vision.ime.usp.br/~casado/matlab/htmltoolbox/
%
%  This program is free software; you can redistribute it and/or
%  modify it under the terms of the GNU General Public License
%  as published by the Free Software Foundation; either version 2
%  of the License, or (at your option) any later version.
%
%  This program is distributed in the hope that it will be useful,
%  but WITHOUT ANY WARRANTY; without even the implied warranty of
%  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%  GNU General Public License for more details.
%
%  You should have received a copy of the GNU General Public License
%  along with this program; if not, write to the Free Software
%  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
%
%   Copyright (c) 2000 by Andre Casado Castano.
%   e-mail:casado@vision.ime.usp.br
%   $Revision: 1.1.1.1 $  $Date: 2006-02-01 19:06:51 $

fprintf(fp,'<hr>\n');