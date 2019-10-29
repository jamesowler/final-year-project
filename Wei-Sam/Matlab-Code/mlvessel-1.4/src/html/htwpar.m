function htwpar(fp)
%HTWPAR Make a paragraph in an HTML document.
%  HTWPAR(FP) inserts a paragraph mark in the HTML document 
%  pointed by FP.
%
%  Example:
%  htwpar(fp)
%
%  See also HTOPEN, HTCLOSE, HTWMATRIX, HTWTEXT, HTWHR, HTWHANDLE
%
%  Further Information: 
%       http://www.vision.ime.usp.br/~casado/matlab/htmltoolbox/
%
%   Copyright (c) 2000 by Andre Casado Castano.
%   e-mail:casado@vision.ime.usp.br
%   $Revision: 1.1.1.1 $  $Date: 2006-02-01 19:06:51 $

fprintf(fp,'<p>\n');