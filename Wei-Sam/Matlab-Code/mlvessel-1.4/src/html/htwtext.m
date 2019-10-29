function htwtext(fp, string) 
%HTWTEXT Insert the string 'string' in the HTML file pointed by FP. 
%  HTWTEXT(FP, 'string') inserts the string 'string' in HTML format 
%  on the file pointed by FP. 
% 
%  This functin does not support special characters like '\n', '\t', etc. 
% 
%  Example: 
%  htwtext(fp,'This is HTML Toolbox for Matlab. Enjoy!') 
% 
%  See also HTOPEN, HTCLOSE, HTWMATRIX, HTWTEXT, HTWPAR, HTWURL 
%  
%  Further Information:  
%       http://www.vision.ime.usp.br/~casado/matlab/htmltoolbox/  
%  
%   Copyright (c) 2000 by Andre Casado Castano. 
%   e-mail:casado@vision.ime.usp.br  
%   $Revision: 1.1.1.1 $  $Date: 2006-02-01 19:06:51 $ 
 
fprintf(fp,'%s \n', string); 
