function smat=htwtable(fp, vheader, hheader, mat)
%HTWTABLE Print the table mat in the HTML file pointed by fp 
%  
%  
% 
%  Example that prints the matrix matvar in fp: 
%  
% 
%  See also HTOPEN, HTCLOSE, HTWHR, HTWTEXT, HTWPAR 
% 
%  Further Information:  
%       http://www.vision.ime.usp.br/~casado/matlab/htmltoolbox/  
%  
%   Copyright (c) 2000 by Andre Casado Castano and Roberto M Cesar Jr. 
%   e-mail:casado@vision.ime.usp.br  
%   $Revision: 1.1.1.1 $  $Date: 2006-02-01 19:06:51 $ 
 
[lin, col] = size(mat); 

for lins=1:lin
 for cols=1:col
  smat(lins,cols) = {num2str(mat(lins,cols), 4)};
 end
end

[vlin, vcol] = size( vheader );
if (vcol > 1)
 vheader = vheader';
 [vlin, vcol] = size( vheader );
end

xcol = {};
if (vlin == lin)
 smat = [vheader smat];
 xcol = {' '};
end

[hlin, hcol] = size( hheader );
if (hlin > 1)
 hheader = hheader';
 [hlin, hcol] = size( hheader );
end

if (hcol == col)
 hheader = [xcol hheader];
 smat = [hheader ; smat];
end

[lin, col] = size(smat);

fprintf(fp,'<table border="1">\n'); 
for i=1:lin 
   fprintf(fp,'  <tr align="center">\n'); 
   for j=1:col 
      fprintf(fp,'    <td> %s </td>\n', char(smat(i,j))); 
   end 
   fprintf(fp,'  </tr>\n'); 
end 

fprintf(fp,'</table>\n');

