function htwimage3name(fp, outputdir, im, imfilename, imsizes)
%HTWTABLE Print the table mat in the HTML file pointed by fp 
%  
%  
% 
%  Outputs image img
%  
% 
%  See also HTOPEN, HTCLOSE, HTWHR, HTWTEXT, HTWPAR, HTWHANDLE 
% 
%  Further Information:  
%       http://www.vision.ime.usp.br/~casado/matlab/htmltoolbox/  
%  
%   Copyright (c) 2000 by Andre Casado Castano and Roberto M Cesar Jr. 
%                 2006, 2012 João Vitor Baldini Soares
%   e-mail:casado@vision.ime.usp.br  
%   $Revision: 1.1.1.1 $  $Date: 2006-02-01 19:06:51 $ 

if size(im, 3) == 3
    [lins,cols,z] = size(im);
else 
    [lins,cols] = size(im);
    im = double(im);
end

iconim = imresize(im, [round(lins/imsizes) round(cols/imsizes)]); 
iconimfilename = ['icon_' imfilename];

imwrite(iconim, [outputdir iconimfilename]);
imwrite(im, [outputdir imfilename]);
  
fprintf(fp,'<a href="%s">',imfilename);  
  
fprintf(fp,'<img src="%s">', iconimfilename);  
  
fprintf(fp,'</a>\n');