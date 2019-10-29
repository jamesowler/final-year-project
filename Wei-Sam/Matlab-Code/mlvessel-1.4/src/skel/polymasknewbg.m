function Pmask = polymasknewbg (img)
% polymasknewbg receives image IMG and returns a polygonal binary mask,
% with vertices collected interactively.

% Jorge de Jesus Gomes Leandro - 08/04/2003

% Cria a mascara a partir de IMG com perimetro em [Xper,Yper]
[Pmask,xi,yi] = roipoly(img);

Pmask=~Pmask;

