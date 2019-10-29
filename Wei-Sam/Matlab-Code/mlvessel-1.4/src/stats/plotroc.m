function handle = plotroc(rocfilenames)
% handle = plotroc(rocfilenames)
%
% Plots the ROC curves.
%
% Plots the ROC curves present in files named in cell array
% "rocfilenames". Returns "handle" to the plot's figure. The files
% used should follow the format of the files created using
% "createstats".
%
% See also: createstats, plot, figure.

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

% Collects curves from all files in G.
j = 1;
for i = 1:size(rocfilenames, 2);
  rocfilename = rocfilenames{i};
  F = load(rocfilename);
    
  for k = 1:size(F.fp, 2)
    G(j).fp = F.fp{k} / F.n(k);
    G(j).tp = F.tp{k} / F.p(k);
    G(j).description = F.description{k};
    
    j = j + 1;
  end
end

legends = [];

% Prepares the figure.
handle = figure('visible', 'off');
hold on; axis([0 1 0 1]); 
xlabel('false positive fraction'); 
ylabel('true positive fraction');
grid on;

% Plots each curve and adds legends.
for j = 1:size(G, 2)
  s = int2symbol(j);
  
  if (size(G(j).fp, 1) == 1)
    h = plot(G(j).fp(:), G(j).tp(:), [s]);
  else
    h = plot(G(j).fp(:), G(j).tp(:), [s(1) '-']);
  end
  legends = strvcat(legends, G(j).description);
end

legend(legends);
