function createstatspage(resultdir, imagenames)
% createstatspage(resultdir, imagenames)
%
% Creates a page containing all data from the statistics, including
% accuracy and ROC curves.
%
% The data should have been created beforehand using "createstats",
% otherwise this won't work. The data used to create the pages comes
% from the results in "resultdir" (created with "createstats"). The
% page created is kept in "resultdir".
%
% See also: createstats.

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

% Opening (creating) the html page.
fp = htopen([resultdir 'stats.html'], ['Statistics']);

htwtext(fp, '<h1>Statistics results</h1>');

htwtext(fp, '<h2>For all images</h2>');

htwtext(fp, '<h3>Accuracies and ROC areas</h3>');

% First, creates a table with results from each image.
for i = 1:size(imagenames, 2)
  imfilename = imagenames{i}.original;
  imfilenames{i} = imfilename;
  
  shortname = imfilename(1:(end-4));
 
  S = load([resultdir filesep shortname filesep 'stats.mat']);

  results(i,:) = S.stats;
end

% Adds global results.
S = load([resultdir filesep 'stats.mat']);
results = [results; S.stats];

% Outputs the table.
htwtable(fp, {imfilenames{:} 'total'}, S.description, results);

% Adds the global ROC graph.
htwtext(fp, '<h3>ROC curve</h3>');
htwtext(fp, '<img src="rocs.jpg" width = 75%>'); 

% Outputs stats and ROC graphs for images individually.
for i = 1:size(imfilenames, 2)
  imfilename = imfilenames{i};
  
  shortname = imfilename(1:(end-4));
  
  htwtext(fp, ['<h2>For image ' imfilename '</h2>']);
  htwtext(fp, '<h3>Accuracies and ROC areas</h3>');

  S = load([resultdir filesep shortname filesep 'stats.mat']);
  htwtable(fp, {imfilename}, S.description, S.stats);
  
  htwtext(fp, '<h3>ROC curve</h3>');
  htwtext(fp, ['<img src="' shortname filesep 'rocs.jpg"  width = 75%>']); 
end

% Close the page.
htclose(fp);
