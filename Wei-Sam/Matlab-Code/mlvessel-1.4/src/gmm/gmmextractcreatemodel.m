function gmmextractcreatemodel(processedfilename, projectionfilename, ...
                               modelfilename, kvessel, krest, type, ndims)
% function gmmextractcreatemodel(processedfilename, projectionfilename, ...
%                                modelfilename, kvessel, krest, type, ndims)
%
% Creates the mixture models using samples in "processedfilename" and
% saving everything in "modelfilename".  The model should have
% "kvessel" gaussians representin vessel pixel samples and "krest"
% gaussians representin "non-vessels". "type" specifies the type of
% covariance matrix for the gaussians.
%
% "projectionfilename" contains the projection applied before creating
% the model. The projection is done into "ndims" dimensions.
%
% See also: createprocessed, gmmcreatemodel, gmmmodel.

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

% Gets projection from file.
P = load(projectionfilename);

projectionmatrix = P.projectionmatrix;

projectioninfo = P.info;

clear P;

T = load(processedfilename);

% Projects training features.
projectedmatrix = [T.featurematrix(:,1) ...
                   T.featurematrix(:,2:end) * projectionmatrix(:,1:ndims)];

description = T.description;

samplemean = T.samplemean;

samplestd = T.samplestd;

clear T;

% "vessels" indicates vessel pixels in featurematrix.
vessels = (projectedmatrix(:, 1) == 1);

% Matrix with vessel samples.
vesselmatrix = projectedmatrix(vessels, 2:end);
% Matrix with non-vessel samples.
restmatrix = projectedmatrix(~vessels, 2:end);

% Number of vessel and non-vessel samples.
nvessel = size(vesselmatrix, 1);
nrest = size(restmatrix, 1);
ntotal = nvessel + nrest;

% Creating the mixture model for the vessels.
[vesselgaussians, vesselQ] = gmmmodel(vesselmatrix, kvessel, type);

% Creating the mixture model for non-vessels.
[restgaussians, restQ] = gmmmodel(restmatrix, krest, type);

% Priors.
vesselprior = nvessel / ntotal;
restprior = nrest / ntotal;

vesselit = size(vesselQ, 2);
restit = size(restQ, 2);

% Saves some info on parameters.
info = strvcat([' File used for training: ' processedfilename],...
               [' kvessel = ' num2str(kvessel)],...
               [' krest = ' num2str(krest)],...
               [' Iterations for vessels: ' num2str(vesselit)],...
               [' Iterations for rest: ' num2str(restit)],...
               [' Covariance matrix type: ' type],...
               [' File used for projection: ' projectionfilename],...
               [' Number of dimensions projected: ' num2str(ndims)], ...
               [' About projection: '], projectioninfo);

% Saves in "modelfilename" in matlab format.
save(modelfilename, 'description', 'info', 'vesselprior', ...
     'vesselgaussians', 'vesselQ', 'ndims', 'projectionmatrix', ...
     'restprior', 'restgaussians', 'restQ', 'samplemean', 'samplestd', ...
     '-MAT');