function varargout = guimain(varargin)
% GUIMAIN Application M-file for guimain.fig
%    FIG = GUIMAIN launch guimain GUI.
%    GUIMAIN('callback_name', ...) invoke the named callback.
%
% Main interface window for mlvessel. Allows user to access
% classifiers and images.

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

if nargin == 0  % LAUNCH GUI

	fig = openfig(mfilename,'reuse');
    
	% Use system color scheme for figure:
	set(fig,'Color',get(0,'defaultUicontrolBackgroundColor'));
    % UserData is used as a permanent image counter.
    if isempty(get(fig, 'UserData'))
        data.imagecount = 0;
        data.children = [];
        set(fig, 'UserData', data);
    end

	% Generate a structure of handles to pass to callbacks, and store it. 
	handles = guihandles(fig);
	guidata(fig, handles);
            
	if nargout > 0
		varargout{1} = fig;
	end

elseif ischar(varargin{1}) % INVOKE NAMED SUBFUNCTION OR CALLBACK

	try
		if (nargout)
			[varargout{1:nargout}] = feval(varargin{:}); % FEVAL switchyard
		else
			feval(varargin{:}); % FEVAL switchyard
		end
	catch
		disp(lasterr);
	end

end


%| ABOUT CALLBACKS:
%| GUIDE automatically appends subfunction prototypes to this file, and 
%| sets objects' callback properties to call them through the FEVAL 
%| switchyard above. This comment describes that mechanism.
%|
%| Each callback subfunction declaration has the following form:
%| <SUBFUNCTION_NAME>(H, EVENTDATA, HANDLES, VARARGIN)
%|
%| The subfunction name is composed using the object's Tag and the 
%| callback type separated by '_', e.g. 'slider2_Callback',
%| 'figure1_CloseRequestFcn', 'axis1_ButtondownFcn'.
%|
%| H is the callback object's handle (obtained using GCBO).
%|
%| EVENTDATA is empty, but reserved for future use.
%|
%| HANDLES is a structure containing handles of components in GUI using
%| tags as fieldnames, e.g. handles.figure1, handles.slider2. This
%| structure is created at GUI startup using GUIHANDLES and stored in
%| the figure's application data using GUIDATA. A copy of the structure
%| is passed to each callback.  You can store additional information in
%| this structure at GUI startup, and you can change the structure
%| during callbacks.  Call guidata(h, handles) after changing your
%| copy to replace the stored original so that subsequent callbacks see
%| the updates. Type "help guihandles" and "help guidata" for more
%| information.
%|
%| VARARGIN contains any extra arguments you have passed to the
%| callback. Specify the extra arguments by editing the callback
%| property in the inspector. By default, GUIDE sets the property to:
%| <MFILENAME>('<SUBFUNCTION_NAME>', gcbo, [], guidata(gcbo))
%| Add any extra arguments after the last argument, before the final
%| closing parenthesis.



% --------------------------------------------------------------------
function varargout = menu_file_Callback(h, eventdata, handles, varargin)




% --------------------------------------------------------------------
function varargout = menu_file_image_Callback(h, eventdata, handles, varargin)




% --------------------------------------------------------------------
function newfigure = menu_file_image_open_Callback(h, eventdata, handles, varargin)

newfigure = [];
if length(varargin) == 0
    filterspec = {'*.jpg;*.jpeg;*.tif;*.tiff;*.gif;*.png;*.bmp', ...
            'All MATLAB Image Files (*.jpg,*.jpeg,*.tif,*.tiff,*.gif,*.png,*.bmp)'; ...
    		'*.jpg;*.jpeg', 'JPEG Files (*.jpg,*.jpeg)'; ...
    		'*.tif;*.tiff', 'TIFF Files (*.tif,*.tiff)'; ...
    		'*.gif', 'GIF Files (*.gif)'; ...
    		'*.png', 'PNG Files (*.png)'; ...
    		'*.bmp', 'Bitmap Files (*.bmp)'; ...
    		'*.*', 'All Files (*.*)'};
    [filename, pathname] = uigetfile(filterspec, 'Open image');
    if isequal(filename,0) | isequal(pathname,0)
    	return;
    end
else
    pathname = varargin{1};
    filename = varargin{2};
end

userdata = get(handles.figure1, 'UserData');
userdata.imagecount = userdata.imagecount + 1;

params.filename = filename;
params.pathname = pathname;
params.imagecount = userdata.imagecount;
params.mainfig = handles.figure1;

% Add new image to children.
child.figure = guiimage(params);
child.type = 'guiimage';
userdata.children = [userdata.children child];
set(handles.figure1, 'UserData', userdata);

newfigure = child.figure;


% --------------------------------------------------------------------
function varargout = menu_file_classifier_Callback(h, eventdata, handles, varargin)




% --------------------------------------------------------------------
function varargout = menu_file_classifier_new_Callback(h, eventdata, handles, varargin)
userdata = get(handles.figure1, 'UserData');
userdata.imagecount = userdata.imagecount + 1;

% Add new classifier to children.
params.mainfig = handles.figure1;
params.imagecount = userdata.imagecount;
params.action = 'new';

child.figure = guiclassifier(params);
child.type = 'guiclassifier';

userdata.children = [userdata.children child];
set(handles.figure1, 'UserData', userdata);



% --------------------------------------------------------------------
function newfigure = menu_file_classifier_open_Callback(h, eventdata, handles, varargin)

newfigure = [];
filterspec = {'*.cla', 'Classifiers (*.cla)'; ...
        '*.*', 'All Files (*.*)'};
[filename, pathname] = uigetfile(filterspec, 'Open classifier');

if isequal(filename,0) | isequal(pathname,0)
	return;
end

userdata = get(handles.figure1, 'UserData');
userdata.imagecount = userdata.imagecount + 1;

params.filename = filename;
params.pathname = pathname;
params.imagecount = userdata.imagecount;
params.mainfig = handles.figure1;
params.action = 'open';

% Add new classifier to children.
child.figure = guiclassifier(params);
child.type = 'guiclassifier';

userdata.children = [userdata.children child];
set(handles.figure1, 'UserData', userdata);

newfigure = child.figure;

% --------------------------------------------------------------------
function varargout = menu_file_exit_Callback(h, eventdata, handles, varargin)
guiexit(h, eventdata, handles, varargin);



% --------------------------------------------------------------------
function varargout = menu_help_Callback(h, eventdata, handles, varargin)




% --------------------------------------------------------------------
function varargout = menu_help_help_Callback(h, eventdata, handles, varargin)
uiwait(warndlg('Help not yet available. Sorry!','Warning','modal'));



% --------------------------------------------------------------------
function varargout = menu_help_about_Callback(h, eventdata, handles, varargin)
% Add about to children.
child.figure = guiabout;
child.type = 'guiabout';
add_child(h, eventdata, handles, child);


% --------------------------------------------------------------------
function varargout = figure1_CloseRequestFcn(h, eventdata, handles, varargin)
guiexit(h, eventdata, handles, varargin);



% --------------------------------------------------------------------
function varargout = figure1_CreateFcn(h, eventdata, handles, varargin)
% Opens an axes and adds the logo.
logo = imread([installdir filesep 'src' filesep 'gui' filesep 'logo.jpg']);
ax = axes;
image(logo);
set(ax, 'XTick', []);
set(ax, 'YTick', []);
set(ax, 'Position', [0.07 0.07 0.86 0.86]);


% --------------------------------------------------------------------
function segment(h, eventdata, handles, params)

% Verify presence of figures
if ~ishandle(params.classifierfigure)
  uiwait(errordlg('Could not find chosen classifier.','Error', ...
                 'modal'));
  return;
end

if ~ishandle(params.imagefigure)
  uiwait(errordlg('Could not find chosen image.','Error', ...
                 'modal'));
  return;
end

if ~isdeployed
  % The gaussian mixture model module.
  addpath([installdir filesep 'src' filesep 'gmm']);
  % The linear classifier module.
  addpath([installdir filesep 'src' filesep 'lmse']);
  % The knn classifier module.
  addpath([installdir filesep 'src' filesep 'knn']);
  % The feature manipulation module.
  addpath([installdir filesep 'src' filesep 'ftrs']);
end

tmpdir = [installdir filesep 'guitmp'];

% Create features file.
imagehandles = guidata(params.imagefigure);
classifierhandles = guidata(params.classifierfigure);

% Setting the name
imagefilename = imagehandles.filename;
[path,shortname,ext,version] = fileparts(imagefilename);
featurefilename = [tmpdir filesep shortname '-features.mat'];

% Getting the image.
img = imagehandles.original;

% Getting the classifier.
classifier = classifierhandles.classifier;

disp(['Creating features for ' shortname]);
guicreatefeatures(img, classifier.imagetype, ...
                  classifier.features, featurefilename);

% Saving classifier, then applying.
switch classifier.type
  case 'gmm'
  classifierfilename = [tmpdir filesep 'gmmclassifier.mat'];

  % Saves data in classifier file.
  vesselprior = classifier.vesselprior;
  restprior = classifier.restprior;
  vesselgaussians = classifier.vesselgaussians;
  restgaussians = classifier.restgaussians;
  samplemean = classifier.samplemean;
  samplestd = classifier.samplestd;

  save(classifierfilename, 'vesselprior', 'restprior', 'vesselgaussians', ...
       'restgaussians', 'samplemean', 'samplestd', '-MAT');
  
  gray = gmmclassifygray(featurefilename, classifierfilename);
  class = gray >= 0.5;
 case 'lmse'
  classifierfilename = [tmpdir filesep 'lmseclassifier.mat'];

  X = classifier.X;
  th = classifier.th;
  samplemean = classifier.samplemean;
  samplestd = classifier.samplestd;

  save(classifierfilename, 'X', 'th', 'samplemean', 'samplestd');
  
  [class, gray] = lmseapply(featurefilename, classifierfilename);
  % Fixes gray for visualization.
  gray = (gray + 1) / 2;
 case 'knn'
  classifierfilename = [tmpdir filesep 'knnclassifier.mat'];
  
  featurematrix = classifier.featurematrix;
  samplemean = classifier.samplemean;
  samplestd = classifier.samplestd;
  
  save(classifierfilename, 'featurematrix', 'samplemean', 'samplestd');
  
  [class, gray] = knnclassify(featurefilename, classifierfilename, ...
                              classifier.k);
end

% Delete features and classifier files.
delete(classifierfilename);
delete(featurefilename);
    
% Call guiresult to show resultsand add to children.
params.title = ['Segmentation results: ' get(params.imagefigure, 'Name') ...
                   ' classified by ' get(params.classifierfigure, 'Name')];

params.results{1}.data = gray;
params.results{1}.name = 'Posterior probabilities';

params.results{2}.data = class;
params.results{2}.name = 'Classification';

child.figure = guiresult(params);
child.type = 'guiresult';

add_child(handles.figure1, [], handles, child);



% --------------------------------------------------------------------
function segment_interface(h, eventdata, handles, params)
params.mainfig = handles.figure1;
params.classifiers = [];
params.images = [];

userdata = get(handles.figure1, 'UserData');
children = userdata.children;

for i = 1:length(children)
    if ishandle(children(i).figure)
        if strcmp(children(i).type, 'guiimage')
            newimage.figure = children(i).figure;
            newimage.name = get(children(i).figure, 'Name');
            params.images = [params.images newimage];
        elseif strcmp(children(i).type, 'guiclassifier')
            classifierhandle = guidata(children(i).figure);
            if classifierhandle.created
                newclassifier.figure = children(i).figure;
                newclassifier.name = get(children(i).figure, 'Name');
                params.classifiers = [params.classifiers newclassifier];
            end
        end
    end

end

child.figure = guisegment(params);
child.type = 'guisegment';
add_child(h, eventdata, handles, child);


% --------------------------------------------------------------------
function add_child(h, eventdata, handles, child)
userdata = get(handles.figure1, 'UserData');
userdata.children = [userdata.children child];
set(handles.figure1, 'UserData', userdata);


% --------------------------------------------------------------------
function exit = guiexit(h, eventdata, handles, varargin)
% Close all children.
userdata = get(handles.figure1, 'UserData');
children = userdata.children;

i = 1;
nchildren = length(children);
exit = 1;
while i <= nchildren & exit
    if ishandle(children(i).figure)
        exit = feval(children(i).type, 'figure1_CloseRequestFcn', children(i).figure, [], guidata(children(i).figure));
    end
    i = i + 1;
end

if exit
    delete(handles.figure1);
end

% Exit MATLAB also???