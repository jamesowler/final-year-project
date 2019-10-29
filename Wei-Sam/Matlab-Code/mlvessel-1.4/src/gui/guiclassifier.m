function varargout = guiclassifier(varargin)
% GUICLASSIFIER Application M-file for guiclassifier.fig
%    FIG = GUICLASSIFIER launch guiclassifier GUI.
%    GUICLASSIFIER('callback_name', ...) invoke the named callback.
%
% Opens a new or existing classifier from a file. Opens the window
% for viewing and configuring the classifier.

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

if nargin <= 1  % LAUNCH GUI
    if nargin == 0
        errordlg('Did not receive input parameters.', ...
            'Input Argument Error!');
        return;
    end
    
  fig = openfig(mfilename,'new');

  % Use system color scheme for figure:
  set(fig,'Color',get(0,'defaultUicontrolBackgroundColor'));
  set(fig, 'CloseRequestFcn', ...
        'guiclassifier(''figure1_CloseRequestFcn'', gcbo, [], guidata(gcbo))');
  
  % Generate a structure of handles to pass to callbacks, and store it. 
  handles = guihandles(fig);
  
  % Initialize classifier parameters uicontrols...
  handles.uicontrols{1}.text1 = uicontrol(fig, ...
            'Style', 'text', ...
            'Units', 'characters', ...
            'String', 'Classifier properties', ...
            'FontWeight', 'bold', ...
            'FontSize', 12, ...
            'HorizontalAlignment', 'left', ...
            'Position', [2 40 25 1.5]);
    
  handles.uicontrols{1}.text2 = uicontrol(fig, ...
            'Style', 'text', ...
            'String', 'Classifier:', ...
            'Units', 'characters', ...
            'HorizontalAlignment', 'left', ...
            'Position', [2 37.5 20 1.5]);
    
  handles.uicontrols{1}.popup2 = uicontrol(fig, ...
            'Style', 'popupmenu', ...
            'String', 'GMM|LMSE|KNN', ...
            'Units', 'characters', ...
            'HorizontalAlignment', 'left', ...
            'Position', [12 37.5 26 1.7], ...
  'Callback', 'guiclassifier(''change_classifier'', gcbo, [], guidata(gcbo))');
        
  handles.uicontrols{1}.text3 = uicontrol(fig, ...
            'Style', 'text', ...
            'String', 'K: ', ...
            'Units', 'characters', ...
            'HorizontalAlignment', 'left', ...
            'Position', [2 35.2 3 1.5]);
    
  handles.uicontrols{1}.edit1 = uicontrol(fig, ...
            'Style', 'edit', ...
            'String', '10', ...
            'Units', 'characters', ...
            'HorizontalAlignment', 'right', ...
            'Position', [12 35.5 7 1.5], ...
           'Callback', 'guiclassifier(''change_k'', gcbo, [], guidata(gcbo))');
    
  handles.uicontrols{1}.text4 = uicontrol(fig, ...
            'Style', 'text', ...
            'String', 'Training samples:', ...
            'Units', 'characters', ...
            'HorizontalAlignment', 'left', ...
            'Position', [2 32 20 1.5]);
    
  handles.uicontrols{1}.popup1 = uicontrol(fig, ...
            'Style', 'popupmenu', ...
            'String', 'All|Random subset', ...
            'Units', 'characters', ...
            'HorizontalAlignment', 'left', ...
            'Position', [14 30.3 24 1.7], ...
    'Callback', 'guiclassifier(''change_sampling'', gcbo, [], guidata(gcbo))');

  handles.uicontrols{1}.text5 = uicontrol(fig, ...
            'Style', 'text', ...
            'String', 'Maximum number of samples:', ...
            'Units', 'characters', ...
            'HorizontalAlignment', 'left', ...
            'Position', [2 28.6 30 1.5], ...
            'Enable', 'off');

  handles.uicontrols{1}.edit2 = uicontrol(fig, ...
            'Style', 'edit', ...
            'String', '1000', ...
            'Units', 'characters', ...
            'HorizontalAlignment', 'right', ...
            'Position', [23 26.9 15 1.5], ...
            'Enable', 'off', ...
     'Callback', 'guiclassifier(''change_samples'', gcbo, [], guidata(gcbo))');

  %
  % Now adding features controls.
  %
  handles.uicontrols{2}.text1 = uicontrol(fig, ...
            'Style', 'text', ...
            'Units', 'characters', ...
            'String', 'Pixel features', ...
            'FontWeight', 'bold', ...
            'FontSize', 12, ...
            'HorizontalAlignment', 'left', ...
            'Position', [42 42.7 22 1.5]);
    
  handles.uicontrols{2}.frame1 = uicontrol(fig, ...
            'Style', 'frame', ...
            'Units', 'characters', ...
            'Position', [42 39.7 46.5 2.5]);

  handles.uicontrols{2}.popupmenu1 = uicontrol(fig, ...
            'Style', 'popupmenu', ...
            'Units', 'characters', ...
            'HorizontalAlignment', 'left', ...
            'String', 'Colored|Fluorescein Angiogram|Red-free', ...
            'Position', [57.5 40 30.5 1.7], ...
   'Callback', 'guiclassifier(''change_imagetype'', gcbo, [], guidata(gcbo))');
        
  handles.uicontrols{2}.text2 = uicontrol(fig, ...
            'Style', 'text', ...
            'Units', 'characters', ...
            'String', 'Image type:', ...
            'Position', [43 40 12 1.5]);

  handles.uicontrols{2}.pushbutton1 = uicontrol(fig, ...
            'Style', 'pushbutton', ...
            'Units', 'characters', ...
            'String', 'New feature', ...
            'Position', [42 37.5 14 1.7], ...
        'Callback', 'guiclassifier(''add_feature'', gcbo, [], guidata(gcbo))');

  %
  % Now adding training images controls.
  %
  handles.uicontrols{3}.text1 = uicontrol(fig, ...
            'Style', 'text', ...
            'Units', 'characters', ...
            'String', 'Training images', ...
            'FontWeight', 'bold', ...
            'FontSize', 12, ...
            'HorizontalAlignment', 'left', ...
            'Position', [112 42.7 28 1.5]);

  handles.uicontrols{3}.pushbutton1 = uicontrol(fig, ...
            'Style', 'pushbutton', ...
            'Units', 'characters', ...
            'String', 'New image', ...
            'Position', [112 40.5 14 1.7], ...
          'Callback', 'guiclassifier(''add_image'', gcbo, [], guidata(gcbo))');

  % Fills in default values.
  params = varargin{1};
  if strcmp(params.action, 'new')
    % Fills in default values.
    handles.classifier.images = {};
    handles.classifier.features = {};
    handles.classifier.imagetype = 'colored';
    handles.classifier.k = 10;
    handles.classifier.type = 'gmm';
    handles.classifier.sampling = 'all';
    handles.classifier.samples = 1000;
    
    handles.created = 0;
    handles.saved = 0;
    handles.filename = [];
    handles.pathname = [];
    handles.featurescolumnexcess = 0;
    handles.imagescolumnexcess = 0;
    handles.mainfig = params.mainfig;
    handles.imagecount = params.imagecount;
    
    % Sets window name
    set(fig, 'Name', ['Untitled_classifier-' int2str(params.imagecount)]);
    
    % Disables "save as" and "Segment image"
    set(handles.pushbutton2, 'Enable', 'off');
    set(handles.menu_classifier_saveas, 'Enable', 'off');
  elseif strcmp(params.action, 'open')
    % Fills in values from classifer file.
    handles.created = 1;
    handles.saved = 1;
    handles.filename = params.filename;
    handles.pathname = params.pathname;
    handles.featurescolumnexcess = 0;
    handles.imagescolumnexcess = 0;
    handles.mainfig = params.mainfig;
    handles.imagecount = params.imagecount;
    
    C = load(fullfile(params.pathname, params.filename), '-MAT');
    classifier = C.classifier;
    handles.classifier = classifier;
    
    switch classifier.type
     case 'gmm'
      set(handles.uicontrols{1}.popup2, 'Value', 1);
     case 'lmse'
      set(handles.uicontrols{1}.popup2, 'Value', 2);
     case 'knn'
      set(handles.uicontrols{1}.popup2, 'Value', 3);
    end
    set(handles.uicontrols{1}.edit1, 'String', int2str(classifier.k));
    
    switch classifier.sampling
     case 'all'
      set(handles.uicontrols{1}.popup1, 'Value', 1);
     case 'random'
      set(handles.uicontrols{1}.popup1, 'Value', 2);
    end
    set(handles.uicontrols{1}.edit2, 'String', int2str(classifier.samples));
    
    handles.classifier.images = {};
    handles.classifier.features = {};
    switch classifier.imagetype
     case 'colored'
      set(handles.uicontrols{2}.popupmenu1, 'Value', 1);
     case 'angiogram'
      set(handles.uicontrols{2}.popupmenu1, 'Value', 2);
     case 'redfree'
      set(handles.uicontrols{2}.popupmenu1, 'Value', 3);
    end
    change_imagetype(handles.uicontrols{2}.popupmenu1, [], handles);
    handles = guidata(fig);
    
    % Fills in features.
    for i=1:length(classifier.features)
      add_feature(handles.uicontrols{2}.pushbutton1, [], handles);
      handles = guidata(fig);
      
      value = get(handles.classifier.features{i}.popup, 'Value');
      string = get(handles.classifier.features{i}.popup, 'String');
      value = 1;
      while ~strcmp(deblank(string(value,:)), classifier.features{i}.type)
        value = value + 1;
      end
      % Call callback and update handles.
      set(handles.classifier.features{i}.popup, 'Value', value);
      change_feature(handles.classifier.features{i}.popup, [], handles);
      handles = guidata(fig);
      
      % If necessary, fill in values
      if findstr('Gabor', classifier.features{i}.type)
        % Add Gabor filter parameters and controls.
        set(handles.classifier.features{i}.uicontrols(2), 'String', ...
                          num2str(classifier.features{i}.parameters.scale));
        set_scale(handles.classifier.features{i}.uicontrols(2), [], handles);
        handles = guidata(fig);
        
        set(handles.classifier.features{i}.uicontrols(4), 'String', ...
                          num2str(classifier.features{i}.parameters.epsilon));
        set_epsilon(handles.classifier.features{i}.uicontrols(4), [], handles);
        handles = guidata(fig);

        set(handles.classifier.features{i}.uicontrols(6), 'String', ...
                          num2str(classifier.features{i}.parameters.k0y));
        set_k0y(handles.classifier.features{i}.uicontrols(6), [], handles);
        handles = guidata(fig);
      end
    end
    
    % Fills in images.
    for i=1:length(classifier.images)
      add_image(handles.uicontrols{3}.pushbutton1, [], handles);
      handles = guidata(fig);
      
      handles.classifier.images{i}.labelspathname = classifier.images{i}.labelspathname;
      handles.classifier.images{i}.labelsfilename = classifier.images{i}.labelsfilename;
      handles.classifier.images{i}.imagepathname = classifier.images{i}.imagepathname;
      handles.classifier.images{i}.imagefilename = classifier.images{i}.imagefilename;
      handles.classifier.images{i}.imagethumb = classifier.images{i}.imagethumb;
      handles.classifier.images{i}.labelsthumb = classifier.images{i}.labelsthumb;
      
      % Set text, images, and enable open button
      set(handles.classifier.images{i}.text2, 'String', classifier.images{i}.imagefilename);
      set(handles.classifier.images{i}.text4, 'String', classifier.images{i}.labelsfilename);
      set(handles.classifier.images{i}.open1, 'Enable', 'on');
      
      ax = handles.classifier.images{i}.axes1;
      axes(ax);
      image(classifier.images{i}.imagethumb);
      set(ax, 'XTick', []);
      set(ax, 'YTick', []);
      set(ax, 'DataAspectRatio', [1 1 1]);
      
      ax = handles.classifier.images{i}.axes2;
      axes(ax);
      image(classifier.images{i}.labelsthumb);
      set(ax, 'XTick', []);
      set(ax, 'YTick', []);
      set(ax, 'DataAspectRatio', [1 1 1]);
    end
    
    % Sets window name
    set(fig, 'Name', [params.filename '-' int2str(params.imagecount)]);
    
    disable_all(handles);
    
    % Turn some people on (sliders, uimenu, segment button)
    set(handles.pushbutton2, 'Enable', 'on');
    set(handles.slider1, 'Enable', 'on');
    set(handles.slider2, 'Enable', 'on');
    set(handles.menu_classifier, 'Enable', 'on');
    set(handles.menu_classifier_new, 'Enable', 'on');
    set(handles.menu_classifier_open, 'Enable', 'on');
    set(handles.menu_classifier_saveas, 'Enable', 'on');
    set(handles.menu_classifier_close, 'Enable', 'on');
  end
  
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
function varargout = pushbutton1_Callback(h, eventdata, handles, varargin)
% Create classifier

% Verifiy features
if length(handles.classifier.features) == 0
    uiwait(warndlg('Please choose at least one pixel feature.','Warning','modal'));
    return;
end

% Verifiy training images
if length(handles.classifier.images) == 0
    uiwait(warndlg('Please choose at least one training image.','Warning','modal'));    
    return;
end

for i=1:length(handles.classifier.images)
    if isempty (handles.classifier.images{i}.imagefilename)
        uiwait(warndlg('Please set all training image filenames.','Warning','modal'));            
        return;
    end

    if isempty (handles.classifier.images{i}.labelsfilename)
        uiwait(warndlg('Please set all training image labels filenames','Warning','modal'));            
        return;
    end
  
end

% Turn everybody off (enable)
disable_all(handles);

% Show message telling user to wait.
frametmp = uicontrol(handles.figure1, ...
       'Style', 'frame', ...
       'Units', 'characters', ...
       'Position', [50 20 50 10]);

texttmp = uicontrol(handles.figure1, ...
      'Style', 'text', ...
      'Units', 'characters', ...
      'String', strvcat('Generating classifier. Please wait...', ...
                        'This may take a while...'), ...
      'Position', [60 23 34 3]);
       
drawnow;

% Creating the classifier...
handles.classifier = guicreateclassifier(handles.classifier);
handles.created = 1;
guidata(h, handles);

% Turn some people on (sliders, uimenu, segment button)
set(handles.pushbutton2, 'Enable', 'on');
set(handles.slider1, 'Enable', 'on');
set(handles.slider2, 'Enable', 'on');
set(handles.menu_classifier, 'Enable', 'on');
set(handles.menu_classifier_new, 'Enable', 'on');
set(handles.menu_classifier_open, 'Enable', 'on');
set(handles.menu_classifier_saveas, 'Enable', 'on');
set(handles.menu_classifier_close, 'Enable', 'on');

% Remove wait message
delete(texttmp);
delete(frametmp);



%------------------------------------------------------------------
function disable_all(handles)
features = handles.classifier.features;
images = handles.classifier.images;
uicontrols = handles.uicontrols;

basehandles = rmfield(handles, {'classifier', 'uicontrols', ...
         'featurescolumnexcess', 'figure1', 'imagescolumnexcess', ...
         'mainfig', 'imagecount', 'created', 'saved', 'pathname', ...
         'filename'});

cell = struct2cell(basehandles);
vector = [];
for k = 1:size(cell, 1)
    vector = [vector cell{k}];
end

for i=1:3
    cell = struct2cell(uicontrols{i});
    for k = 1:size(cell, 1)
        vector = [vector cell{k}];
    end
end

for i = 1:length(features)
    vector = [vector features{i}.popup, features{i}.remove, ...
                features{i}.frame, features{i}.uicontrols];
end

for i = 1:length(images)
    newhandles = rmfield(images{i}, {'imagefilename', ...
            'imagepathname', 'labelsfilename', ...
            'labelspathname', 'axes1', 'axes2', ...
            'imagethumb', 'labelsthumb'});

    cell = struct2cell(newhandles);
    for k = 1:size(cell, 1)
        vector = [vector cell{k}];
    end
end

for i=1:length(vector)
   set(vector(i), 'Enable', 'off');
end


% --------------------------------------------------------------------
function varargout = pushbutton2_Callback(h, eventdata, handles, varargin)
% Segment an image.
params.classifierfigure = handles.figure1;
params.imagefigure = [];
guimain('segment_interface', handles.mainfig, [], guidata(handles.mainfig), params);



% --------------------------------------------------------------------
function varargout = menu_classifier_Callback(h, eventdata, handles, varargin)




% --------------------------------------------------------------------
function varargout = menu_classifier_new_Callback(h, eventdata, handles, varargin)
guimain('menu_file_classifier_new_Callback', handles.mainfig, [], guidata(handles.mainfig));



% --------------------------------------------------------------------
function varargout = menu_classifier_open_Callback(h, eventdata, handles, varargin)
guimain('menu_file_classifier_open_Callback', handles.mainfig, [], guidata(handles.mainfig));



% --------------------------------------------------------------------
function saved = menu_classifier_saveas_Callback(h, eventdata, handles, varargin)

saved = 0;

% Get image name for saving.
filterspec = {'*.cla', 'Classifiers (*.cla)'; ...
		'*.*', 'All Files (*.*)'};
[filename, pathname] = uiputfile(filterspec, 'Save classifier as');

% Check if user cancelled.
if isequal(filename,0) | isequal(pathname,0)
	return;
end

% Check filename extension.
[path,name,ext,version] = fileparts(filename);
if strcmp(ext, '.cla')
    classifier = handles.classifier;
    % Clean up classifier...
    for i = 1:length(classifier.features)
        classifier.features{i} = rmfield(classifier.features{i}, ...
            {'frame', 'popup', 'remove', 'uicontrols'});
    end

    for i = 1:length(classifier.images)
        classifier.images{i} = rmfield(classifier.images{i}, {'text1', ...
                'text2', 'text3', 'text4', ...
                'browse1', 'browse2', ...
                'open1', 'open2' ...
                'remove', 'frame1', ...
                'axes1', 'axes2'});
    end
    
    save([pathname filesep filename], 'classifier', '-MAT');
    set(handles.figure1, 'Name', [filename '-' int2str(handles.imagecount)]);
    saved = 1;
    handles.saved = 1;
    guidata(h, handles);
else
    uiwait(warndlg('File not saved: incorrect or unavailable filename extension.', ...
        'Warning','modal'));
end



% --------------------------------------------------------------------
function varargout = menu_classifier_close_Callback(h, eventdata, ...
                                                  handles, varargin)

guiexit(h, eventdata, handles, varargin);



% --------------------------------------------------------------------
function varargout = slider2_Callback(h, eventdata, handles, ...
                                      varargin)

value = get(h, 'Value');

excess = handles.featurescolumnexcess;
textpos = get(handles.uicontrols{2}.text1, 'Position');

textcurrentdelta = textpos(2) - 42.7;
textnewdelta = (1 - value) * excess;

deltadelta = textnewdelta - textcurrentdelta;

% Move everybody up!!
movehandles = featuresmovevector(handles);

% loop over the remaining handles and change their positions
for k=1:length(movehandles)
  set(movehandles(k), 'Position', ...
                    get(movehandles(k), 'Position') + [0 deltadelta 0 0]);
end


% --------------------------------------------------------------------
function varargout = slider1_Callback(h, eventdata, handles, varargin)

value = get(h, 'Value');

excess = handles.imagescolumnexcess;
textpos = get(handles.uicontrols{3}.text1, 'Position');

textcurrentdelta = textpos(2) - 42.7;
textnewdelta = (1 - value) * excess;

deltadelta = textnewdelta - textcurrentdelta;

% Move everybody up!!
movehandles = imagesmovevector(handles);

% loop over the remaining handles and change their positions
for k=1:length(movehandles)
  set(movehandles(k), 'Position', ...
                    get(movehandles(k), 'Position') + [0 deltadelta 0 0]);
end



% --------------------------------------------------------------------
function varargout = change_classifier(h, eventdata, handles, varargin)
value = get(h, 'Value');
string = get(h, 'String');
newtype = deblank(string(value,:));

if ~strcmp(handles.classifier.type, newtype)
switch value
    case 1
        handles.classifier.type = 'gmm';
        set(handles.uicontrols{1}.text3, 'Enable', 'on');
        set(handles.uicontrols{1}.edit1, 'Enable', 'on');
    case 2
        handles.classifier.type = 'lmse';
        set(handles.uicontrols{1}.text3, 'Enable', 'off');
        set(handles.uicontrols{1}.edit1, 'Enable', 'off');
    case 3
        handles.classifier.type = 'knn';
        set(handles.uicontrols{1}.text3, 'Enable', 'on');
        set(handles.uicontrols{1}.edit1, 'Enable', 'on');
    end
end
    
guidata(h, handles);



% --------------------------------------------------------------------
function varargout = change_sampling(h, eventdata, handles, varargin)
value = get(h, 'Value');

switch value
    case 1 % all
        handles.classifier.sampling = 'all';
        set(handles.uicontrols{1}.text5, 'Enable', 'off');
        set(handles.uicontrols{1}.edit2, 'Enable', 'off');
    case 2 % random subset
        handles.classifier.sampling = 'random';
        set(handles.uicontrols{1}.text5, 'Enable', 'on');
        set(handles.uicontrols{1}.edit2, 'Enable', 'on');
end
    
guidata(h, handles);


% --------------------------------------------------------------------
function varargout = change_samples(h, eventdata, handles, varargin)
newstrvalue = get(h, 'String');
newvalue = str2double(newstrvalue);

% Check that the entered value falls within the allowable range
if  isempty(newvalue) | (newvalue <= 0)
    % Revert to last value
    oldvalue = handles.classifier.samples;
    set(h, 'String', oldvalue);
else % Use new value
    handles.classifier.samples = newvalue;
end

guidata(h, handles);



% --------------------------------------------------------------------
function varargout = change_k(h, eventdata, handles, varargin)
newstrvalue = get(h, 'String');
newvalue = str2double(newstrvalue);

% Check that the entered value falls within the allowable range
if  isempty(newvalue) | (newvalue <= 0)
    % Revert to last value
    oldvalue = handles.classifier.k;
    set(h, 'String', oldvalue);
else % Use new value
    handles.classifier.k = newvalue;
end

guidata(h, handles);


% --------------------------------------------------------------------
function varargout = change_imagetype(h, eventdata, handles, varargin)

value = get(h, 'Value');
oldtype = handles.classifier.imagetype;
switch value
case 1 % User selected colored
    handles.classifier.imagetype = 'colored';
case 2 % User selected angiogram
    handles.classifier.imagetype = 'angiogram';
case 3 % User selected red-free
    handles.classifier.imagetype = 'redfree';
end

% If type changed, erase old features.
if ~strcmp(handles.classifier.imagetype, oldtype)

    nfeatures = size(handles.classifier.features, 2);
    
    if nfeatures >= 7
      % Make everyone visible.
      set(handles.slider2, 'Value', 1);
      guiclassifier('slider2_Callback', handles.slider2, [], ...
                    handles);
      
      % Remove slider
      set(handles.slider2, 'Visible', 'off');
    end

    % Erasing features
    for i = 1:nfeatures
        delete(handles.classifier.features{i}.popup);
        delete(handles.classifier.features{i}.remove);
        delete(handles.classifier.features{i}.frame);
        for j=1:length(handles.classifier.features{i}.uicontrols)
            delete(handles.classifier.features{i}.uicontrols(j));
        end
    end
    
    handles.classifier.features = {};
    handles.featurescolumnexcess = 0;

    % Bring pushbutton up.
    set(handles.uicontrols{2}.pushbutton1, 'Position', [42 37.5 14 1.7]);
end 

guidata(h, handles); % Save the handles structure after adding data



% --------------------------------------------------------------------
function varargout = remove_feature(h, eventdata, handles, varargin)

% Find uicontrol in features vector
features = handles.classifier.features;
i = 1;
while features{i}.remove ~= h
    i = i + 1;
end

% Remove feature i
dyingfeature = features{i};
features = {features{1:i-1} features{i+1:end}};

handles.classifier.features = features;

% Move bottom controls up!!
movehandles = featuresmovevector(handles, i);
for k=1:length(movehandles)
   set(movehandles(k), 'Position', ...
                     get(movehandles(k), 'Position') + [0 5.5 0 0]);
end

% Update slidebar
if length(handles.classifier.features) > 6
    % Reset slider parameters.
    handles.featurescolumnexcess = handles.featurescolumnexcess - 5.5;
    step = 1 / handles.featurescolumnexcess;

    % Calls slider's callback to update position
    set(handles.slider2, 'SliderStep', [min(1, step), min(1, step * 5)]);
    guiclassifier('slider2_Callback', handles.slider2, [], ...
                  handles);
    
elseif length(handles.classifier.features) == 6
  % Make everyone visible.
  set(handles.slider2, 'Value', 1);
  guiclassifier('slider2_Callback', handles.slider2, [], ...
                handles);

  handles.featurescolumnexcess = 0;
  
  % Remove slider
  set(handles.slider2, 'Visible', 'off');
end

% Save changes to features
guidata(h, handles);

% Delete uicontrols.
delete(dyingfeature.remove);
delete(dyingfeature.popup);
delete(dyingfeature.frame);
for j = 1:size(dyingfeature.uicontrols, 2)
    delete(dyingfeature.uicontrols(j));
end


% --------------------------------------------------------------------
function varargout = add_feature(h, eventdata, handles, varargin)

% Gets possible and default image features.
switch handles.classifier.imagetype
case 'colored'
    newfeature.type = 'Inverted green channel';
    possiblefeatures = ['Inverted green channel|' ...
            'Gabor processed inverted green channel'];
case 'angiogram'
    newfeature.type = 'Gray channel';
    possiblefeatures = ['Gray channel|', ...
                'Gabor processed gray channel'];
case 'redfree'
    newfeature.type = 'Inverted gray channel';
    possiblefeatures = ['Inverted gray channel|', ...
            'Gabor processed inverted gray channel'];
end

% Pushes "new feature" pushbutton down.
pushpos = get(handles.uicontrols{2}.pushbutton1, 'Position');
pushpos = pushpos - [0 5.5 0 0];
set(handles.uicontrols{2}.pushbutton1, 'Position', pushpos);
pushpos = [pushpos(1:2) 0 0];

% Finally, adds new feature and uicontrols.
newfeature.frame = uicontrol('Style', 'frame', ...
    'Units', 'characters', ...
    'Position', pushpos + [0 2.2 62.5 5]);

newfeature.popup = uicontrol('Style', 'popup', ...
    'String', possiblefeatures, ...
    'Units', 'characters', ...
    'Position', pushpos + [1 5 45 1.7], ...
    'Callback', 'guiclassifier(''change_feature'', gcbo, [], guidata(gcbo))');

newfeature.remove = uicontrol('Style', 'pushbutton', ...
    'String', 'Remove feature', ...
    'Units', 'characters', ...
    'Position', pushpos + [41.7 2.6 20 1.7], ...
    'Callback', 'guiclassifier(''remove_feature'', gcbo, [], guidata(gcbo))');

newfeature.uicontrols = [];
newfeature.parameters = [];

handles.classifier.features = {handles.classifier.features{:}, newfeature};

if length(handles.classifier.features) >= 7
    % Make pushbutton visible by moving everybody up.
    deltay = pushpos(2) - 1;
    
    % Move everybody up!!
    movehandles = featuresmovevector(handles);

    % loop over the remaining handles and change their positions
    for k=1:length(movehandles)
      set(movehandles(k), 'Position', ...
                        get(movehandles(k), 'Position') - [0 deltay 0 0]);
    end
    
    % Set slider parameters.
    if length(handles.classifier.features) == 7 % first excess
      handles.featurescolumnexcess = - deltay;
    else
      handles.featurescolumnexcess = ...
          handles.featurescolumnexcess + 5.5;
    end
    step = 1 / handles.featurescolumnexcess;
    
    set(handles.slider2, 'Visible', 'on');
    set(handles.slider2, 'Value', 0);
    set(handles.slider2, 'SliderStep', [min(1, step), min(1, step * 5)]);
end

guidata(h, handles);

%----------------------------------------------------------------------
function varargout = change_feature(h, eventdata, handles, varargin)
% Find uicontrol in features vector
features = handles.classifier.features;
i = 1;
while features{i}.popup ~= h
    i = i + 1;
end

value = get(h, 'Value');
string = get(h, 'String');
newtype = deblank(string(value,:));

if ~strcmp(features{i}.type, newtype)
    % Delete old uicontrols and parameters.
    for j = 1:size(features{i}.uicontrols, 2)
        delete(features{i}.uicontrols(j));
    end
    features{i}.uicontrols = [];
    features{i}.parameters = [];
    
    % Add new uicontrols and parameters
    if findstr('Gabor', newtype)
        % Add Gabor filter parameters and controls.
        features{i}.parameters.scale = 2;
        features{i}.parameters.epsilon = 4;
        features{i}.parameters.k0y = 3;
        
        poppos = get(h, 'Position');
        features{i}.uicontrols(1) = uicontrol('Style', 'text', ...
            'String', 'Scale: ', ...
            'Units', 'characters', ...
            'Position', [poppos(1) (- 2.4 + poppos(2)) 8 1.6]);
        
        features{i}.uicontrols(2) = uicontrol('Style', 'edit', ...
            'String', '2', ...
            'Units', 'characters', ...
            'Position', [(7 + poppos(1)) (- 2.1 + poppos(2)) 6 1.6], ...
            'HorizontalAlignment', 'right', ...
            'Callback', 'guiclassifier(''set_scale'', gcbo, [], guidata(gcbo))');

        features{i}.uicontrols(3) = uicontrol('Style', 'text', ...
            'String', 'Epsilon: ', ...
            'Units', 'characters', ...
            'Position', [(14 + poppos(1)) (- 2.4 + poppos(2)) 8 1.6]);
        
        features{i}.uicontrols(4) = uicontrol('Style', 'edit', ...
            'String', '4', ...
            'Units', 'characters', ...
            'Position', [(22 + poppos(1)) (- 2.1 + poppos(2)) 6 1.6], ...
            'HorizontalAlignment', 'right', ...
            'Callback', 'guiclassifier(''set_epsilon'', gcbo, [], guidata(gcbo))');
        
        features{i}.uicontrols(5) = uicontrol('Style', 'text', ...
            'String', 'k0y: ', ...
            'Units', 'characters', ...
            'Position', [(28 + poppos(1)) (- 2.4 + poppos(2)) 8 1.6]);
        
        features{i}.uicontrols(6) = uicontrol('Style', 'edit', ...
            'String', '3', ...
            'Units', 'characters', ...
            'Position', [(34 + poppos(1)) (- 2.1 + poppos(2)) 5 1.6], ...
            'HorizontalAlignment', 'right', ...
            'Callback', 'guiclassifier(''set_k0y'', gcbo, [], guidata(gcbo))');
        
    end
    features{i}.type = newtype;
    handles.classifier.features = features;
    guidata(h, handles);
end


%----------------------------------------------------------------------
function varargout = set_scale(h, eventdata, handles, varargin)
% Find uicontrol in features vector
features = handles.classifier.features;
i = 1;
while length(features{i}.uicontrols) < 2 | features{i}.uicontrols(2) ~= h
    i = i + 1;
end

newstrvalue = get(h, 'String');
newvalue = str2double(newstrvalue);

% Check that the entered value falls within the allowable range
if  isempty(newvalue) | (newvalue <= 0) | (newvalue > 500)
    % Revert to last value
    oldvalue = handles.classifier.features{i}.parameters.scale;
    set(h, 'String', oldvalue);
else % Use new value
    handles.classifier.features{i}.parameters.scale = newvalue;
end

guidata(h, handles);


%----------------------------------------------------------------------
function varargout = set_epsilon(h, eventdata, handles, varargin)
% Find uicontrol in features vector
features = handles.classifier.features;
i = 1;
while length(features{i}.uicontrols) < 4 | features{i}.uicontrols(4) ~= h
    i = i + 1;
end

newstrvalue = get(h, 'String');
newvalue = str2double(newstrvalue);

% Check that the entered value falls within the allowable range
if  isempty(newvalue) | (newvalue <= 0) | (newvalue > 500)
    % Revert to last value
    oldvalue = handles.classifier.features{i}.parameters.epsilon;
    set(h, 'String', oldvalue);
else % Use new value
    handles.classifier.features{i}.parameters.epsilon = newvalue;
end

guidata(h, handles);



%----------------------------------------------------------------------
function varargout = set_k0y(h, eventdata, handles, varargin)
% Find uicontrol in features vector
features = handles.classifier.features;
i = 1;
while length(features{i}.uicontrols) < 6 | features{i}.uicontrols(6) ~= h
    i = i + 1;
end

newstrvalue = get(h, 'String');
newvalue = str2double(newstrvalue);

% Check that the entered value falls within the allowable range
if  isempty(newvalue) | (newvalue <= 0) | (newvalue > 20)
    % Revert to last value
    oldvalue = handles.classifier.features{i}.parameters.k0y;
    set(h, 'String', oldvalue);
else % Use new value
    handles.classifier.features{i}.parameters.k0y = newvalue;
end

guidata(h, handles);



%----------------------------------------------------------------------
function vector = featuresmovevector(handles, i)

features = handles.classifier.features;

basichandles = handles.uicontrols{2};

if nargin < 2
  i = 1;
else
  basichandles = rmfield(basichandles, ...
                         {'frame1', 'text1', 'text2', 'popupmenu1'});
end

cell = struct2cell(basichandles);
for k = 1:size(cell, 1)
    vector(k) = cell{k};
end

for j = i:length(features)
    vector = [vector, features{j}.popup, features{j}.remove, ...
                features{j}.frame, features{j}.uicontrols];
end



% --------------------------------------------------------------------
function varargout = add_image(h, eventdata, handles, varargin)

% Pushes "new image" pushbutton down.
pushpos = get(handles.uicontrols{3}.pushbutton1, 'Position');
pushpos = pushpos - [0 26 0 0];
set(handles.uicontrols{3}.pushbutton1, 'Position', pushpos);
pushpos = [pushpos(1:2) 0 0];

% Adds uicontrols.
newimage.text1 = uicontrol('Style', 'text', ...
    'String', 'Image', ...
    'HorizontalAlignment', 'left', ...
    'Units', 'characters', ...
    'Position', pushpos + [1 26.8 15 1]);

newimage.text2 = uicontrol('Style', 'text', ...
    'String', 'No image loaded', ...
    'HorizontalAlignment', 'left', ...
    'Units', 'characters', ...
    'Position', pushpos + [29 25.3 30 1]);

newimage.browse1 = uicontrol('Style', 'pushbutton', ...
    'String', 'Browse...', ...
    'Units', 'characters', ...
    'Position', pushpos + [29 23 12 1.7], ...
    'Callback', 'guiclassifier(''browse_image'', gcbo, [], guidata(gcbo))');

newimage.open1 = uicontrol('Style', 'pushbutton', ...
    'String', 'Open', ...
    'Units', 'characters', ...
    'Enable', 'off', ...
    'Position', pushpos + [42 23 9 1.7], ...
    'Callback', 'guiclassifier(''open_image'', gcbo, [], guidata(gcbo))');

ax = axes;
image(128);
colormap(gray(256));
set(ax, 'Units', 'characters');
set(ax, 'Position', pushpos + [1 16 26.5 10.5]);
set(ax, 'XTick', []);
set(ax, 'YTick', []);
set(ax, 'DataAspectRatio', [1 1 1]);
newimage.axes1 = ax;

newimage.text3 = uicontrol('Style', 'text', ...
    'String', 'Labels', ...
    'HorizontalAlignment', 'left', ...
    'Units', 'characters', ...
    'Position', pushpos + [1 14.2 15 1]);

newimage.text4 = uicontrol('Style', 'text', ...
    'String', 'No image loaded', ...
    'HorizontalAlignment', 'left', ...
    'Units', 'characters', ...
    'Position', pushpos + [29 12.8 30 1]);

newimage.browse2 = uicontrol('Style', 'pushbutton', ...
    'String', 'Browse...', ...
    'Units', 'characters', ...
    'Position', pushpos + [29 10.5 12 1.7], ...
    'Callback', 'guiclassifier(''browse_labels'', gcbo, [], guidata(gcbo))');

newimage.open2 = uicontrol('Style', 'pushbutton', ...
    'String', 'Open', ...
    'Units', 'characters', ...
    'Enable', 'off', ...
    'Position', pushpos + [42 10.5 9 1.7], ...
    'Visible', 'off', ... 
    'Callback', 'guiclassifier(''open_labels'', gcbo, [], guidata(gcbo))');

ax = axes;
image(128);
colormap(gray(256));
set(ax, 'Units', 'characters');
set(ax, 'Position', pushpos + [1 3.5 26.5 10.5]);
set(ax, 'XTick', []);
set(ax, 'YTick', []);
set(ax, 'DataAspectRatio', [1 1 1]);
newimage.axes2 = ax;

newimage.remove = uicontrol('Style', 'pushbutton', ...
    'String', 'Remove image', ...
    'Units', 'characters', ...
    'Position', pushpos + [43 3.3 20 1.7], ...
    'Callback', 'guiclassifier(''remove_image'', gcbo, [], guidata(gcbo))');

newimage.frame1 = uicontrol('Style', 'frame', ...
    'Units', 'characters', ...
    'Position', pushpos + [-2 2.5 66.5 0.2]);

newimage.imagefilename = [];
newimage.imagepathname = [];
newimage.labelsfilename = [];
newimage.labelspathname = [];

handles.classifier.images = {handles.classifier.images{:}, newimage};

if length(handles.classifier.images) >= 2
    % Make pushbutton visible by moving everybody up.
    deltay = pushpos(2) - 1;
    
    % Move everybody up!!
    movehandles = imagesmovevector(handles);

    % loop over the remaining handles and change their positions
    for k=1:length(movehandles)
        set(movehandles(k), 'Position', ...
            get(movehandles(k), 'Position') - [0 deltay 0 0]);
    end
    
    % Set slider parameters.
    if length(handles.classifier.images) == 2 % first excess
        handles.imagescolumnexcess = - deltay;
    else
        handles.imagescolumnexcess = ...
            handles.imagescolumnexcess + 26;
    end
    
    step = 1 / handles.imagescolumnexcess;
    
    set(handles.slider1, 'Visible', 'on');
    set(handles.slider1, 'Value', 0);
    set(handles.slider1, 'SliderStep', [min(1, step), min(1, step * 10)]);
end

guidata(h, handles);


% --------------------------------------------------------------------
function varargout = remove_image(h, eventdata, handles, varargin)

% Find uicontrol in features vector
images = handles.classifier.images;
i = 1;
while images{i}.remove ~= h
    i = i + 1;
end

% Remove feature i
dyingimage = images{i};
images = {images{1:i-1} images{i+1:end}};

handles.classifier.images = images;

% Move bottom controls up!!
movehandles = imagesmovevector(handles, i);
for k=1:length(movehandles)
   set(movehandles(k), 'Position', ...
                     get(movehandles(k), 'Position') + [0 26 0 0]);
end

% Update slidebar
if length(handles.classifier.images) > 1
    % Reset slider parameters.
    handles.imagescolumnexcess = handles.imagescolumnexcess - 26;
    step = 1 / handles.imagescolumnexcess;

    % Calls slider's callback to update position
    set(handles.slider1, 'SliderStep', [min(1, step), min(1, step * 10)]);
    guiclassifier('slider1_Callback', handles.slider1, [], ...
                  handles);
    
elseif length(handles.classifier.images) == 1
  % Make everyone visible.
  set(handles.slider1, 'Value', 1);
  guiclassifier('slider1_Callback', handles.slider1, [], ...
                handles);

  handles.imagescolumnexcess = 0;
  
  % Remove slider
  set(handles.slider1, 'Visible', 'off');
end

% Save changes to features
guidata(h, handles);

% Delete uicontrols.
delete(dyingimage.remove);
delete(dyingimage.axes1);
delete(dyingimage.axes2);
delete(dyingimage.text1);
delete(dyingimage.text2);
delete(dyingimage.text3);
delete(dyingimage.text4);
delete(dyingimage.browse1);
delete(dyingimage.browse2);
delete(dyingimage.open1);
delete(dyingimage.open2);
delete(dyingimage.frame1);


%----------------------------------------------------------------------
function varargout = browse_image(h, eventdata, handles, varargin)
% Find uicontrol in features vector
images = handles.classifier.images;
i = 1;
while images{i}.browse1 ~= h
    i = i + 1;
end

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

% Set new image filename.
images{i}.imagefilename = filename;
images{i}.imagepathname = pathname;

original = imread(fullfile(pathname, filename));
scale = 200 / max(size(original));
thumbnail = imresize(original, scale);
images{i}.imagethumb = thumbnail;

ax = images{i}.axes1;
axes(ax);
image(thumbnail);
set(ax, 'XTick', []);
set(ax, 'YTick', []);
set(ax, 'DataAspectRatio', [1 1 1]);

set(images{i}.open1, 'Enable', 'on');
set(images{i}.text2, 'String', filename);

% Save changed data.
handles.classifier.images = images;
guidata(h, handles);


%----------------------------------------------------------------------
function varargout = open_image(h, eventdata, handles, varargin)
% Find uicontrol in features vector
images = handles.classifier.images;
i = 1;
while images{i}.open1 ~= h
    i = i + 1;
end

guimain('menu_file_image_open_Callback', ...
        handles.mainfig, [], guidata(handles.mainfig), ...
        images{i}.imagepathname, images{i}.imagefilename);



%----------------------------------------------------------------------
function varargout = browse_labels(h, eventdata, handles, varargin)
% Find uicontrol in features vector
images = handles.classifier.images;
i = 1;
while images{i}.browse2 ~= h
    i = i + 1;
end

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

% Set new image filename.
images{i}.labelsfilename = filename;
images{i}.labelspathname = pathname;

original = imread(fullfile(pathname, filename));
scale = 200 / max(size(original));
thumbnail = imresize(original, scale);
images{i}.labelsthumb = thumbnail;

ax = images{i}.axes2;
axes(ax);
image(thumbnail);
set(ax, 'XTick', []);
set(ax, 'YTick', []);
set(ax, 'DataAspectRatio', [1 1 1]);

set(images{i}.open2, 'Enable', 'on');
set(images{i}.text4, 'String', filename);

% Save changed data.
handles.classifier.images = images;
guidata(h, handles);



%----------------------------------------------------------------------
function vector = imagesmovevector(handles, i)

images = handles.classifier.images;

basichandles = handles.uicontrols{3};

if nargin < 2
  i = 1;
else
  basichandles = rmfield(basichandles, 'text1');
end

cell = struct2cell(basichandles);
for k = 1:size(cell, 1)
    vector(k) = cell{k};
end

for j = i:length(images)
    vector = [vector, images{j}.frame1, images{j}.remove ...
             images{j}.text1, images{j}.text2, ...
             images{j}.browse1, images{j}.open1, images{j}.axes1, ...
             images{j}.text3, images{j}.text4, ...
             images{j}.browse2, images{j}.open2, images{j}.axes2];
end



%----------------------------------------------------------------------
function exit = figure1_CloseRequestFcn(h, eventdata, handles, varargin)
exit = guiexit(h, eventdata, handles, varargin);



%----------------------------------------------------------------------
function exit = guiexit(h, eventdata, handles, varargin)
if handles.created & ~handles.saved
    answer = questdlg(['Untitled_classifier-' int2str(handles.imagecount) ' has not been saved. ' ...
            'Save classifier before closing?'], ...
        ['Close Untitled_classifier-' int2str(handles.imagecount)], ...
        'Discard', 'Cancel', 'Save', 'Cancel');
    switch answer
        case 'Discard'
            exit = 1;
        case 'Cancel'
            exit = 0;
            % Nothing to do
        case 'Save'
            exit = guiclassifier('menu_classifier_saveas_Callback', handles.menu_classifier_saveas, [], handles);
    end
else
    exit = 1;
end

if exit
    delete(handles.figure1);
end