function varargout = guifeaturesdialog(varargin)
% GUIFEATURESDIALOG Application M-file for guifeaturesdialog.fig
%    FIG = GUIFEATURESDIALOG launch guifeaturesdialog GUI.
%    GUIFEATURESDIALOG('callback_name', ...) invoke the named callback.
%
% Opens a dialog for user to define features to be generated.

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
        errordlg('Did not receive input parameters.',...
            'Input Argument Error!');
        return;
    end
    
	fig = openfig(mfilename,'new');

  	% Use system color scheme for figure:
	set(fig, 'Color', get(0,'defaultUicontrolBackgroundColor'));
    set(fig, 'CloseRequestFcn', ...
        'guifeaturesdialog(''figure1_CloseRequestFcn'', gcbo, [], guidata(gcbo))');
    
    params = varargin{1};
    set(fig, 'Name', ['Choose features - ' params.title]);

	% Generate a structure of handles to pass to callbacks, and store it. 
	handles = guihandles(fig);

    handles.imagetype = 'colored';
    handles.features = {};
    handles.fatherfigure = params.fatherfigure;
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
function varargout = pushbutton2_Callback(h, eventdata, handles, varargin)
% VIEW FEATURES

if length(handles.features) == 0
    uiwait(warndlg('Please choose at least one feature.','Warning','modal'));
    return;
end

% disable all handles
vector = controlsvector(handles);
for i=1:length(vector)
    set(vector(i), 'Enable', 'off');
end

% Displays message
uicontrol('Style', 'text', ...
    'String', 'Generating features. Please wait...', ...
    'Units', 'characters', ...
    'Position', [10 2.4 40 1.5]);
drawnow;

% generate and show features (progress bar?)
guiimage('show_features', ...
    handles.fatherfigure, [], guidata(handles.fatherfigure), ...
    handles.imagetype, handles.features);

% Close window
delete(handles.figure1);

% --------------------------------------------------------------------
function varargout = popupmenu1_Callback(h, eventdata, handles, varargin)
% SWITCH IMAGE TYPE

value = get(h, 'Value');
oldtype = handles.imagetype;
switch value
case 1 % User selected colored
    handles.imagetype = 'colored';
case 2 % User selected angiogram
    handles.imagetype = 'angiogram';
case 3 % User selected red-free
    handles.imagetype = 'redfree';
end

% If type changed, erase old features.
if ~strcmp(handles.imagetype, oldtype)
    % Erasing features
    nfeatures = size(handles.features, 2);
    for i = 1:nfeatures
        delete(handles.features{i}.popup);
        delete(handles.features{i}.remove);
        delete(handles.features{i}.frame);
        for j=1:length(handles.features{i}.uicontrols)
            delete(handles.features{i}.uicontrols(j));
        end
    end
    handles.features = {};
    
    % Changing window size and repositions uicontrols.
    figpos = get(handles.figure1, 'Position');
    set(handles.figure1, 'Position', figpos - nfeatures * [0 -5.5 0 5.5]);
    
    % Move almost everybody up.
    % exclude the handles of the lower buttons
    movehandles = movevector(handles);

    % loop over the remaining handles and change their positions
    for k=1:length(movehandles)
      set(movehandles(k), 'Position', ...
                 (get(movehandles(k), 'Position')) - nfeatures * [0 5.5 0 0]);
    end

end 

guidata(h, handles); % Save the handles structure after adding data


% --------------------------------------------------------------------
function varargout = pushbutton3_Callback(h, eventdata, handles, varargin)
% ADD NEW FEATURE

if size(handles.features, 2) == 7
    uiwait(warndlg('Addition of more features not yet implemented. Sorry!', ...
        'Warning', 'modal'));
    return;
end

% Gets possible and default image features.
switch handles.imagetype
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

% Changes window size and repositions uicontrols.
figpos = get(handles.figure1, 'Position');
set(handles.figure1, 'Position', figpos + [0 -5.5 0 5.5]);

% Move almost everybody down.
% exclude the handles of the lower buttons
movehandles = movevector(handles);

% loop over the remaining handles and change their positions
for k=1:length(movehandles)
  set(movehandles(k), 'Position', ...
                    get(movehandles(k), 'Position') + [0 5.5 0 0]);
end

% Finally, adds new feature and uicontrols.
newfeature.frame = uicontrol('Style', 'frame', ...
    'Units', 'characters', ...
    'Position', [1 6.8 62.5 5]);

newfeature.popup = uicontrol('Style', 'popup', ...
    'String', possiblefeatures, ...
    'Units', 'characters', ...
    'Position', [2 9.6 45 1.7], ...
    'Callback', 'guifeaturesdialog(''change_feature'', gcbo, [], guidata(gcbo))');

newfeature.remove = uicontrol('Style', 'pushbutton', ...
    'String', 'Remove feature', ...
    'Units', 'characters', ...
    'Position', [42.7 7.2 20 1.7], ...
    'Callback', 'guifeaturesdialog(''remove_feature'', gcbo, [], guidata(gcbo))');

newfeature.uicontrols = [];
newfeature.parameters = [];

handles.features = {handles.features{:}, newfeature};
guidata(h, handles);

%----------------------------------------------------------------------
function varargout = remove_feature(h, eventdata, handles, varargin)

% Find uicontrol in features vector
features = handles.features;
i = 1;
while features{i}.remove ~= h
    i = i + 1;
end

% Remove feature i
dyingfeature = features{i};
features = {features{1:i-1} features{i+1:end}};

% Save changes to features
handles.features = features;
guidata(h, handles);

% Delete uicontrols.
delete(dyingfeature.remove);
delete(dyingfeature.popup);
delete(dyingfeature.frame);
for j = 1:size(dyingfeature.uicontrols, 2)
    delete(dyingfeature.uicontrols(j));
end

% Changes window size and repositions uicontrols.
figpos = get(handles.figure1, 'Position');
set(handles.figure1, 'Position', figpos - [0 -5.5 0 5.5]);

% Move almost everybody around.
% exclude the handles of the lower buttons
movehandles = movevector(handles, i);

% loop over the remaining handles and change their positions
for k=1:length(movehandles)
  set(movehandles(k), 'Position', ...
                    get(movehandles(k), 'Position') - [0 5.5 0 0]);
end


%----------------------------------------------------------------------
function varargout = change_feature(h, eventdata, handles, varargin)
% Find uicontrol in features vector
features = handles.features;
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
        
        pushpos = get(h, 'Position');
        features{i}.uicontrols(1) = uicontrol('Style', 'text', ...
            'String', 'Scale: ', ...
            'Units', 'characters', ...
            'Position', [pushpos(1) (- 2.4 + pushpos(2)) 8 1.6]);
        
        features{i}.uicontrols(2) = uicontrol('Style', 'edit', ...
            'String', '2', ...
            'Units', 'characters', ...
            'Position', [(7 + pushpos(1)) (- 2.1 + pushpos(2)) 6 1.6], ...
            'HorizontalAlignment', 'right', ...
            'Callback', 'guifeaturesdialog(''set_scale'', gcbo, [], guidata(gcbo))');

        features{i}.uicontrols(3) = uicontrol('Style', 'text', ...
            'String', 'Epsilon: ', ...
            'Units', 'characters', ...
            'Position', [(14 + pushpos(1)) (- 2.4 + pushpos(2)) 8 1.6]);
        
        features{i}.uicontrols(4) = uicontrol('Style', 'edit', ...
            'String', '4', ...
            'Units', 'characters', ...
            'Position', [(22 + pushpos(1)) (- 2.1 + pushpos(2)) 6 1.6], ...
            'HorizontalAlignment', 'right', ...
            'Callback', 'guifeaturesdialog(''set_epsilon'', gcbo, [], guidata(gcbo))');
        
        features{i}.uicontrols(5) = uicontrol('Style', 'text', ...
            'String', 'k0y: ', ...
            'Units', 'characters', ...
            'Position', [(28 + pushpos(1)) (- 2.4 + pushpos(2)) 8 1.6]);
        
        features{i}.uicontrols(6) = uicontrol('Style', 'edit', ...
            'String', '3', ...
            'Units', 'characters', ...
            'Position', [(34 + pushpos(1)) (- 2.1 + pushpos(2)) 5 1.6], ...
            'HorizontalAlignment', 'right', ...
            'Callback', 'guifeaturesdialog(''set_k0y'', gcbo, [], guidata(gcbo))');
        
    end
    features{i}.type = newtype;
    handles.features = features;
    guidata(h, handles);
end


%----------------------------------------------------------------------
function varargout = set_scale(h, eventdata, handles, varargin)
% Find uicontrol in features vector
features = handles.features;
i = 1;
while length(features{i}.uicontrols) < 2 | features{i}.uicontrols(2) ~= h
    i = i + 1;
end

newstrvalue = get(h, 'String');
newvalue = str2double(newstrvalue);

% Check that the entered value falls within the allowable range
if  isempty(newvalue) | (newvalue <= 0) | (newvalue > 500)
    % Revert to last value
    oldvalue = handles.features{i}.parameters.scale;
    set(h, 'String', oldvalue);
else % Use new value
    handles.features{i}.parameters.scale = newvalue;
end

guidata(h, handles);


%----------------------------------------------------------------------
function varargout = set_epsilon(h, eventdata, handles, varargin)
% Find uicontrol in features vector
features = handles.features;
i = 1;
while length(features{i}.uicontrols) < 4 | features{i}.uicontrols(4) ~= h
    i = i + 1;
end

newstrvalue = get(h, 'String');
newvalue = str2double(newstrvalue);

% Check that the entered value falls within the allowable range
if  isempty(newvalue) | (newvalue <= 0) | (newvalue > 500)
    % Revert to last value
    oldvalue = handles.features{i}.parameters.epsilon;
    set(h, 'String', oldvalue);
else % Use new value
    handles.features{i}.parameters.epsilon = newvalue;
end

guidata(h, handles);



%----------------------------------------------------------------------
function varargout = set_k0y(h, eventdata, handles, varargin)
% Find uicontrol in features vector
features = handles.features;
i = 1;
while length(features{i}.uicontrols) < 6 | features{i}.uicontrols(6) ~= h
    i = i + 1;
end

newstrvalue = get(h, 'String');
newvalue = str2double(newstrvalue);

% Check that the entered value falls within the allowable range
if  isempty(newvalue) | (newvalue <= 0) | (newvalue > 20)
    % Revert to last value
    oldvalue = handles.features{i}.parameters.k0y;
    set(h, 'String', oldvalue);
else % Use new value
    handles.features{i}.parameters.k0y = newvalue;
end

guidata(h, handles);


%----------------------------------------------------------------------
function vector = controlsvector(handles)
features = handles.features;

allhandles = rmfield(handles, {'features','imagetype' ...
        'fatherfigure', 'figure1'});

cell = struct2cell(allhandles);
for k = 1:size(cell, 1)
    vector(k) = cell{k};
end

for j = 1:size(features, 2)
    vector = [vector, features{j}.popup, features{j}.remove, ...
                features{j}.frame, features{j}.uicontrols];
end


%----------------------------------------------------------------------
function vector = movevector(handles, i)
features = handles.features;

movehandles = rmfield(handles, {'figure1','pushbutton1', ...
        'pushbutton2','pushbutton3','features','imagetype' ...
        'fatherfigure'});

cell = struct2cell(movehandles);
for k = 1:size(cell, 1)
    vector(k) = cell{k};
end

if nargin < 2
    i = length(features);
else
    i = i - 1;
end

for j = 1:i
    vector = [vector, features{j}.popup, features{j}.remove, ...
                features{j}.frame, features{j}.uicontrols];
end

%----------------------------------------------------------------------
function exit = figure1_CloseRequestFcn(h, eventdata, handles, varargin)
% Close
delete(handles.figure1);
exit = 1;

% --------------------------------------------------------------------
function varargout = pushbutton1_Callback(h, eventdata, handles, varargin)
% Close
delete(handles.figure1);