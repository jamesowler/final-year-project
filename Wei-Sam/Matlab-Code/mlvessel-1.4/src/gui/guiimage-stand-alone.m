function varargout = guiimage(varargin)
% GUIIMAGE Application M-file for guiimage.fig
%    FIG = GUIIMAGE launch guiimage GUI.
%    GUIIMAGE('callback_name', ...) invoke the named callback.
%
% Window for viewing an image, with callbacks for viewing features
% and segmenting the image.

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
	set(fig,'Color',get(0,'defaultUicontrolBackgroundColor'));

    % Opens image and inicializes names.
    params = varargin{1};
    title = [params.filename '-' int2str(params.imagecount)];
    set(fig, 'Name', title);
    original = imread(fullfile(params.pathname, params.filename));
    
    % Showing image preserving oringinal aspect ratio.
    origsize = size(original);

    oldunit = get(fig, 'Unit');
    set(fig, 'Unit', 'pixels');
    pos = get(fig, 'Position');

    yratio = pos(4) / origsize(1);
    xratio = pos(3) / origsize(2);
    if xratio < yratio
        pos = [pos(1), pos(2) + pos(4) - origsize(1) * xratio, pos(3), origsize(1) * xratio];
    else
        pos = [pos(1:2), origsize(2) * yratio, pos(4)];
    end
    set(fig, 'Position', pos);
    set(fig, 'Unit', oldunit);

    figure(fig);
    ax = axes;
    image(original);
    colormap(gray(256));
    set(ax, 'XTick', []);
    set(ax, 'YTick', []);
    set(ax, 'Position', [0.03 0.03 0.94 0.94]);
    set(ax, 'DataAspectRatio', [1 1 1]);
    
    % Generate a structure of handles to pass to callbacks, and
    % store it. 
    handles = guihandles(fig);
    handles.pathname = params.pathname;
    handles.filename = params.filename;
    handles.mainfig = params.mainfig;
    handles.title = title;
    handles.original = original;
    handles.children = [];
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
function varargout = menu_image_Callback(h, eventdata, handles, varargin)




% --------------------------------------------------------------------
function varargout = menu_image_open_Callback(h, eventdata, handles, varargin)
guimain('menu_file_image_open_Callback', handles.mainfig, [], guidata(handles.mainfig));



% --------------------------------------------------------------------
function varargout = menu_image_features_Callback(h, eventdata, handles, varargin)

% Check to see if a features dialog is already open.
open = 0;
children = handles.children;
for i=1:length(children)
    child = children(i);
    if strcmp(child.type, 'guifeaturesdialog') & ishandle(child.figure)
        open = i;
    end
end
   
if open == 0 % No dialog open yet, open new.
    params.title = handles.title;
    params.fatherfigure = handles.figure1;
    child.figure = guifeaturesdialog(params);
    child.type = 'guifeaturesdialog';
    handles.children = [children child];
    guidata(h, handles);
else % Call focus to open dialog
    figure(children(i).figure);
end


% --------------------------------------------------------------------
function varargout = menu_image_segment_Callback(h, eventdata, handles, varargin)
params.imagefigure = handles.figure1;
params.classifierfigure = [];
guimain('segment_interface', handles.mainfig, [], guidata(handles.mainfig), params);



% --------------------------------------------------------------------
function varargout = menu_image_close_Callback(h, eventdata, handles, varargin)
guiexit(h, eventdata, handles, varargin);



% --------------------------------------------------------------------
function exit = figure1_CloseRequestFcn(h, eventdata, handles, varargin)
exit = guiexit(h, eventdata, handles, varargin);



% --------------------------------------------------------------------
function exit = guiexit(h, eventdata, handles, varargin)
children = handles.children;

i = 1;
exit = 1;
nchildren = length(children);
while i <= nchildren & exit
    if ishandle(children(i).figure)
        exit = feval(children(i).type, 'figure1_CloseRequestFcn', children(i).figure, [], guidata(children(i).figure));
    end
    i = i + 1;
end

if exit
    delete(handles.figure1);
end



% --------------------------------------------------------------------
function varargout = show_features(h, eventdata, handles, imagetype, features)

% Generates specified features.
if ~isdeployed
  addpath([installdir filesep 'src' filesep 'ftrs']);
end

[features, mask] = guigeneratefeatures(handles.original, imagetype, features);

params.title = ['Features for ' handles.title];
params.results = features;
for i = 1:length(params.results)
  data = params.results{i}.data;
  data = data .* double(mask);
  params.results{i}.data = data / max(data(:));
end

child.figure = guiresult(params);
child.type = 'guiresult';

guimain('add_child', handles.mainfig, [], guidata(handles.mainfig), child);


% --------------------------------------------------------------------
function varargout = figure1_CreateFcn(h, eventdata, handles, varargin)

% --------------------------------------------------------------------
function varargout = figure1_ResizeFcn(h, eventdata, handles, varargin)
