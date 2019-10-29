function varargout = guisegment(varargin)
% GUISEGMENT Application M-file for guisegment.fig
%    FIG = GUISEGMENT launch guisegment GUI.
%    GUISEGMENT('callback_name', ...) invoke the named callback.
%
% Window for choosing classifier and image to be segmented.

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

	fig = openfig(mfilename,'reuse');

	% Use system color scheme for figure:
	set(fig,'Color',get(0,'defaultUicontrolBackgroundColor'));

  	handles = guihandles(fig);

      	set(handles.text1, 'FontSize', 11);
      	set(handles.text2, 'FontSize', 11);

       	set(handles.pushbutton2, 'String', 'Open...');
	set(handles.pushbutton3, 'String', 'Open...');
        
    params = varargin{1};
    % Set general info.
    handles.mainfig = params.mainfig;

    % Set images info.
    handles.images = params.images;
    handles.imagefigure = params.imagefigure;
    % Set images popupmenu.
    if length(handles.images) == 0
        set(handles.popupmenu2, 'String', 'None available');
        set(handles.popupmenu2, 'Enable', 'off');
    else
        popstring = handles.images(1).name;
        for i=2:length(handles.images)
            popstring = [popstring '|' handles.images(i).name];
        end
        set(handles.popupmenu2, 'String', popstring);
        set(handles.popupmenu2, 'Enable', 'on');
        
        % Choose default value if available.
        if ~isempty(handles.imagefigure)
            value = 1;
            while handles.images(value).figure ~= handles.imagefigure
                value = value + 1;
            end
            set(handles.popupmenu2, 'Value', value);
        else % Set image as first available.
          handles.imagefigure = handles.images(1).figure;
        end
    end
    
    % Set classifier info.
    handles.classifiers = params.classifiers;
    handles.classifierfigure = params.classifierfigure;
    % Set classifiers popupmenu.
    if length(handles.classifiers) == 0
        set(handles.popupmenu1, 'String', 'None available');
        set(handles.popupmenu1, 'Enable', 'off');
    else
        popstring = handles.classifiers(1).name;
        for i=2:length(handles.classifiers)
            popstring = [popstring '|' handles.classifiers(i).name];
        end
        set(handles.popupmenu1, 'String', popstring);
        set(handles.popupmenu1, 'Enable', 'on');
        
        % Choose default value if available.
        if ~isempty(handles.classifierfigure)
            value = 1;
            while handles.classifiers(value).figure ~= handles.classifierfigure
                value = value + 1;
            end
            set(handles.popupmenu1, 'Value', value);
        else % Set classifier as first available.
          handles.classifierfigure = handles.classifiers(1).figure;
        end
        
    end
        
    % Generate a structure of handles to pass to callbacks, and store it. 
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
function varargout = figure1_CloseRequestFcn(h, eventdata, handles, varargin)
exit = guiexit(h, eventdata, handles, varargin);



% --------------------------------------------------------------------
function varargout = popupmenu1_Callback(h, eventdata, handles, varargin)
% Classifier choice.
value = get(h, 'Value');
handles.classifierfigure = handles.classifiers(value).figure;

guidata(h, handles);



% --------------------------------------------------------------------
function varargout = popupmenu2_Callback(h, eventdata, handles, varargin)
% Image choice.
value = get(h, 'Value');
handles.imagefigure = handles.images(value).figure;

guidata(h, handles);




% --------------------------------------------------------------------
function varargout = pushbutton2_Callback(h, eventdata, handles, varargin)
% Open new classifier.
classifierfigure = guimain('menu_file_classifier_open_Callback', handles.mainfig, [], guidata(handles.mainfig));

% Reset segment window.
if ~isempty(classifierfigure)
    params.classifierfigure = classifierfigure;
else
    params.classifierfigure = handles.classifierfigure;
end
params.imagefigure = handles.imagefigure;
guimain('segment_interface', handles.mainfig, [], guidata(handles.mainfig), params);


% --------------------------------------------------------------------
function varargout = pushbutton3_Callback(h, eventdata, handles, varargin)
% Open new image
imagefigure = guimain('menu_file_image_open_Callback', handles.mainfig, [], guidata(handles.mainfig));

% Reset segment window.
if ~isempty(imagefigure)
    params.imagefigure = imagefigure;
else
    params.imagefigure = handles.imagefigure;
end
params.classifierfigure = handles.classifierfigure;
guimain('segment_interface', handles.mainfig, [], guidata(handles.mainfig), params);


% --------------------------------------------------------------------
function varargout = pushbutton4_Callback(h, eventdata, handles, varargin)
% Segment image.

% Verify if there is both a chosen classifier and image.
if isempty(handles.classifierfigure)
  uiwait(warndlg('Please open and select a classifier.','Warning', ...
                 'modal'));
  return;
end

if isempty(handles.imagefigure)
  uiwait(warndlg('Please open and select an image.','Warning', ...
                 'modal'));
  return;
end

% Disable handles and print wait message.
basichandles = rmfield(handles, 'text1');

set(handles.text1, 'Enable', 'off');
set(handles.text2, 'Enable', 'off');
set(handles.popupmenu1, 'Enable', 'off');
set(handles.popupmenu2, 'Enable', 'off');
set(handles.pushbutton2, 'Enable', 'off');
set(handles.pushbutton3, 'Enable', 'off');
set(handles.pushbutton4, 'Enable', 'off');
set(handles.pushbutton5, 'Enable', 'off');

uicontrol(handles.figure1, 'Style', 'frame', ...
          'Units', 'normalized', ...
          'Position', [0.2 0.1 0.6 0.3]);

uicontrol(handles.figure1, 'Style', 'text', ...
          'Units', 'characters', ...
          'String', 'Segmenting image. Please wait...', ...
          'Position', [30 1.5 20 2]);

drawnow;

params.imagefigure = handles.imagefigure;
params.classifierfigure = handles.classifierfigure;

% call guimain to segment and open new window.
guimain('segment', handles.mainfig, [], guidata(handles.mainfig), params);

guiexit(h, [], handles);



% --------------------------------------------------------------------
function exit = pushbutton5_Callback(h, eventdata, handles, varargin)
exit = guiexit(h, eventdata, handles, varargin);

function exit = guiexit(h, eventdata, handles, varargin)
delete(handles.figure1);
exit = 1;