function varargout = guiresult(varargin)
% GUIRESULT Application M-file for guiresult.fig
%    FIG = GUIRESULT launch guiresult GUI.
%    GUIRESULT('callback_name', ...) invoke the named callback.
%
% Shows results of imges and allows user to save them.

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
    set(fig, 'Name', params.title);
    
    figpos = get(fig, 'Position');

    figure(fig);
    ax = axes;
    imagehandle = image(params.results{1}.data);
    set(imagehandle, 'CDataMapping', 'scaled');
    colormap(gray);
    set(ax, 'XTick', []);
    set(ax, 'YTick', []);
    set(ax, 'Unit', 'characters');
    set(ax, 'Position', [1.7 0.7 (figpos(3) - 3.4) (figpos(4) - 3.5)]);
    set(ax, 'DataAspectRatio', [1 1 1]);
    
    handles = guihandles(fig);
    
    set(handles.text1, 'Unit', 'characters');
    set(handles.popupmenu1, 'Unit', 'characters');
    poppos = get(handles.popupmenu1, 'Position');
    textpos = get(handles.text1, 'Position');
    
    poppos(2) = figpos(4) - poppos(2);
    textpos(2) = figpos(4) - textpos(2);
        
    % Prepares popup-menu
    popupstring = params.results{1}.name;
    for i=2:length(params.results)
        popupstring = [popupstring '|' params.results{i}.name];
    end
    set(handles.popupmenu1, 'String', popupstring);
   
    handles.activeimg = 1;
    handles.results = params.results;
    handles.poppos = poppos;
    handles.textpos = textpos;
    guidata(fig, handles);
    handles.imagehandle = imagehandle;
    handles.axes = ax;

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
function varargout = menu_result_Callback(h, eventdata, handles, varargin)



% --------------------------------------------------------------------
function varargout = menu_result_saveas_Callback(h, eventdata, handles, varargin)

% Get image name for saving.
filterspec = {'*.jpg;*.jpeg;*.tif;*.tiff;*.png;*.bmp', ...
        'All MATLAB Image Files (*.jpg,*.jpeg,*.tif,*.tiff,*.png,*.bmp)'; ...
		'*.jpeg;*.jpg', 'JPEG Files (*.jpeg,*.jpg)'; ...
		'*.tiff;*.tif', 'TIFF Files (*.tiff,*.tif)'; ...
		'*.png', 'PNG Files (*.png)'; ...
		'*.bmp', 'Bitmap Files (*.bmp)'; ...
		'*.*', 'All Files (*.*)'};
[filename, pathname] = uiputfile(filterspec, 'Save image as');

% Check if user cancelled.
if isequal(filename,0) | isequal(pathname,0)
	return;
end

% Check filename extension.
[path,name,ext,version] = fileparts(filename);
if strcmp(ext, '.jpg') | strcmp(ext, '.jpeg') | strcmp(ext, '.tif') | ...
        strcmp(ext, '.tiff') | strcmp(ext, '.png') | strcmp(ext, '.bmp')
    imwrite(handles.results{handles.activeimg}.data, ...
        [pathname filesep filename]);
else
    uiwait(warndlg('File not saved: incorrect or unavailable filename extension.', ...
        'Warning','modal'));
end
    

% --------------------------------------------------------------------
function varargout = menu_result_close_Callback(h, eventdata, handles, varargin)
guiexit(h, eventdata, handles, varargin);


% --------------------------------------------------------------------
function varargout = popupmenu1_Callback(h, eventdata, handles, varargin)
value = get(h, 'Value');

axes(handles.axes);
newimage = handles.results{value}.data;
set(handles.imagehandle, 'CData', newimage);
handles.activeimg = value;
guidata(h, handles);


% --------------------------------------------------------------------
function varargout = figure1_CreateFcn(h, eventdata, handles, varargin)



% --------------------------------------------------------------------
function varargout = figure1_ResizeFcn(h, eventdata, handles, varargin)

if isfield(handles, 'axes')
  poppos = handles.poppos;
  textpos = handles.textpos;

  figpos = get(handles.figure1, 'Position');

  set(handles.axes, 'Position', [1.7 0.7 (figpos(3) - 3.4) (figpos(4) - 3.5)]);
  set(handles.popupmenu1, 'Position', [poppos(1) (figpos(4) - poppos(2)) poppos(3:4)]);
  set(handles.text1, 'Position', [textpos(1) (figpos(4) - textpos(2)) textpos(3:4)]);
end

% --------------------------------------------------------------------
function exit = figure1_CloseRequestFcn(h, eventdata, handles, varargin)
exit = guiexit(h, eventdata, handles, varargin);



% --------------------------------------------------------------------
function exit = guiexit(h, eventdata, handles, varargin)
delete(handles.figure1);
exit = 1;