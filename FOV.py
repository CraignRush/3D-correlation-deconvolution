import pandas as pd
import numpy as np
from readlif.reader import LifFile
from readlif.utilities import get_xml
import xmltodict


# @HACK Add a parameter to select for a certain name
class LIF_info:
    def __init__(self,lif_path, FOV_num=0, FOV_naming = ['Pos','FOV'], overview=False):
        """Initialize the LIF_info object with path, naming conventions, and setup the initial data parsing.
            
            Args:
                lif_path (str): The path to the .lif file.
                FOV_num (int): The field of view number, [0:end]
                FOV_naming (list, optional): A list containing strings used for naming conventions of individual FOVs. Defaults to ['Pos', 'FOV'].
        """

        self.FOV_naming = FOV_naming
        self.FOV_num = FOV_num

        self.lif_path = lif_path
        self.param_dict = self.parse_lif_header(lif_path)
        self.multiAcquisitionFile, self.tilescans = self.get_tilescans()
        self.lif_file = LifFile(lif_path) # @TODO wrap with execption handler    
        self.tileset_index = 1

        if overview:
            self.multiAcquisitionFile, self.tilescans = self.get_tilescans()
            self.tilescan_mosaic, self.tilescan_merged, self.index_mosaic, self.index_merged = self.find_tilescans(self.tilescans)           
            tiles = self.get_mosaic_tiles(self.tilescan_mosaic)
            self.tileset_index = self.index_merged

        else:

            self.tileset_index = self.get_tilescan_index(self.tilescans)

            self.FOV_count = self.get_FOV_count()
            self.FOV_list_names = self.get_FOV_list_names()
            self.FOV_selection = self.get_FOV_selection()
            self.attachment_num = self.get_attachment_num(self.FOV_selection)


            self.microscope_params = self.FOV_selection['Data']['Image']['Attachment'][self.attachment_num]['ATLConfocalSettingDefinition'] #Small helper for dict depth

            self.FOV_name = self.FOV_selection['@Name']
            self.NA = float(self.microscope_params['@NumericalAperture'])
            self.mag = float(self.microscope_params['@Magnification'])
            self.StagePosX, self.StagePosY = self.get_stage_pos()
            self.pinholeAiry = float(self.microscope_params['@PinholeAiry'])
            self.zoom = float(self.microscope_params['@Zoom'])
            self.stack_sections = int(self.microscope_params['@Sections'])
            self.channel_order = self.microscope_params['@ScanMode']
            self.resolution = self.get_resolution_df()
            self.microscope = self.FOV_selection['Data']['Image']['Attachment'][self.attachment_num]['@SystemTypeName']
            if self.microscope == 'TCS SP8':
                self.working_distance_mm = 0.28            
            self.fluo_channels = []
            self.channels = self.get_channels(self.microscope_params)
            self.num_channels = self.get_image().channels

    def find_tilescans(self,tilescans):
        tilescan_merged, tilescan_mosaic = None, None
        for i in range(len(tilescans)):
            el = tilescans[i][1]
            if ' Merged' in el['@Name']:
                tilescan_merged = el
                searchstring = el['@Name'][:-7]
                index_merged = i
        for i in range(len(tilescans)):
            el = tilescans[i][1]
            if searchstring in el['@Name'] and ' Merged' not in el['@Name']:
                tilescan_mosaic = el
                index_mosaic = i

        return tilescan_mosaic,tilescan_merged, index_mosaic, index_merged



    def get_stage_pos(self, FOV_num=None):
        """
        Retrieves the stage position coordinates (X and Y) for a given Field of View (FOV).

        This method determines the stage position from the microscope settings based on the FOV number provided.
        If no FOV number is specified, it uses the default microscope parameters already stored in the object.
        Otherwise, it fetches the specific FOV's parameters to find the relevant stage positions.

        Args:
            FOV_num (int, optional): The index of the FOV for which to retrieve the stage position. 
                                    If None, uses default microscope parameters. Defaults to None.

        Returns:
            tuple: A tuple containing the X and Y coordinates of the stage position as floats.
        """
        # Use default microscope parameters if no FOV number is provided.
        if FOV_num is None:
            FOV = self.FOV_selection
        else:
            # Retrieve the specified FOV's settings based on the provided FOV number.
            FOV = self.get_FOV_selection(FOV_num)  # Fetch the specific FOV configuration.

        x = float(FOV['Data']['Image']['Attachment'][0]['Tile']['@PosX'])
        y = float(FOV['Data']['Image']['Attachment'][0]['Tile']['@PosY'])

        #x,y = self._flip_swap_correction(x,y, FOV)
 
        # Return the stage X and Y positions from the microscope parameters.
        return (x,y)
    def _flip_swap_correction(self, x,y,FOV):
        if FOV['Data']['Image']['Attachment'][self.attachment_num]['ATLConfocalSettingDefinition']['@FlipX'] == '1':
            x = -x
        if FOV['Data']['Image']['Attachment'][self.attachment_num]['ATLConfocalSettingDefinition']['@FlipY'] == '1':
            y = -y
        if FOV['Data']['Image']['Attachment'][self.attachment_num]['ATLConfocalSettingDefinition']['@SwapXY'] == '1':
            x, y = y, x
        return x,y
    def get_mosaic_tiles(self,tilescan):     
        tiles_list = tilescan['Data']['Image']['Attachment'][0]['Tile']  
        tiles = pd.DataFrame(tiles_list)   
        tiles = tiles.astype({'@PosX': float,'@PosY': float})
        return tiles

    def get_image(self,tileset_index = None, FOV_num = None):
        if tileset_index is None:
            tileset_index = self.tileset_index
        if FOV_num is None:
            FOV_num = self.FOV_num

        return self.lif_file.get_image(tileset_index + FOV_num)
    
    def get_FOV_selection(self, FOV_num = None):    
        """Select the specific field of view from the parsed .lif data based on FOV number.  

        Returns:
            dict: The dictionary containing the selected field of view information.
        """    
        if FOV_num is None:
            FOV_num = self.FOV_num
            
        if self.multiAcquisitionFile:
            return self.param_dict['LMSDataContainerHeader']['Element']['Children']['Element'][self.tileset_index]['Children']['Element'][FOV_num]
        else:            
            return self.param_dict['LMSDataContainerHeader']['Element']['Children']['Element']['Children']['Element'][FOV_num]
        
    def get_tilescans(self):
        """
        Retrieves all tile scans from the .lif file's parsed header and checks for multiple acquisition setups.

        This function iterates through elements within the LMSDataContainerHeader of the parsed .lif file. 
        It identifies elements that are specifically marked as 'TileScan' and collects their indices and details.
        The method also determines whether the file contains a single or multiple acquisition setups based on the structure of the data.

        Returns:
            list: A list of lists, where each sub-list contains an index and the corresponding element (tile scan) from the .lif file data.
        """

        tilescans = []
        if type(self.param_dict['LMSDataContainerHeader']['Element']['Children']['Element']) is list:
            multiAcquisitionFile = True
            for i, el in enumerate(self.param_dict['LMSDataContainerHeader']['Element']['Children']['Element']):
                if 'TileScan' in el['@Name']:
                    tilescans.append([i,el])
        else: 
            multiAcquisitionFile = False
            tilescans.append([0,self.param_dict['LMSDataContainerHeader']['Element']['Children']['Element']['@Name']]) 
        return multiAcquisitionFile, tilescans  
    
    
    def get_tilescan_index(self,tilescans):
        """Determine the index of the tilescan that contains the field of view names matching the naming convention.      

        Returns:
            int: The index of the tilescan.      
              
        Raises:
            Exception: If no tilescan matches the naming convention, an exception is raised.
        """

        if len(tilescans) > 1:  # @TODO check for Pos or FOV string            
            for el in tilescans:
                if any(s in self.param_dict['LMSDataContainerHeader']['Element']['Children']['Element'][el[0]]['Children']['Element'][0]['@Name'] for s in self.FOV_naming):
                    tileset_index = el[0]
                    break
        elif len(tilescans) == 1:
            tileset_index = tilescans[0][0]
        else:
            raise Exception("Error, I couldn't detect any TileScans containing {}".format(' or '.join(self.FOV_naming)))
        return tileset_index
    
    def get_resolution_df(self):    
        """Construct a DataFrame containing the resolution details for each dimension of the image.
        
        Returns:
            DataFrame: A DataFrame with the resolution details for xyz
        """    
        resolution_params = self.FOV_selection['Data']['Image']['ImageDescription']['Dimensions']['DimensionDescription']
        resolution = pd.DataFrame({'dimension_name': None, 'dimension_number': None, 'pixels': None, 'length': None, 'unit': None, 'resolution_nm': None}, index=[0])
        
        for dim in range(len(resolution_params)):
            resolution.loc[dim,'dimension_number'] = int(resolution_params[dim]['@DimID'])
            resolution.loc[dim,'dimension_name']= self.channel_order[resolution.loc[dim,'dimension_number'] - 1]
            resolution.loc[dim,'pixels'] = int(resolution_params[dim]['@NumberOfElements'])
            resolution.loc[dim,'length'] = float(resolution_params[dim]['@Length'])
            resolution.loc[dim,'unit'] = resolution_params[dim]['@Unit']
            if resolution.loc[dim,'unit']  == 'm':
                resolution.loc[dim,'resolution_nm'] = float(resolution.loc[dim,'length']) / int(resolution.loc[dim,'pixels']) * 1e9
            return resolution
    
    def get_attachment_num(self, FOV):
        """Identify the number/index of the attachment within the FOV's data containing specific settings.
        
        Args:
            FOV (dict): The current FOV containing settings definitions.
        
        Returns:
            int: The index of the attachment containing the 'ATLConfocalSettingDefinition'.
        """
        for i in range(len(FOV['Data']['Image']['Attachment'])):
            if 'ATLConfocalSettingDefinition' in FOV['Data']['Image']['Attachment'][i]:
                return i

    def get_FOV_list_names(self):
        """Retrieve a list with the names of all field of views from the tile scan index.
        
        Returns:
            list: A list containing the names of all fields of view.
        """

        if self.multiAcquisitionFile:
            return [self.param_dict['LMSDataContainerHeader']['Element']['Children']['Element'][self.tileset_index]['Children']['Element'][i]['@Name'] for i in range(0,self.FOV_count)]
        else:
            return [self.param_dict['LMSDataContainerHeader']['Element']['Children']['Element']['Children']['Element'][i]['@Name'] for i in range(0,self.FOV_count)]
            
    def get_FOV_dict_coords(self):        
        FOV_coords = pd.DataFrame({'FOV_num': None, 'FOV_name': None, 'X': None, 'Y': None}, index=[0])
        for i in range(0,self.FOV_count):   
            FOV_coords.loc[i,'FOV_num'] = i
            FOV_coords.loc[i,'FOV_name'] = self.FOV_list_names[i]
            FOV_coords.loc[i,'X'] = self.get_stage_pos(i)[0]
            FOV_coords.loc[i,'Y'] = self.get_stage_pos(i)[1]
        return FOV_coords        

        
    def get_channel_stack(self,channel_num=0,all_channels=False):
        """Retrieve image data for a specific channel or all channels from the .lif file.
        
        Args:
            channel_num (int, list, optional): The channel number(s) to retrieve. Defaults to 0.
            all_channels (bool, optional): Whether to retrieve all channels. Defaults to False.
        
        Returns:
            ndarray: An array containing the image data for the specified channels.
        """

        self.lif_stack = self.lif_file.get_image(self.tileset_index + self.FOV_num)
        if all_channels:
            return np.array([[np.array(i) for i in self.lif_stack.get_iter_z(t=0, c=chan)] for chan in range(self.lif_stack.channels)])
        elif len(channel_num) > 1:
            return np.array([[np.array(i) for i in self.lif_stack.get_iter_z(t=0, c=chan)] for chan in channel_num])
        else:
            return np.array([np.array(i) for i in self.lif_stack.get_iter_z(t=0, c=channel_num)])

    def get_FOV_count(self):
        """Determine the number of fields of view available in the .lif file for the current tile scan.
        
        Returns:
            int: The count of fields of view.
        """
        if self.multiAcquisitionFile:
            return len(self.param_dict['LMSDataContainerHeader']['Element']['Children']['Element'][self.tileset_index]['Children']['Element'])
        else:
            return len(self.param_dict['LMSDataContainerHeader']['Element']['Children']['Element']['Children']['Element'])

    def parse_lif_header(self,lif_path=None):
        """Parse the header of the .lif file to extract parameters for the images.
        
        Args:
            lif_path (str, optional): Path to the .lif file. If not provided, use the instance's lif_path.
        
        Returns:
            dict: A dictionary representing the parsed XML header.
        """
        if lif_path is not None:
            lif_path = self.lif_path
        param_xml = get_xml(lif_path)
        param_dict = xmltodict.parse(param_xml[1])
        return param_dict
    
    def get_channels(self,FOV_params):
        """Parse channel information from the microscope parameters for the field of view.
        
        Args:
            FOV_params (dict): The parameters containing details about the channels.
        
        Returns:
            list: A list of dictionaries, each representing a channel with its settings.
        """
        #@FIXME there is an error, there are more channels detected than are active. Maybe a sequnce thingy...
        channels = [{'number': 0,
                    'name': "",
                    'scan_type': '',
                    'dye_preset': '',
                    'detection_window': ''
                    } for i in range(len(FOV_params['DetectorList']['Detector']))]
        for c in range(len(FOV_params['DetectorList']['Detector'])):
            chan_params = dict(FOV_params['DetectorList']['Detector'][c])
            #if chan_params['@IsActive'] == '0':
            #    continue
            channels[c]['number'] = c
            channels[c]['name'] = chan_params['@Name'] 
            channels[c]['scan_type'] = chan_params['@ScanType']    
            channels[c]['gain'] = float(chan_params['@Gain'])
            if chan_params['@ScanType']  == 'Internal':
                self.fluo_channels.append(channels[c]['number'])

        for b in range(len(FOV_params['Spectro']['MultiBand'])):
            if type(FOV_params['Spectro']['MultiBand'][b]) is dict:
                ind = int(FOV_params['Spectro']['MultiBand'][b]['@Channel'])-1
                channels[ind]['dye_preset'] = FOV_params['Spectro']['MultiBand'][b]['@DyeName']
                channels[ind]['detection_window'] = (float(FOV_params['Spectro']['MultiBand'][b]['@LeftWorld']), float(FOV_params['Spectro']['MultiBand'][b]['@RightWorld']))
                channels[ind]['center_wavelength'] = (channels[ind]['detection_window'][1] + channels[ind]['detection_window'][0]) / 2
        return channels
                    
    def print(self, unselected_keys = ['param_xml', 'param_dict', 'FOV_selection', 'tilescans'] ):
        """Print a filtered set of the FOV object's attributes excluding certain keys.
        
        Returns:
            dict: A dictionary of the filtered variables, excluding some internal ones for clarity.
        """
        # Create a new dictionary excluding the unselected_keys
        filtered_vars = {key: value for key, value in vars(self).items() if key not in unselected_keys}

        try:
            from pprint import pprint
            pprint(filtered_vars)
            return(filtered_vars)
        except ImportError as e:            
            print(filtered_vars)

    def plot_FOVs(self):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        df = self.get_FOV_dict_coords()
        df['X'] = df['X']*1e3
        df['Y'] = df['Y']*1e3
        # Plotting
        fig, ax = plt.subplots()
        scale = self.resolution[self.resolution['dimension_name'] == 'x']['length'][0] *1e3 # Example scale factor for the size of the rectangles

        # Add rectangles for each FOV
        for index, row in df.iterrows():
            rect = patches.Rectangle((row['X'], row['Y']), scale, scale, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            # Optionally add text labels
            ax.text(row['X'] + scale / 2, row['Y'] + scale*0.1, row['FOV_name'], ha='center', va='center',size=8)

        ax.set_xlim(df['X'].min() - scale*3, df['X'].max() + scale*3)
        ax.set_ylim(df['Y'].min() - scale*3, df['Y'].max() + scale*3)
        ax.set_xlabel('X Position / mm')
        ax.set_ylabel('Y Position / mm')
        ax.set_title('Field of View (FOV) Positions')
        return fig,ax
