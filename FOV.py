import pandas as pd
import numpy as np
from readlif.reader import LifFile
from readlif.utilities import get_xml
import xmltodict

class FOV:
    def __init__(self,lif_path, FOV_num):
        self.FOV_num = FOV_num
        self.lif_path = lif_path

        self.parse_lif_header(lif_path)

        tilescans=[]         # @TODO find tilescan with Pos or FOV
        for i, el in enumerate(self.param_dict['LMSDataContainerHeader']['Element']['Children']['Element']):
            if 'TileScan' in el['@Name']:
                tilescans.append([i,el])

        if len(tilescans) > 1:  # @TODO check for Pos or FOV string            
            for el in tilescans:
                if ('Pos' in self.param_dict['LMSDataContainerHeader']['Element']['Children']['Element'][el[0]]['Children']['Element'][0]['@Name']) or \
                ('FOV' in self.param_dict['LMSDataContainerHeader']['Element']['Children']['Element'][el[0]]['Children']['Element'][0]['@Name']):
                    self.tileset_index = el[0]
                    break
        elif len(tilescans) == 1:
            self.tileset_index = tilescans[0][0]
        else:
            Exception("Error, I couldn't detect any TileScans containing 'FOV' or 'Pos'")

        self.FOV_count = self.get_FOV_count()

        self.selected_FOV = self.param_dict['LMSDataContainerHeader']['Element']['Children']['Element'][self.tileset_index]['Children']['Element'][FOV_num]

        microscope_params = self.selected_FOV['Data']['Image']['Attachment'][4]['ATLConfocalSettingDefinition'] #Small helper for dict depth
        self.FOV_name = self.selected_FOV['@Name']
        self.NA = float(microscope_params['@NumericalAperture'])
        self.mag = float(microscope_params['@Magnification'])
        self.StagePosX = float(microscope_params['@StagePosX'])
        self.StagePosY = float(microscope_params['@StagePosY'])
        self.pinholeAiry = float(microscope_params['@PinholeAiry'])
        self.zoom = float(microscope_params['@Zoom'])
        self.stack_sections = float(microscope_params['@Sections'])

        resolution_params = self.selected_FOV['Data']['Image']['ImageDescription']['Dimensions']['DimensionDescription']
        self.channel_order = microscope_params['@ScanMode']
        self.resolution = pd.DataFrame({'dimension_name': None, 'dimension_number': None, 'pixels': None, 'length': None, 'unit': None, 'resolution_nm': None}, index=[0])
        for dim in range(len(resolution_params)):
            self.resolution.loc[dim,'dimension_number'] = int(resolution_params[dim]['@DimID'])
            self.resolution.loc[dim,'dimension_name']= self.channel_order[self.resolution.loc[dim,'dimension_number'] - 1]
            self.resolution.loc[dim,'pixels'] = int(resolution_params[dim]['@NumberOfElements'])
            self.resolution.loc[dim,'length'] = float(resolution_params[dim]['@Length'])
            self.resolution.loc[dim,'unit'] = resolution_params[dim]['@Unit']
            if self.resolution.loc[dim,'unit']  == 'm':
                self.resolution.loc[dim,'resolution_nm'] = float(self.resolution.loc[dim,'length']) / int(self.resolution.loc[dim,'pixels']) * 1e9




        self.microscope = self.selected_FOV['Data']['Image']['Attachment'][4]['@SystemTypeName']
        if self.microscope == 'TCS SP8':
            self.working_distance_mm = 0.28
        
        self.fluo_channels = 0
        self.set_channels(microscope_params)

        self.lif_file = LifFile(lif_path) # @TODO wrap with execption handler

        del self.param_dict

        
        
    def get_channel_stack(self,channel_num=0,all_channels=False):
        self.lif_stack = self.lif_file.get_image(self.tileset_index + self.FOV_num)
        if all_channels:
            return np.array([[i for i in self.lif_stack.get_iter_z(t=0, c=chan)] for chan in range(self.lif_stack.channels)])
        else:
            return np.array([[i for i in self.lif_stack.get_iter_z(t=0, c=channel_num)]])

    def get_FOV_count(self):
        return len(self.param_dict['LMSDataContainerHeader']['Element']['Children']['Element'][self.tileset_index]['Children']['Element'])

    def parse_lif_header(self,lif_path=None):
        if lif_path is not None:
            lif_path = self.lif_path
        self.param_xml = get_xml(lif_path)
        self.param_dict = xmltodict.parse(self.param_xml[1])
        return self.param_dict
    
    def set_channels(self,FOV_params):
            self.channels = [{'number': 0,
                              'name': "",
                              'scan_type': '',
                              'dye_preset': '',
                              'detection_window': ''
                              } for i in range(len(FOV_params['DetectorList']['Detector']))]
            for c in range(len(FOV_params['DetectorList']['Detector'])):
                chan_params = dict(FOV_params['DetectorList']['Detector'][c])
                self.channels[c]['number'] = c
                self.channels[c]['name'] = chan_params['@Name'] 
                self.channels[c]['scan_type'] = chan_params['@ScanType']    
                self.channels[c]['gain'] = float(chan_params['@Gain'])
                if chan_params['@ScanType']  == 'Internal':
                    self.fluo_channels += 1

            for b in range(len(FOV_params['Spectro']['MultiBand'])):
                if type(FOV_params['Spectro']['MultiBand'][b]) is dict:
                    ind = int(FOV_params['Spectro']['MultiBand'][b]['@Channel'])-1
                    self.channels[ind]['dye_preset'] = FOV_params['Spectro']['MultiBand'][b]['@DyeName']
                    self.channels[ind]['detection_window'] = (float(FOV_params['Spectro']['MultiBand'][b]['@LeftWorld']), float(FOV_params['Spectro']['MultiBand'][b]['@RightWorld']))
                    self.channels[ind]['center_wavelength'] = (self.channels[ind]['detection_window'][1] + self.channels[ind]['detection_window'][0]) / 2
                    
    def print(self):
        try:
            from pprint import pprint
            pprint(vars(self))
        except ImportError as e:            
            print(vars(self))