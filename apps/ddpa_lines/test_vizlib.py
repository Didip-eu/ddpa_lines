
    def Dtest_segmentation_dict_to_polygons_lines( self ):
        """
        To be removed (visualization code)
        """
        with open( self.data_path.joinpath('segdict_NA-ACK_14201223_01485_r-r1+model_20.json'), 'r') as segdict_file, Image.open( self.data_path.joinpath('NA-ACK_14201223_01485_r-r1_reduced.png'), 'r') as input_image:
            segdict = json.load( segdict_file )
            lbl, polygons = seglib.dict_to_polygon_lines( segdict, input_image )
            self.assertTrue( True )

