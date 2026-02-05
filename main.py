class PieceData:
    def __init__(self, num, rio_time, image_time, type, distance, angle):
        self.image_num = num
        self.rio_time = rio_time
        self.image_time = time_time
        self.type = type
        self.distance = distance
        self.angle = angle


def fuel_regress_distance(y):
    terms = [
     5.6042530346620970e+002,
    -2.1541292739721818e+001,
     4.6098851143728686e-001,
    -5.9768933371784367e-003,
     4.8838220799736427e-005,
    -2.5636133459344433e-007,
     8.6119866667699354e-010,
    -1.7871693995785495e-012,
     2.0846186505905067e-015,
    -1.0446668966595392e-018
    ]
    
    t = 1
    r = 0
    for c in terms:
        r += c * t
        t *= y
    return r

def fuel_regress_px_per_deg(x):
    terms = [
     3.0950248997394785e+000,
     1.5607142915418259e-001,
    -1.9222516478834060e-003,
     8.9708743706644227e-006
    ]   

    t = 1
    r = 0
    for c in terms:
        r += c * t
        t *= x
    return r

def pose_data_string(sequence_num, rio_time, time, tags, tag_poses, nt_objects):
    string_header = ""
    string_header = f'num={sequence_num} t_rio={rio_time:1.3f} t_img={time:1.3f} len={len(tags)}'

    string_data_rot = f'tags={len(tags)} '
    string_data_t = f'tags={len(tags)} '
    tag_pose = 0
    
    for tag in tags:
        
        x_deg = math.degrees(tag_poses[tag_pose].rotation().X())
        y_deg = math.degrees(tag_poses[tag_pose].rotation().Y())
        z_deg = math.degrees(tag_poses[tag_pose].rotation().Z())
        x_in = (tag_poses[tag_pose].translation().X() * 39.3701)
        y_in = (tag_poses[tag_pose].translation().Y() * 39.3701)
        z_in = (tag_poses[tag_pose].translation().Z() * 39.3701)

        x_deg_str = f'{x_deg:3.1f}'
        y_deg_str = f'{y_deg:3.1f}'
        z_deg_str = f'{z_deg:3.1f}'

        x_in_str = f'{x_in:3.1f}'
        y_in_str = f'{y_in:3.1f}'
        z_in_str = f'{z_in:3.1f}'

        string_data_rot += f'id={tag.getId()} \
        x_deg={x_deg_str} \
        y_deg={y_deg_str} \
        z_deg={z_deg_str} '

        id = tag.getId()
        dm = tag.getDecisionMargin()
        errors = tag.getHamming()
    
        id_str = f'{id}'
        dm_str = f'{dm:5.1f}'
        errors_str = f'{errors}'

        string_data_t += f'id={id_str} dm={dm_str} e={errors_str} \
        x_in={x_in_str} \
        #y_in={(tag_poses[tag_pose].translation().Y() - (0.0075 * tag_poses[tag_pose].translation().Z())  * 39.37):3.1f} \
        y_in={y_in_str} \
        z_in={z_in_str} '
        tag_pose +=1
    
        nt_objects[0].set(id)
        nt_objects[1].set(dm)
        nt_objects[2].set(errors)
        nt_objects[3].set(x_deg)
        nt_objects[4].set(y_deg)
        nt_objects[5].set(z_deg)
        nt_objects[6].set(x_in)
        nt_objects[7].set(y_in)
        nt_objects[8].set(z_in)

    return string_header, string_data_rot, string_data_t

def piece_pose_data_string(sequence_num, rio_time, time, dist, angle):
    string_header = f'num={sequence_num} t_rio={rio_time:1.3f} t_img={time:1.3f} z_in={dist:3.1f} y_deg={angle:3.1f}'
    
    return string_header


def draw_tags(img, tags, tag_poses, rVector, tVector, camMatrix, distCoeffs, crop_top):
    tag_pose = 0

    # need to add crop_top back to all the y dimensions in order to draw everything 
    # at the right place on the full image 
    # because the tag detect locations are for a (potentially) cropped image from the full image
    
    for tag in tags:
        x0 = int(tag.getCorner(0).x)
        y0 = int(tag.getCorner(0).y) + crop_top
        x1 = int(tag.getCorner(1).x)
        y1 = int(tag.getCorner(1).y) + crop_top
        x2 = int(tag.getCorner(2).x)
        y2 = int(tag.getCorner(2).y) + crop_top
        x3 = int(tag.getCorner(3).x)
        y3 = int(tag.getCorner(3).y) + crop_top
        cv2.line(img, (x0, y0), (x1, y1), (0,255,0), 3) #starts at top left corner of apriltag
        cv2.line(img, (x1, y1), (x2, y2), (0,255,0), 3) #top left to bottom left
        cv2.line(img, (x2, y2), (x3, y3), (0,255,0), 3) #bottom left to bottom right
        cv2.line(img, (x3, y3), (x0, y0), (0,255,0), 3) #bottom right to top right
        fontFace = cv2.FONT_HERSHEY_TRIPLEX
        fontScale = 1
        fontColor = (0, 0, 255)
        thickness = 2
        IdStr = str(tag.getId())
        textWidth, textHeight = cv2.getTextSize(IdStr, fontFace, fontScale, thickness)[0]
        CenterCoordinates = (int(tag.getCenter().x) - int(textWidth / 2), int( tag.getCenter().y + crop_top) + int(textHeight / 2))
        cv2.putText(img, IdStr, CenterCoordinates, fontFace, fontScale, fontColor, thickness) # ID in center

        rVector[0][0] = tag_poses[tag_pose].rotation().X()
        rVector[1][0] = tag_poses[tag_pose].rotation().Y()
        rVector[2][0] = tag_poses[tag_pose].rotation().Z()
        tVector[0][0] = tag_poses[tag_pose].translation().X()
        tVector[1][0] = tag_poses[tag_pose].translation().Y() + crop_top
        tVector[2][0] = tag_poses[tag_pose].translation().Z()
        tag_pose += 1
        #for rotation, ask if its shrinking on each axis 
        cv2.drawFrameAxes(img, camMatrix, distCoeffs, rVector, tVector, .076, 1)
    return img

def file_write_tags(file, 
               threads,
                decimate, 
                blur, 
                refine, 
                sharpen, 
                atdebug, 
                decisionmargin_min,
                decisionmargin_max,
                crop_x,
                crop_y,
                errors,
                ):

    parser = configparser.ConfigParser()

    parser.add_section('VISION')
    parser.set('VISION', THREADS_TOPIC_NAME, str(int(threads)))
    parser.set('VISION', BLUR_TOPIC_NAME, str(blur))
    parser.set('VISION', REFINE_EDGES_TOPIC_NAME, str(refine))
    parser.set('VISION', SHARPENING_TOPIC_NAME, str(round(sharpen,2)))
    parser.set('VISION', APRILTAG_DEBUG_MODE_TOPIC_NAME, str(atdebug))
    parser.set('VISION', DECISION_MARGIN_MIN_TOPIC_NAME, str(round(decisionmargin_min)))
    parser.set('VISION', DECIMATE_TOPIC_NAME, str(round(decimate,2)))
    parser.set('VISION', TAG_CONFIG_FILE_TOPIC_NAME, str(file))
    parser.set('VISION', DECISION_MARGIN_MAX_TOPIC_NAME, str(round(decisionmargin_max)))
    parser.set('VISION', TAG_CROP_TOP_TOPIC_NAME, str(round(crop_x)))
    parser.set('VISION', TAG_CROP_BOTTOM_TOPIC_NAME, str(round(crop_y)))
    parser.set('VISION', TAG_ERRORS_TOPIC_NAME, str(errors))

    with open(file, 'w') as config:
        parser.write(config)
        print('wrote tag file:')
        print({'VISION': dict(parser['VISION'])})
    
    #print(f'file={file} mh={str(min_h)} ms={str(min_s)} mv={str(min_v)} xh={str(max_h)} xs={str(max_s)} xv={str(max_v)}')

    '''        
    with open(file, 'w') as config:
        parser.write(config)
        print({'VISION': dict(parser['VISION'])})
    '''

def file_write_fuel(file,
                min_h,
                min_s,
                min_v,
                max_h,
                max_s,
                max_v,
                min_area):

    parser = configparser.ConfigParser()

    parser.add_section('VISION')
    parser.set('VISION', FUEL_CONFIG_FILE_TOPIC_NAME, str(file))
    parser.set('VISION', FUEL_MIN_HUE_TOPIC_NAME, str(round(min_h)))
    parser.set('VISION', FUEL_MIN_SAT_TOPIC_NAME, str(round(min_s)))
    parser.set('VISION', FUEL_MIN_VAL_TOPIC_NAME, str(round(min_v)))
    parser.set('VISION', FUEL_MAX_HUE_TOPIC_NAME, str(round(max_h)))
    parser.set('VISION', FUEL_MAX_SAT_TOPIC_NAME, str(round(max_s)))
    parser.set('VISION', FUEL_MAX_VAL_TOPIC_NAME, str(round(max_v)))
    parser.set('VISION', FUEL_MIN_AREA_TOPIC_NAME, str(round(min_area)))
    
    #print(f'file={file} mh={str(min_h)} ms={str(min_s)} mv={str(min_v)} xh={str(max_h)} xs={str(max_s)} xv={str(max_v)}')

    #HEY HEY HEY!!! LOOK AT MEEEEE!!!! >>>pscp.exe pi@10.2.33.177:/home/pi/config.ini C:\Users\23JMurphy\Downloads will copy any file from pi to windows<<<
    with open(file, 'w') as config:
        parser.write(config)
        print('wrote fuel file:')
        print({'VISION': dict(parser['VISION'])})

def get_type():
    parser= configparser.ConfigParser()
    parser.read('gen_config.ini')
    return(parser.get('GENERAL', 'type'))

def camera_upside_down():
    parser= configparser.ConfigParser()
    parser.read('gen_config.ini')
    return(parser.getboolean('GENERAL', 'camera_upside_down'))

def file_read_tag(parser, configfile_failure_ntt):
    config_exists = os.path.isfile(TAG_CONFIG_FILE_DEFAULT)
    if config_exists == True:
        parser.read(TAG_CONFIG_FILE_DEFAULT)
        configfile_failure_ntt.set(False) #if it works mark no error
        print('read tag file:')
        print({'VISION': dict(parser['VISION'])})

    else: # re-create config and container file to default
        configfile_failure_ntt.set(True) # set error for config file

        parser.add_section('VISION')
        parser.set('VISION', THREADS_TOPIC_NAME, str(THREADS_DEFAULT))
        parser.set('VISION', BLUR_TOPIC_NAME, str(BLUR_DEFAULT))
        parser.set('VISION', REFINE_EDGES_TOPIC_NAME, str(REFINE_EDGES_DEFAULT))
        parser.set('VISION', SHARPENING_TOPIC_NAME, str(SHARPENING_DEFAULT))
        parser.set('VISION', APRILTAG_DEBUG_MODE_TOPIC_NAME, str(APRILTAG_DEBUG_MODE_DEFAULT))
        parser.set('VISION', DECISION_MARGIN_MIN_TOPIC_NAME, str(DECISION_MARGIN_DEFAULT))
        parser.set('VISION', DECIMATE_TOPIC_NAME, str(DECIMATE_DEFAULT))
        parser.set('VISION', TAG_CONFIG_FILE_TOPIC_NAME, str(TAG_CONFIG_FILE_DEFAULT))
        parser.set('VISION', DECISION_MARGIN_MAX_TOPIC_NAME, str(DECISION_MARGIN_DEFAULT))
        parser.set('VISION', TAG_CROP_TOP_TOPIC_NAME, str(TAG_CROP_DEFAULT))
        parser.set('VISION', TAG_CROP_BOTTOM_TOPIC_NAME, str(TAG_CROP_DEFAULT))
        parser.set('VISION', TAG_ERRORS_TOPIC_NAME, str(TAG_ERRORS_DEFAULT))

        with open("/home/pi/" + TAG_CONFIG_FILE_DEFAULT, 'w') as config:
            parser.write(config)
            print('wrote tag file:')
            print({'VISION': dict(parser['VISION'])})

        configfile_failure_ntt.set(True) # recreated config file

def file_read_gen(parser, configfile_failure_ntt):
    config_exists = os.path.isfile(GEN_CONFIG_FILE_DEFAULT)
    if config_exists == True:
        parser.read(GEN_CONFIG_FILE_DEFAULT)
        configfile_failure_ntt.set(False) #if it works mark no error
        print('read gen file:')
        print({'GENERAL': dict(parser['GENERAL'])})

    else: # re-create config and container file to default
        configfile_failure_ntt.set(True) # set error for config file

        parser.add_section('GENERAL')
        parser.set('GENERAL', 'Type', 'None')
        parser.set('GENERAL', 'Camera Upside Down', False)
        parser.set('GENERAL', 'Brightness', str(BRIGHTNESS_DEFAULT))
        parser.set('GENERAL', 'Contrast', str(CONTRAST_DEFAULT))
        parser.set('GENERAL', 'Auto Exposure', str(AE_DEFAULT))
        parser.set('GENERAL', 'Manual Exposure Time', str(EXPOSURE_DEFAULT))
        parser.set('GENERAL', 'Y Offset', str(5))

        with open("/home/pi/" + GEN_CONFIG_FILE_DEFAULT, 'w') as config:
            parser.write(config)
            print('wrote gen file:')
            print({'GENERAL': dict(parser['GENERAL'])})

        configfile_failure_ntt.set(True) # recreated config file



def tag_check(tag,config_tag):
    #print(f'id={tag.getId()} e={tag.getHamming()} DM={int(round(tag.getDecisionMargin()))} x={int(round(tag_pose.translation().X()*39.37))} z={int(round(tag_pose.translation().Z()*39.37))}')
    #e_max = int(float(config_tag.get('VISION', TAG_ERRORS_TOPIC_NAME)))
    #print(f'id={tag.getId()} e={tag.getHamming()} e_max={e_max} DM={tag.getDecisionMargin()}')

    if tag.getDecisionMargin() > float(config_tag.get('VISION', DECISION_MARGIN_MIN_TOPIC_NAME)) and \
        tag.getDecisionMargin() < float(config_tag.get('VISION', DECISION_MARGIN_MAX_TOPIC_NAME)) and \
        (tag.getHamming() <= int(float(config_tag.get('VISION', TAG_ERRORS_TOPIC_NAME)))) and (tag.getId() >= 1 and tag.getId() <= 16):
        return True
    else:
        e_max = float(config_tag.get('VISION', TAG_ERRORS_TOPIC_NAME))
        dm_min = float(config_tag.get('VISION', DECISION_MARGIN_MIN_TOPIC_NAME))
        dm_max = float(config_tag.get('VISION', DECISION_MARGIN_MAX_TOPIC_NAME)) 
        print(f'id={tag.getId()} e={tag.getHamming()} e_max={e_max} DM={tag.getDecisionMargin()} dm_min={dm_min} dm_max={dm_max}')
        return False

def file_read_fuel(parser, configfile_failure_ntt):
    config_exists = os.path.isfile(FUEL_CONFIG_FILE_DEFAULT)
    if config_exists == True:
        parser.read(FUEL_CONFIG_FILE_DEFAULT)
        configfile_failure_ntt.set(False) #if it works mark no error
        print('read fuel file:')
        print({'VISION': dict(parser['VISION'])})
    else: # re-create config and container file to default
        configfile_failure_ntt.set(True) # set error for config file

        parser.add_section('VISION')
        
        parser.set('VISION', FUEL_CONFIG_FILE_TOPIC_NAME, str(FUEL_CONFIG_FILE_DEFAULT))
        parser.set('VISION', FUEL_MIN_HUE_TOPIC_NAME, str(FUEL_MIN_HUE))
        parser.set('VISION', FUEL_MIN_SAT_TOPIC_NAME, str(FUEL_MIN_SAT))
        parser.set('VISION', FUEL_MIN_VAL_TOPIC_NAME, str(FUEL_MIN_VAL))
        parser.set('VISION', FUEL_MAX_HUE_TOPIC_NAME, str(FUEL_MAX_HUE))
        parser.set('VISION', FUEL_MAX_SAT_TOPIC_NAME, str(FUEL_MAX_SAT))
        parser.set('VISION', FUEL_MAX_VAL_TOPIC_NAME, str(FUEL_MAX_VAL))
        parser.set('VISION', FUEL_MIN_AREA_TOPIC_NAME, str(FUEL_MIN_AREA))

        with open("/home/pi/" + FUEL_CONFIG_FILE_DEFAULT, 'w') as config:
            parser.write(config)
            print('wrote fuel file:')
            print({'VISION': dict(parser['VISION'])})
        configfile_failure_ntt.set(False) # config file recreated

def file_write_gen(brightness, contrast, ae_mode, man_exposure_time, y_offset):

    parser = configparser.ConfigParser()
    parser.read("/home/pi/" + GEN_CONFIG_FILE_DEFAULT)
    parser.set('GENERAL', 'Brightness', str(brightness))
    parser.set('GENERAL', 'Contrast', str(contrast))
    parser.set('GENERAL', 'Auto Exposure', str(ae_mode))
    parser.set('GENERAL', 'Manual Exposure Time', str(man_exposure_time))
    parser.set('GENERAL', 'Y Offset', str(y_offset))

    with open("/home/pi/" + GEN_CONFIG_FILE_DEFAULT, 'w') as config:
        parser.write(config)
        print('wrote gen file: ' + "/home/pi/" + GEN_CONFIG_FILE_DEFAULT)
        print({'GENERAL': dict(parser['GENERAL'])})

def nt_update_tags(config,
              threads,
              quadDecimate,
              blur,
              refineEdges,
              decodeSharpening,
              ATDebug,
              decision_min,
              decision_max,
              crop_x,
              crop_y,
              errors,
              configfile
            ):
    # sync the stuff in the file with matching values in the file
    t = float(config.get('VISION', THREADS_TOPIC_NAME))
    qd = float(config.get('VISION', DECIMATE_TOPIC_NAME))
    b = float(config.get('VISION', BLUR_TOPIC_NAME))
    re = ast.literal_eval(config.get('VISION', REFINE_EDGES_TOPIC_NAME))
    ds = float(config.get('VISION', SHARPENING_TOPIC_NAME))
    atd = ast.literal_eval(config.get('VISION', APRILTAG_DEBUG_MODE_TOPIC_NAME))
    dm_min = float(config.get('VISION', DECISION_MARGIN_MIN_TOPIC_NAME))
    dm_max = float(config.get('VISION', DECISION_MARGIN_MAX_TOPIC_NAME))
    c_x = float(config.get('VISION', TAG_CROP_TOP_TOPIC_NAME))
    c_y = float(config.get('VISION', TAG_CROP_BOTTOM_TOPIC_NAME))
    e = float(config.get('VISION', TAG_ERRORS_TOPIC_NAME))

    threads.set(t)
    quadDecimate.set(qd)
    blur.set(b)
    refineEdges.set(re)
    decodeSharpening.set(ds)
    ATDebug.set(atd)
    decision_min.set(dm_min)
    decision_max.set(dm_max)
    crop_x.set(c_x)
    crop_y.set(c_y)
    errors.set(e)
    #configfile.set(str(config.get('VISION', TAG_CONFIG_FILE_TOPIC_NAME)))

def nt_update_fuel(config,
              configfile,
              min_h,
              min_s,
              min_v,
              max_h,
              max_s,
              max_v,
              min_area):
    # sync the stuff in the file with matching values in the file

    print('dump fuel file:')
    print({'VISION': dict(config['VISION'])})

    mi_h = float(config.get('VISION', FUEL_MIN_HUE_TOPIC_NAME))
    mi_s = float(config.get('VISION', FUEL_MIN_SAT_TOPIC_NAME))
    mi_v = float(config.get('VISION', FUEL_MIN_VAL_TOPIC_NAME))

    mx_h = float(config.get('VISION', FUEL_MAX_HUE_TOPIC_NAME))
    mx_s = float(config.get('VISION', FUEL_MAX_SAT_TOPIC_NAME))
    mx_v = float(config.get('VISION', FUEL_MAX_VAL_TOPIC_NAME))

    #configfile.set(str(config.get('VISION', FUEL_CONFIG_FILE_TOPIC_NAME)))
    min_h.set(mi_h)
    min_s.set(mi_s)
    min_v.set(mi_v)
    max_h.set(mx_h)
    max_s.set(mx_s)
    max_v.set(mx_v)
    #min_area.set(float(config.get('VISION', FUEL_MIN_AREA_TOPIC_NAME)))

def nt_update_gen(type,
                  config,
              tag_brightness,
              tag_contrast,
              tag_ae,
              tag_exposure,
              fuel_brightness,
              fuel_contrast,
              fuel_ae,
              fuel_exposure,
              y_offset):
    # sync the stuff in the file with matching values in the file
    b = float(config.get('GENERAL', 'Brightness'))
    c = float(config.get('GENERAL', 'Contrast'))
    ae = bool(config.get('GENERAL', 'Auto Exposure'))
    exp = float(config.get('GENERAL', 'Manual Exposure Time'))
    y = float(config.get('GENERAL', 'Y Offset'))

    if type == 'tag':
        tag_brightness.set(b)
        tag_contrast.set(c)
        tag_ae.set(ae)
        tag_exposure.set(exp)
    else:
        fuel_brightness.set(b)
        fuel_contrast.set(c)
        fuel_ae.set(ae)
        fuel_exposure.set(exp)
        y_offset.set(y)

'''
all data to send is packaged as an array of bytes, using a Python bytearray, in big-endian format:
sequence number: unsigned long (4 bytes)
rio time: float (4 bytes)
image time:float (4 bytes)
type (tag = 1, fuel = 3): unsigned char (1 byte)
length: how many tags/fuel follow
what follows these first 3 items depends on the type:
tag:
number of tags detected: unsigned char (1 byte)
for each tag: tag id unsigned char (1 byte), pose x: float (4 bytes), pose y: float (4 bytes), pose z: float (4 bytes), pose x angle: float (4 bytes), pose y angle: float (4 bytes), pose z angle: float (4 bytes)

fuel:
number of fuel detected: unsigned char (1 byte)
for each fuel: pose x: float (4 bytes), pose y: float (4 bytes), pose z: float (4 bytes), pose x angle: float (4 bytes), pose y angle: float (4 bytes), pose z angle: float (4 bytes)s)
'''
def pose_data_bytes(sequence_num, rio_time, image_time, tags, tag_poses):
    byte_array = bytearray()
    # get a list of tags that were detected
    # start the array with sequence number, the RIO's time, image time, and tag type
    tag_pose = 0
    byte_array += struct.pack(">LffBB", sequence_num, rio_time, image_time, 1, len(tags))
    # subtract 3% of the distance Z from the y because of camera tilt
    for tag in tags:
        byte_array += struct.pack(">BBfffffff", tag.getId(), tag.getHamming(), tag.getDecisionMargin(), \
            tag_poses[tag_pose].rotation().X(), tag_poses[tag_pose].rotation().Y(), tag_poses[tag_pose].rotation().Z(), \
            tag_poses[tag_pose].translation().X(), tag_poses[tag_pose].translation().Y(), tag_poses[tag_pose].translation().Z())
        tag_pose += 1
    return byte_array

def piece_pose_data_bytes(sequence_num, rio_time, image_time, type, dist, angle):
    byte_array = bytearray()
    # start the array with sequence number, the RIO's time, image time, and tag type
    byte_array += struct.pack(">LffBB", sequence_num, rio_time, image_time, type, 1)
    byte_array += struct.pack(">ff", angle, dist) 
    return byte_array

def remove_image_files(path):
    for filename in os.listdir(path): 
        file_path = os.path.join(path, filename)  
        if os.path.isfile(file_path):
            os.remove(file_path)  

class Fuel_Config_Save_Class:
    def __init__(self, save, fileName, minH, maxH, minS, maxS, minV, maxV, minA):
        self.save = save
        self.fileName = fileName
        self.minH = minH
        self.maxH = maxH
        self.minS = minS
        self.maxS = maxS
        self.minV = minV
        self.maxV = maxV
        self.minA = minA


def fuel_config_save(config):
    if config.save.get() == True:
        print("saving...")
        file_write_fuel(config.fileName.get(), \
            config.minH.get(), \
            config.minS.get(), \
            config.minV.get(), \
            config.maxH.get(), \
            config.maxS.get(), \
            config.maxV.get(), \
            config.minA.get())
        config.save.set(False)


def main():

    print("Hello")

    vision_type = get_type()
    camera_orientation = camera_upside_down()

    # start NetworkTables
    ntconnect = NTConnectType(NTConnectType.SERVER)    #use CLIENT when running with rio
    ntinst = NetworkTableInstance.getDefault()
    if ntconnect == NTConnectType.SERVER:
        ntinst.startServer()
    else:
        print("connect as client")
        ntinst.startClient4("raspberrypi910")
        ntinst.setServerTeam(910)
 
    # Wait for NetworkTables to start
    time.sleep(1)
    
    rio_time_ntt = NTGetDouble(ntinst.getDoubleTopic(RIO_TIME_TOPIC_NAME), 0, 0, 0)
    
    if ntconnect == NTConnectType.CLIENT:
        while rio_time_ntt.get() == 0:
            time.sleep(1)
            print("Waiting to receive data from Rio...")
        print("Received data from Rio")

    # Table for vision output information
    tag_uptime_ntt = NTGetDouble(ntinst.getDoubleTopic("/Vision/Tag Uptime"), 0, 0, -1)
    fuel_uptime_ntt = NTGetDouble(ntinst.getDoubleTopic("/Vision/Fuel Uptime"), 0, 0, -1)

    debug_tag_ntt = NTGetBoolean(ntinst.getBooleanTopic("/Vision/Tag Debug Mode"), DEBUG_MODE_DEFAULT, DEBUG_MODE_DEFAULT, DEBUG_MODE_DEFAULT)
    debug_fuel_ntt = NTGetBoolean(ntinst.getBooleanTopic("/Vision/Fuel Debug Mode"), DEBUG_MODE_DEFAULT, DEBUG_MODE_DEFAULT, DEBUG_MODE_DEFAULT)
    threads_ntt = NTGetDouble(ntinst.getDoubleTopic(THREADS_TOPIC_NAME),THREADS_DEFAULT, THREADS_DEFAULT, THREADS_DEFAULT)
    quadDecimate_ntt = NTGetDouble(ntinst.getDoubleTopic(DECIMATE_TOPIC_NAME),DECIMATE_DEFAULT, DECIMATE_DEFAULT, DECIMATE_DEFAULT)
    blur_ntt = NTGetDouble(ntinst.getDoubleTopic(BLUR_TOPIC_NAME),BLUR_DEFAULT, BLUR_DEFAULT, BLUR_DEFAULT) 
    refineEdges_ntt = NTGetBoolean(ntinst.getBooleanTopic(REFINE_EDGES_TOPIC_NAME),REFINE_EDGES_DEFAULT, REFINE_EDGES_DEFAULT, REFINE_EDGES_DEFAULT) 
    decodeSharpening_ntt = NTGetDouble(ntinst.getDoubleTopic(SHARPENING_TOPIC_NAME), SHARPENING_DEFAULT, SHARPENING_DEFAULT, SHARPENING_DEFAULT)
    ATDebug_ntt = NTGetBoolean(ntinst.getBooleanTopic(APRILTAG_DEBUG_MODE_TOPIC_NAME), APRILTAG_DEBUG_MODE_DEFAULT, APRILTAG_DEBUG_MODE_DEFAULT, APRILTAG_DEBUG_MODE_DEFAULT)
    decision_margin_min_ntt = NTGetDouble(ntinst.getDoubleTopic(DECISION_MARGIN_MIN_TOPIC_NAME), DECISION_MARGIN_DEFAULT, DECISION_MARGIN_DEFAULT, DECISION_MARGIN_DEFAULT)
    tagconfigfile_ntt = NTGetString(ntinst.getStringTopic(TAG_CONFIG_FILE_TOPIC_NAME), TAG_CONFIG_FILE_DEFAULT,TAG_CONFIG_FILE_DEFAULT, TAG_CONFIG_FILE_DEFAULT)
    fuelconfigfile_ntt = NTGetString(ntinst.getStringTopic(FUEL_CONFIG_FILE_TOPIC_NAME), FUEL_CONFIG_FILE_DEFAULT,FUEL_CONFIG_FILE_DEFAULT, FUEL_CONFIG_FILE_DEFAULT)
    configfilefail_ntt = NTGetBoolean(ntinst.getBooleanTopic("/Vision/Config File Fail"), False, False, False)
    tag_active_ntt = NTGetBoolean(ntinst.getBooleanTopic(TAG_ACTIVE_TOPIC_NAME), True, True, True)
    fuel_active_ntt = NTGetBoolean(ntinst.getBooleanTopic(FUEL_ACTIVE_TOPIC_NAME), True, True, True)
    pose_data_bytes_ntt = NTGetRaw(ntinst, POSE_DATA_RAW_TOPIC_NAME, None, None, None)
    fuel_pose_data_bytes_ntt = NTGetRaw(ntinst, FUEL_POSE_DATA_RAW_TOPIC_NAME, None, None, None)
    pose_data_string_header_ntt = NTGetString(ntinst.getStringTopic(POSE_DATA_STRING_TOPIC_NAME_HEADER),"", "", "")
    fuel_pose_data_string_header_ntt = NTGetString(ntinst.getStringTopic(FUEL_POSE_DATA_STRING_TOPIC_NAME_HEADER),"", "", "") 
    pose_data_string_data_translation_ntt = NTGetString(ntinst.getStringTopic(POSE_DATA_STRING_TOPIC_NAME_DATA_TRANSLATION),"", "", "")
    pose_data_string_data_rotation_ntt = NTGetString(ntinst.getStringTopic(POSE_DATA_STRING_TOPIC_NAME_DATA_ROTATION),"", "", "")
    temp_tag_ntt = NTGetDouble(ntinst.getDoubleTopic(TAG_PI_TEMP_TOPIC_NAME), 0, 0, 0)
    temp_fuel_ntt = NTGetDouble(ntinst.getDoubleTopic(FUEL_PI_TEMP_TOPIC_NAME), 0, 0, 0)
    z_in_ntt = NTGetDouble(ntinst.getDoubleTopic(Z_IN_TOPIC_NAME), 0.0, 0.0, 0.0)
    fuel_min_h_ntt = NTGetDouble(ntinst.getDoubleTopic(FUEL_MIN_HUE_TOPIC_NAME), FUEL_MIN_HUE, FUEL_MIN_HUE, FUEL_MIN_HUE)
    fuel_min_s_ntt = NTGetDouble(ntinst.getDoubleTopic(FUEL_MIN_SAT_TOPIC_NAME), FUEL_MIN_SAT, FUEL_MIN_SAT, FUEL_MIN_SAT)
    fuel_min_v_ntt = NTGetDouble(ntinst.getDoubleTopic(FUEL_MIN_VAL_TOPIC_NAME), FUEL_MIN_VAL, FUEL_MIN_VAL, FUEL_MIN_VAL)
    fuel_max_h_ntt = NTGetDouble(ntinst.getDoubleTopic(FUEL_MAX_HUE_TOPIC_NAME), FUEL_MAX_HUE, FUEL_MAX_HUE, FUEL_MAX_HUE)
    fuel_max_s_ntt = NTGetDouble(ntinst.getDoubleTopic(FUEL_MAX_SAT_TOPIC_NAME), FUEL_MAX_SAT, FUEL_MAX_SAT, FUEL_MAX_SAT)
    fuel_max_v_ntt = NTGetDouble(ntinst.getDoubleTopic(FUEL_MAX_VAL_TOPIC_NAME), FUEL_MAX_VAL, FUEL_MAX_VAL, FUEL_MAX_VAL)
    tag_enable = NTGetBoolean(ntinst.getBooleanTopic(TAG_ENABLE_TOPIC_NAME), False, False, False)
    fuel_enable_ntt = NTGetBoolean(ntinst.getBooleanTopic(FUEL_ENABLE_TOPIC_NAME), False, False, False)
    fuel_min_area_ntt = NTGetDouble(ntinst.getDoubleTopic(FUEL_MIN_AREA_TOPIC_NAME), FUEL_MIN_AREA, FUEL_MIN_AREA, FUEL_MIN_AREA)
    fuel_angle_ntt = NTGetDouble(ntinst.getDoubleTopic(FUEL_ANGLE_TOPIC_NAME), 0.0, 0.0, 0.0)
    tag_record_ntt = NTGetBoolean(ntinst.getBooleanTopic(TAG_RECORD_ENABLE_TOPIC_NAME), False, False, False)
    tag_record_remove_ntt = NTGetBoolean(ntinst.getBooleanTopic(TAG_RECORD_REMOVE_TOPIC_NAME), False, False, False)
    fuel_record_data_ntt = NTGetBoolean(ntinst.getBooleanTopic(FUEL_RECORD_DATA_TOPIC_NAME), False, False, False)
    decision_margin_max_ntt = NTGetDouble(ntinst.getDoubleTopic(DECISION_MARGIN_MAX_TOPIC_NAME), DECISION_MARGIN_DEFAULT, DECISION_MARGIN_DEFAULT, DECISION_MARGIN_DEFAULT)
    fuel_distance_ntt = NTGetDouble(ntinst.getDoubleTopic("/Vision/Fuel Distance"), 0.0, 0.0, 0.0)
    tag_crop_x_ntt = NTGetDouble(ntinst.getDoubleTopic(TAG_CROP_TOP_TOPIC_NAME), TAG_CROP_TOP_DEFAULT, TAG_CROP_TOP_DEFAULT, TAG_CROP_TOP_DEFAULT)
    tag_crop_y_ntt = NTGetDouble(ntinst.getDoubleTopic(TAG_CROP_BOTTOM_TOPIC_NAME), TAG_CROP_BOTTOM_DEFAULT, TAG_CROP_BOTTOM_DEFAULT, TAG_CROP_BOTTOM_DEFAULT)
    tag_corrected_errors_ntt = NTGetDouble(ntinst.getDoubleTopic(TAG_ERRORS_TOPIC_NAME), TAG_ERRORS_DEFAULT, TAG_ERRORS_DEFAULT, TAG_ERRORS_DEFAULT)
    pose_data_x_deg_ntt = NTGetDouble(ntinst.getDoubleTopic(POSE_DATA_X_DEG_TOPIC_NAME), 0.0, 0.0, 0.0)
    pose_data_y_deg_ntt = NTGetDouble(ntinst.getDoubleTopic(POSE_DATA_Y_DEG_TOPIC_NAME), 0.0, 0.0, 0.0)
    pose_data_z_deg_ntt = NTGetDouble(ntinst.getDoubleTopic(POSE_DATA_Z_DEG_TOPIC_NAME), 0.0, 0.0, 0.0)
    pose_data_x_in_ntt = NTGetDouble(ntinst.getDoubleTopic(POSE_DATA_X_IN_TOPIC_NAME), 0.0, 0.0, 0.0)
    pose_data_y_in_ntt = NTGetDouble(ntinst.getDoubleTopic(POSE_DATA_Y_IN_TOPIC_NAME), 0.0, 0.0, 0.0)
    tag_detected_id_ntt = NTGetDouble(ntinst.getDoubleTopic(TAG_DETECTED_ID_TOPIC_NAME), 0, 0, 0)
    tag_detected_dm_ntt = NTGetDouble(ntinst.getDoubleTopic(TAG_DETECTED_DM_TOPIC_NAME), 0, 0, 0)
    tag_detected_errors_ntt = NTGetDouble(ntinst.getDoubleTopic(TAG_DETECTED_ERRORS_TOPIC_NAME), 0, 0, 0)

    tag_brightness_ntt = NTGetDouble(ntinst.getDoubleTopic(TAG_BRIGHTNESS_TOPIC_NAME), BRIGHTNESS_DEFAULT, BRIGHTNESS_DEFAULT, BRIGHTNESS_DEFAULT)
    tag_contrast_ntt = NTGetDouble(ntinst.getDoubleTopic(TAG_CONTRAST_TOPIC_NAME), CONTRAST_DEFAULT, CONTRAST_DEFAULT, CONTRAST_DEFAULT)
    tag_ae_ntt = NTGetBoolean(ntinst.getBooleanTopic(TAG_AE_TOPIC_NAME), AE_DEFAULT, AE_DEFAULT, AE_DEFAULT)
    tag_exposure_ntt = NTGetDouble(ntinst.getDoubleTopic(TAG_EXPOSURE_TOPIC_NAME), EXPOSURE_DEFAULT, EXPOSURE_DEFAULT, EXPOSURE_DEFAULT)

    fuel_brightness_ntt = NTGetDouble(ntinst.getDoubleTopic(FUEL_BRIGHTNESS_TOPIC_NAME), BRIGHTNESS_DEFAULT, BRIGHTNESS_DEFAULT, BRIGHTNESS_DEFAULT)
    fuel_contrast_ntt = NTGetDouble(ntinst.getDoubleTopic(FUEL_CONTRAST_TOPIC_NAME), CONTRAST_DEFAULT, CONTRAST_DEFAULT, CONTRAST_DEFAULT)
    fuel_ae_ntt = NTGetBoolean(ntinst.getBooleanTopic(FUEL_AE_TOPIC_NAME), AE_DEFAULT, AE_DEFAULT, AE_DEFAULT)
    fuel_exposure_ntt = NTGetDouble(ntinst.getDoubleTopic(FUEL_EXPOSURE_TOPIC_NAME), EXPOSURE_DEFAULT, EXPOSURE_DEFAULT, EXPOSURE_DEFAULT)

    tag_config_savefile_ntt = NTGetBoolean(ntinst.getBooleanTopic("/Vision/Tag Config Save"), False, False, False)
    fuel_config_savefile_ntt = NTGetBoolean(ntinst.getBooleanTopic("/Vision/Fuel Config Save"), False, False, False)
    tag_camera_safefile_ntt = NTGetBoolean(ntinst.getBooleanTopic("/Vision/Tag Camera Save"), False, False, False)
    fuel_camera_savefile_ntt = NTGetBoolean(ntinst.getBooleanTopic("/Vision/Fuel Camera Save"), False, False, False)
    fuel_camera_refresh_nt_ntt = NTGetBoolean(ntinst.getBooleanTopic("/Vision/Fuel Camera Refresh Nt"), False, False, False)
    gen_fuel_y_offset_ntt =  NTGetDouble(ntinst.getDoubleTopic(GEN_FUEL_Y_OFFSET_TOPIC_NAME), 0, 0, 0)

    detector = robotpy_apriltag.AprilTagDetector()
    #detector.addFamily("tag16h5")
    detector.addFamily("tag36h11")

    # use for file
    config_tag = configparser.ConfigParser()
    config_fuel = configparser.ConfigParser()
    config_gen = configparser.ConfigParser()

    fuelConfigSave = Fuel_Config_Save_Class(fuel_config_savefile_ntt, \
        fuelconfigfile_ntt, \
        fuel_min_h_ntt, \
        fuel_max_h_ntt, \
        fuel_min_s_ntt, \
        fuel_max_s_ntt, \
        fuel_min_v_ntt, \
        fuel_max_v_ntt, \
        fuel_min_area_ntt)

    '''
    print('*****')
    parser= configparser.ConfigParser()
    parser.read('gen_config.ini')
    print('read gen config file:')
    print({'GENERAL': dict(parser['GENERAL'])})
    return(parser.get('GENERAL', 'type'))
    '''

    file_read_tag(config_tag, configfilefail_ntt)
    nt_update_tags(config_tag,threads_ntt, quadDecimate_ntt, blur_ntt, refineEdges_ntt, \
        decodeSharpening_ntt, ATDebug_ntt, decision_margin_min_ntt, decision_margin_max_ntt, \
        tag_crop_x_ntt, tag_crop_y_ntt, tag_corrected_errors_ntt, tagconfigfile_ntt)
        
    file_read_fuel(config_fuel, configfilefail_ntt)
    nt_update_fuel(config_fuel, fuelconfigfile_ntt, \
        fuel_min_h_ntt, fuel_min_s_ntt, fuel_min_v_ntt, fuel_max_h_ntt, fuel_max_s_ntt, fuel_max_v_ntt, \
        fuel_min_area_ntt)

    file_read_gen(config_gen, configfilefail_ntt)
    nt_update_gen(vision_type, config_gen, tag_brightness_ntt, tag_contrast_ntt, tag_ae_ntt, tag_exposure_ntt, \
        fuel_brightness_ntt, fuel_contrast_ntt, fuel_ae_ntt, fuel_exposure_ntt, gen_fuel_y_offset_ntt)
    
    detectorConfig = robotpy_apriltag.AprilTagDetector.Config()

    detectorConfig.numThreads = int(float(config_tag.get('VISION', THREADS_TOPIC_NAME)))
    detectorConfig.quadDecimate = float(config_tag.get('VISION', DECIMATE_TOPIC_NAME))
    detectorConfig.quadSigma = float (config_tag.get('VISION', BLUR_TOPIC_NAME))
    detectorConfig.refineEdges = ast.literal_eval(config_tag.get('VISION', REFINE_EDGES_TOPIC_NAME))
    detectorConfig.decodeSharpening = float(config_tag.get('VISION', SHARPENING_TOPIC_NAME))
    detectorConfig.debug = ast.literal_eval(config_tag.get('VISION', APRILTAG_DEBUG_MODE_TOPIC_NAME))
    detector.setConfig(detectorConfig)
    

    fuel_min_h = int(config_fuel.get('VISION', FUEL_MIN_HUE_TOPIC_NAME))
    fuel_min_s = int(config_fuel.get('VISION', FUEL_MIN_SAT_TOPIC_NAME))
    fuel_min_v = int(config_fuel.get('VISION', FUEL_MIN_VAL_TOPIC_NAME))
    fuel_max_h = int(config_fuel.get('VISION', FUEL_MAX_HUE_TOPIC_NAME))
    fuel_max_s = int(config_fuel.get('VISION', FUEL_MAX_SAT_TOPIC_NAME))
    fuel_max_v = int(config_fuel.get('VISION', FUEL_MAX_VAL_TOPIC_NAME))
    fuel_min_area = int(config_fuel.get('VISION', FUEL_MIN_AREA_TOPIC_NAME))
    fuel_max_area = 1000

    #set up pose estimation
    ''' old way for camera calibration
    calib_data_path = "calib_data"
    calib_data = np.load(f"{calib_data_path}/{CAMERA_CAL_FILE_NAME}")
    camMatrix = calib_data["camMatrix"]
    distCoeffs = calib_data["distCoef"]
    '''
    # new way for camera calibration
    with open('cameraMatrix.pkl', 'rb') as c:
        camMatrix = pickle.load(c)
    with open('dist.pkl', 'rb') as d:
        distCoeffs = pickle.load(d)

    #camMatrix[0][0] = Focal point distance x (fx) 
    #camMatrix[1][1] = Focal point distance y (fy) 
    #camMatrix[0][2] = camera center  (cx) 
    #camMatrix[1][2] = camera center  (cy) 

    apriltag_est_config = robotpy_apriltag.AprilTagPoseEstimator.Config(0.153, camMatrix[0][0], camMatrix[1][1], camMatrix[0][2], camMatrix[1][2])
    apriltag_est = robotpy_apriltag.AprilTagPoseEstimator(apriltag_est_config)
    rVector = np.zeros((3,1))
    tVector = np.zeros((3,1))
    
    #load camera settings set from web console
    with open('/boot/frc.json') as f:
        web_settings = json.load(f)
    cam_config = web_settings['cameras'][0]

    w = cam_config['width']
    h = cam_config['height']
    fps = cam_config['fps']

    picam2 = Picamera2()
    server = FrameServer(picam2)

    print(f'{len(picam2.sensor_modes)} camera image sensor modes')
    pprint(picam2.sensor_modes[0])
    #picam2_config = picam2.create_preview_configuration(main={"size" : (w,h)})
    #picam2_config = picam2.create_preview_configuration(main={"size" : (w,h)}, raw=sensor_modes[0])
    picam2_config = picam2.create_video_configuration( {'size': (w, h), 'format' : 'RGB888'})
    
    # When the camera bolt hole is facing up, the camera is upside down
    # When the camera bolt hole is facing down, the camera is mounted right side up.
    # If flip needed, flip every image using cv2.flip(img,-1) or tell Picamera2 at config time (now) to flip every image it gives
    #img = cv2.flip(img, -1) 
    #picam2_config['transform'] = libcamera.Transform(hflip=1, vflip=1)
    #picam2.configure(picam2_config)
    picam2.set_controls({"FrameRate": fps})
    
    # added @ Troy 3-23
    picam2.set_controls({'Brightness': float(config_gen.get('GENERAL', 'Brightness'))})
    picam2.set_controls({'Contrast': float(config_gen.get('GENERAL', 'Contrast'))})
    # end added @ Troy 3-23

    # AeFlickerPeriod from https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf
    # The period of the lighting cycle in microseconds. For example, for 50Hz mains
    # lighting the flicker occurs at 100Hz, so the
    # period would be 10000 microseconds.
    # but this doesn't work: 'libcamera._libcamera.controls' has no attribute 'AeFlickerModeEnum
    # picam2.set_controls({"FrameRate": fps}, \
    #   {"AeFlickerMode": controls.AeFlickerModeEnum.Manual}, \
    #   {"AeFlickerPeriod": 12000})

    server.start()
    picam2.start()
    time.sleep(3)
    
    
    #picam2.set_controls({'AeEnable': False})
    #picam2.set_controls({'AwbMode': controls.AwbModeEnum.Indoor})
    
    #picam2.set_controls({'AwbMode': controls.AeExposureModeEnum.Short}) # didn't change tag fps
    #picam2.set_controls({"ExposureTime": 25000, "AnalogueGain": 2.0})

    # (optional) Setup a CvSource. This will send images back to the Dashboard
    if vision_type == 'tag':
        outputStreamTag = CameraServer.putVideo("tag image", cam_config['width'], cam_config['height'])
    else:
        outputStreamFuel = CameraServer.putVideo("fuel image", cam_config['width'], cam_config['height'])
        outputMask = CameraServer.putVideo("mask image", cam_config['width'], cam_config['height'])

    # Allocating new images is very expensive, always try to preallocate
    #img = np.zeros(shape=(cam_config['height'], cam_config['width'], 3), dtype=np.uint8)

    image_num = 0
    image_counter = 0
    image_time_av_total = 0
    fps_av = 0
    fps_av_min = 24601
    fps_av_max = -1
    tag_image_counter = 0
    tag_image_time_av_total = 0
    tag_fps_av = 0
    tag_fps_av_min = 24601
    tag_fps_av_max = -1
    seconds = 0
    current_seconds = 0
    prev_seconds = 0
    temp_sec = 30
    tag_recording = False
    
    tag_last_brightness = None
    tag_last_contrast = None
    tag_last_ae_mode = None
    fuel_last_brightness = None
    fuel_last_contrast = None
    fuel_last_ae_mode = None

    brightness = BRIGHTNESS_DEFAULT
    contrast = CONTRAST_DEFAULT
    ae_mode = AE_DEFAULT
    exp_time = EXPOSURE_DEFAULT
    cam_settings_changed = False

    FUEL_Y_OFFSET = int(config_gen.get('GENERAL', 'Y Offset'))

    while True:

        rio_time = rio_time_ntt.get()
        current_seconds = time.time()
        time_check = False
        if current_seconds - prev_seconds >= UPTIME_UPDATE_INTERVAL:
            prev_seconds = current_seconds
            seconds = seconds + 1
            temp_sec = temp_sec + 1

            if vision_type == 'tag':

                db_t = debug_tag_ntt.get()

                if db_t == True:

                    if tag_last_brightness != tag_brightness_ntt.get():
                        brightness = float(tag_brightness_ntt.get())
                        picam2.set_controls({'Brightness': brightness})
                        config_gen.set('GENERAL', 'Brightness', str(brightness))
                        tag_last_brightness = brightness
                        cam_settings_changed = True

                    if tag_last_contrast != tag_contrast_ntt.get():
                        contrast = float(tag_contrast_ntt.get())
                        picam2.set_controls({'Contrast': contrast})
                        config_gen.set('GENERAL', 'Contrast', str(contrast))
                        tag_last_contrast = contrast
                        cam_settings_changed = True

                    if tag_last_ae_mode != tag_ae_ntt.get():
                        ae_mode = bool(tag_ae_ntt.get())
                        picam2.set_controls({'AeEnable': bool(ae_mode)})
                        config_gen.set('GENERAL', 'AeEnable', str(ae_mode))
                        tag_last_ae_mode = ae_mode
                        cam_settings_changed = True

                    if tag_last_ae_mode == False:
                        exp_time = int(round(tag_exposure_ntt.get()))
                        picam2.set_controls({"ExposureTime": \
                            exp_time, "AnalogueGain": 1.0})
                        config_gen.set('GENERAL', 'Manual Exposure Time', str(exp_time))
                        cam_settings_changed = True

                    if cam_settings_changed == True and tag_camera_safefile_ntt.get() == True:
                        file_write_gen(brightness, contrast, ae_mode, exp_time, FUEL_Y_OFFSET)
                        tag_camera_safefile_ntt.set(False)
                        cam_settings_changed = False

            else:

                db_n = debug_fuel_ntt.get()

                if db_n == True:
                    
                    if fuel_camera_refresh_nt_ntt.get() == True:
                        file_read_fuel(config_fuel, configfilefail_ntt)
                        nt_update_fuel(config_fuel, fuelconfigfile_ntt, \
                            fuel_min_h_ntt, fuel_min_s_ntt, fuel_min_v_ntt, fuel_max_h_ntt, fuel_max_s_ntt, fuel_max_v_ntt, \
                            fuel_min_area_ntt)
                        file_read_gen(config_gen, configfilefail_ntt)
                        nt_update_gen(vision_type, config_gen, tag_brightness_ntt, tag_contrast_ntt, tag_ae_ntt, tag_exposure_ntt, \
                            fuel_brightness_ntt, fuel_contrast_ntt, fuel_ae_ntt, fuel_exposure_ntt, gen_fuel_y_offset_ntt)
                        fuel_camera_refresh_nt_ntt.set(False)

                    if fuel_last_brightness != fuel_brightness_ntt.get():
                        brightness = float(fuel_brightness_ntt.get())
                        picam2.set_controls({'Brightness': float(brightness)})
                        config_gen.set('GENERAL', 'Brightness', str(brightness))
                        fuel_last_brightness = brightness
                        cam_settings_changed = True

                    if fuel_last_contrast != fuel_contrast_ntt.get():
                        contrast = float(fuel_contrast_ntt.get())
                        picam2.set_controls({'Contrast': float(contrast)})
                        config_gen.set('GENERAL', 'Contrast', str(contrast))
                        fuel_last_contrast = contrast
                        cam_settings_changed = True

                    if fuel_last_ae_mode != fuel_ae_ntt.get():
                        ae_mode = bool(tag_ae_ntt.get())
                        picam2.set_controls({'AeEnable': bool(ae_mode)})
                        config_gen.set('GENERAL', 'AeEnable', str(ae_mode))
                        fuel_last_ae_mode = ae_mode
                        cam_settings_changed = True

                    if fuel_last_ae_mode == False:
                        exp_time = int(round(fuel_exposure_ntt.get(),0))
                        picam2.set_controls({"ExposureTime": \
                            exp_time, "AnalogueGain": 1.0})
                        config_gen.set('GENERAL', 'Manual Exposure Time', str(exp_time))
                        cam_settings_changed = True

                    if FUEL_Y_OFFSET != gen_fuel_y_offset_ntt.get():
                        FUEL_Y_OFFSET = int(round(gen_fuel_y_offset_ntt.get(),0))
                        config_gen.set('GENERAL', 'Y Offset', str(FUEL_Y_OFFSET))
                        cam_settings_changed = True
                    
                    if cam_settings_changed == True and fuel_camera_savefile_ntt.get() == True:
                        print("saving camera settings")
                        file_write_gen(brightness, contrast, ae_mode, exp_time, FUEL_Y_OFFSET)
                        fuel_camera_savefile_ntt.set(False)
                        cam_settings_changed = False
                    elif cam_settings_changed == False and fuel_camera_savefile_ntt.get() == True:
                        print("values unchanged")
                        fuel_camera_savefile_ntt.set(False)



            if vision_type == 'tag':
                tag_uptime_ntt.set(seconds)
            else:
                fuel_uptime_ntt.set(seconds)
            time_check = True

            if vision_type == 'tag':
                if db_t == True:
                    print(f'sec={seconds} tags: ave fps={round(tag_fps_av,0)} min fps={round(tag_fps_av_min,0)} max fps={round(tag_fps_av_max,0)}')
                else:
                    print(f'{seconds}')
            else:                
                if db_n == True:
                    print(f'sec={seconds} fuel: ave fps={round(fps_av,0)} fps min={round(fps_av_min,0)} fps max={round(fps_av_max,0)}')
                    #print(f'FUEL_Y_OFFSET={FUEL_Y_OFFSET}')
                else:
                    print(f'{seconds}')
            
        if temp_sec >= TEMP_UPDATE_INTERVAL:
            with open("/sys/class/thermal/thermal_zone0/temp", 'r') as f:
                current_temp = int(f.readline()) / 1000 #converting milidegrees C to degrees C
                if vision_type == 'tag':
                    temp_tag_ntt.set(current_temp)
                else:
                    temp_fuel_ntt.set(current_temp)
                '''
                temp_ntt.set(int(f.readline()) / 1000) #converting milidegrees C to degrees C
                temp_sec = 0
                '''
                 
        t1_time = time.perf_counter()
        #img = picam2.capture_array()
        img = None
        img = server.wait_for_frame(img)
        #image_time = time.perf_counter() - t1_time
        # When the camera bolt hole is facing up, the camera is upside down
        # When the camera bolt hole is facing down, the camera is mounted right side up.
        # If flip needed, flip every image using cv2.flip(img,-1) or tell Picamera2 at config time (now) to flip every image it gives
        #picam2_config['transform'] = libcamera.Transform(hflip=1, vflip=1)
        #picam2.configure(picam2_config)
        #picam2.set_controls({"FrameRate": fps})
        if camera_orientation == True:
            img = cv2.flip(img, -1)

        '''
        tag_image_counter += 1
        tag_image_time_av_total += image_time
        outputStream.putFrame(img)

        if tag_image_counter == FPS_NUM_SAMPLES: 
            #print("glob")
            tag_fps_av = 1/(tag_image_time_av_total/tag_image_counter)
            if tag_fps_av < tag_fps_av_min:
                tag_fps_av_min = tag_fps_av
            if tag_fps_av > tag_fps_av_max:
                tag_fps_av_max = tag_fps_av
            tag_image_time_av_total = 0
            tag_image_counter = 0
        continue 
        '''

        '''
        if frame_time == 0:
            # Send the output the error.
            outputStream.notifyError(cvSink.getError())
            # skip the rest of the current iteration
            continue
        '''
        
        #
        # Insert your image processing logic here!

        # Tags
        if vision_type == 'tag':
            #if tag_enable.get() == True:

            dm_list = {'1': [999999, 0], '2': [999999, 0], '3' : [999999, 0], '4' : [999999, 0], '5' : [999999, 0], '6' : [999999, 0], '7' : [999999, 0], '8' : [999999, 0]}

            # crop the image by y
            # crop is done by this format
            # img[y_start:y_end,x_start:x_end]

            # crop from y = 0 to this number
            crop_top = float(config_tag.get('VISION', TAG_CROP_TOP_TOPIC_NAME))
            # crop from y = this number to y = max y
            crop_bottom = float(config_tag.get('VISION', TAG_CROP_BOTTOM_TOPIC_NAME))
            crop_top = int(round(crop_top,0))
            crop_bottom = int(round(crop_bottom,0))

            # keep below the top and above the bottom
            cropped_image = img[crop_top:crop_bottom,0:w-1]
            
            ###
            # convert to RGB for displaying
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ###

            #print(f'{cropped_image.shape}')
            # black out above the top and below the bottom for displaying
            img[0:crop_top,0:w-1] = 0
            img[crop_bottom:h-1,0:w-1] = 0
            #print(f'{img.shape}')

            gimg = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

            if db_t == True:
                detectorConfig.numThreads = int(threads_ntt.get())
                detectorConfig.quadDecimate = float(quadDecimate_ntt.get())
                detectorConfig.quadSigma = float(blur_ntt.get())
                detectorConfig.refineEdges = refineEdges_ntt.get()
                detectorConfig.decodeSharpening = float(decodeSharpening_ntt.get())
                detectorConfig.debug = ATDebug_ntt.get()
                detector.setConfig(detectorConfig)
                config_tag.set('VISION', DECISION_MARGIN_MIN_TOPIC_NAME, str(decision_margin_min_ntt.get()))
                config_tag.set('VISION', DECISION_MARGIN_MAX_TOPIC_NAME, str(decision_margin_max_ntt.get()))
                config_tag.set('VISION', TAG_CROP_TOP_TOPIC_NAME, str(tag_crop_x_ntt.get()))
                config_tag.set('VISION', TAG_CROP_BOTTOM_TOPIC_NAME, str(tag_crop_y_ntt.get()))
                config_tag.set('VISION', TAG_ERRORS_TOPIC_NAME, str(tag_corrected_errors_ntt.get()))

            #d_time = time.perf_counter()
            detected = detector.detect(gimg)
            #print(f'tag d = {(time.perf_counter() - d_time):1.3f}')

            tag_poses = []
            tags = []

            for tag in detected:
                if tag_check(tag,config_tag) == True:
                    tag_pose = apriltag_est.estimateHomography(tag)
                    #print(f'id={tag.getId()} e={tag.getHamming()} DM={int(round(tag.getDecisionMargin()))} x={int(round(tag_pose.translation().X()*39.37))} z={int(round(tag_pose.translation().Z()*39.37))}')
                    tag_poses.append(tag_pose)
                    tags.append(tag)

            if len(tags) > 0:
                image_num += 1
                tag_image_counter += 1
                image_time = time.perf_counter() - t1_time
                tag_image_time_av_total += image_time

                if tag_image_counter == FPS_NUM_SAMPLES: 
                    #print("glob")
                    tag_fps_av = 1/(tag_image_time_av_total/tag_image_counter)
                    if tag_fps_av < tag_fps_av_min:
                        tag_fps_av_min = tag_fps_av
                    if tag_fps_av > tag_fps_av_max:
                        tag_fps_av_max = tag_fps_av
                    tag_image_time_av_total = 0
                    tag_image_counter = 0                    
                pose_data = pose_data_bytes(image_num, rio_time, image_time, tags, tag_poses)
                pose_data_bytes_ntt.set(pose_data)
                NetworkTableInstance.getDefault().flush()

            db_t = debug_tag_ntt.get()

            if db_t == True:
                if len(tags) > 0:
                    header, rot_data, trans_data = \
                        pose_data_string(image_num, rio_time, image_time, tags, tag_poses, \
                        (tag_detected_id_ntt, tag_detected_dm_ntt, tag_detected_errors_ntt, \
                        pose_data_x_deg_ntt, pose_data_y_deg_ntt, pose_data_z_deg_ntt, \
                        pose_data_x_in_ntt, pose_data_y_in_ntt, z_in_ntt))
                    pose_data_string_header_ntt.set(header)
                    pose_data_string_data_translation_ntt.set(trans_data)
                    pose_data_string_data_rotation_ntt.set(rot_data)
                    img = draw_tags(img, tags, tag_poses, rVector, tVector, camMatrix, distCoeffs, crop_top)
                outputStreamTag.putFrame(img) # send to dashboard
                if tag_record_remove_ntt.get() == True:
                    remove_image_files('/home/pi/tag_images')
                    tag_record_remove_ntt.set(False)
                if tag_record_ntt.get() == True:
                    # the ID's of all tags in images with > 1 tag should be all on the same side
                    mismatch = False
                    '''
                    if len(tags) > 1:
                        i = None
                        for t in tags:
                            i = t
                            break
                        id = i.getID()
                        if id == 5 or id == 6 or id == 7 or id == 8:
                            blue = True
                        else:
                            blue = False
                        for j in tags:
                            id = j.getID()
                            if (blue == True and (id == 1 or id == 2 or id == 3 or id == 4)) or \
                                (blue == False and (id == 5 or id == 6 or id == 7 or id == 8)):
                                cv2.imwrite(f'tag_images/ERROR_TAG_{str(rio_time)}.jpg', img)
                                mismatch = True
                                break
                    '''
                    if mismatch == False:
                        cv2.imwrite(f'tag_images/tag_{str(rio_time)}.jpg', img)
                NetworkTableInstance.getDefault().flush()
                if tag_config_savefile_ntt.get() == True:
                    print("write tags")
                    file_write_tags(tagconfigfile_ntt.get(), threads_ntt.get(), \
                        quadDecimate_ntt.get(), blur_ntt.get(), refineEdges_ntt.get(), \
                        decodeSharpening_ntt.get(), ATDebug_ntt.get(), \
                        decision_margin_min_ntt.get(), decision_margin_max_ntt.get(), \
                        tag_crop_x_ntt.get(), tag_crop_y_ntt.get(), tag_corrected_errors_ntt.get()) 
                    tag_config_savefile_ntt.set(False)

        #FUEL!!!
        elif vision_type == 'fuel':
        
            #if fuel_enable_ntt.get() == True:

            db_n = debug_fuel_ntt.get()

            if db_n == True:
                fuel_min_h = int(fuel_min_h_ntt.get())
                fuel_min_s = int(fuel_min_s_ntt.get())
                fuel_min_v = int(fuel_min_v_ntt.get())
                fuel_max_h = int(fuel_max_h_ntt.get())
                fuel_max_s = int(fuel_max_s_ntt.get())
                fuel_max_v = int(fuel_max_v_ntt.get())
                fuel_min_area = int(fuel_min_area_ntt.get())

            '''
            # filter colors in HSV space
            img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            '''
            # even though image capture format is RGB888, images are stored as BGR
            # for HSV filtering / masking / detecting, convert input image from BGR to HSV
            # but for displaying the image, convert input image from BGR to RGB
            original_image = img.copy()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_HSV = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

            # only keep pixels with colors that match the range in color_config
            orange_low = np.array([fuel_min_h, fuel_min_s, fuel_min_v])
            orange_high = np.array([fuel_max_h, fuel_max_s, fuel_max_v])
            img_mask = cv2.inRange(img_HSV, orange_low, orange_high)
            # fuel should appear in the region below the bottom of this region
            #   img_mask[0:260,0:640] = 0
            
            orange, useless = cv2.findContours(img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #sorting the orange pixels from largest to smallest
            orangeSorted = sorted(orange, key=lambda x: cv2.contourArea(x), reverse=True)
            
            fuel = []
            fuel_contours = []
           
            max_contour = None
            center_y_max = -24
            area = 1
            for y in orangeSorted:

                area = cv2.contourArea(y)

                #uncomment the following block to get raw data output for debugging and calibrating distance / angle
                '''
                r_x,r_y,r_w,r_h = cv2.boundingRect(y)
                center_x = r_x + int(round(r_w / 2)) + FUEL_X_OFFSET
                center_y = r_y + int(round(r_h / 2)) + FUEL_Y_OFFSET
                extent = float(area) / (r_w * r_h)
                print(f'ar={area:4.1f} ex={extent:1.2f} fuel_x={center_x} fuel_y={center_y}')
                '''

                if area > 300:
                    r_x,r_y,r_w,r_h = cv2.boundingRect(y)
                    center_y = r_y + int(round(r_h / 2)) + FUEL_Y_OFFSET

                    if center_y > center_y_max:
                        center_y_max = center_y
                        max_contour = y     
            
            # at this point, max_contour points to closest shape by vertical y or None if the area of all were too small
            # now need to determine if this shape is a fuel
            if max_contour is not None:

                area = cv2.contourArea(max_contour)

                r_x,r_y,r_w,r_h = cv2.boundingRect(max_contour)
                center_x = r_x + int(round(r_w / 2)) + FUEL_X_OFFSET
                center_y = r_y + int(round(r_h / 2)) + FUEL_Y_OFFSET

                if (center_y > 17  and center_y < 240*2):

                    if (center_y > 240*2): # at really close, can't see the bottom, aspect ratio goes way up 
                        extent_min = 0.25
                    else:
                        extent_min = 0.25

                    #Extent is the ratio of contour area to bounding rectangle area.
                    extent = float(area) / (r_w * r_h)

                    #extent goes way down when we get real close
                    if (extent > extent_min and extent < 1.0):

                        if center_y >= 390: # don't see a full fuel this close, so y value for this distance is a bit off so force it to 0
                            distance = 0
                        else:
                            distance = fuel_regress_distance(center_y) # get distance (inches) using y location
                        px_per_deg = fuel_regress_px_per_deg(distance) # get pixel per degree
                        angle = (1 / px_per_deg) * (center_x - w/2)
                        if (distance >= 0 and distance < 360) and (angle >= -70 and angle < 70): # sanity check'''
                    
                            image_num += 1
                            image_counter += 1
                            image_time = time.perf_counter() - t1_time
                            image_time_av_total += image_time

                            if image_counter == FPS_NUM_SAMPLES:
                                fps_av = 1/(image_time_av_total/image_counter)
                                if fps_av < fps_av_min:
                                    fps_av_min = fps_av
                                if fps_av > fps_av_max:
                                    fps_av_max = fps_av
                                image_time_av_total = 0
                                image_counter = 0

                            pose_data = piece_pose_data_bytes(image_num, rio_time, image_time, 3, distance, angle)
                            fuel_pose_data_bytes_ntt.set(pose_data)
                            NetworkTableInstance.getDefault().flush()

                            if db_n == True:
                                txt = piece_pose_data_string(image_num, rio_time, image_time, distance, angle)
                                fuel_pose_data_string_header_ntt.set(txt)
                                fuel_distance_ntt.set(round(distance,2))
                                fuel_angle_ntt.set(round(angle,2))                       
                                cv2.circle(img, (center_x, center_y), 6, (255,0,255), -1)
                                cv2.drawContours(img, [max_contour], 0, (200,0,0), 4)
                                fuel_config_save(fuelConfigSave)
                                outputStreamFuel.putFrame(img) # send to dashboard
                                outputMask.putFrame(img_mask) # send to dashboard
                                if fuel_record_data_ntt.get() == True:
                                    fuel_data = f'{area:4.1f},{extent:2.1f},{center_x},{center_y},{distance:3.1f},{angle:2.1f}'
                                    with open('fuel_data.txt', 'a') as f:
                                        f.write(fuel_data)
                                        f.write('\n')
                                    fuel_record_data_ntt.set(False)
                                continue
            if db_n == True:
                fuel_config_save(fuelConfigSave)
                outputStreamFuel.putFrame(img) # send to dashboard
                outputMask.putFrame(img_mask) # send to dashboard

                    
        else:
            continue

main()