from cscore import CameraServer
import ntcore
from ntcore import NetworkTableInstance
from enum import Enum
import configparser
import robotpy_apriltag
import cv2
import numpy as np
import time
import os
import os.path
import ast
import math
import struct
from math import log10, floor
import json
from picamera2 import Picamera2
import libcamera
from libcamera import controls
import threading
from pprint import *
import sys
import pickle




X_RES = 320
Y_RES = 240
UPTIME_UPDATE_INTERVAL = 1
TEMP_UPDATE_INTERVAL= 30
DEBUG_MODE_DEFAULT = False
THREADS_DEFAULT = 3
DECIMATE_DEFAULT = 1.0
BLUR_DEFAULT = 0.0
REFINE_EDGES_DEFAULT = True
SHARPENING_DEFAULT = 0.25
APRILTAG_DEBUG_MODE_DEFAULT = False
DECISION_MARGIN_DEFAULT = 125
CAMERA_CAL_FILE_NAME = "MultiMatrix.npz.PiGS.640.480" # "MultiMatrix.npz" #"MultiMatrix.npz.PiGS.320.240" #"MultiMatrix.npz" #MultiMatrix.npz.PiGS.640.480" # "MultiMatrix.npz.PiGS.320.240" # "MultiMatrix.npz.webcam.320.240" # "MultiMatrix.npz.webcam.640.480"
THREADS_TOPIC_NAME = "/Vision/Threads"
DECIMATE_TOPIC_NAME = "/Vision/Decimate"
BLUR_TOPIC_NAME = "/Vision/Blur"
REFINE_EDGES_TOPIC_NAME = "/Vision/Edge Refine"
SHARPENING_TOPIC_NAME = "/Vision/Sharpening"
APRILTAG_DEBUG_MODE_TOPIC_NAME = "/Vision/April Tag Debug"
DECISION_MARGIN_MIN_TOPIC_NAME = "/Vision/Decision Margin Min"
DECISION_MARGIN_MAX_TOPIC_NAME = "/Vision/Decision Margin Max"
TAG_CONFIG_FILE_TOPIC_NAME = "/Vision/Tag Config File"
#ACTIVE_TOPIC_NAME = "/Vision/Active"
TAG_ACTIVE_TOPIC_NAME = "/Vision/Tag Active" 
NOTE_ACTIVE_TOPIC_NAME = "/Vision/Note Active" 
POSE_DATA_RAW_TOPIC_NAME = "Tag Pose Data Bytes" #cannot say /Vision becuase we already do in NTGetRaw
NOTE_POSE_DATA_RAW_TOPIC_NAME = "Note Pose Data Bytes" #cannot say /Vision becuase we already do in NTGetRaw
POSE_DATA_STRING_TOPIC_NAME_HEADER ="/Vision/Pose Data Header"
NOTE_POSE_DATA_STRING_TOPIC_NAME_HEADER = "/Vision/Note Pose Data Header"
POSE_DATA_STRING_TOPIC_NAME_DATA_TRANSLATION ="/Vision/Pose Data Trans"
POSE_DATA_STRING_TOPIC_NAME_DATA_ROTATION ="/Vision/Pose Data Rot"
TAG_PI_TEMP_TOPIC_NAME = "/Vision/Tag Temperature"
NOTE_PI_TEMP_TOPIC_NAME = "/Vision/Note Temperature"
RIO_TIME_TOPIC_NAME = "/Vision/RIO Time"

Z_IN_TOPIC_NAME = "/Vision/Z In"
NOTE_MIN_HUE_TOPIC_NAME = "/Vision/Note Min Hue"
NOTE_MIN_SAT_TOPIC_NAME = "/Vision/Note Min Sat"
NOTE_MIN_VAL_TOPIC_NAME = "/Vision/Note Min Val"
NOTE_MAX_HUE_TOPIC_NAME = "/Vision/Note Max Hue"
NOTE_MAX_SAT_TOPIC_NAME = "/Vision/Note Max Sat"
NOTE_MAX_VAL_TOPIC_NAME = "/Vision/Note Max Val"
NOTE_CONFIG_FILE_TOPIC_NAME = "/Vision/Note Config File"
NOTE_CONFIG_FILE_DEFAULT = "note_config.ini"
TAG_CONFIG_FILE_DEFAULT = "tag_config.ini"
GEN_CONFIG_FILE_DEFAULT = "gen_config.ini"
NOTE_MIN_HUE = 0
NOTE_MIN_SAT = 0
NOTE_MIN_VAL = 0
NOTE_MAX_HUE = 179
NOTE_MAX_SAT = 255
NOTE_MAX_VAL = 255
TAG_ENABLE_TOPIC_NAME = "/Vision/Tag Enable"
NOTE_ENABLE_TOPIC_NAME = "/Vision/Note Enable"
TOP_LINE_DIST_FROM_TOP = 0.15
BOTTOM_LINE_DIST_FROM_TOP = 0.7
NOTE_MIN_AREA_TOPIC_NAME = "/Vision/Note Min Area"
NOTE_MIN_AREA = 44 #275
NOTE_ANGLE_TOPIC_NAME = "/Vision/Note Angle"
WRITE_TAG_IMAGE = False
TAG_RECORD_ENABLE_TOPIC_NAME = "/Vision/Tag Record"
TAG_RECORD_REMOVE_TOPIC_NAME = "/Vision/Tag Remove"
NOTE_RECORD_DATA_TOPIC_NAME = "/Vision/Note Record"
NOTE_X_OFFSET = 0
NOTE_Y_OFFSET = -3
FPS_NUM_SAMPLES = 100 #after this number of images the fps average is calulated
TAG_CROP_TOP_TOPIC_NAME = "/Vision/Tag Crop Top"
TAG_CROP_BOTTOM_TOPIC_NAME = "/Vision/Tag Crop Bottom"
TAG_CROP_TOP_DEFAULT = 0 # this is a %; default to no crop
TAG_CROP_BOTTOM_DEFAULT = 100 # this is a %; default to no crop
NOTE_NUM_PIXELS_FROM_CENTER_BLANK = 15
TAG_BRIGHTNESS_TOPIC_NAME = "/Vision/Tag Brightness"
NOTE_BRIGHTNESS_TOPIC_NAME = "/Vision/Note Brightness"

BRIGHTNESS_DEFAULT = 0.0
TAG_CONTRAST_TOPIC_NAME = "/Vision/Tag Contrast"
NOTE_CONTRAST_TOPIC_NAME = "/Vision/Note Contrast"
CONTRAST_DEFAULT = 1.0
TAG_ERRORS_TOPIC_NAME = "/Vision/Tag Corrected Errors"
TAG_ERRORS_DEFAULT = 0
TAG_AE_TOPIC_NAME = "/Vision/Tag Auto Exposure"
NOTE_AE_TOPIC_NAME = "/Vision/Note Auto Exposure"
AE_DEFAULT = True
TAG_EXPOSURE_TOPIC_NAME = "/Vision/Tag Manual Exposure" # only used if AE_TOPIC_NAME is disabled
NOTE_EXPOSURE_TOPIC_NAME = "/Vision/Note Manual Exposure" # only used if AE_TOPIC_NAME is disabled
EXPOSURE_DEFAULT = 1000 # in microseconds - total guess as default 
POSE_DATA_X_DEG_TOPIC_NAME = "/Vision/X Deg"
POSE_DATA_Y_DEG_TOPIC_NAME = "/Vision/Y Deg"
POSE_DATA_Z_DEG_TOPIC_NAME = "/Vision/Z Deg"
POSE_DATA_X_IN_TOPIC_NAME = "/Vision/X In"
POSE_DATA_Y_IN_TOPIC_NAME = "/Vision/Y In"
TAG_DETECTED_ID_TOPIC_NAME = "/Vision/Tag Id"
TAG_DETECTED_DM_TOPIC_NAME = "/Vision/Tag DM"
TAG_DETECTED_ERRORS_TOPIC_NAME = "/Vision/Tag Errors"
class NTConnectType(Enum):
    SERVER = 1
    CLIENT = 2
class NTGetString:
    def __init__(self, stringTopic: ntcore.StringTopic, init, default, failsafe):
        self.init = init
        self.default = default
        self.failsafe = failsafe
        # start subscribing; the return value must be retained.
        # the parameter is the default value if no value is available when get() is called
        self.stringTopic = stringTopic.getEntry(failsafe)

        self.stringTopic.setDefault(default)
        self.stringTopic.set(init)

    def get(self):
        return self.stringTopic.get(self.failsafe)

    def set(self, string):
        self.stringTopic.set(string)

    def unpublish(self):
        # you can stop publishing while keeping the subscriber alive
        self.stringTopic.unpublish()

    def close(self):
        # stop subscribing/publishing
        self.stringTopic.close()
class NTGetBoolean:
    def __init__(self, boolTopic: ntcore.BooleanTopic, init, default, failsafe):
        self.init = init
        self.default = default
        self.failsafe = failsafe

        # start subscribing; the return value must be retained.
        # the parameter is the default value if no value is available when get() is called
        self.boolTopic = boolTopic.getEntry(failsafe)

        self.boolTopic.setDefault(default)
        self.boolTopic.set(init)

    def get(self):
        return self.boolTopic.get(self.failsafe)
    def set(self, boolean):
        self.boolTopic.set(boolean)
    def unpublish(self):
        # you can stop publishing while keeping the subscriber alive
        self.boolTopic.unpublish()

    def close(self):
        # stop subscribing/publishing
        self.boolTopic.close()
class NTGetDouble:
    def __init__(self, dblTopic: ntcore.DoubleTopic, init, default, failsafe):
        self.init = init
        self.default = default
        self.failsafe = failsafe
        # start subscribing; the return value must be retained.
        # the parameter is the default value if no value is available when get() is called
        self.dblTopic = dblTopic.getEntry(failsafe)
        self.dblTopic.setDefault(default)
        self.dblTopic.set(init)

    def get(self):
        return self.dblTopic.get(self.failsafe)

    def set(self, double):
        self.dblTopic.set(double)

    def unpublish(self):
        # you can stop publishing while keeping the subscriber alive
        self.dblTopic.unpublish()

    def close(self):
        # stop subscribing/publishing
        self.dblTopic.close()
class NTGetRaw:
    def __init__(self, ntinst, topicname, init, default, failsafe):
        self.init = init
        self.default = default
        self.failsafe = failsafe
        self.table = ntinst.getTable("/Vision")

        self.pub = self.table.getRawTopic(topicname).publish("raw")

    def set(self, raw):
        self.pub.set(raw)

    def unpublish(self):
        # you can stop publishing while keeping the subscriber alive
        self.pub.unpublish()

    def close(self):
        # stop subscribing/publishing
        self.pub.close()
#!/usr/bin/python3

# These two are only needed for the demo code below the FrameServer class.
import time
from threading import Condition, Thread

from picamera2 import Picamera2


class FrameServer:
    def __init__(self, picam2, stream='main'):
        """A simple class that can serve up frames from one of the Picamera2's configured streams to multiple other threads.

        Pass in the Picamera2 object and the name of the stream for which you want
        to serve up frames.
        """
        self._picam2 = picam2
        self._stream = stream
        self._array = None
        self._condition = Condition()
        self._running = True
        self._count = 0
        self._thread = Thread(target=self._thread_func, daemon=True)

    @property
    def count(self):
        """A count of the number of frames received."""
        return self._count

    def start(self):
        """To start the FrameServer, you will also need to start the Picamera2 object."""
        self._thread.start()

    def stop(self):
        """To stop the FrameServer

        First stop any client threads (that might be
        blocked in wait_for_frame), then call this stop method. Don't stop the
        Picamera2 object until the FrameServer has been stopped.
        """
        self._running = False
        self._thread.join()

    def _thread_func(self):
        count = 0
        while self._running:
            array = self._picam2.capture_array(self._stream)
            self._count += 1
            ''' # uncomment this block to see exposure time for images
            count += 1
            if (count > 1000):
                print(self._picam2.capture_metadata()['ExposureTime'])
                count = 0
            '''
            with self._condition:
                self._array = array
                self._condition.notify_all()

    def wait_for_frame(self, previous=None):
        """You may optionally pass in the previous frame that you got last time you called this function.

        This will guarantee that you don't get duplicate frames
        returned in the event of spurious wake-ups, and it may even return more
        quickly in the case where a new frame has already arrived.
        """
        with self._condition:
            if previous is not None and self._array is not previous:
                return self._array
            while True:
                self._condition.wait()
                if self._array is not previous:
                    return self._array


class PieceData:
    def __init__(self, num, rio_time, image_time, type, distance, angle):
        self.image_num = num
        self.rio_time = rio_time
        self.image_time = time_time
        self.type = type
        self.distance = distance
        self.angle = angle


def note_regress_distance(y):
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

def note_regress_px_per_deg(x):
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

def file_write_notes(file,
                min_h,
                min_s,
                min_v,
                max_h,
                max_s,
                max_v,
                min_area):

    parser = configparser.ConfigParser()

    parser.add_section('VISION')
    parser.set('VISION', NOTE_CONFIG_FILE_TOPIC_NAME, str(file))
    parser.set('VISION', NOTE_MIN_HUE_TOPIC_NAME, str(round(min_h)))
    parser.set('VISION', NOTE_MIN_SAT_TOPIC_NAME, str(round(min_s)))
    parser.set('VISION', NOTE_MIN_VAL_TOPIC_NAME, str(round(min_v)))
    parser.set('VISION', NOTE_MAX_HUE_TOPIC_NAME, str(round(max_h)))
    parser.set('VISION', NOTE_MAX_SAT_TOPIC_NAME, str(round(max_s)))
    parser.set('VISION', NOTE_MAX_VAL_TOPIC_NAME, str(round(max_v)))
    parser.set('VISION', NOTE_MIN_AREA_TOPIC_NAME, str(round(min_area)))
    
    #print(f'file={file} mh={str(min_h)} ms={str(min_s)} mv={str(min_v)} xh={str(max_h)} xs={str(max_s)} xv={str(max_v)}')

    #HEY HEY HEY!!! LOOK AT MEEEEE!!!! >>>pscp.exe pi@10.2.33.177:/home/pi/config.ini C:\Users\23JMurphy\Downloads will copy any file from pi to windows<<<
    with open(file, 'w') as config:
        parser.write(config)
        print('wrote note file:')
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

def file_read_note(parser, configfile_failure_ntt):
    config_exists = os.path.isfile(NOTE_CONFIG_FILE_DEFAULT)
    if config_exists == True:
        parser.read(NOTE_CONFIG_FILE_DEFAULT)
        configfile_failure_ntt.set(False) #if it works mark no error
        print('read note file:')
        print({'VISION': dict(parser['VISION'])})
    else: # re-create config and container file to default
        configfile_failure_ntt.set(True) # set error for config file

        parser.add_section('VISION')
        
        parser.set('VISION', NOTE_CONFIG_FILE_TOPIC_NAME, str(NOTE_CONFIG_FILE_DEFAULT))
        parser.set('VISION', NOTE_MIN_HUE_TOPIC_NAME, str(NOTE_MIN_HUE))
        parser.set('VISION', NOTE_MIN_SAT_TOPIC_NAME, str(NOTE_MIN_SAT))
        parser.set('VISION', NOTE_MIN_VAL_TOPIC_NAME, str(NOTE_MIN_VAL))
        parser.set('VISION', NOTE_MAX_HUE_TOPIC_NAME, str(NOTE_MAX_HUE))
        parser.set('VISION', NOTE_MAX_SAT_TOPIC_NAME, str(NOTE_MAX_SAT))
        parser.set('VISION', NOTE_MAX_VAL_TOPIC_NAME, str(NOTE_MAX_VAL))
        parser.set('VISION', NOTEMIN_AREA_TOPIC_NAME, str(NOTE_MIN_AREA))

        with open("/home/pi/" + NOTE_CONFIG_FILE_DEFAULT, 'w') as config:
            parser.write(config)
            print('wrote note file:')
            print({'VISION': dict(parser['VISION'])})
        configfile_failure_ntt.set(False) # config file recreated

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

    threads.set(float(config.get('VISION', THREADS_TOPIC_NAME)))
    quadDecimate.set(float(config.get('VISION', DECIMATE_TOPIC_NAME)))
    blur.set(float(config.get('VISION', BLUR_TOPIC_NAME)))
    refineEdges.set(ast.literal_eval(config.get('VISION', REFINE_EDGES_TOPIC_NAME)))
    decodeSharpening.set(float(config.get('VISION', SHARPENING_TOPIC_NAME)))
    ATDebug.set(ast.literal_eval(config.get('VISION', APRILTAG_DEBUG_MODE_TOPIC_NAME)))
    decision_min.set(float(config.get('VISION', DECISION_MARGIN_MIN_TOPIC_NAME)))
    decision_max.set(float(config.get('VISION', DECISION_MARGIN_MAX_TOPIC_NAME)))
    crop_x.set(float(config.get('VISION', TAG_CROP_TOP_TOPIC_NAME)))
    crop_y.set(float(config.get('VISION', TAG_CROP_BOTTOM_TOPIC_NAME)))
    errors.set(float(config.get('VISION', TAG_ERRORS_TOPIC_NAME)))
    configfile.set(str(config.get('VISION', TAG_CONFIG_FILE_TOPIC_NAME)))

def nt_update_notes(config,
              configfile,
              min_h,
              min_s,
              min_v,
              max_h,
              max_s,
              max_v,
              min_area):
    # sync the stuff in the file with matching values in the file

    configfile.set(str(config.get('VISION', NOTE_CONFIG_FILE_TOPIC_NAME)))
    min_h.set(float(config.get('VISION', NOTE_MIN_HUE_TOPIC_NAME)))
    min_s.set(float(config.get('VISION', NOTE_MIN_SAT_TOPIC_NAME)))
    min_v.set(float(config.get('VISION', NOTE_MIN_VAL_TOPIC_NAME)))
    max_h.set(float(config.get('VISION', NOTE_MAX_HUE_TOPIC_NAME)))
    max_s.set(float(config.get('VISION', NOTE_MAX_SAT_TOPIC_NAME)))
    max_v.set(float(config.get('VISION', NOTE_MAX_VAL_TOPIC_NAME)))
    min_area.set(float(config.get('VISION', NOTE_MIN_AREA_TOPIC_NAME)))

def nt_update_gen(config,
              tag_brightness,
              tag_contrast,
              tag_ae,
              tag_exposure,
              note_brightness,
              note_contrast,
              note_ae,
              note_exposure):
    # sync the stuff in the file with matching values in the file

    if get_type() == 'tag':
        tag_brightness.set(float(config.get('GENERAL', 'Brightness')))
        tag_contrast.set(float(config.get('GENERAL', 'Contrast')))
        tag_ae.set(bool(config.get('GENERAL', 'Auto Exposure')))
        tag_exposure.set(float(config.get('GENERAL', 'Manual Exposure Time')))
    else:
        note_brightness.set(float(config.get('GENERAL', 'Brightness')))
        note_contrast.set(float(config.get('GENERAL', 'Contrast')))
        note_ae.set(bool(config.get('GENERAL', 'Auto Exposure')))
        note_exposure.set(float(config.get('GENERAL', 'Manual Exposure Time')))

'''
all data to send is packaged as an array of bytes, using a Python bytearray, in big-endian format:
sequence number: unsigned long (4 bytes)
rio time: float (4 bytes)
image time:float (4 bytes)
type (tag = 1, note = 3): unsigned char (1 byte)
length: how many tags/notes follow
what follows these first 3 items depends on the type:
tag:
number of tags detected: unsigned char (1 byte)
for each tag: tag id unsigned char (1 byte), pose x: float (4 bytes), pose y: float (4 bytes), pose z: float (4 bytes), pose x angle: float (4 bytes), pose y angle: float (4 bytes), pose z angle: float (4 bytes)

note:
number of notes detected: unsigned char (1 byte)
for each note: pose x: float (4 bytes), pose y: float (4 bytes), pose z: float (4 bytes), pose x angle: float (4 bytes), pose y angle: float (4 bytes), pose z angle: float (4 bytes)s)
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

def main():

    print("Hello")

    vision_type = get_type()
    camera_orientation = camera_upside_down()

    # start NetworkTables
    ntconnect = NTConnectType(NTConnectType.CLIENT)    #use CLIENT when running with rio
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
    note_uptime_ntt = NTGetDouble(ntinst.getDoubleTopic("/Vision/Note Uptime"), 0, 0, -1)

    debug_tag_ntt = NTGetBoolean(ntinst.getBooleanTopic("/Vision/Tag Debug Mode"), DEBUG_MODE_DEFAULT, DEBUG_MODE_DEFAULT, DEBUG_MODE_DEFAULT)
    debug_note_ntt = NTGetBoolean(ntinst.getBooleanTopic("/Vision/Note Debug Mode"), DEBUG_MODE_DEFAULT, DEBUG_MODE_DEFAULT, DEBUG_MODE_DEFAULT)
    threads_ntt = NTGetDouble(ntinst.getDoubleTopic(THREADS_TOPIC_NAME),THREADS_DEFAULT, THREADS_DEFAULT, THREADS_DEFAULT)
    quadDecimate_ntt = NTGetDouble(ntinst.getDoubleTopic(DECIMATE_TOPIC_NAME),DECIMATE_DEFAULT, DECIMATE_DEFAULT, DECIMATE_DEFAULT)
    blur_ntt = NTGetDouble(ntinst.getDoubleTopic(BLUR_TOPIC_NAME),BLUR_DEFAULT, BLUR_DEFAULT, BLUR_DEFAULT) 
    refineEdges_ntt = NTGetBoolean(ntinst.getBooleanTopic(REFINE_EDGES_TOPIC_NAME),REFINE_EDGES_DEFAULT, REFINE_EDGES_DEFAULT, REFINE_EDGES_DEFAULT) 
    decodeSharpening_ntt = NTGetDouble(ntinst.getDoubleTopic(SHARPENING_TOPIC_NAME), SHARPENING_DEFAULT, SHARPENING_DEFAULT, SHARPENING_DEFAULT)
    ATDebug_ntt = NTGetBoolean(ntinst.getBooleanTopic(APRILTAG_DEBUG_MODE_TOPIC_NAME), APRILTAG_DEBUG_MODE_DEFAULT, APRILTAG_DEBUG_MODE_DEFAULT, APRILTAG_DEBUG_MODE_DEFAULT)
    decision_margin_min_ntt = NTGetDouble(ntinst.getDoubleTopic(DECISION_MARGIN_MIN_TOPIC_NAME), DECISION_MARGIN_DEFAULT, DECISION_MARGIN_DEFAULT, DECISION_MARGIN_DEFAULT)
    tagconfigfile_ntt = NTGetString(ntinst.getStringTopic(TAG_CONFIG_FILE_TOPIC_NAME), TAG_CONFIG_FILE_DEFAULT,TAG_CONFIG_FILE_DEFAULT, TAG_CONFIG_FILE_DEFAULT)
    noteconfigfile_ntt = NTGetString(ntinst.getStringTopic(NOTE_CONFIG_FILE_TOPIC_NAME), NOTE_CONFIG_FILE_DEFAULT,NOTE_CONFIG_FILE_DEFAULT, NOTE_CONFIG_FILE_DEFAULT)
    savefile_ntt = NTGetBoolean(ntinst.getBooleanTopic("/Vision/Save File"), False, False, False)
    configfilefail_ntt = NTGetBoolean(ntinst.getBooleanTopic("/Vision/Config File Fail"), False, False, False)
    tag_active_ntt = NTGetBoolean(ntinst.getBooleanTopic(TAG_ACTIVE_TOPIC_NAME), True, True, True)
    note_active_ntt = NTGetBoolean(ntinst.getBooleanTopic(NOTE_ACTIVE_TOPIC_NAME), True, True, True)
    pose_data_bytes_ntt = NTGetRaw(ntinst, POSE_DATA_RAW_TOPIC_NAME, None, None, None)
    note_pose_data_bytes_ntt = NTGetRaw(ntinst, NOTE_POSE_DATA_RAW_TOPIC_NAME, None, None, None)
    pose_data_string_header_ntt = NTGetString(ntinst.getStringTopic(POSE_DATA_STRING_TOPIC_NAME_HEADER),"", "", "")
    note_pose_data_string_header_ntt = NTGetString(ntinst.getStringTopic(NOTE_POSE_DATA_STRING_TOPIC_NAME_HEADER),"", "", "") 
    pose_data_string_data_translation_ntt = NTGetString(ntinst.getStringTopic(POSE_DATA_STRING_TOPIC_NAME_DATA_TRANSLATION),"", "", "")
    pose_data_string_data_rotation_ntt = NTGetString(ntinst.getStringTopic(POSE_DATA_STRING_TOPIC_NAME_DATA_ROTATION),"", "", "")
    temp_tag_ntt = NTGetDouble(ntinst.getDoubleTopic(TAG_PI_TEMP_TOPIC_NAME), 0, 0, 0)
    temp_note_ntt = NTGetDouble(ntinst.getDoubleTopic(NOTE_PI_TEMP_TOPIC_NAME), 0, 0, 0)
    z_in_ntt = NTGetDouble(ntinst.getDoubleTopic(Z_IN_TOPIC_NAME), 0.0, 0.0, 0.0)
    note_min_h_ntt = NTGetDouble(ntinst.getDoubleTopic(NOTE_MIN_HUE_TOPIC_NAME), NOTE_MIN_HUE, NOTE_MIN_HUE, NOTE_MIN_HUE)
    note_min_s_ntt = NTGetDouble(ntinst.getDoubleTopic(NOTE_MIN_SAT_TOPIC_NAME), NOTE_MIN_SAT, NOTE_MIN_SAT, NOTE_MIN_SAT)
    note_min_v_ntt = NTGetDouble(ntinst.getDoubleTopic(NOTE_MIN_VAL_TOPIC_NAME), NOTE_MIN_VAL, NOTE_MIN_VAL, NOTE_MIN_VAL)
    note_max_h_ntt = NTGetDouble(ntinst.getDoubleTopic(NOTE_MAX_HUE_TOPIC_NAME), NOTE_MAX_HUE, NOTE_MAX_HUE, NOTE_MAX_HUE)
    note_max_s_ntt = NTGetDouble(ntinst.getDoubleTopic(NOTE_MAX_SAT_TOPIC_NAME), NOTE_MAX_SAT, NOTE_MAX_SAT, NOTE_MAX_SAT)
    note_max_v_ntt = NTGetDouble(ntinst.getDoubleTopic(NOTE_MAX_VAL_TOPIC_NAME), NOTE_MAX_VAL, NOTE_MAX_VAL, NOTE_MAX_VAL)
    tag_enable = NTGetBoolean(ntinst.getBooleanTopic(TAG_ENABLE_TOPIC_NAME), False, False, False)
    note_enable_ntt = NTGetBoolean(ntinst.getBooleanTopic(NOTE_ENABLE_TOPIC_NAME), False, False, False)
    note_min_area_ntt = NTGetDouble(ntinst.getDoubleTopic(NOTE_MIN_AREA_TOPIC_NAME), NOTE_MIN_AREA, NOTE_MIN_AREA, NOTE_MIN_AREA)
    note_angle_ntt = NTGetDouble(ntinst.getDoubleTopic(NOTE_ANGLE_TOPIC_NAME), 0.0, 0.0, 0.0)
    tag_record_ntt = NTGetBoolean(ntinst.getBooleanTopic(TAG_RECORD_ENABLE_TOPIC_NAME), False, False, False)
    tag_record_remove_ntt = NTGetBoolean(ntinst.getBooleanTopic(TAG_RECORD_REMOVE_TOPIC_NAME), False, False, False)
    note_record_data_ntt = NTGetBoolean(ntinst.getBooleanTopic(NOTE_RECORD_DATA_TOPIC_NAME), False, False, False)
    decision_margin_max_ntt = NTGetDouble(ntinst.getDoubleTopic(DECISION_MARGIN_MAX_TOPIC_NAME), DECISION_MARGIN_DEFAULT, DECISION_MARGIN_DEFAULT, DECISION_MARGIN_DEFAULT)
    note_distance_ntt = NTGetDouble(ntinst.getDoubleTopic("/Vision/Note Distance"), 0.0, 0.0, 0.0)
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

    note_brightness_ntt = NTGetDouble(ntinst.getDoubleTopic(NOTE_BRIGHTNESS_TOPIC_NAME), BRIGHTNESS_DEFAULT, BRIGHTNESS_DEFAULT, BRIGHTNESS_DEFAULT)
    note_contrast_ntt = NTGetDouble(ntinst.getDoubleTopic(NOTE_CONTRAST_TOPIC_NAME), CONTRAST_DEFAULT, CONTRAST_DEFAULT, CONTRAST_DEFAULT)
    note_ae_ntt = NTGetBoolean(ntinst.getBooleanTopic(NOTE_AE_TOPIC_NAME), AE_DEFAULT, AE_DEFAULT, AE_DEFAULT)
    note_exposure_ntt = NTGetDouble(ntinst.getDoubleTopic(NOTE_EXPOSURE_TOPIC_NAME), EXPOSURE_DEFAULT, EXPOSURE_DEFAULT, EXPOSURE_DEFAULT)

    detector = robotpy_apriltag.AprilTagDetector()
    #detector.addFamily("tag16h5")
    detector.addFamily("tag36h11")

    # use for file
    config_tag = configparser.ConfigParser()
    config_note = configparser.ConfigParser()
    config_gen = configparser.ConfigParser()

    '''
    print('*****')
    parser= configparser.ConfigParser()
    parser.read('gen_config.ini')
    print('read gen config file:')
    print({'GENERAL': dict(parser['GENERAL'])})
    return(parser.get('GENERAL', 'type'))
    '''

    file_read_tag(config_tag, configfilefail_ntt)
    file_read_note(config_note, configfilefail_ntt)
    file_read_gen(config_gen, configfilefail_ntt)

    nt_update_tags(config_tag,threads_ntt, quadDecimate_ntt, blur_ntt, refineEdges_ntt, \
        decodeSharpening_ntt, ATDebug_ntt, decision_margin_min_ntt, decision_margin_max_ntt, \
        tag_crop_x_ntt, tag_crop_y_ntt, tag_corrected_errors_ntt, tagconfigfile_ntt)
    nt_update_notes(config_note, noteconfigfile_ntt, \
        note_min_h_ntt, note_min_s_ntt, note_min_v_ntt, note_max_h_ntt, note_max_s_ntt, note_max_v_ntt, \
        note_min_area_ntt)
    nt_update_gen(config_gen, tag_brightness_ntt, tag_contrast_ntt, tag_ae_ntt, tag_exposure_ntt, \
        note_brightness_ntt, note_contrast_ntt, note_ae_ntt, note_exposure_ntt)
    
    detectorConfig = robotpy_apriltag.AprilTagDetector.Config()

    detectorConfig.numThreads = int(float(config_tag.get('VISION', THREADS_TOPIC_NAME)))
    detectorConfig.quadDecimate = float(config_tag.get('VISION', DECIMATE_TOPIC_NAME))
    detectorConfig.quadSigma = float (config_tag.get('VISION', BLUR_TOPIC_NAME))
    detectorConfig.refineEdges = ast.literal_eval(config_tag.get('VISION', REFINE_EDGES_TOPIC_NAME))
    detectorConfig.decodeSharpening = float(config_tag.get('VISION', SHARPENING_TOPIC_NAME))
    detectorConfig.debug = ast.literal_eval(config_tag.get('VISION', APRILTAG_DEBUG_MODE_TOPIC_NAME))
    detector.setConfig(detectorConfig)
    

    note_min_h = int(config_note.get('VISION', NOTE_MIN_HUE_TOPIC_NAME))
    note_min_s = int(config_note.get('VISION', NOTE_MIN_SAT_TOPIC_NAME))
    note_min_v = int(config_note.get('VISION', NOTE_MIN_VAL_TOPIC_NAME))
    note_max_h = int(config_note.get('VISION', NOTE_MAX_HUE_TOPIC_NAME))
    note_max_s = int(config_note.get('VISION', NOTE_MAX_SAT_TOPIC_NAME))
    note_max_v = int(config_note.get('VISION', NOTE_MAX_VAL_TOPIC_NAME))
    note_min_area = int(config_note.get('VISION', NOTE_MIN_AREA_TOPIC_NAME))
    note_max_area = 1000

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
        outputStreamNote = CameraServer.putVideo("note image", cam_config['width'], cam_config['height'])
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
    note_last_brightness = None
    note_last_contrast = None
    note_last_ae_mode = None



    while True:

        rio_time = rio_time_ntt.get()
        current_seconds = time.time()
        time_check = False
        if current_seconds - prev_seconds >= UPTIME_UPDATE_INTERVAL:
            prev_seconds = current_seconds
            seconds = seconds + 1
            temp_sec = temp_sec + 1

            db_t = debug_tag_ntt.get()
            db_n = debug_note_ntt.get()

            if db_t == True and vision_type == 'tag':

                if tag_last_brightness != tag_brightness_ntt.get():
                    picam2.set_controls({'Brightness': float(tag_brightness_ntt.get())})
                    config_gen.set('GENERAL', 'Brightness', str(tag_brightness_ntt.get()))
                    tag_last_brightness = tag_brightness_ntt.get()

                if tag_last_contrast != tag_contrast_ntt.get():
                    picam2.set_controls({'Contrast': float(tag_contrast_ntt.get())})
                    config_gen.set('GENERAL', 'Contrast', str(tag_contrast_ntt.get()))
                    tag_last_contrast = tag_contrast_ntt.get()

                if tag_last_ae_mode != tag_ae_ntt.get():
                    picam2.set_controls({'AeEnable': bool(tag_ae_ntt.get())})
                    config_gen.set('GENERAL', 'AeEnable', str(tag_ae_ntt.get()))
                    tag_last_ae_mode = tag_ae_ntt.get()

                if tag_last_ae_mode == False:
                    exp_time = int(round(tag_exposure_ntt.get()))
                    picam2.set_controls({"ExposureTime": \
                        exp_time, "AnalogueGain": 1.0})
                    config_gen.set('GENERAL', 'Manual Exposure Time', str(exp_time))

            if db_n == True and vision_type == 'note':

                if note_last_brightness != note_brightness_ntt.get():
                    picam2.set_controls({'Brightness': float(note_brightness_ntt.get())})
                    config_gen.set('GENERAL', 'Brightness', str(note_brightness_ntt.get()))
                    note_last_brightness = note_brightness_ntt.get()

                if note_last_contrast != note_contrast_ntt.get():
                    picam2.set_controls({'Contrast': float(note_contrast_ntt.get())})
                    config_gen.set('GENERAL', 'Contrast', str(note_contrast_ntt.get()))
                    note_last_contrast = note_contrast_ntt.get()

                if note_last_ae_mode != note_ae_ntt.get():
                    picam2.set_controls({'AeEnable': bool(note_ae_ntt.get())})
                    config_gen.set('GENERAL', 'AeEnable', str(note_ae_ntt.get()))
                    note_last_ae_mode = note_ae_ntt.get()

                if note_last_ae_mode == False:
                    exp_time = int(round(note_exposure_ntt.get()))
                    picam2.set_controls({"ExposureTime": \
                        exp_time, "AnalogueGain": 1.0})
                    config_gen.set('GENERAL', 'Manual Exposure Time', str(exp_time))

            tag_uptime_ntt.set(seconds)
            note_uptime_ntt.set(seconds)
            time_check = True

            if vision_type == 'tag':
                if db_t == True:
                    print(f'sec={seconds} tags: ave fps={round(tag_fps_av,0)} min fps={round(tag_fps_av_min,0)} max fps={round(tag_fps_av_max,0)}')
                else:
                    print(f'{seconds}')
            else:                
                if db_n == True:
                    print(f'sec={seconds} notes: ave fps={round(fps_av,0)} fps min={round(fps_av_min,0)} fps max={round(fps_av_max,0)}')
                else:
                    print(f'{seconds}')
            
        if temp_sec >= TEMP_UPDATE_INTERVAL:
            with open("/sys/class/thermal/thermal_zone0/temp", 'r') as f:
                current_temp = int(f.readline()) / 1000 #converting milidegrees C to degrees C
                if vision_type == 'tag':
                    temp_tag_ntt.set(current_temp)
                else:
                    temp_note_ntt.set(current_temp)
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
                if savefile_ntt.get() == True:
                    print("write tags")
                    file_write_tags(tagconfigfile_ntt.get(), threads_ntt.get(), \
                        quadDecimate_ntt.get(), blur_ntt.get(), refineEdges_ntt.get(), \
                        decodeSharpening_ntt.get(), ATDebug_ntt.get(), \
                        decision_margin_min_ntt.get(), decision_margin_max_ntt.get(), \
                        tag_crop_x_ntt.get(), tag_crop_y_ntt.get(), tag_corrected_errors_ntt.get()) 
                    savefile_ntt.set(False)

        #NOTE!!!
        elif vision_type == 'note':
            #if note_enable_ntt.get() == True:

            if db_n == True:
                note_min_h = int(note_min_h_ntt.get())
                note_min_s = int(note_min_s_ntt.get())
                note_min_v = int(note_min_v_ntt.get())
                note_max_h = int(note_max_h_ntt.get())
                note_max_s = int(note_max_s_ntt.get())
                note_max_v = int(note_max_v_ntt.get())
                note_min_area = int(note_min_area_ntt.get())

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
            orange_low = np.array([note_min_h, note_min_s, note_min_v])
            orange_high = np.array([note_max_h, note_max_s, note_max_v])
            img_mask = cv2.inRange(img_HSV, orange_low, orange_high)
            # notes should appear in the region below the bottom of this region
            #   img_mask[0:260,0:640] = 0
            
            orange, useless = cv2.findContours(img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #sorting the orange pixels from largest to smallest
            orangeSorted = sorted(orange, key=lambda x: cv2.contourArea(x), reverse=True)
            
            notes = []
            note_contours = []
            for y in orangeSorted:

                area = cv2.contourArea(y)

                r_x,r_y,r_w,r_h = cv2.boundingRect(y)

                #ar = float(r_w)/r_h  at the far distance we are going for the aspect ratio is so small we won't get much change

                center_x = r_x + int(round(r_w / 2)) + NOTE_X_OFFSET
                center_y = r_y + int(round(r_h / 2)) + NOTE_Y_OFFSET
                #distance = note_regress_distance(center_y) # get distance (inches) using y location
                #px_per_deg = note_regress_px_per_deg(distance) # get pixel per degree
                #angle = (1 / px_per_deg) * (center_x - w/2)
                
                #Extent is the ratio of contour area to bounding rectangle area.
                extent = float(area) / (r_w * r_h)

                if db_n == True:
                    if note_record_data_ntt.get() == True:
                        note_data = f'{len(orangeSorted)},{area:4.1f},{center_x},{center_y},{ar:2.1f},{extent:2.1f},{distance:3.1f},{angle:2.1f}'
                        with open('note_data.txt', 'a') as f:
                            f.write(note_data)
                            f.write('\n')
                        note_record_data_ntt.set(False)
                    #if time_check == True and area > 222 and ar > 0.6 and ar < 2.3:
                    
                    if time_check == True and area > 4:
                        #print(f'num={len(orangeSorted)} ar={area:4.1f} note_x={center_x} note_y={center_y} a_r={ar:2.1f} ex={extent:2.1f} d={distance:3.1f} {angle:2.1f} deg')
                        print(f'ar={area:4.1f} ex={extent:1.2f} note_x={center_x} note_y={center_y} ppd={px_per_deg:1.1f} d={distance:3.1f} {angle:2.1f} deg')
                        #if area > 4:
                        #    print(f'num={len(orangeSorted)} ar={area:4.1f} note_x={center_x} note_y={center_y} ex={extent:2.1f}')
                        #print(f'num={len(orangeSorted)} ar={area:4.1f} note_x={center_x} note_y={center_y} a_r={ar:2.1f} ex={extent:2.1f}')
                    
                #cv2.drawContours(img, y, -1, (0,255,0), 2)

                #(ar > 0.65 and ar < 4)                      and
                if (area > 25 and area < 54000 ) and \
                    (center_y > 17  and center_y < 240*2):

                        if (center_y > 240*2): # at really close, can't see the bottom, aspect ratio goes way up 
                            extent_min = 0.25
                        else:
                            extent_min = 0.25

                        #extent goes way down when we get real close
                        if (extent > extent_min and extent < 1.0): 
                
                            # found a note
                            notes.append((center_x,center_y))
                            note_contours.append(y)
                            
                            if len(notes) > 0:

                                closest_idx = -1
                                x_diff_min = 999
                                y_diff_min = 999

                                # find the note with smallest diff between: center x and center (320) and center y and max y (480)
                                for note in range(len(notes)):

                                    note_x = notes[note][0]
                                    note_y = notes[note][1]
                                    if abs(note_x - 320) < x_diff_min and abs(note_y - 480) < y_diff_min:
                                        x_diff_min = abs(note_x - 320)
                                        y_diff_min = abs(note_y - 480)
                                        closest_idx = note
                                        
                                if closest_idx != -1:
                                    if notes[closest_idx][1] >= 390: # don't see a full note this close, so y value for this distance is a bit off so force it to 0
                                        distance = 0
                                    else:
                                        distance = note_regress_distance(notes[closest_idx][1]) # get distance (inches) using y location
                                    px_per_deg = note_regress_px_per_deg(distance) # get pixel per degree
                                    angle = (1 / px_per_deg) * (notes[closest_idx][0] - w/2)
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
                                        note_pose_data_bytes_ntt.set(pose_data)
                                        NetworkTableInstance.getDefault().flush()

                                        #print(f'ar={area:4.1f} ex={extent:1.2f} note_x={center_x} note_y={center_y} ppd={px_per_deg:1.1f} d={distance:3.1f} {angle:2.1f} deg')

                                        if db_n == True:
                                            txt = piece_pose_data_string(image_num, rio_time, image_time, distance, angle)
                                            note_pose_data_string_header_ntt.set(txt)
                                            note_distance_ntt.set(round(distance,2))
                                            note_angle_ntt.set(round(angle,2))                            
                                            cv2.circle(img, (notes[closest_idx][0], notes[closest_idx][1]), 4, (0,255,0), -1)
                                            cv2.drawContours(img, [note_contours[closest_idx]], 0, (0,255,0), 1)
                                            outputStreamNote.putFrame(img) # send to dashboard
                                            outputMask.putFrame(img_mask) # send to dashboard
                                            continue
                    
            if db_n == True:
                outputStreamNote.putFrame(img) # send to dashboard
                outputMask.putFrame(img_mask) # send to dashboard
                if savefile_ntt.get() == True:
                    file_write_notes(noteconfigfile_ntt.get(), \
                        note_min_h_ntt.get(), \
                        note_min_s_ntt.get(), \
                        note_min_v_ntt.get(), \
                        note_max_h_ntt.get(), \
                        note_max_s_ntt.get(), \
                        note_max_v_ntt.get(), \
                        note_min_area_ntt.get())
                    savefile_ntt.set(False)

        else:
            continue

main()