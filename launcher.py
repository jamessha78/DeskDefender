# Gutted from https://github.com/codedance/Retaliation.git

############################################################################
#
# Copyright 2011 PaperCut Software Int. Pty. Ltd. http://www.papercut.com/
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
# 
############################################################################
#
# RETALIATION - A Jenkins "Extreme Feedback" Contraption
#
#    Lava Lamps are for pussies! Retaliate to a broken build with a barrage 
#    of foam missiles.
#
#  Author:  Chris Dance <chris.dance@papercut.com>
#  Version: 1.0 : 2011-08-15
#
############################################################################

import sys
import platform
import time
import socket
import re
import json
import urllib2
import base64

import usb.core
import usb.util


# Protocol command bytes
DOWN    = 0x01
UP      = 0x02
LEFT    = 0x04
RIGHT   = 0x08
FIRE    = 0x10
STOP    = 0x20


class Launcher(object):
    def __init__(self):
        self.setup_usb()

    def setup_usb(self):
        # Tested only with the Cheeky Dream Thunder
        # and original USB Launcher

        self.DEVICE = usb.core.find(idVendor=0x2123, idProduct=0x1010)

        if self.DEVICE is None:
            self.DEVICE = usb.core.find(idVendor=0x0a81, idProduct=0x0701)
            if self.DEVICE is None:
                raise ValueError('Missile device not found')
            else:
                self.DEVICE_TYPE = "Original"
        else:
            self.DEVICE_TYPE = "Thunder"

        

        # On Linux we need to detach usb HID first
        if "Linux" == platform.system():
            try:
                self.DEVICE.detach_kernel_driver(0)
            except Exception, e:
                pass # already unregistered    

        self.DEVICE.set_configuration()

    def send_cmd(self, cmd):
        if "Thunder" == self.DEVICE_TYPE:
            self.DEVICE.ctrl_transfer(0x21, 0x09, 0, 0, [0x02, cmd, 0x00,0x00,0x00,0x00,0x00,0x00])
        elif "Original" == self.DEVICE_TYPE:
            self.DEVICE.ctrl_transfer(0x21, 0x09, 0x0200, 0, [cmd])

    def led(self, cmd):
        if "Thunder" == self.DEVICE_TYPE:
            self.DEVICE.ctrl_transfer(0x21, 0x09, 0, 0, [0x03, cmd, 0x00,0x00,0x00,0x00,0x00,0x00])
        elif "Original" == self.DEVICE_TYPE:
            print("There is no LED on this device")

    def send_move(self, cmd, duration_ms):
        self.send_cmd(cmd)
        time.sleep(duration_ms / 1000.0)
        self.send_cmd(STOP)

    def run_command(self, command, value):
        command = command.lower()
        if command == "right":
            self.send_move(RIGHT, value)
        elif command == "left":
            self.send_move(LEFT, value)
        elif command == "up":
            self.send_move(UP, value)
        elif command == "down":
            self.send_move(DOWN, value)
        elif command == "zero" or command == "park" or command == "reset":
            # Move to bottom-left
            self.send_move(DOWN, 2000)
            self.send_move(LEFT, 8000)
        elif command == "pause" or command == "sleep":
            time.sleep(value / 1000.0)
        elif command == "led":
            if value == 0:
                self.led(0x00)
            else:
                self.led(0x01)
        elif command == "fire" or command == "shoot":
            if value < 1 or value > 4:
                value = 1
            # Stabilize prior to the shot, then allow for reload time after.
            time.sleep(0.5)
            for i in range(value):
                self.send_cmd(FIRE)
                time.sleep(4.5)
        else:
            print "Error: Unknown command: '%s'" % command

    def run_command_set(self, commands):
        for cmd, value in commands:
            self.run_command(cmd, value)

def main(args):
    if len(args) < 2:
        usage()
        sys.exit(1)
    
    launcher = Launcher()

    # Process any passed commands or command_sets
    command = args[1]
    value = 0
    if len(args) > 2:
        value = int(args[2])

    launcher.run_command(command, value)

if __name__ == '__main__':
    main(sys.argv)
