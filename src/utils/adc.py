# Copyright 2019 The OpenRadar Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# ------------------------------------------------------------------------------
# Modified by Leon Braungardt on 2025-05-09:
#  - Converted static ADC constants (ADC_PARAMS) to dynamic constructor parameters
#  - Converted ADC_PARAMS "IQ" param (int) to "cmplx_valued" (bool)
#  - Moved calculation of UDP packet / frame variables which were based on ADC 
#       constants (ADC_PARAMS) to read() function
#  - Replaced parameters of CONFIG_FPGA_GEN to match xWRL6432 and in order to
#       replace mmWave studio
#  - Add comment to CONFIG_PACKET_DATA
#  - Add reset() function for resetting the FPGA
#  - Add start_stream() function for starting ADC data capture
#  - Rename _stop_stream() to stop_stream() and add comment
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Modified by Leon Braungardt on 2025-05-10:
#  - Increase default data socket buffer
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Modified by Leon Braungardt on 2025-05-13:
#  - Changed self.frame_buff from list to dict
#  - Added self.uint16_in_frame to __init__()
#  - Completely refactor (replace) read() function in order to fix issues 
#       concerning data corruption and not all frames being captured
#  - Added _place_data_packet_in_frame_buffer() function
#  - Added _delete_incomplete_frames() function
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Modified by Leon Braungardt on 2025-05-17:
#  - Added frame_period to DCA1000 class and adapt the timeout after which frames
#       get deleted to it
# ------------------------------------------------------------------------------

import codecs
import socket
import struct
from enum import Enum

import numpy as np
import time


class CMD(Enum):
    RESET_FPGA_CMD_CODE = '0100'
    RESET_AR_DEV_CMD_CODE = '0200'
    CONFIG_FPGA_GEN_CMD_CODE = '0300'
    CONFIG_EEPROM_CMD_CODE = '0400'
    RECORD_START_CMD_CODE = '0500'
    RECORD_STOP_CMD_CODE = '0600'
    PLAYBACK_START_CMD_CODE = '0700'
    PLAYBACK_STOP_CMD_CODE = '0800'
    SYSTEM_CONNECT_CMD_CODE = '0900'
    SYSTEM_ERROR_CMD_CODE = '0a00'
    CONFIG_PACKET_DATA_CMD_CODE = '0b00'
    CONFIG_DATA_MODE_AR_DEV_CMD_CODE = '0c00'
    INIT_FPGA_PLAYBACK_CMD_CODE = '0d00'
    READ_FPGA_VERSION_CMD_CODE = '0e00'

    def __str__(self):
        return str(self.value)


# MESSAGE = codecs.decode(b'5aa509000000aaee', 'hex')
CONFIG_HEADER = '5aa5'
CONFIG_STATUS = '0000'
CONFIG_FOOTER = 'aaee'

# STATIC
MAX_PACKET_SIZE = 4096
BYTES_IN_PACKET = 1456

class DCA1000:
    """Software interface to the DCA1000 EVM board via ethernet.

    Attributes:
        static_ip (str): IP to receive data from the FPGA
        adc_ip (str): IP to send configuration commands to the FPGA
        data_port (int): Port that the FPGA is using to send data
        config_port (int): Port that the FPGA is using to read configuration commands from


    General steps are as follows:
        1. Power cycle DCA1000 and XWR1xxx sensor
        2. Open mmWaveStudio and setup normally until tab SensorConfig or use lua script
        3. Make sure to connect mmWaveStudio to the board via ethernet
        4. Start streaming data
        5. Read in frames using class

    Examples:
        >>> dca = DCA1000()
        >>> adc_data = dca.read(timeout=.1)
        >>> frame = dca.organize(adc_data, 128, 4, 256)

    """

    def __init__(self,
                 num_chirp_loops: int,
                 num_rx_ant: int,
                 num_tx_ant: int,
                 num_adc_samples: int,
                 frame_period,
                 num_bytes_per_sample: int = 2,
                 cmplx_valued: bool = False,
                 static_ip='192.168.33.30', adc_ip='192.168.33.180',
                 data_port=4098, config_port=4096):
        # Save network data
        # self.static_ip = static_ip
        # self.adc_ip = adc_ip
        # self.data_port = data_port
        # self.config_port = config_port

        self.frame_period_seconds = frame_period / 1000

        # Calculate bytes per frame
        self.bytes_in_frame = (num_chirp_loops * num_rx_ant * num_tx_ant * (2 if cmplx_valued else 1) *
                            num_adc_samples * num_bytes_per_sample)
        self.uint16_in_frame = self.bytes_in_frame // 2

        # Create configuration and data destinations
        self.cfg_dest = (adc_ip, config_port)
        self.cfg_recv = (static_ip, config_port)
        self.data_recv = (static_ip, data_port)

        # Create sockets
        self.config_socket = socket.socket(socket.AF_INET,
                                           socket.SOCK_DGRAM,
                                           socket.IPPROTO_UDP)
        self.data_socket = socket.socket(socket.AF_INET,
                                         socket.SOCK_DGRAM,
                                         socket.IPPROTO_UDP)

        # Bind data socket to fpga
        self.data_socket.bind(self.data_recv)

        # Modified by Leon Braungardt on 2025-05-10: Increase default socket buffer
        self.data_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 512 * 1024)

        # Bind config socket to fpga
        self.config_socket.bind(self.cfg_recv)

        self.data = []
        self.packet_count = []
        self.byte_count = []

        self.frame_buff = {}

        self.curr_buff = None
        self.last_frame = None

        self.lost_packets = None

    def configure(self):
        """Initializes and connects to the FPGA

        Returns:
            None

        """
        # SYSTEM_CONNECT_CMD_CODE
        # 5a a5 09 00 00 00 aa ee
        print(self._send_command(CMD.SYSTEM_CONNECT_CMD_CODE))

        # READ_FPGA_VERSION_CMD_CODE
        # 5a a5 0e 00 00 00 aa ee
        print(self._send_command(CMD.READ_FPGA_VERSION_CMD_CODE))

        # CONFIG_FPGA_GEN_CMD_CODE
        # Set FPGA config, equivalent to captureCardModeCfg(1, 1, 1, 2, 1, 25)
        # Parameters can be found on page 36 in:
        # ti/mmwave_studio_04_01_00_06/mmWaveStudio/ReferenceCode/DCA1000/Docs/
        #       TI_DCA1000EVM_CLI_Software_DeveloperGuide.pdf
        #   0x01: Logging Mode:         raw
        #   0x01: Num LVDS lanes:       4
        #   0x01: Data Transfer Mode:   LVDS
        #   0x02: Data Capture Mode:    Ethernet
        #   0x01: Data Format Mode:     12-Bit
        #   0x19: Packet delay in us:   25
        # 5a a5 03 00 06 00 01 01 01 02 01 19 aa ee
        print(self._send_command(CMD.CONFIG_FPGA_GEN_CMD_CODE, '0600', '010101020119'))

        # CONFIG_PACKET_DATA_CMD_CODE
        # set UDP params (CONFIG_PACKET_DATA_CMD_CODE), seems to be LE
        # source: https://e2e.ti.com/support/sensors-group/sensors/f/sensors-forum/702269/dca1000evm-expected-packet-loss-rate
        #   0x05c0: Packet Size (UDP payload):  1472
        #   0x0c35: Packet Delay:               3125 (API value for 25us)
        # 5a a5 0b 00 06 00 c0 05 35 0c 00 00 aa ee
        print(self._send_command(CMD.CONFIG_PACKET_DATA_CMD_CODE, '0600', 'c005350c0000'))
    
    def reset(self):
        """Resets the FPGA

        Returns:
            None
        """
        # RESET_FPGA_CMD_CODE
        # Page 42 in ti/mmwave_studio_04_01_00_06/mmWaveStudio/ReferenceCode/DCA1000/Docs/
        #       TI_DCA1000EVM_CLI_Software_DeveloperGuide.pdf
        # 5a a5 01 00 00 00 aa ee
        return self._send_command(CMD.RESET_FPGA_CMD_CODE)

    def start_stream(self):
        """Helper function to send the start command to the FPGA

        Returns:
            None
        """
        # RECORD_START_CMD_CODE
        # Page 47 in ti/mmwave_studio_04_01_00_06/mmWaveStudio/ReferenceCode/DCA1000/Docs/
        #       TI_DCA1000EVM_CLI_Software_DeveloperGuide.pdf
        # 5a a5 05 00 00 00 aa ee
        return self._send_command(CMD.RECORD_START_CMD_CODE)

    def stop_stream(self):
        """Helper function to send the stop command to the FPGA

        Returns:
            None
        """
        # RECORD_STOP_CMD_CODE
        # Page 53 in ti/mmwave_studio_04_01_00_06/mmWaveStudio/ReferenceCode/DCA1000/Docs/
        #       TI_DCA1000EVM_CLI_Software_DeveloperGuide.pdf
        # 5a a5 06 00 00 00 aa ee
        return self._send_command(CMD.RECORD_STOP_CMD_CODE)

    def close(self):
        """Closes the sockets that are used for receiving and sending data

        Returns:
            None

        """
        self.data_socket.close()
        self.config_socket.close()

    def read(self, timeout=1):
        """ Read in a single frame via UDP

        Args:
            timeout (float): Time to wait for packet before moving on

        Returns:
            Full frame as array if successful, else None

        """
        # Configure
        self.data_socket.settimeout(timeout)

        timeout_incomplete_frames = self.frame_period_seconds * 2

        # Read packets until a full frame is read
        while True: 
            # Read UDP packet
            packet_num, byte_count, packet_data = self._read_data_packet()

            # Place data from UDP packet in frame buffer
            frame_num, frame_data = self._place_data_packet_in_frame_buffer(
                byte_count=byte_count, 
                payload=packet_data
            )

            if frame_data is not None:
                # Remove incomplete frames from frame buffer which exceed a timeout
                dropped_frames = self._delete_incomplete_frames(timeout_seconds=timeout_incomplete_frames)
                if dropped_frames:
                    ids = ", ".join(str(f) for f in dropped_frames)
                    print(f"WARNING: Dropped Frame(s) {ids} since they weren't complete.")
                # Return the complete frame
                return frame_data


    def _send_command(self, cmd, length='0000', body='', timeout=1):
        """Helper function to send a single commmand to the FPGA

        Args:
            cmd (CMD): Command code to send to the FPGA
            length (str): Length of the body of the command (if any)
            body (str): Body information of the command
            timeout (int): Time in seconds to wait for socket data until timeout

        Returns:
            str: Response message

        """
        # Create timeout exception
        self.config_socket.settimeout(timeout)

        # Create and send message
        resp = ''
        msg = codecs.decode(''.join((CONFIG_HEADER, str(cmd), length, body, CONFIG_FOOTER)), 'hex')
        try:
            self.config_socket.sendto(msg, self.cfg_dest)
            resp, addr = self.config_socket.recvfrom(MAX_PACKET_SIZE)
        except socket.timeout as e:
            print(e)
        return resp

    def _read_data_packet(self):
        """Helper function to read in a single ADC packet via UDP

        Returns:
            int: Current packet number, byte count of data that has already been read, raw ADC data in current packet

        """
        data, addr = self.data_socket.recvfrom(MAX_PACKET_SIZE)
        packet_num = struct.unpack('<1l', data[:4])[0]
        byte_count = struct.unpack('>Q', b'\x00\x00' + data[4:10][::-1])[0]
        packet_data = np.frombuffer(data[10:], dtype=np.uint16)
        return packet_num, byte_count, packet_data
    
    def _place_data_packet_in_frame_buffer(self, byte_count: int, payload: np.ndarray):
        """Helper function to place one UDP packet at the correct position in the frame buffer
        
        Args:
            byte_count (int):        cumulative Bytes before this payload (from DCA1000 header)
            payload (np.ndarray):    uint16 from the UDP packet
        
        Returns:
            (int, np.ndarray): Complete frame as a tuple of (frame_num, frame_data),
                                (None, None) if no frame is complete yet
        """

        offset = byte_count // 2 # Absolute position in UDP packet stream
        idx = 0                  # Read-index of payload
        remaining = payload.size # Number of uint16 to process
        completed = (None, None) # Tuple of (frame_id, frame_data) for complete captured frame

        while remaining > 0:
            # Determine which frame_id this data chunk belongs to
            frame_id = offset // self.uint16_in_frame
            # Determine which packet number this is within the frame
            packet_num_within_frame = offset % self.uint16_in_frame
            n_uint16_to_frame_end = self.uint16_in_frame - packet_num_within_frame

            # Determine the size chunk of the data which is written to buffer
            # (detect if the frame border is within this packet or not)
            chunk_size = min(remaining, n_uint16_to_frame_end)

            # print(f"pkt start off={offset}, take={chunk_size}, frame={frame_id}")

            # Create buffer within frame_buff obj for this frame if neccessary
            buf = self.frame_buff.setdefault(
                frame_id,
                {
                    'data':   np.empty(self.uint16_in_frame, dtype=np.uint16),
                    'filled': np.zeros(self.uint16_in_frame, dtype=bool),
                    'first_seen': time.time()
                }
            )

            # Write chunk to appropriate position in the frame's buffer
            start   = packet_num_within_frame
            end     = packet_num_within_frame + chunk_size
            buf['data'][start:end]   = payload[idx:idx+chunk_size]
            buf['filled'][start:end] = True

            # If all packets for the frame have been read, add it to completed tuple
            # (but do not return yet, as otherwise the rest of the packet data is lost)
            if buf['filled'].all():
                completed = (frame_id, buf['data'].copy())
                del self.frame_buff[frame_id]

            # Persist in helper vars that chunk has been read
            offset    += chunk_size
            idx       += chunk_size
            remaining -= chunk_size
        
        return completed

    def _delete_incomplete_frames(self, timeout_seconds: float=0.2):
        """Helper function to delete incomplete frames from frame buffer which exceed a given timeout

        Args:
            timeout_seconds (float): Time after which incomplete frames are deleted
        
        Returns:
            List[int]: List of frame numbers which were deleted (can be empty)
        """
        now = time.time()
        to_delete = []
        for frame_number, buf in self.frame_buff.items():
            if now - buf['first_seen'] > timeout_seconds:
                to_delete.append(frame_number)
        for frame_number in to_delete:
            del self.frame_buff[frame_number]
        
        return to_delete

    def _listen_for_error(self):
        """Helper function to try and read in for an error message from the FPGA

        Returns:
            None

        """
        self.config_socket.settimeout(None)
        msg = self.config_socket.recvfrom(MAX_PACKET_SIZE)
        if msg == b'5aa50a000300aaee':
            print('stopped:', msg)

    @staticmethod
    def organize(raw_frame, num_chirps, num_rx, num_samples):
        """Reorganizes raw ADC data into a full frame

        Args:
            raw_frame (ndarray): Data to format
            num_chirps: Number of chirps included in the frame
            num_rx: Number of receivers used in the frame
            num_samples: Number of ADC samples included in each chirp

        Returns:
            ndarray: Reformatted frame of raw data of shape (num_chirps, num_rx, num_samples)

        """
        ret = np.zeros(len(raw_frame) // 2, dtype=complex)

        # Separate IQ data
        ret[0::2] = raw_frame[0::4] + 1j * raw_frame[2::4]
        ret[1::2] = raw_frame[1::4] + 1j * raw_frame[3::4]
        return ret.reshape((num_chirps, num_rx, num_samples))
