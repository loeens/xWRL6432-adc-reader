"""
Module for acquiring and ADC data from an IWRL6432 radar sensor via DCA100
without the need for mmWave Studio.

This module defines the `XWRL6432AdcReader` class, which encapsulates the
functionality required to interface with a Texas Instruments IWRL6432 radar
evaluation module (EVM) and a DCA1000EVM data capture card.

The primary class, `XWRL6432AdcReader`, handles:
- Parsing radar configuration files (.cfg).
- Initializing and configuring the radar and DCA1000EVM hardware.
- Managing data acquisition in a separate thread.
- Performing data processing, specifically RDIF (Radar Data Interface Format) 
  unswizzling to reconstruct ADC samples.
- Outputting processed data frames via a queue for consumption by other 
  applications.
  

This module relies on helper utilities:
- `utils.ADC.DCA1000` for DCA1000EVM control, which is taken from the OpenRadar project 
  https://github.com/PreSenseRadar/OpenRadar (/mmwave/dataloader/ADC.py) and was adapted
  for the needs of this project and the xWRL6432
- `utils.radar_cli.RadarCLI` for communication with the radar EVM via mmWave CLI.

Typical Usage:
    from queue import Queue
    from XWRL6432AdcReader import XWRL6432AdcReader

    data_queue = Queue()
    adc_reader = XWRL6432AdcReader(
        radar_serial_port="/dev/ttyACM1",
        radar_cfg_path="path/to/your/iwrl6432.cfg",
        out_queue=data_queue
    )
    
    try:
        adc_reader.start_acquisition()
        # ... Application logic to consume data from data_queue ...
        # e.g., while True: frame = data_queue.get() ...
    except KeyboardInterrupt:
        print("Stopping acquisition...")
    finally:
        adc_reader.stop_acquisition()
        adc_reader.close()
"""
import threading
from queue import Queue
from pathlib import Path
import numpy as np
import time

from .utils.adc import DCA1000
from .utils.radar_cli import RadarCLI

class XWRL6432AdcReader(threading.Thread):
    """
    A class to manage data acquisition from an xWRL6432 radar sensor
    using a DCA1000EVM data capture card, removing the need for mmWave
    studio.

    This class handles radar configuration, data capture, raw data processing
    (including RDIF unswizzling), and makes the processed ADC data available
    through an output queue.

    Attributes:
        radar_serial_port (str): The serial port for radar communication.
        radar_cfg_path (Path):  Path to the radar configuration file.
        out_queue (Queue):      Queue to output processed ADC data.
        cli (RadarCLI | None):  Instance for radar command-line interface communication.
        dca (DCA1000 | None):   Instance for DCA1000 EVM control.
        num_chirps_per_frame (int): Total number of chirps in a frame.
        num_tx_ant (int):       Number of active transmitter antennas.
        num_rx_ant (int):       Number of active receiver antennas.
        num_adc_samples (int):  Number of ADC samples per chirp.
        num_chirp_loops (int):  Number of chirp loops (frames / Tx antennas).
    """
    def __init__(self, radar_serial_port: str, radar_cfg_path: str, out_queue: Queue):
        """
        Initializes the XWRL6432AdcReader.

        Args:
            radar_serial_port (str): The serial port name for communicating with the radar EVM
                                     (e.g., "COM3" or "/dev/ttyUSB0").
            radar_cfg_path (str):   The file path to the radar configuration (.cfg) file.
            out_queue (Queue):      A Queue instance where processed ADC data frames
                               (as NumPy arrays) will be placed.

        Raises:
            FileNotFoundError: If the radar configuration file does not exist.
            ValueError: If the supplied config file is not a .cfg file or if
                        parsed num_tx_ant is zero or negative.
        """
        super().__init__(daemon=True)

        self.radar_serial_port = radar_serial_port
        self.radar_cfg_path = Path(radar_cfg_path)
        self.out_queue = out_queue

        self._running = False # Don't start running immediately

        self.cli = None
        self.dca = None
        
        # Ensure supplied radar_cfg_path exists
        if not self.radar_cfg_path.is_file():
            raise FileNotFoundError(f"Config file not found: {self.radar_cfg_path}")
        if self.radar_cfg_path.suffix != '.cfg': 
            raise ValueError("Supplied config file must be of type/format .cfg")

        # Parse radar config into values
        try:
            self.num_chirps_per_frame, self.num_tx_ant, self.num_rx_ant, self.num_adc_samples, self.frame_period = self._parse_radar_config(self.radar_cfg_path)
            
            if self.num_tx_ant <= 0:
                raise ValueError("Parsed num_tx_ant is zero or negative.")
            self.num_chirp_loops = self.num_chirps_per_frame // self.num_tx_ant
            
            num_chirps_per_frame, num_tx_ant, num_rx_ant, num_adc_samples = self.num_chirps_per_frame, self.num_tx_ant, self.num_rx_ant, self.num_adc_samples
            print(f"num_chirps_per_frame: {num_chirps_per_frame}, num_tx_ant: {num_tx_ant}, \
            \num_rx_ant: {num_rx_ant}, num_adc_samples: {num_adc_samples}")
        except Exception as e:
            print(f"Failed to parse config: {e}")
            raise
        
    def run(self):
        """
        Main execution method for the thread.

        This method continuously reads raw data from the DCA1000,
        interprets it, and places the processed ADC data into the output queue.
        The loop continues as long as the internal `_running` flag is True and
        no errors occur.
        """
        if not self._running or self.dca is None or self.cli is None:
            print("ADC Reader: Run condition not met (not running or DCA/Radar CLI not initialized). Exiting thread.")
            return

        print("ADC Reader thread: Starting data acquisition loop.")
        last_frame_time = time.perf_counter()
        while self._running:
            try:
                raw_frame = self.dca.read()

                # Optional: Calculate and output duration since last call
                current_time = time.perf_counter()
                time_interval_sec = current_time - last_frame_time
                time_interval_ms = time_interval_sec * 1000
                last_frame_time = current_time
                print(f"Time since last frame start: {time_interval_ms:.2f} ms")
                
                if raw_frame is not None:
                    adc_data = self._interpret_raw_data(raw_frame)
                    self.out_queue.put(adc_data)
                elif self._running:
                    raise RuntimeError("No data from DCA read while reader was active")
                else: 
                    break 
            except Exception as e: 
                if self._running: 
                    print(f"ADC Reader thread: Error during data read/process: {e}")
                break
        print("ADC Reader thread: Data acquisition loop finished.")

    def start_acquisition(self):
        """
        Initializes hardware if necessary, starts the data stream from DCA1000,
        sends start command to the radar, and starts this thread's execution.
        """
        try:
            if self.cli is None or self.dca is None:
                try:
                    self._init_hardware()
                except Exception as e:
                    raise SystemExit(f"Failed to initialize hardware.") from e
            
            if self.cli is None or self.dca is None:
                print("Cannot start acquisition: Hardware initialization failed.")
                return

            self.dca.start_stream()
            self.cli.send_start_cmd()
            self._running = True
            super().start()
        except Exception as e:
            print(f"ADC Reader: Failed to start acquisition: {e}")
    
    def stop_acquisition(self, wait_for_thread=True, timeout=2.0):
        """
        Stops the data acquisition process.

        Sets the internal running flag to False, sends stop commands to the radar
        and DCA1000, and optionally waits for the acquisition thread to terminate.

        Args:
            wait_for_thread (bool): If True, waits for the acquisition thread to
                                    join (finish its current operations).
            timeout (float): Maximum time in seconds to wait for the thread if
                             `wait_for_thread` is True.
        """
        print("ADC Reader: Stopping acquisition...")
        self._running = False

        if self.cli:
            try:
                self.cli.send_stop_cmd()
            except Exception as e:
                print(f"ADC Reader: Error sending CLI stop command: {e}")
        if self.dca:
            try:
                self.dca.stop_stream()
            except Exception as e:
                print(f"ADC Reader: Error stopping DCA stream: {e}")

        if wait_for_thread and self.is_alive():
            print(f"ADC Reader: Waiting for thread to finish (timeout: {timeout}s)...")
            self.join(timeout) 
            if self.is_alive():
                print("ADC Reader: Thread did not terminate in time.")
        print("ADC Reader: Acquisition stopped.")

    def get_radar_config(self) -> dict:
        """
        Returns the essential radar configuration parameters as a dictionary.

        Returns:
            A dictionary containing key radar configuration parameters.
        """
        config_data = {
            "num_chirps_per_frame": self.num_chirps_per_frame,
            "num_tx_ant": self.num_tx_ant,
            "num_rx_ant": self.num_rx_ant,
            "num_adc_samples": self.num_adc_samples,
            "num_chirp_loops": self.num_chirp_loops,
        }
        return config_data

    
    def close(self):
        """
        Closes connections to the radar CLI and DCA1000.
        This should be called to release hardware resources.
        """
        if self.cli:
            self.cli.close()
        if self.dca:
            self.dca.close()

    def _init_hardware(self):
        """
        Initializes and configures the radar EVM via RadarCLI (mmWave CLI) 
        and the DCA1000EVM.

        Raises:
            Exception: If radar or DCA1000 configuration fails.
        """
        if self.cli is not None and self.dca is not None:
            print("Hardware already initialized.")
            return

        try:
            self.cli = RadarCLI(self.radar_serial_port)
            self.cli.send_config(self.radar_cfg_path)
            print("Radar EVM initialized and configured.")
        except Exception as e:
            print(f"Failed to configure radar: {e}")
            self.cli = None
            raise
        
        try:
            self.dca = DCA1000(
                self.num_chirp_loops, 
                self.num_rx_ant, 
                self.num_tx_ant, 
                self.num_adc_samples, 
                self.frame_period
            )
            self.dca.configure()
            print("DCA1000 initialized and configured.")
        except Exception as e:
            print(f"Failed to configure DCA1000: {e}")
            self.dca = None
            raise
    
    def _interpret_raw_data(self, raw_data):
        """
        Processes raw ADC data from the DCA1000.

        This involves unswizzling the RDIF data, reshaping it into the
        desired format (chirp_loops x channel x adc_samples), and
        converting 12-bit unsigned ADC values to signed values.

        Args:
            raw_data (bytes or np.ndarray): Raw data (uint16) from the DCA1000.

        Returns:
            np.ndarray: Processed ADC data as a NumPy array of dtype float32,
                        with shape (num_chirp_loops, num_tx_ant * num_rx_ant, num_adc_samples).
        """
        # Perform RDIF unswizzling
        data = self._unswizzle_rdif_data(raw_data)

        # Reshape the data into the structure it comes in
        data = np.reshape(data, [self.num_chirp_loops, self.num_tx_ant, self.num_adc_samples, self.num_rx_ant])
        # Reformat into chirp_loops x adc_samples x tx_channel x rx_channel
        data = np.transpose(data, [0, 2, 1, 3])
        # Reformat into chirp_loops x adc_samples x channel
        # Channel then is organized as: [0]tx0->rx0 | [1]tx0->rx1 | [2]tx0->rx2 | [3]tx1->rx0 | [4]tx1->rx1 | [5]tx1->rx2 
        data = np.reshape(data, [self.num_chirp_loops, self.num_adc_samples, (self.num_tx_ant*self.num_rx_ant)])
        # Bring it into format chirp_loops x channel x adc_samples so it is compatible with OpenRadar lib
        data = data.swapaxes(1, 2)

        # Convert the 12-bit unsigned values [0, 4095] to signed [-2048, 2047]
        data = data.astype(np.float32)
        max_signed_val = 2**(12 - 1) - 1 # 2047
        # Subtract 2^12 (4096) from values which exceed the max positive to wrap them to negative
        data[data > max_signed_val] -= 2**12

        return data
    
    @staticmethod
    def _parse_radar_config(config_path: Path) -> tuple[int, int, int, int]:
        """
        Parses a radar configuration file to extract essential parameters.

        Args:
            config_path (Path): Path object pointing to the radar .cfg file.

        Returns:
            tuple: A tuple containing:
                - chirps_per_frame (int): Total number of chirps per frame.
                - num_tx_ant (int): Number of active transmitter antennas.
                - num_rx_ant (int): Number of active receiver antennas.
                - num_adc_samples (int): Number of ADC samples per chirp.
                - frame_period (int): Frame period in ms
        
        Raises:
            ValueError: If essential configuration lines are missing or malformed.
        """
        chirps_per_frame = 0
        num_tx_ant = 0
        num_rx_ant = 0
        num_adc_samples = 0
        num_chirps_per_burst = 0
        num_bursts_per_frame = 0
        frame_period = 0

        with open(config_path, 'r') as file:
            for line in file:
                # Strip trailing spaces
                line = line.strip()

                # Only channelCfg, chirpComnCfg and frameCfg are of interest
                if line.startswith("channelCfg"):
                    parts = line.split()
                    vals = parts[1:]
                    num_rx_ant = bin(int(vals[0])).count('1')
                    num_tx_ant = bin(int(vals[1])).count('1')
                    
                elif line.startswith("chirpComnCfg"):
                    parts = line.split()
                    vals = parts[1:]
                    num_adc_samples = int(vals[3])
                    
                elif line.startswith("frameCfg"):
                    parts = line.split()
                    vals = parts[1:]
                    num_chirps_per_burst = int(vals[0])
                    num_bursts_per_frame = int(vals[3])
                    frame_period         = int(vals[4])
        chirps_per_frame = num_chirps_per_burst * num_bursts_per_frame
        return chirps_per_frame, num_tx_ant, num_rx_ant, num_adc_samples, frame_period

    @staticmethod
    def _unswizzle_rdif_data(raw_data) -> np.ndarray:
        """
        Unpacks RDIF data (un-swizzles) from a 1D array of uint16 values, each ADC
        sample being made up of 12 bits.
        IWRL6432 uses RDIF instead of LVDS, and RDIF data is swizzled.
        Only compatible with RDIF Swizzling Mode 2 (aka. Pin0-bit0-Cycle3 Mode)
        More info here: 
        https://e2e.ti.com/support/sensors-group/sensors/f/sensors-forum/1232378/iwrl6432boost-enabling-data-streaming-via-ldvs-in-software

        Args:
            raw_data (np.ndarray): 1D NumPy array of dtype uint16.

        Returns:
            np.ndarray: 1D NumPy array of dtype uint16 containing the unpacked
                        12-bit samples.
        """
        # Ensure data is a NumPy array of uint16
        raw_data = np.array(raw_data, dtype=np.uint16)

        # Ensure data length is a multiple of 4 (RDIF uses 64 bit blocks)
        num_elements = raw_data.shape[0]
        assert num_elements > 0 and num_elements % 4 == 0, (
            f"raw_data length must be non-zero and a multiple of 4, got {num_elements}"
        )

        # RDIF unpacking works on 64-bit (4 * uint16) blocks, so reshape data into blocks with each 4 columns (4 uint16 values)
        # each row data_chunk is [W0, W1, W2, W3]
        data_reshaped = np.reshape(raw_data, [-1, 4])
        num_chunks = data_reshaped.shape[0]

        # The bits of one sample are scattered across all 4 words (w0, w1, w2, w3). So for each Sample (S0, S1, S2, S3):
        #       MSB                                                                                                LSB
        # S0 = w3_b11 | w2_b11 | w1_b11 | w0_b11 | w3_b10 | w2_b10 | w1_b10 | w0_b10 | w3_b09 | w2_b09 | w1_b09 | w0_b09
        # S1 = w3_b08 | w2_b08 | w1_b08 | w0_b08 | w3_b07 | w2_b07 | w1_b07 | w0_b07 | w3_b06 | w2_b06 | w1_b06 | w0_b06
        # S2 = w3_b05 | w2_b05 | w1_b05 | w0_b05 | w3_b04 | w2_b04 | w1_b04 | w0_b04 | w3_b03 | w2_b03 | w1_b03 | w0_b03
        # S3 = w3_b02 | w2_b02 | w1_b02 | w0_b02 | w3_b01 | w2_b01 | w1_b01 | w0_b01 | w3_b00 | w2_b00 | w1_b00 | w0_b00

        w_arrays = [] # arrays for the 4 words (4 columns of RDIF block over all RDIF blocks)
        s_arrays = [] # arrays for the final samples derived from the 4 columns

        # Extract the four words for all chunks
        for i in range(4):
            w_arrays.append(data_reshaped[:, i])
        
        # Initialize all output sample arrays with zeros
        for i in range(4):
            s_arrays.append(np.zeros(num_chunks, dtype=np.uint16))
    
        # Bit shifts to get the 3-bit groups from input words Wi for S0, S1, S2, S3 respectively
        input_word_group_shifts = [9, 6, 3, 0]

        # Loop for each output sample type (S0, S1, S2, S3)
        for s_idx in range(4):
            current_input_word_bit_shift = input_word_group_shifts[s_idx]

            # Extract the relevant 3-bit groups from ALL W0, W1, W2, W3 for current sample
            bitgroups_for_current_s = [] # e.g. for S0: [ ((W0>>9)&7), ((W1>>9)&7), ((W2>>9)&7), ((W3>>9)&7) ]
            for w_array in w_arrays:
                three_bits_array_from_word = (w_array >> current_input_word_bit_shift) & 0x7
                bitgroups_for_current_s.append(three_bits_array_from_word)

            # Construct output samples
            for idx_of_bit_in_3bit_group in range(3):  # iterate 0 (LSB), 1 (Mid), 2 (MSB) of the 3-bit groups
                for source_word_idx in range(4): # iterate over bits extracted from W0, W1, W2, W3
                    # Determine where this bit goes in the 12-bit output sample
                    bit_pos_in_s = idx_of_bit_in_3bit_group * 4 + source_word_idx

                    # Get the specific bit (0 or 1) from the current 3-bit group
                    source_bit_val_array = (bitgroups_for_current_s[source_word_idx] >> idx_of_bit_in_3bit_group) & 1

                    # Place the bit in the correct position in the current output sample array
                    s_arrays[s_idx] |= (source_bit_val_array << bit_pos_in_s)
        
        # Stack S0, S1, S2, S3 arrays side-by-side (columns) and then flatten row-wise
        unswizzled_output = np.stack(s_arrays, axis=1).flatten()

        return unswizzled_output
    
    

    