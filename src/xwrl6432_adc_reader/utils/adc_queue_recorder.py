
import time
import threading
from queue import Queue
from queue import Empty as QueueEmpty
import numpy as np
from pathlib import Path

class ADCRecorder(threading.Thread):
    """
    A thread-based class to record a specified number of data frames
    from an input queue

    It consumes ADC frames from the queue, stores them, and stops once the 
    target number of frames is reached or when intentionally stopped
    """
    def __init__(self, input_queue: Queue, num_frames: int):
        """
        Initializes the ADCRecorder

        Args:
            input_queue: The queue from which frames are read. 
            num_frames (int): The total number of frames to record.

        Raises:
            ValueError: If num_frames is negative.
        """
        super().__init__(daemon=True)

        if not isinstance(num_frames, int):
            raise ValueError("num_frames must be an integer")

        self.input_queue = input_queue
        self.num_frames_to_record = num_frames
        
        self.recorded_frames = []
        self._num_recorded_frames = 0
        self._running = False
        
        self.rec_complete_event = threading.Event()

    def run(self):
        """
        Continuously read frames from the input_queue until the set number
        of frames is recorded or the recording is stopped via _running flag
        """
        try:
            while self._running and self._num_recorded_frames < self.num_frames_to_record:
                try:
                    # Read frame from input_queue and add to recorded_frames[]
                    frame = self.input_queue.get(timeout = 5.0)
                    self.recorded_frames.append(frame)
                    self._num_recorded_frames += 1
                except QueueEmpty:
                    print("Input Queue was empty for 5 seconds.")
                    raise
            
            if self._num_recorded_frames == self.num_frames_to_record:
                print(f"Successfully recorded all {self.num_frames_to_record} targeted frames.")
            elif not self._running:
                print(f"Recording stopped externally. Recorded {self._num_recorded_frames} of {self.num_frames_to_record} targeted frames.")

        except Exception as e:
            print(f"Error during recording: {e}")
        finally:
            self._running = False
            self.rec_complete_event.set()

    def start_recording(self) -> bool:
        """
        Starts the recording process

        Returns:
            success (bool): True if the recording thread was started successfully
        """
        if self._running:
            return False
        
        # Init vars
        self.recorded_frames = []
        self._num_recorded_frames = 0
        self.rec_complete_event.clear()
        self._running = True
        
        try:
            # Start run() loop
            super().start()
            return True
        except Exception as e:
            self._running = False
            self.rec_complete_event.set()
            print(f"Error during start of recording thread: {e}")
            return False

    def stop_recording(self, timeout: float = 2.0):
        """
        Signals the recording thread to stop and and join

        Args:
            timeout (float):  Maximum number of seconds to wait for the thread to join
        """
        self._running = False

        # Join threads
        if self.is_alive():
            self.join(timeout) 
            if self.is_alive():
                print("Recording thread did not terminate within timeout!")
            else:
                print("Recording thread joined successfully")

    def get_recorded_frames(self) -> list:
        """
        Returns the list of frames that have been recorded

        Returns:
            list: containing the recorded frames
        """
        return self.recorded_frames

    def save_to_npz(self, file_path: str | Path, config_metadata: dict = None) -> bool:
        """
        Saves the recorded ADC frames and optional configuration to .npz file

        Args:
            file_path (str | Path): The path of the .npz file
            config_metadata (dict, optional): A dictionary containing metadata to be saved

        Returns:
            bool: True if saving was successful, False otherwise.
        """
        if not self.recorded_frames:
            print("Error: No frames recorded to save")
            return False
        
        if self._running and not self.rec_complete_event.is_set():
             print("Warning: Saving data while recording is in progress!")

        try:
            file_path_str = str(file_path)

            # Stack frames into a 1D array
            frames_array = np.array(self.recorded_frames)

            data_to_save = {
                'adc_data': frames_array,
                'num_frames_recorded_actual': np.array(self._num_recorded_frames),
                'num_frames_target_config': np.array(self.num_frames_to_record)
            }

            if config_metadata is not None:
                # Store the dictionary as a 1D array of type object
                # => allows retrieving it as a dictionary using .item()
                data_to_save['config_metadata'] = np.array(config_metadata, dtype=object)
            
            np.savez_compressed(file_path_str, **data_to_save)
            print(f"Successfully saved {self._num_recorded_frames} frames to {file_path_str}")
            return True
        except Exception as e:
            print(f"Error saving data to NPZ file '{file_path}': {e}")
            return False

    def get_num_recorded_frames(self) -> int:
        """
        Returns the number of frames that have been successfully recorded
        """
        return self._num_recorded_frames

    def is_active(self) -> bool:
        """
        Checks if the recording thread is currently alive and its _running flag is set
        """
        return self.is_alive() and self._running

    def wait_for_completion(self, timeout: float = None) -> bool:
        """
        Blocks the calling thread until the recording task completes

        Args:
            timeout (float, optional): Maximum time in seconds to wait for the thread to complete
        Returns:
            bool: True if the recording task completed within the timeout,
                  False if not
        """
        # Check if the thread has already completed
        if not self.is_alive() and self.rec_complete_event.is_set():
            return True
        # Else, wait until completion
        return self.rec_complete_event.wait(timeout)
