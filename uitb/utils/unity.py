import zmq
import subprocess
import socket
import cv2
import numpy as np
import os
import shutil
import time


class UnityClient:
  """ This class defines an object that 1) opens up a Unity standalone application, and 2) communicates with the
  application through ZMQ. """

  def __init__(self, unity_executable, port=None, standalone=True, app_args=[]):

    # If a port number hasn't been given, grab one randomly
    if port is None:
      self.port = self._find_free_port()
    else:
      self.port = port

    self._app = None

    # Check if we should open the app (during debugging the app is typically run in Unity Editor)
    if standalone:

      # Hack for headless environments
      env_with_display = os.environ.copy()
      if "DISPLAY" not in env_with_display:
        env_with_display["DISPLAY"] = ":0"

      # Get build folder
      build_folder = os.path.split(unity_executable)[0]

      # Define path for log file; use current time (in microseconds) since epoch, should be high resolution enough
      log_name = f"{int(time.time_ns()/1e3)}"
      log_path = os.path.join(build_folder, "app_logs")

      # Create a "log" folder in the build folder
      os.makedirs(log_path, exist_ok=True)

      # Open the app
      self._app = subprocess.Popen([unity_executable,
                                    '-simulated',
                                    '-port', f'{self.port}',
                                    '-logFile', f'{os.path.join(log_path, log_name)}'] + app_args, env=env_with_display)

    # Create zmq client
    self._context = zmq.Context()
    self._client = self._context.socket(zmq.REQ)
    self._client.connect(f"tcp://localhost:{self.port}")

  def close(self):
    if self._client:
      self._client.send_json({"quitApplication": True})
      self._client.close()
    if self._context:
      self._context.destroy()

  def _receive(self):

    # Receive message
    msg = self._client.recv_json()

    # Convert byte array to image
    image = cv2.imdecode(np.asarray(msg["image"], dtype=np.uint8), -1)

    # Convert BGRA format to RGBA format
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

    return {"image": image, "time": msg["timeFeature"]}, msg["reward"], msg["isFinished"], msg["logDict"]

  def step(self, state, is_finished):
    msg = {**state, "isFinished": is_finished, "reset": False}
    self._client.send_json(msg)
    return self._receive()

  def reset(self, state):
    msg = {**state, "isFinished": False, "reset": True}
    self._client.send_json(msg)
    return self._receive()[0]

  def handshake(self, time_options):
    # Send time options, receive an empty msg
    print(f"Attempting to connect to Unity app on port {self.port}")
    self._client.send_json(time_options)
    msg = self._client.recv_json()
    print("Connection confirmed")

  def _find_free_port(self):
    # Not perfect but works for our purpose. Susceptible to race conditions.
    with socket.socket() as s:
      s.bind(('', 0))
      return s.getsockname()[1]


def images_to_video(recording_folder, action_sample_freq, resolution, evaluate_dir=None):

  # There can be several folders with images, loop through them
  for key in os.listdir(recording_folder):

    maybe_folder = os.path.join(recording_folder, key)

    # Only process folders (there shouldn't be anything else anyways)
    if os.path.isdir(maybe_folder):
      # Save to evaluate_dir if it is given, otherwise save to same folder where images are read from

      # Create the video
      subprocess.call([
        'ffmpeg',
        '-y', '-r', f'{action_sample_freq}', '-f', 'image2', '-s', resolution,
        '-i', f"{os.path.join(maybe_folder, 'image%d.png')}",
        '-vcodec', 'libx264', '-crf', '15', '-pix_fmt', 'yuv420p',
        f"{os.path.join(evaluate_dir if evaluate_dir else recording_folder, f'{key}.mp4')}"])

      # Remove the folder containing images
      shutil.rmtree(maybe_folder)