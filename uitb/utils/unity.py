import zmq
import subprocess
import socket
import cv2
import numpy as np
import os


class UnityClient:
  """ This class defines an object that 1) opens up a Unity standalone application, and 2) communicates with the
  application through ZMQ. """

  def __init__(self, unity_executable, port=None, standalone=True):

    # If a port number hasn't been given, grab one randomly
    if port is None:
      port = self._find_free_port()

    # Open Unity application
    self._app = None
    if standalone:
      # Hack for headless environments
      env_with_display = os.environ.copy()
      if "DISPLAY" not in env_with_display:
        env_with_display["DISPLAY"] = ":0"
      # Open the app
      self._app = subprocess.Popen([unity_executable, '-port', f'{port}'], env=env_with_display)

    # Create zmq client
    self._context = zmq.Context()
    self._client = self._context.socket(zmq.REQ)
    self._client.connect(f"tcp://localhost:{port}")

  def __del__(self):
    if self._client:
      self._client.send_json({"quitApplication": True})
      self._client.close()
    if self._context:
      self._context.destroy()

  def _receive(self):
    msg = self._client.recv_json()
    image = np.flip(cv2.imdecode(np.asarray(msg["image"], dtype=np.uint8), -1), 2)
    return image, msg["reward"], msg["isFinished"]

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
    print("Attempting to connect to Unity app")
    self._client.send_json(time_options)
    msg = self._client.recv_json()
    print("Connection confirmed")

  def _find_free_port(self):
    # Not perfect but works for our purpose. Susceptible to race conditions.
    with socket.socket() as s:
      s.bind(('', 0))
      return s.getsockname()[1]
