from connection.client import Client
from detection.gesture_system import GestureSystem
from detection.cnn.cnn_gesture_system import CnnGestureSystem


if __name__ == "__main__":

    # cnn_gesture_system = CnnGestureSystem()
    # cnn_gesture_system.start_detection("../../models/model4.h5")

    # For Matlab
    c = Client("192.168.43.84", 7777)  # 172.22.32.1 192.168.6.177
    c.connect()
    c.send("Hello server!")

    def on_gesture(message):
        global c
        c.send(message)

    '''def on_gesture(message):
        print(message)'''

    gesture_system = GestureSystem()
    gesture_system.start_detection(on_gesture)


