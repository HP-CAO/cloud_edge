import numpy as np
import math
from quanser.hardware import HIL, Clock, HILError

card = HIL("q2_usb", "0")
analog_channels = np.array([0], dtype=np.uint32)
encoder_channels = np.array([0, 1], dtype=np.int32)
num_analog_channels = len(analog_channels)
num_encoder_channels = len(encoder_channels)
frequency = 20.00  # increase periods
samples = np.iinfo(np.int32).max
samples_in_buffer = int(frequency)
samples_to_read = 1
samples_to_write = 1
analog_buffer = np.zeros(num_analog_channels, dtype=np.float64)
analog_write_buffer = np.zeros(num_analog_channels, dtype=np.float64)
encoder_buffer = np.zeros(num_encoder_channels, dtype=np.int32)

analog_task = card.task_create_analog_reader(samples_in_buffer, analog_channels, num_analog_channels)
encoder_task = card.task_create_encoder_reader(samples_in_buffer, encoder_channels, num_encoder_channels)
# analog_task_control = card.task_create_analog_writer(samples_in_buffer, analog_channels, num_analog_channels)

try:
    card.task_start(analog_task, Clock.HARDWARE_CLOCK_0, frequency, samples)
    card.task_start(encoder_task, Clock.HARDWARE_CLOCK_0, frequency, samples)
    # card.task_start(analog_task_control, Clock.HARDWARE_CLOCK_0, frequency, samples)
    analog_write_buffer += 2
    i = 0

    while True:
        card.task_read_analog(analog_task, samples_to_read, analog_buffer)
        card.task_read_encoder(encoder_task, samples_to_read, encoder_buffer)
        # card.task_write_analog(analog_task_control, samples_to_write, analog_write_buffer)
        # print('Analog_write_buffer', analog_write_buffer)
        # card.write_analog(analog_channels, num_analog_channels, analog_write_buffer)
        print("Encoder: ", encoder_buffer)
        print("Analog ", analog_buffer)

        if i % 2 == 0:
            analog_write_buffer = -1 * analog_write_buffer
        i += 1

except HILError:
    print("HILError--")
    card.task_stop_all()
    card.task_delete_all()

card.close()

