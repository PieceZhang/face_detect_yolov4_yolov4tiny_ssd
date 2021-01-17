import cv2

image_origin = cv2.imread('ssd.png')
assert image_origin is not None, 'Image is not found, No such file or directory'
image_data = cv2.resize(image_origin, (300, 300))

cost_time = 136.06
image_data = cv2.putText(image_data, 'Model: SSD_300',
                         (10, 25), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 1)
image_data = cv2.putText(image_data, 'Device: VPU',
                         (10, 45), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 1)
image_data = cv2.putText(image_data, 'Cost: {:2.2f} ms'.format(cost_time),
                         (10, 65), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 1)
image_data = cv2.putText(image_data, 'FPS: {:2.2f}'.format(1000 / cost_time) if cost_time > 0 else 'FPS: --',
                         (10, 85), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 1)
cv2.imwrite('ssd_300-3.jpg', image_data)
