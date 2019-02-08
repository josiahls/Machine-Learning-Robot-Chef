from CameraCapture import MyVideoCapture
from FoodDetectionModel import FoodDetectionModel
from util import *
import cv2


class TrackedObject:
    def __init__(self, binary_mask, classification, name, label_loc, cut_line):
        self.binary_mask = binary_mask
        self.classification = classification
        self.name = name
        self.is_in_scene = True
        self.label_loc = label_loc
        self.cut_line = cut_line
        self.life = 10  # Stay alive for 10 frames


class Tracker(object):
    def __init__(self, video_name='20181209_231536.mp4'):
        root = get_absolute_data_path('videos_in')

        self.name = 'Tracker'
        self.camera = MyVideoCapture(root + video_name)
        root = get_absolute_data_path('videos_out')
        self.out = cv2.VideoWriter(root + f'{video_name[:-4]}.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                                   (int(self.camera.width), int(self.camera.height)))
        # cv2.namedWindow(self.name, cv2.WND_PROP_FULLSCREEN)
        # cv2.setWindowProperty(self.name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        self.model = FoodDetectionModel('food20181208T1618')
        self.tracked_objects = []

    def run(self):
        while True:
            ret, frame = self.camera.get_frame()

            if ret:
                aug_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                lines, detections, label_loc = self.model.predict(aug_frame, show_matplots=False)

                # Set current detected objects to False
                for tracked_obj in self.tracked_objects:
                    tracked_obj.is_in_scene = False

                # Label tracked objects from previous frames
                for i in range(detections['masks'].shape[2]):
                    found = False
                    # Is this mask already in the list of detected objects?
                    for tracked_obj in self.tracked_objects:
                        # Set the reset of the objects to False
                        if np.sum(tracked_obj.binary_mask[np.where(detections['masks'][:, :, i])]) > 0 and \
                                tracked_obj.classification == detections['class_ids'][i]:
                            found = True
                            tracked_obj.label_loc = label_loc[i]
                            tracked_obj.binary_mask = detections['masks'][:, :, i]
                            tracked_obj.is_in_scene = True
                            tracked_obj.life = 10
                            break

                    if not found:
                        self.tracked_objects.append(TrackedObject(detections['masks'][:, :, i],
                                                                  detections['class_ids'][i],
                                                                  f'Object {len(self.tracked_objects)+1}',
                                                                  label_loc[i], lines[i]))
                # Remove objects that failed to track
                # Set current detected objects to False
                for i in range(len(self.tracked_objects)-1, -1, -1):
                    if not self.tracked_objects[i].is_in_scene:
                        self.tracked_objects[i].life -= 1
                        if self.tracked_objects[i].life < 1:
                            self.tracked_objects.pop(i)

                for tracked_obj in self.tracked_objects:
                    cv2.putText(img=frame, text=tracked_obj.name, org=tracked_obj.label_loc,
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255), fontScale=2, thickness=2)
                    cv2.line(frame, tracked_obj.cut_line[0], tracked_obj.cut_line[1], (0, 0, 255), 2)

                # Write the frame into the file 'output.avi'
                self.out.write(frame)

                # # Display the resulting frame
                cv2.imshow(self.name, frame)

                # Press Q on keyboard to stop recording
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            # Break the loop
            else:
                break

    def __del__(self):
        self.out.release()

        # Closes all the frames
        cv2.destroyAllWindows()


if __name__ == '__main__':
    tracker = Tracker()
    tracker.run()
