import cv2

def take_pictures(width=0, height=0, folder="."):
    cap = cv2.VideoCapture(0)
    if width > 0 and height > 0:
        print("Set Width and Height")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    counter   = 0
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    file  = "%s/%s_%d_%d_" %(folder, 'img', w, h)
    while True:
        ret, frame = cap.read()
        cv2.imshow('camera image', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            print("Took image ", counter)
            cv2.imwrite("%s%d.jpg"%(file, counter), frame)
            counter += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    folder = "images"
    width = 640
    height = 480
    take_pictures(width=width, height=height, folder=folder)

    print("Done")