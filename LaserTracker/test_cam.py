import cv2

for i in range(5):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Use CAP_DSHOW for Windows compatibility
    if cap.isOpened():
        print(f"Camera {i} is available. Showing preview window...")
        ret, frame = cap.read()
        if ret and frame is not None:
            cv2.imshow(f"Camera {i}", frame)
            cv2.waitKey(2000)  # Show for 2 seconds
            cv2.destroyAllWindows()
        else:
            print(f"Camera {i} opened but could not read frame.")
        cap.release()
    else:
        print(f"Camera {i} is not available.")