import os
import cv2

def main():
    target_dir = './datasets/target_person'
    cap = cv2.VideoCapture(0)
    count = 0
    cleared = False
    print("Press 'c' to capture an image (5 total).")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        cv2.imshow('Capture - Press c', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            if not cleared:
                # Clear directory only before first capture
                if os.path.exists(target_dir):
                    for f in os.listdir(target_dir):
                        os.remove(os.path.join(target_dir, f))
                else:
                    os.makedirs(target_dir)
                cleared = True
            count += 1
            filename = os.path.join(target_dir, f'{count}.jpg')
            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")
            if count >= 5:
                print("Captured 5 images. Exiting.")
                break
        elif key == ord('q'):
            print("Exit before capturing 5 images.")
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()