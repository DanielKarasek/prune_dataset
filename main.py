import cv2


def main():
    query_image = cv2.imread("image_video/275.jpg")
    train_image = cv2.imread("image_video/921.jpg")
    query_image_bw = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
    train_image_bw = cv2.cvtColor(train_image, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()

    query_keypoints, query_descriptor = orb.detectAndCompute(query_image_bw, None)
    train_keypoints, train_descriptor = orb.detectAndCompute(train_image_bw, None)

    print(len(query_keypoints), len(train_keypoints))
    matcher = cv2.BFMatcher(crossCheck=True)
    matches = matcher.match(query_descriptor, train_descriptor)
    matches = sorted(matches, key=lambda x: x.distance)
    print(len(matches))

if __name__ == "__main__":
    main()
