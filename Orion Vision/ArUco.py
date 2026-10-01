import cv2

# Gebruik exact hetzelfde dictionary als in de tracker
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# Genereer Marker ID 0 (Robot)
marker_robot = cv2.aruco.generateImageMarker(dictionary, 0, 400)
cv2.imwrite("aruco_id0_robot.png", marker_robot)

# Genereer Marker ID 1 (Tegenstander)
marker_opp = cv2.aruco.generateImageMarker(dictionary, 1, 400)
cv2.imwrite("aruco_id1_opponent.png", marker_opp)

print("✅ ArUco markers succesvol opgeslagen als PNG!")