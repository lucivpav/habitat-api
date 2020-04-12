import habitat
import cv2
import os.path
import habitat_sim
from habitat_sim.utils.common import quat_from_magnum
import quaternion
import numpy as np

from habitat.sims.habitat_simulator.actions import HabitatSimActionsSingleton

def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

def capture_sequence():
    outputDir = '/Volumes/GoogleDrive/Můj disk/ARTwin/personal/lucivpav/habitat'
    posesDir = os.path.join(outputDir, 'poses')
    posesPath = os.path.join(posesDir, 'poses.csv')

    if not os.path.isdir(posesDir):
        os.mkdir(posesDir)

    env = habitat.Env(config=habitat.get_config("configs/datasets/pointnav/mp3d.yaml"))

    print("Environment creation successful")
    observations = env.reset()
    cv2.imshow("RGB", transform_rgb_bgr(observations["rgb"]))

    print("Agent stepping around inside environment.")
    posesFile = open(posesPath, 'w')
    posesFile.write('id x y z dirx diry dirz\n')

    count_steps = 0
    while not env.episode_over:
        keystroke = cv2.waitKey(0)

        if keystroke == ord('w'):
            action = HabitatSimActions.MOVE_FORWARD
            print("action: FORWARD")
        elif keystroke == ord('a'):
            action = HabitatSimActions.TURN_LEFT
            print("action: LEFT")
        elif keystroke == ord('d'):
            action = HabitatSimActions.TURN_RIGHT
            print("action: RIGHT")
        elif keystroke == ord('f'):
            action = HabitatSimActions.STOP
            print("action: FINISH")
        elif keystroke == 0:
            action = HabitatSimActions.LOOK_UP
            print("action: LOOK UP")
        elif keystroke == 1:
            action = HabitatSimActions.LOOK_DOWN
            print("action: LOOK DOWN")
        else:
            print("INVALID KEY")
            continue

        observations = env.step(action)
        count_steps += 1

        state = env._sim.get_agent_state(0)
        rotation1 = state.rotation
        rotation2 = state.sensor_states['rgb'].rotation
        rotation = rotation1 * rotation2
        print(f'Position: {state.position}')
        x = state.position[0]
        y = state.position[1]
        z = state.position[2]
        R = quaternion.as_rotation_matrix(rotation)
        initialCameraDirection = np.array([0.0, 0.0, -1.0]).T
        cameraDirection = R @ initialCameraDirection
        dirx = cameraDirection[0]
        diry = cameraDirection[1]
        dirz = cameraDirection[2]
        print(f'Rotation: {dirx},{diry},{dirz}')
        posesFile.write('%0.2f %0.2f %0.2f %0.2f %0.2f %0.2f\n' % (x,y,z, dirx,diry,dirz))

        cv2.imwrite(os.path.join(posesDir, f'{count_steps}.png'), transform_rgb_bgr(observations["rgb"]))
        cv2.imshow("RGB", transform_rgb_bgr(observations["rgb"]))

    posesFile.close()

if __name__ == "__main__":
    print(os.path.abspath('./'))
    HabitatSimActions: HabitatSimActionsSingleton = HabitatSimActionsSingleton()
    capture_sequence()
