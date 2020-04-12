import habitat
import cv2
import os.path
import habitat_sim

from habitat.sims.habitat_simulator.actions import HabitatSimActionsSingleton

def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

def example():
    outputDir = '/Volumes/GoogleDrive/Můj disk/ARTwin/personal/lucivpav/habitat'
    env = habitat.Env(config=habitat.get_config("configs/datasets/pointnav/mp3d.yaml"))

    print("Environment creation successful")
    observations = env.reset()
    cv2.imshow("RGB", transform_rgb_bgr(observations["rgb"]))

    print("Agent stepping around inside environment.")

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
        cv2.imwrite(os.path.join(outputDir, f'export_imgs/im_{count_steps}.png'), transform_rgb_bgr(observations["rgb"]))

        cv2.imshow("RGB", transform_rgb_bgr(observations["rgb"]))

    print("Episode finished after {} steps.".format(count_steps))

    if action == HabitatSimActions.STOP and observations["pointgoal_with_gps_compass"][0] < 0.2:
        print("you successfully navigated to destination point")
    else:
        print("your navigation was unsuccessful")


if __name__ == "__main__":
    print(os.path.abspath('./'))
    HabitatSimActions: HabitatSimActionsSingleton = HabitatSimActionsSingleton()
    example()
