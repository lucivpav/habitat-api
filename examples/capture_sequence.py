import habitat
import cv2
import os.path
import habitat_sim
from habitat_sim.utils.common import quat_from_magnum
import quaternion
import numpy as np
from habitat_sim.agent.controls.default_controls import _rotate_local, _Z_AXIS
from habitat_sim.agent.controls.controls import ActuationSpec, SceneNodeControl
from habitat_sim.scene import SceneNode
from habitat.sims.habitat_simulator.actions import (
    HabitatSimActions,
    HabitatSimV1ActionSpaceConfiguration,
    HabitatSimActionsSingleton
)
from habitat.tasks.nav.nav import SimulatorTaskAction

def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

@habitat_sim.registry.register_move_fn(body_action=False)
class LookLeftish(SceneNodeControl):
    def __call__(self, scene_node: SceneNode, actuation_spec: ActuationSpec):
        _rotate_local(
            scene_node, actuation_spec.amount, _Z_AXIS, actuation_spec.constraint
        )

@habitat_sim.registry.register_move_fn(body_action=False)
class LookRightish(SceneNodeControl):
    def __call__(self, scene_node: SceneNode, actuation_spec: ActuationSpec):
        _rotate_local(
            scene_node, -actuation_spec.amount, _Z_AXIS, actuation_spec.constraint
        )

@habitat.registry.register_task_action
class LookLeftish(SimulatorTaskAction):
    def _get_uuid(self, *args, **kwargs) -> str:
        return "look_leftish"

    def step(self, *args, **kwargs):
        return self._sim.step(HabitatSimActions.LOOK_LEFTISH)

@habitat.registry.register_task_action
class LookRightish(SimulatorTaskAction):
    def _get_uuid(self, *args, **kwargs) -> str:
        return "look_rightish"

    def step(self, *args, **kwargs):
        return self._sim.step(HabitatSimActions.LOOK_RIGHTISH)

@habitat.registry.register_task_action
class MoveBackwardAction(SimulatorTaskAction):
    def _get_uuid(self, *args, **kwargs) -> str:
        return "move_backward"

    def step(self, *args, **kwargs):
        return self._sim.step(HabitatSimActions.MOVE_BACKWARD)

@habitat.registry.register_action_space_configuration(name="v1Extended")
class HabitatSimV1ExtendedActionSpaceConfiguration(HabitatSimV1ActionSpaceConfiguration):
    def get(self):
        config = super().get()

        config[HabitatSimActions.LOOK_LEFTISH] = habitat_sim.ActionSpec(
            "look_leftish",
            habitat_sim.ActuationSpec(amount=self.config.TILT_ANGLE)
        )
        config[HabitatSimActions.LOOK_RIGHTISH] = habitat_sim.ActionSpec(
            "look_rightish",
            habitat_sim.ActuationSpec(amount=self.config.TILT_ANGLE)
        )
        config[HabitatSimActions.MOVE_BACKWARD] = habitat_sim.ActionSpec(
            "move_backward",
            habitat_sim.ActuationSpec(amount=self.config.FORWARD_STEP_SIZE)
        )

        return config

def capture_sequence():
    environmentConfigPath = "configs/datasets/pointnav/ciirc.yaml"
    outputDir = '/Volumes/GoogleDrive/Můj disk/ARTwin/personal/lucivpav/habitat'
    posesDir = os.path.join(outputDir, 'poses')

    if not os.path.isdir(posesDir):
        os.mkdir(posesDir)

    newActions = ['LOOK_LEFTISH', 'LOOK_RIGHTISH', 'MOVE_BACKWARD']

    for newAction in newActions:
        HabitatSimActions.extend_action_space(newAction)

    config = habitat.get_config(environmentConfigPath)
    config.defrost()

    config.TASK.POSSIBLE_ACTIONS.extend(['LOOK_UP', 'LOOK_DOWN'])
    config.TASK.POSSIBLE_ACTIONS.extend(newActions)
    config.TASK.ACTIONS.LOOK_LEFTISH = habitat.config.Config()
    config.TASK.ACTIONS.LOOK_LEFTISH.TYPE = "LookLeftish"
    config.TASK.ACTIONS.LOOK_RIGHTISH = habitat.config.Config()
    config.TASK.ACTIONS.LOOK_RIGHTISH.TYPE = "LookRightish"
    config.TASK.ACTIONS.MOVE_BACKWARD = habitat.config.Config()
    config.TASK.ACTIONS.MOVE_BACKWARD.TYPE = "MoveBackwardAction"
    config.SIMULATOR.ACTION_SPACE_CONFIG = "v1Extended"
    config.freeze()

    env = habitat.Env(config=config)

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
        elif keystroke == ord('s'):
            action = HabitatSimActions.MOVE_BACKWARD
            print("action: BACKWARD")
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
        elif keystroke == 2:
            action = HabitatSimActions.LOOK_LEFTISH
            print("action: LOOK LEFTISH")
        elif keystroke == 3:
            action = HabitatSimActions.LOOK_RIGHTISH
            print("action: LOOK RIGHTISH")
        else:
            print("INVALID KEY")
            continue

        observations = env.step(action)
        count_steps += 1

        state = env._sim.get_agent_state(0)
        rotation = state.sensor_states['rgb'].rotation
        position = state.sensor_states['rgb'].position
        print(f'Position: {position}')
        t = np.reshape(position, (3,1))
        R = quaternion.as_rotation_matrix(rotation)
        P = np.vstack((np.hstack((R,-R @ t)), np.array([0,0,0,1])))

        posePath = os.path.join(posesDir, str(count_steps) + '.txt')
        poseFile = open(posePath, 'w')
        for i in range(4):
            poseFile.write('%0.2f %0.2f %0.2f %0.2f\n' % (P[i,0],P[i,1],P[i,2],P[i,3]))
        poseFile.close()

        cv2.imwrite(os.path.join(posesDir, f'{count_steps}.png'), transform_rgb_bgr(observations["rgb"]))
        cv2.imshow("RGB", transform_rgb_bgr(observations["rgb"]))

if __name__ == "__main__":
    print(os.path.abspath('./'))
    HabitatSimActions: HabitatSimActionsSingleton = HabitatSimActionsSingleton()
    capture_sequence()
