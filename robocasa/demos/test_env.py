import argparse
import robosuite
from robosuite.controllers import load_composite_controller_config
from termcolor import colored
from robosuite.devices import Keyboard
from robocasa.scripts.collect_demos import collect_human_trajectory

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="BasicEnvChef", help="task")
    parser.add_argument("--robot", type=str, default="PandaOmron", help="robot")
    args = parser.parse_args()

    # 固定使用 layout=3 和 style=3
    layout = 3
    style = 3

    # 创建控制器配置
    controller_config = load_composite_controller_config(robot=args.robot)

    # 创建环境配置
    config = {
        "env_name": args.task,
        "robots": args.robot,
        "controller_configs": controller_config,
        "translucent_robot": False,
    }

    args.renderer = "mjviewer"

    print(colored("Initializing environment...", "yellow"))

    # 创建环境
    env = robosuite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera=None,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
        renderer=args.renderer,
    )

    # 设置布局和样式
    env.layout_and_style_ids = [[layout, style]]

    # 初始化键盘控制
    device = Keyboard(env=env, pos_sensitivity=4.0, rot_sensitivity=4.0)

    print(
        colored(
            f"Showing configuration:\n    Layout: {layout}\n    Style: {style}", "green"
        )
    )
    print()
    print(colored("Spawning environment...\n(Press Q to quit)", "yellow"))

    # 单次展示环境，注意 render 参数设置
    collect_human_trajectory(
        env,
        device,
        "right",
        "single-arm-opposed",
        mirror_actions=True,
        render=(args.renderer != "mjviewer"),
        max_fr=30,
        print_info=False,
    )
