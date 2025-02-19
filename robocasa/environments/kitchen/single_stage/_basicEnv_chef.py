from robocasa.environments.kitchen._kitchen_chef import KitchenChef, FixtureType

# from robocasa_extension.utilities.scene_utils import SceneConfigGenerator
import numpy as np
import inspect


class BasicEnvChef(KitchenChef):
    def __init__(self, *args, **kwargs):
        self.cab_id = kwargs.pop("cab_id", FixtureType.CABINET)
        self.objects_to_spawn = kwargs.pop("objects_to_spawn", [])
        self.open_fixtures = kwargs.pop("open_fixtures", [])

        # 初始化基础属性
        self.objects = {}
        self.tool_types = {"tool", "utensil"}

        # 保存初始对象配置
        self.initial_objects = self.objects_to_spawn.copy()
        self.current_objects = self.objects_to_spawn.copy()

        # 延迟初始化对象配置
        self._obj_cfg = None

        super().__init__(*args, **kwargs)

    @property
    def obj_cfg(self):
        """延迟加载对象配置"""
        if self._obj_cfg is None:
            print("\n=== 首次创建对象配置 ===")
            self._obj_cfg = self._get_obj_cfgs()
        return self._obj_cfg

    def _is_tool(self, obj):
        """判断对象是否为工具或餐具"""
        return isinstance(obj, dict) and (
            obj.get("type") in {"tool", "utensil"}
            or obj.get("is_tool", False)  # 通过类型判断
        )  # 通过标记判断

    def _setup_kitchen_references(self):
        """设置厨房引用"""
        super()._setup_kitchen_references()

        # 获取基础设备引用
        self.cab = self.register_fixture_ref("cab", dict(id=self.cab_id))
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER, ref=self.cab)
        )
        self.coffee_machine = self.get_fixture("coffee_machine")
        self.microwave = self.get_fixture(FixtureType.MICROWAVE)
        self.fridge = next(
            (
                fixture
                for name, fixture in self.fixtures.items()
                if "fridge" in name.lower()
            ),
            None,
        )
        self.stove = next(
            (
                fixture
                for name, fixture in self.fixtures.items()
                if "stove" in name.lower()
            ),
            None,
        )
        self.shelves = next(
            (
                fixture
                for name, fixture in self.fixtures.items()
                if "shelves" in name.lower() or "shelf" in name.lower()
            ),
            None,
        )
        self.cab_main = self.get_fixture("cab_main_main_group")

        self.init_robot_base_pos = self.cab_main

        # 打印机器人末端执行器位置信息
        if hasattr(self, "sim"):
            try:
                # 使用正确的 site 名称
                eef_site_id = self.sim.model.site_name2id("gripper0_right_grip_site")
                sim_eef_pos = self.sim.data.site_xpos[eef_site_id]
                print(f"\n=== Robot End Effector Info ===")
                print(f"Simulator Position: {sim_eef_pos}")
            except Exception as e:
                print(f"\n获取末端执行器位置失败: {str(e)}")
                # 打印所有可用的 site 名称，用于调试
                print("可用的 site 名称:")
                for i in range(self.sim.model.nsite):
                    print(f"- {self.sim.model.site_id2name(i)}")

        # 获取所有可开启设备
        self.all_openable_fixtures = []
        for fixture_name in self.fixtures:
            if (
                "cabinet" in fixture_name.lower()
                or "cab" in fixture_name.lower()
                or "microwave" in fixture_name.lower()
                or "fridge" in fixture_name.lower()
            ):
                fixture = self.get_fixture(fixture_name)
                if hasattr(fixture, "set_door_state"):
                    self.all_openable_fixtures.append(fixture)

        # print("\n=== Debug: Openable fixtures ===")
        # for fixture in self.all_openable_fixtures:
        #     print(f"- {fixture.name if hasattr(fixture, 'name') else 'Unknown'}")

    def _reset_internal(self):
        """初始化重置 - 仅用于环境首次创建的基础组件初始化"""
        try:
            print("\n=== 初始化基础环境组件 ===")

            # 1. 保存当前对象配置
            current_configs = (
                self.objects_to_spawn.copy()
                if hasattr(self, "objects_to_spawn")
                else []
            )
            # print(f"current_configs: {current_configs}")

            # 2. 调用父类的重置方法
            super()._reset_internal()

            # 3. 处理柜门状态
            print("\n=== Debug: Setting fixture states ===")
            print(f"Debug - Open fixtures list: {self.open_fixtures}")

            for fixture in self.all_openable_fixtures:
                try:
                    if hasattr(fixture, "set_door_state"):
                        fixture_name = (
                            fixture.name if hasattr(fixture, "name") else "Unknown"
                        )
                        should_open = any(
                            [
                                fixture_name in self.open_fixtures,
                                (
                                    "fridge" in self.open_fixtures
                                    and any(
                                        x in fixture_name
                                        for x in [
                                            "fridge_housing",
                                            "fridge_cab",
                                            "fridge_top",
                                        ]
                                    )
                                ),
                            ]
                        )

                        if should_open:
                            print(f"Opening {fixture_name}")
                            fixture.set_door_state(
                                min=0.98, max=1.0, env=self, rng=self.rng
                            )
                        else:
                            print(f"Closing {fixture_name}")
                            fixture.set_door_state(
                                min=0.0, max=0.02, env=self, rng=self.rng
                            )
                    else:
                        print(
                            f"Warning: Fixture {fixture} does not have set_door_state method"
                        )
                except Exception as e:
                    print(f"Error setting state for fixture {fixture}: {str(e)}")

            # 4. 处理对象在餐具上的放置
            print("\n=== 处理对象在餐具上的放置 ===")
            for obj_config in current_configs:
                if obj_config.get("is_on_utensil"):
                    try:
                        obj_name = obj_config["name"]
                        utensil_name = obj_config["location"]

                        if utensil_name in self.objects and obj_name in self.objects:
                            # 获取餐具对象和关节
                            utensil_obj = self.objects[utensil_name]
                            utensil_joint = utensil_obj.joints[0]

                            # 获取餐具位置
                            utensil_pos = self.sim.data.get_joint_qpos(utensil_joint)[
                                :3
                            ]
                            print(f"\n放置对象 {obj_name} 到 {utensil_name}:")
                            print(f"- 餐具位置: {utensil_pos}")

                            # 计算偏移量
                            offset = np.array(
                                {
                                    "pot": (0, 0, 0.02),
                                    "pan": (0, 0, 0.05),
                                    "plate": (0, 0, 0.05),
                                    "mug": (0, 0, 0.02),
                                }.get(utensil_name, (0, 0, 0.02))
                            )

                            # 计算目标位置
                            target_pos = utensil_pos + offset

                            # 获取对象关节并更新位置
                            obj = self.objects[obj_name]
                            obj_joint = obj.joints[0]

                            # 保持当前的方向
                            current_quat = self.sim.data.get_joint_qpos(obj_joint)[3:7]

                            # 设置新的位置和方向
                            joint_pos = np.concatenate([target_pos, current_quat])
                            self.sim.data.set_joint_qpos(obj_joint, joint_pos)

                            # 更新模拟器状态
                            self.sim.forward()

                            # 验证更新后的位置
                            current_pos = self.sim.data.get_joint_qpos(obj_joint)[:3]
                            print(f"- 目标位置: {target_pos}")
                            print(f"- 当前位置: {current_pos}")

                    except Exception as e:
                        print(f"放置对象 {obj_name} 到餐具 {utensil_name} 上时出错: {str(e)}")
                        import traceback

                        traceback.print_exc()

            # 5. 处理工具在机器人夹持器上的放置
            print("\n=== 处理工具在机器人夹持器上的放置 ===")

            # 获取所有工具对象
            tools = ["spatula", "fork", "knife", "spoon"]

            for obj_name in tools:
                if obj_name in self.objects:
                    try:
                        # 检查工具是否在夹持器中 - 修改检查条件
                        tool_config = next(
                            (
                                cfg
                                for cfg in current_configs
                                if cfg["name"] == obj_name
                                and cfg.get("is_in_gripper") == True
                                and cfg.get("state") == "in_use"
                            ),
                            None,
                        )

                        if tool_config:
                            print(f"\n处理工具: {obj_name}")
                            tool_obj = self.objects[obj_name]

                            # 获取末端执行器位置
                            eef_site_id = self.sim.model.site_name2id(
                                "gripper0_right_grip_site"
                            )
                            eef_pos = self.sim.data.site_xpos[eef_site_id]
                            default_quat = np.array([1.0, 0.0, 0.0, 0.0])

                            # 添加工具偏移量
                            tool_offset = np.array([0.0, 0.0, 0.05])
                            tool_pos = eef_pos + tool_offset

                            # 设置工具位置
                            joint_name = tool_obj.joints[0]
                            joint_pos = np.concatenate([tool_pos, default_quat])

                            # 更新工具位置
                            self.sim.data.set_joint_qpos(joint_name, joint_pos)
                            self.sim.forward()

                            print(f"工具 {obj_name} 已放置到机器人夹持器")

                    except Exception as e:
                        print(f"处理工具 {obj_name} 时出错: {str(e)}")
                        import traceback

                        traceback.print_exc()

        except Exception as e:
            print(f"重置内部状态失败: {str(e)}")
            import traceback

            traceback.print_exc()

    def update_environment_state(self, changed_configs=None):
        """更新环境状态
        Args:
            changed_configs: 需要更新的配置列表，如果为None则更新所有配置
        """
        try:
            print("\n=== 更新环境状态 ===")

            # 1. 更新普通对象位置
            if changed_configs:  # 检查changed_configs而不是self.objects_to_spawn
                print(f"需要更新的对象: {[cfg['name'] for cfg in changed_configs]}")
                self._update_object_placements(changed_configs)
            else:
                print("没有对象需要更新位置")

            # 2. 更新设备状态
            if hasattr(self, "all_openable_fixtures") and self.all_openable_fixtures:
                self._update_fixture_states()

        except Exception as e:
            print(f"更新环境状态失败: {str(e)}")
            import traceback

            traceback.print_exc()

    def _update_fixture_states(self):
        """更新设备状态（如柜门开关）"""
        try:
            print("\n=== 更新设备状态 ===")
            print(f"需要打开的设备: {self.open_fixtures}")

            for fixture in self.all_openable_fixtures:
                try:
                    if hasattr(fixture, "set_door_state"):
                        fixture_name = (
                            fixture.name if hasattr(fixture, "name") else "Unknown"
                        )
                        should_open = any(
                            [
                                fixture_name in self.open_fixtures,
                                (
                                    "fridge" in self.open_fixtures
                                    and any(
                                        x in fixture_name
                                        for x in [
                                            "fridge_housing",
                                            "fridge_cab",
                                            "fridge_top",
                                        ]
                                    )
                                ),
                            ]
                        )

                        if should_open:
                            print(f"打开设备 {fixture_name}")
                            fixture.set_door_state(
                                min=0.98, max=1.0, env=self, rng=self.rng
                            )
                        else:
                            print(f"关闭设备 {fixture_name}")
                            fixture.set_door_state(
                                min=0.0, max=0.02, env=self, rng=self.rng
                            )
                    else:
                        print(f"警告: 设备 {fixture} 不支持状态设置")
                except Exception as e:
                    print(f"设置设备 {fixture} 状态失败: {str(e)}")

        except Exception as e:
            print(f"更新设备状态失败: {str(e)}")
            import traceback

            traceback.print_exc()

    def _update_object_placements(self, changed_configs=None):
        try:
            if not self.objects_to_spawn:
                print("没有对象需要更新位置")
                return False

            print("\n=== 更新对象位置 ===")

            print(f"changed_configs: {changed_configs}")
            # 分离普通对象和在餐具上的对象
            regular_objects = []
            utensil_objects = []

            for obj in changed_configs:
                if obj.get("placement", {}).get("use_utensil_reference"):
                    utensil_objects.append(obj)
                    print(f"use utensil reference item: {obj['name']}")
                else:
                    regular_objects.append(obj)
                    print(f"use regular item: {obj['name']}")

            # 1. 处理普通对象
            if regular_objects:
                placement_initializer = self._get_placement_initializer(regular_objects)
                object_placements = placement_initializer.sample()

                for obj_name, placement_data in object_placements.items():
                    if obj_name in self.objects:
                        obj = self.objects[obj_name]
                        pos, quat = placement_data[0], placement_data[1]
                        joint_name = obj.joints[0]
                        joint_pos = np.concatenate([np.array(pos), np.array(quat)])
                        self.sim.data.set_joint_qpos(joint_name, joint_pos)

            # 2. 处理在餐具上的对象
            for obj in utensil_objects:
                obj_name = obj["name"]
                if obj_name not in self.objects:
                    continue

                utensil_name = obj["placement"]["utensil_name"]
                if utensil_name not in self.objects:
                    continue

                # 获取餐具位置
                utensil_obj = self.objects[utensil_name]
                utensil_joint = utensil_obj.joints[0]
                utensil_pos = self.sim.data.get_joint_qpos(utensil_joint)[:3]

                # 应用偏移
                offset = obj["placement"]["offset"]
                target_pos = utensil_pos + np.array(offset)

                # 更新对象位置
                obj_joint = self.objects[obj_name].joints[0]
                current_quat = self.sim.data.get_joint_qpos(obj_joint)[3:7]
                joint_pos = np.concatenate([target_pos, current_quat])
                self.sim.data.set_joint_qpos(obj_joint, joint_pos)

            # 更新物理引擎
            self.sim.forward()
            return True

        except Exception as e:
            print(f"更新对象位置失败: {str(e)}")
            import traceback

            traceback.print_exc()
            return False

    def _get_obj_cfgs(self):
        """获取对象配置"""
        # 直接返回已经配置好的对象列表
        return self.objects_to_spawn

    def _check_success(self):
        """检查任务是否成功"""
        return True

    def reset(self):
        """重置环境状态"""
        print("\n=== 重置环境状态 ===")
        # 恢复初始对象配置
        self.objects_to_spawn = self.initial_objects.copy()
        self.current_objects = self.initial_objects.copy()
        return super().reset()
