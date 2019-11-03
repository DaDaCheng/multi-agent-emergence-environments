{
    make_env: {
        "function": "mae_envs.envs.hide_and_seek:make_env",
        args: {
            # Agents
            n_hiders: 1,
            n_seekers: 1,
            # Agent Actions
            grab_box: false,
            grab_out_of_vision: false,
            grab_selective: false,
            grab_exclusive: false,

            lock_box: true,
            lock_type: "all_lock_team_specific",
            lock_out_of_vision: false,

            # Scenario
            n_substeps: 15,
            horizon: 240,
            scenario: "randomwalls",
            n_rooms: 1,
            random_room_number: false,
            prob_outside_walls: 0.5,
            prep_fraction: 0.4,
            rew_type: "joint_zero_sum",
            restrict_rect: [-6.0, -6.0, 12.0, 12.0],

            hiders_together_radius: 0.5,
            seekers_together_radius: 0.5,

            # Objects
            n_boxes: [3, 9],
            n_elongated_boxes: [3, 9],
            box_only_z_rot: true,
            boxid_obs: false,

            n_ramps: 2,


            # Observations
            n_lidar_per_agent: 30,
            prep_obs: true,
        },
    },
}
