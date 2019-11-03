#!/usr/bin/env python3
import click
from RL_brain import PolicyGradient
from mujoco_worldgen.util.parse_arguments import parse_arguments
@click.command()
@click.argument('argv', nargs=-1, required=False)




def main(argv):

    RL = PolicyGradient(
        n_actions=9,
        n_features=8,
        learning_rate=0.1,
        reward_decay=0.5,
        # output_graph=True,
    )

    RLL = PolicyGradient(
        n_actions=9,
        n_features=8,
        learning_rate=0.1,
        reward_decay=0.5,
        # output_graph=True,
    )

if __name__ == '__main__':
    main()