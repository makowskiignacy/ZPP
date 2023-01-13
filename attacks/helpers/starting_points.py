import math
from foolbox.attacks.blended_noise import LinearSearchBlendedUniformNoiseAttack
from foolbox.distances import LpDistance

import eagerpy as ep

from eagerpy.astensor import astensor

import time


def generate_starting_points(self, data, model, distance, directions=1000, steps=1000):
    # default params for LinearSearchBlendedUniformNoiseAttack: distance=None, directions=1000, steps=1000
    init_attack = LinearSearchBlendedUniformNoiseAttack(distance=distance, directions=directions, steps=steps)
    originals, _ = astensor(data.input)
    starting_points = init_attack.run(model, originals, self.criterion)

    if starting_points is None:
        print(f'Wrong starting points ({starting_points}) for params: distance={distance}, directions={directions}, steps={steps}')
    else:
        print(f'Successful starting points ({starting_points}) for params: distance={distance}, directions={directions}, steps={steps}')
        # sys.exit(f'Wrong starting points ({starting_points}) for params: distance={dist}, directions={dirs}, steps={stps}')

    return starting_points

def generate_starting_points_set(self, data, model):
    dist = LpDistance(p=2)
    directions2check = [10, 100, 1000, 10000, 100000, 1000000]
    steps2check = [10, 100, 1000, 10000, 100000, 1000000]
    distances2check = [LpDistance(0), LpDistance(1), LpDistance(2), LpDistance(math.inf)]
    ptsSet = []
    # stps = 100000
    # default params for LinearSearchBlendedUniformNoiseAttack: distance=None, directions=1000, steps=1000
    for dist in distances2check:
        for dir in directions2check:
            for steps in steps2check:
                time_start = time.time()

                init_attack = LinearSearchBlendedUniformNoiseAttack(distance=dist, directions=dir, steps=steps)
                originals, _ = astensor(data.input)
                starting_points = init_attack.run(model, originals, self.criterion)
                
                time_end = time.time()

                if starting_points is None:
                    print(f'Wrong starting points ({starting_points})')
                else:
                    print(f'Successful starting points ({starting_points})')
                    # sys.exit(f'Wrong starting points ({starting_points}) for params: distance={dist}, directions={dirs}, steps={stps}')
                    ptsSet.append(starting_points)
                
                print(f"Checked for params: distance={dist}, directions={dir}, steps={steps}")
                print(f"Took {time_end - time_start}\n")

    return ptsSet