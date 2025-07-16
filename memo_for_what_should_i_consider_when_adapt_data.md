# Dataset transform

## Driving command

- Type (Straight, Left, Lane Follow, etc...)
- Command distance (like in carla,Near & Far)

## Status

- Unit of each status
- Velocity
- Acceleration
- Positions

## Current strategy

*DD = Diffusion Drive
*B2D = bench2drive

Below is the current strategy:

- Must change
  - Change  coordinate to ego-centric (DD)
- Tryout setup
  - Merge 3 frontal cameras’ RGB  (Follow DD)
  - Map B2D command to DD command
  - Map B2D categories to DD command
