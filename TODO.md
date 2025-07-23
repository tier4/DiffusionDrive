# TODO List

- [ ] BEV can be decide which part to use see below

```makrdown
ased on the code in bench2drive_scene.py, here's what happens with the BEV cache:

  BEV Cache Usage

  The code handles both full_bev and front_bev formats:

  1. Priority: It first looks for front_bev (128×256)
  2. Fallback: If only full_bev exists, it extracts the front half
  3. Extraction: map_bev = full_bev[:128, :] - takes the front 128 rows

  So to answer your question: If your cache contains full_bev, the script will automatically extract and use only the front half (128×256) to match the expected NavSim format.

  Key Points:

  1. NavSim expects 128×256 BEV maps (front-focused rectangular format)
  2. Your full_bev is likely 256×256 (square format covering all directions)
  3. The code automatically handles this by taking only the front portion

  The extraction logic (lines 578-583):

  if 'front_bev' in cached_data:
      map_bev = cached_data['front_bev']  # Use if available
  elif 'full_bev' in cached_data:
      # Extract front half
      full_bev = cached_data['full_bev']
      map_bev = full_bev[:128, :]  # Takes rows 0-127 (front half)

  What this means spatially:

  - Full BEV (256×256): Covers 360° around the vehicle
  - Front BEV (128×256): Covers forward 180° view
  - The extraction [:128, :] takes the top half, which represents the front view

  So you don't need to worry about regenerating your BEV cache - the code will properly extract the front portion from your full BEV cache automatically!
  ```

- [ ] he normalization profiles in the normalization scripts
