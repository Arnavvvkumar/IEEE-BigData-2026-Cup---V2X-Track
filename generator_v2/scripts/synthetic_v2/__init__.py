"""Procedural roadside LiDAR scene generator, version 2.

Modules:
    geometry  - frames, rotations, box conversions and box/ray tests
    meshes    - procedural triangle meshes (vehicles, trucks, people, street furniture)
    layout    - road-layout sampling and static background construction
    traffic   - lane/sidewalk-aware placement of annotated road users
    sensor    - beam patterns, Open3D ray casting, noise and dropout
    config    - configuration loading, merging and validation

Everything is procedural; no real clouds, labels, statistics or assets are used.
"""
