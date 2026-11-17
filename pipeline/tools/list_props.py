"""List CARLA static prop blueprints with their bounding boxes (spawn each once far away, read bbox, destroy)."""
import carla

c = carla.Client('127.0.0.1', 2200)
c.set_timeout(120)
w = c.get_world()
lib = w.get_blueprint_library()
sp = w.get_map().get_spawn_points()[0]
rows = []
for i, bp in enumerate(sorted(lib.filter('static.prop.*'), key=lambda b: b.id)):
    t = carla.Transform(carla.Location(x=sp.location.x, y=sp.location.y, z=sp.location.z + 200 + 5 * i))
    a = w.try_spawn_actor(bp, t)
    if a is None:
        rows.append((bp.id, None))
        continue
    e = a.bounding_box.extent
    rows.append((bp.id, (round(2 * e.x, 2), round(2 * e.y, 2), round(2 * e.z, 2))))
    a.destroy()
for r in rows:
    if r[1] is None or max(r[1]) > 1.5 or any(k in r[0] for k in ('trash', 'bin', 'contain', 'dumpster', 'garbage')):
        print(r)
print('total props', len(rows))
