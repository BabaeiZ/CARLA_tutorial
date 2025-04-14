import os
import time
import sys
import random

sys.path.append(
    'D:\\CARLA\\WindowsNoEditor\\PythonAPI\\carla\\dist\\carla-0.9.11-py%d.%d-win-amd64.egg' %
    (sys.version_info.major, sys.version_info.minor)
)
import carla

output_dir = 'outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
for folder in ['lidar', 'radar', 'collision']:
    path = os.path.join(output_dir, folder)
    if not os.path.exists(path):
        os.makedirs(path)

def save_lidar_data(data):
    """Save LIDAR data to a PLY file."""
    try:
        data.save_to_disk(os.path.join(output_dir, 'lidar', f'{data.frame}.ply'))
        print(f"Saved LIDAR data for frame {data.frame}")
    except Exception as e:
        print(f"Error saving LIDAR data: {e}")

def save_radar_data(data):
    """Save radar data to a text file."""
    try:
        filename = os.path.join(output_dir, 'radar', f'{data.frame}.radar')
        with open(filename, 'w') as f:
            for detection in data:
                f.write(f"{detection.azimuth},{detection.altitude},{detection.depth},{detection.velocity}\n")
        print(f"Saved radar data for frame {data.frame}")
    except Exception as e:
        print(f"Error saving radar data: {e}")

def save_collision_data(data):
    """Save collision data to a text file."""
    try:
        collision_filename = os.path.join(output_dir, 'collision', f"collision_{data.frame}.txt")
        with open(collision_filename, 'w') as f:
            f.write(f"Frame: {data.frame}\n")
            f.write(f"Normal Impulse: {data.normal_impulse.x}, {data.normal_impulse.y}, {data.normal_impulse.z}\n")
            f.write(f"Other Actor: {data.other_actor.type_id if data.other_actor else 'None'}\n")
        print(f"Collision detected! Saved data for frame {data.frame} with actor {data.other_actor.type_id if data.other_actor else 'None'}")
    except Exception as e:
        print(f"Error saving collision data: {e}")

def main():
    # Connect to CARLA client
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    vehicle = None
    obstacle = None
    lidar_sensor = None
    radar_sensor = None
    collision_sensor = None

    try:
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()

        # Spawn a vehicle (Tesla Model 3)
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        spawn_points = world.get_map().get_spawn_points()
        vehicle_spawn_point = random.choice(spawn_points)
        vehicle = world.spawn_actor(vehicle_bp, vehicle_spawn_point)
        print("Spawned vehicle: Tesla Model 3")

        # Spawn a second vehicle closer to ensure collision
        obstacle_bp = blueprint_library.find('vehicle.audi.a2')
        obstacle_spawn_point = carla.Transform(
            carla.Location(
                x=vehicle_spawn_point.location.x + 5.0,
                y=vehicle_spawn_point.location.y,
                z=vehicle_spawn_point.location.z
            ),
            vehicle_spawn_point.rotation
        )
        obstacle = world.spawn_actor(obstacle_bp, obstacle_spawn_point)
        print("Spawned obstacle vehicle: Audi A2")

        # Add LIDAR sensor
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', '50')
        lidar_spawn_point = carla.Transform(carla.Location(x=0.0, z=2.5))
        lidar_sensor = world.spawn_actor(lidar_bp, lidar_spawn_point, attach_to=vehicle)
        lidar_sensor.listen(lambda data: save_lidar_data(data))
        print("LIDAR sensor attached")

        # Add RADAR sensor
        radar_bp = blueprint_library.find('sensor.other.radar')
        radar_spawn_point = carla.Transform(carla.Location(x=2.0, z=1.0))
        radar_sensor = world.spawn_actor(radar_bp, radar_spawn_point, attach_to=vehicle)
        radar_sensor.listen(lambda data: save_radar_data(data))
        print("RADAR sensor attached")

        # Add Collision sensor
        collision_bp = blueprint_library.find('sensor.other.collision')
        collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)
        collision_sensor.listen(lambda data: save_collision_data(data))
        print("Collision sensor attached")

        # Move vehicle forward with higher throttle to cause a collision
        vehicle.apply_control(carla.VehicleControl(throttle=0.7, steer=0.0))
        print("Vehicle moving forward to collide with obstacle")

        # Run simulation for 15 seconds to allow collision
        time.sleep(15)

    except Exception as e:
        print(f"Error in simulation: {e}")
        raise

    finally:
        # Cleanup
        print("Cleaning up actors and sensors...")
        for sensor in [lidar_sensor, radar_sensor, collision_sensor]:
            if sensor and sensor.is_alive:
                sensor.stop()
                sensor.destroy()
        for actor in [vehicle, obstacle]:
            if actor and actor.is_alive:
                actor.destroy()
        print("Cleanup complete")

if __name__ == '__main__':
    main()