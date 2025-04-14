import sys
import os
import numpy as np
import time
import random
from queue import Queue

# Set up CARLA path
sys.path.append(
    'D:\\CARLA\\WindowsNoEditor\\PythonAPI\\carla\\dist\\carla-0.9.11-py%d.%d-win-amd64.egg' % (
        sys.version_info.major, sys.version_info.minor
    )
)

import carla

def connect_to_carla():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    return client, world

# Set weather conditions with synchronization
def set_weather(world, weather_condition, weather_name):
    world.set_weather(weather_condition)
    print(f"Weather set to {weather_name}: {weather_condition}")
    world.wait_for_tick()
    time.sleep(3.0)

def process_image(image, image_queue):
    image_queue.put(image)

# Generate dataset with different weather conditions
def generate_dataset(world, output_dir='dataset', num_samples=100):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    weather_conditions_with_names = [
        (carla.WeatherParameters(
            cloudiness=80.0,
            precipitation=80.0,
            sun_altitude_angle=60.0,
            fog_density=0.0,
            wind_intensity=20.0
        ), 'rain'),

        (carla.WeatherParameters(
            cloudiness=80.0,
            precipitation=0.0,
            sun_altitude_angle=10.0,
            fog_density=0.7,
            wind_intensity=5.0
        ), 'fog'),

        (carla.WeatherParameters(
            cloudiness=60.0,
            precipitation=0.0,
            sun_altitude_angle=-15.0,
            fog_density=0.3,
            wind_intensity=5.0
        ), 'night'),

        (carla.WeatherParameters(
            cloudiness=20.0,
            precipitation=0.0,
            sun_altitude_angle=70.0,
            fog_density=0.0,
            wind_intensity=5.0
        ), 'day'),
    ]

    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('model3')[0]
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '224')
    camera_bp.set_attribute('image_size_y', '224')

    spawn_points = world.get_map().get_spawn_points()

    for weather, condition_name in weather_conditions_with_names:
        set_weather(world, weather, condition_name)

        condition_dir = os.path.join(output_dir, condition_name)
        os.makedirs(condition_dir, exist_ok=True)

        for i in range(num_samples):
            try:
                spawn_point = random.choice(spawn_points)
                vehicle = world.spawn_actor(vehicle_bp, spawn_point)
                camera = world.spawn_actor(
                    camera_bp,
                    carla.Transform(carla.Location(x=2.5, z=1.2)),
                    attach_to=vehicle
                )

                image_queue = Queue()
                camera.listen(lambda image: process_image(image, image_queue))

                world.tick()
                time.sleep(0.1)
                image = image_queue.get()

                # Save image
                image_path = os.path.join(condition_dir, f'{i}.png')
                image.save_to_disk(image_path)
                print(f"Saved image: {image_path}")

                # Stop and destroy objects
                camera.stop()
                vehicle.destroy()
                time.sleep(0.1)

            except Exception as e:
                print(f"Error during sample {i} for {condition_name}: {e}")
                if 'vehicle' in locals():
                    vehicle.destroy()
                if 'camera' in locals():
                    camera.stop()
                continue

    print("Dataset generation completed!")

def main():
    try:
        client, world = connect_to_carla()
        generate_dataset(world)
    finally:
        print("Cleanup completed.")

if __name__ == '__main__':
    main()
