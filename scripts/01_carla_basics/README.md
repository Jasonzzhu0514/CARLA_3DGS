# Stage 01: CARLA basics

Check the configured CARLA installation and Python API:

```bash
./scripts/01_carla_basics/check_environment.sh
```

Start CARLA and keep the terminal open:

```bash
./scripts/01_carla_basics/start_carla.sh
```

In another terminal, spawn a stationary vehicle:

```bash
./scripts/01_carla_basics/run_spawn_vehicle.sh
```

Press Enter or `Ctrl+C` in the vehicle terminal to remove the actor. Run
`run_spawn_vehicle.sh --help` to view client options.
