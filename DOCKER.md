# Docker Setup for AIProctor

## Prerequisites

- Docker Engine
- Docker Compose v2
- A working local environment with enough resources for Python and MongoDB containers

## Architecture

This application is a small monolith split into three runtime pieces:

- Backend API: FastAPI service running on port 8000
- Frontend: React + Vite app running on port 5173
- Database: MongoDB container for persistence

The backend reads configuration from environment variables and connects to MongoDB using the service name `mongo` inside the Docker network.

## Configuration

Copy the example environment file before running the app:

```bash
cp .env.example .env
```

Edit `.env` with the same values you need for your environment. Do not commit secrets. Default values are safe placeholders for local development.

## Start the app

```bash
docker compose up -d --build
```

## Stop the app

```bash
docker compose down
```

## View logs

```bash
docker compose logs -f
```

## Check running status

```bash
docker compose ps
```

## Database notes

MongoDB data is stored in the Docker volume `mongo-data`, so it survives container restarts and recreation unless you explicitly remove the volume.

The app does not run automatic destructive resets. If database migrations are ever introduced, they should be executed intentionally via dedicated commands.

## Common troubleshooting

### Port already in use

Check whether another service is already using port 8000 or 5173 and stop it or change the host port mapping in `docker-compose.yml`.

### Database connection failure

Verify MongoDB is healthy and that `MONGO_URL` points to `mongo` instead of `localhost` when running inside Docker.

### Container not starting

Inspect the container logs:

```bash
docker compose logs api
```

and:

```bash
docker compose logs mongo
```

### Missing environment variables

Make sure `.env` exists and includes required settings such as `PORT`, `MONGO_URL`, and `VITE_API_URL`.

### Frontend cannot reach backend

Confirm the React app is configured with `VITE_API_URL` pointing to the API service, and that the API is listening on `0.0.0.0:8000`.

## Useful commands

```bash
docker compose build

docker compose up
docker compose up -d
docker compose down
docker compose logs
docker compose ps
```
