FROM python:3.11-slim

WORKDIR /app

# Copy the lightweight node runner
COPY node.py /app/node.py

# Expose internal mesh communication port
EXPOSE 8080

# Run in unbuffered mode for real-time streaming output logs
ENTRYPOINT ["python", "-u", "/app/node.py"]
