# Use an official Python runtime as a base image
FROM python:3.9-slim-buster

# Set working directory
# WORKDIR /app
WORKDIR /movie_review_model_api


# Copy the requirements file   into the container at /app
ADD ./movie_review_model_api /movie_review_model_api/

# Install dependencies
COPY requirements/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files
COPY . .

# Make port 8001 available to the world outside this container
EXPOSE 8001

# Expose port and run FastAPI
# CMD ["uvicorn", "movie_review_model_api.app.main:app", "--host", "0.0.0.0", "--port", "8001"]

# Run the application using uvicorn
CMD ["python", "app/main.py"]
