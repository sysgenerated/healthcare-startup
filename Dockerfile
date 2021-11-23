# Load the base python3 image
FROM python:3

# Set a working directory
WORKDIR /app

# Repo is small so just copy everything to container
COPY . .

# Update packages
RUN apt-get -y update

# Install all API requirements
RUN pip3 install -r requirements.txt

# Allow traffic on port 8080 (changed from 81 to be outside of reserved port range)
EXPOSE 8080

# Run the flask python file to start a web server
CMD ["python", "./api.py"]
